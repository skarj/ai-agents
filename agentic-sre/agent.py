import os
import asyncio
import sqlite3
from typing import Annotated, TypedDict

import ollama
from mcp import ClientSession
from mcp.client.sse import sse_client
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from telebot.async_telebot import AsyncTeleBot
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# Configuration
MCP_URL = os.getenv("MCP_URL", "http://mcp-k8s:8080/sse")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL = "qwen2.5:latest"

bot = AsyncTeleBot(TG_TOKEN)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The conversation history"]

# --- DYNAMIC MCP TOOL BUILDER ---
async def create_mcp_tools():
    """Builds tools that maintain their own session context when called."""
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            tools = []
            for t in mcp_tools.tools:
                # We define the function inside a factory to capture tool name
                def create_fn(tool_name):
                    async def func(args: dict):
                        async with sse_client(MCP_URL) as (r, w):
                            async with ClientSession(r, w) as sess:
                                await sess.initialize()
                                res = await sess.call_tool(tool_name, args)
                                return str(res.content)
                    return func

                tools.append(StructuredTool.from_function(
                    coroutine=create_fn(t.name),
                    name=t.name,
                    description=t.description
                ))
            return tools

# --- AGENT LOGIC ---
async def call_model(state: AgentState, config):
    messages = []
    for m in state['messages']:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        if isinstance(m, ToolMessage): role = "tool"
        messages.append({"role": role, "content": m.content})

    # Call Ollama with tools
    response = await asyncio.to_thread(
        ollama.chat,
        model=MODEL,
        messages=messages,
    )

    # Logic to translate Ollama response to AIMessage with tool_calls
    content = response['message'].get('content', "")
    # Check if the model is trying to call a tool (Model dependent logic)
    # For this fix, we assume standard AI Message return
    return {"messages": [AIMessage(content=content)]}

# --- GRAPH BUILDER ---
def build_agent_graph(tools, checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("agent", call_model)
    builder.add_node("action", ToolNode(tools))

    builder.set_entry_point("agent")

    def router(state):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "action"
        return END

    builder.add_conditional_edges("agent", router)
    builder.add_edge("action", "agent")

    # The magic: Interrupt before 'action' to allow SRE approval
    return builder.compile(checkpointer=checkpointer, interrupt_before=["action"])

# --- TELEGRAM HANDLERS ---
def register_bot_handlers(graph, config):
    @bot.callback_query_handler(func=lambda call: True)
    async def handle_query(call):
        if call.data == "approve":
            await bot.answer_callback_query(call.id, "🚀 Execution Approved.")
            # Resume the graph with None input to move past the interrupt
            async for event in graph.astream(None, config, stream_mode="values"):
                await process_graph_event(event)
        else:
            await bot.answer_callback_query(call.id, "❌ Action Cancelled.")
            await bot.send_message(CHAT_ID, "Fix discarded by operator.")

async def process_graph_event(event):
    """Utility to send updates to Telegram as they happen."""
    if "messages" in event:
        last_msg = event["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            # Check if this is a proposal needing approval
            if "proposal" in last_msg.content.lower():
                markup = InlineKeyboardMarkup()
                markup.add(InlineKeyboardButton("✅ Approve", callback_data="approve"))
                markup.add(InlineKeyboardButton("❌ Reject", callback_data="reject"))
                await bot.send_message(CHAT_ID, f"🚨 **PROPOSAL:**\n{last_msg.content}",
                                     reply_markup=markup, parse_mode="Markdown")

# --- MAIN RUNNER ---
async def main():
    print("🛰️ Initializing SRE Agent...")
    tools = await create_mcp_tools()

    # Persistent SQLite Connection
    conn = sqlite3.connect("agent_state.db", check_same_thread=False)
    saver = AsyncSqliteSaver(conn)

    graph = build_agent_graph(tools, saver)
    config = {"configurable": {"thread_id": "sre_session_v1"}}

    register_bot_handlers(graph, config)

    # Initial Prompt
    print("🤖 Agent Online. Monitoring Cluster...")
    initial_input = {"messages": [HumanMessage(content="Check cluster health and propose fixes.")]}

    # Start the graph
    async for event in graph.astream(initial_input, config, stream_mode="values"):
        await process_graph_event(event)

    # Keep the bot polling alive
    await bot.polling(non_stop=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
