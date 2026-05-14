import os
import asyncio
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

# --- CONFIGURATION ---
MCP_URL = os.getenv("MCP_URL", "http://mcp-k8s:8080/sse")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL = "qwen2.5:latest"
DB_PATH = "agent_state.db"

bot = AsyncTeleBot(TG_TOKEN)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The conversation history"]

# --- DYNAMIC MCP TOOL BUILDER ---
async def create_mcp_tools():
    """Builds tools that connect to MCP SSE endpoint."""
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            tools = []
            for t in mcp_tools.tools:
                def create_fn(tool_name):
                    async def func(args: dict):
                        # Each tool call creates its own temporary session to remain stateless
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
async def call_model(state: AgentState):
    """Bridge between LangGraph messages and Ollama API."""
    messages = []
    for m in state['messages']:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        if isinstance(m, ToolMessage): role = "tool"
        messages.append({"role": role, "content": m.content})

    # Execute request via thread pool to avoid blocking the event loop
    response = await asyncio.to_thread(
        ollama.chat,
        model=MODEL,
        messages=messages,
        host=OLLAMA_URL
    )

    content = response['message'].get('content', "")
    return {"messages": [AIMessage(content=content)]}

# --- GRAPH BUILDER ---
def build_agent_graph(tools, checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("agent", call_model)
    builder.add_node("action", ToolNode(tools))

    builder.set_entry_point("agent")

    def router(state):
        last_message = state["messages"][-1]
        # Check for tool_calls attribute (requires compatible model/prompting)
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "action"
        return END

    builder.add_conditional_edges("agent", router)
    builder.add_edge("action", "agent")

    # Set the 'interrupt' point for Human-in-the-loop SRE approval
    return builder.compile(checkpointer=checkpointer, interrupt_before=["action"])

# --- TELEGRAM LOGIC ---
async def process_graph_event(event):
    """Parses graph updates and notifies via Telegram."""
    if "messages" in event:
        last_msg = event["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            # Trigger buttons if the agent proposes a fix
            if any(keyword in last_msg.content.lower() for keyword in ["propose", "fix", "execute"]):
                markup = InlineKeyboardMarkup()
                markup.add(InlineKeyboardButton("✅ Approve", callback_data="approve"))
                markup.add(InlineKeyboardButton("❌ Reject", callback_data="reject"))
                await bot.send_message(CHAT_ID, f"🚨 **SRE PROPOSAL:**\n\n{last_msg.content}",
                                     reply_markup=markup, parse_mode="Markdown")
            else:
                await bot.send_message(CHAT_ID, f"ℹ️ **Agent Update:**\n{last_msg.content}")

def register_bot_handlers(graph, config):
    @bot.callback_query_handler(func=lambda call: True)
    async def handle_query(call):
        if call.data == "approve":
            await bot.answer_callback_query(call.id, "🚀 Execution Approved.")
            # Resume graph execution
            async for event in graph.astream(None, config, stream_mode="values"):
                await process_graph_event(event)
        else:
            await bot.answer_callback_query(call.id, "🛑 Action Aborted.")
            await bot.send_message(CHAT_ID, "Manual override: The proposed action was cancelled.")

# --- MAIN RUNNER ---
async def main():
    print("🛰️ Connecting to MCP SSE endpoint...")
    try:
        tools = await create_mcp_tools()
        print(f"✅ Loaded {len(tools)} tools.")
    except Exception as e:
        print(f"❌ MCP Connection Error: {e}")
        return

    # from_conn_string manages the aiosqlite connection context internally
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:

        graph = build_agent_graph(tools, saver)
        config = {"configurable": {"thread_id": "sre_session_1"}}

        register_bot_handlers(graph, config)

        print("🤖 Agent Online. Monitoring Cluster...")

        # Initial diagnostic run
        initial_input = {"messages": [HumanMessage(content="Scan the cluster for issues and propose a remediation plan.")]}

        async for event in graph.astream(initial_input, config, stream_mode="values"):
            await process_graph_event(event)

        # Non-blocking Telegram bot polling
        await bot.polling(non_stop=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 SRE Agent shutting down gracefully.")
