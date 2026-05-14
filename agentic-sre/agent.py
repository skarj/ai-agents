import os
import asyncio
from typing import Annotated, TypedDict

import ollama
from mcp import ClientSession
from mcp.client.sse import sse_client
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from telebot.async_telebot import AsyncTeleBot
from langgraph.checkpoint.sqlite import AsyncSqliteSaver
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton


MCP_URL = os.getenv("MCP_URL", "http://mcp-k8s:8080/sse")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL = "qwen3.6:27b"

bot = AsyncTeleBot(TG_TOKEN)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The conversation history"]

# --- DYNAMIC MCP TOOL BUILDER ---
async def create_mcp_tools():
    """
    Connects to MCP and builds real executable tools.
    We use StructuredTool.from_function because it handles dynamic naming better than the decorator.
    """
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            tools = []
            for t in mcp_tools.tools:
                async def tool_wrapper(args: dict, mcp_tool_name=t.name):
                    async with sse_client(MCP_URL) as (r, w):
                        async with ClientSession(r, w) as sess:
                            await sess.initialize()
                            result = await sess.call_tool(mcp_tool_name, args)
                            return str(result.content)

                tools.append(StructuredTool.from_function(
                    coroutine=tool_wrapper,
                    name=t.name,
                    description=t.description
                ))
            return tools

# --- AGENT LOGIC ---
async def call_model(state: AgentState):
    messages = state['messages']
    response = await asyncio.to_thread(
        ollama.chat,
        model=MODEL,
        messages=[{"role": m.type, "content": m.content} for m in messages],
        host=OLLAMA_URL,
        options={'think': False} # Disabling internal chain-of-thought for speed
    )
    return {"messages": [AIMessage(content=response['message']['content'])]}

# --- HUMAN IN THE LOOP ---
async def ask_human(state: AgentState):
    last_message = state['messages'][-1].content
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("✅ Approve", callback_data="approve"))
    markup.add(InlineKeyboardButton("❌ Reject", callback_data="reject"))

    await bot.send_message(CHAT_ID, f"🚨 **SRE PROPOSAL:**\n\n{last_message}", reply_markup=markup, parse_mode="Markdown")

# --- GRAPH BUILDER ---
def build_agent_graph(tools, checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("action", ToolNode(tools))

    builder.set_entry_point("agent")

    def router(state):
        if state["messages"][-1].additional_kwargs.get("tool_calls"):
            return "action"
        return END

    builder.add_conditional_edges("agent", router)
    builder.add_edge("action", "agent")

    # Use the passed-in async checkpointer
    return builder.compile(checkpointer=checkpointer, interrupt_before=["action"])

# --- MAIN ---
async def main():
    print("🛰️ Connecting to MCP and building tools...")
    tools = await create_mcp_tools()

    # 1. Initialize the Async Checkpointer using the context manager
    async with AsyncSqliteSaver.from_conn_string("/app/data/agent_state.db") as saver:

        # 2. Build graph with the async saver
        graph = build_agent_graph(tools, saver)
        config = {"configurable": {"thread_id": "sre_session_1"}}

        # Telegram callback remains async
        @bot.callback_query_handler(func=lambda call: True)
        async def handle_query(call):
            if call.data == "approve":
                await bot.answer_callback_query(call.id, "Executing fix...")
                # Use the 'saver' context to resume
                async for event in graph.astream(None, config):
                    print(f"Graph Resumed: {event}")
            else:
                await bot.answer_callback_query(call.id, "Action cancelled.")

        # 3. Start the initial monitoring run
        initial_input = {"messages": [HumanMessage(content="Scan the cluster for issues.")]}

        async for event in graph.astream(initial_input, config):
            if "__interrupt__" in event:
                # Retrieve state to send to Telegram
                state = await graph.aget_state(config)
                await ask_human(state.values)

        print("🤖 Agent and Bot are online!")
        await bot.polling()


if __name__ == "__main__":
    asyncio.run(main())
