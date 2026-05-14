import os
import asyncio
import sqlite3
from typing import Annotated, TypedDict

import ollama
from mcp import ClientSession
from mcp.client.sse import sse_client
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton


MCP_URL = os.getenv("MCP_URL", "http://mcp-k8s:8080/sse")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL = "qwen3.6:27b"

bot = telebot.TeleBot(TG_TOKEN)

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The conversation history"]

# --- MCP TOOL BRIDGE ---
async def get_mcp_tools():
    """Fetch tools from the MCP server and wrap them for LangGraph"""
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            # Helper to wrap MCP tools into LangChain tools
            def create_tool(mcp_t):
                @tool(name=mcp_t.name)
                def k8s_tool(args: dict):
                    # In a real setup, this would call session.call_tool
                    # For this script, we return the plan for user approval
                    return f"Plan to execute {mcp_t.name} with {args}"
                k8s_tool.description = mcp_t.description
                return k8s_tool

            return [create_tool(t) for t in mcp_tools.tools]

# --- AGENT LOGIC ---
def call_model(state: AgentState):
    messages = state['messages']
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": m.type, "content": m.content} for m in messages],
        host=OLLAMA_URL,
        options={'think': False}
    )
    return {"messages": [AIMessage(content=response['message']['content'])]}

# --- HUMAN IN THE LOOP (TELEGRAM) ---
def ask_human_via_telegram(state: AgentState):
    last_message = state['messages'][-1].content
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("✅ Approve", callback_data="approve"))
    markup.add(InlineKeyboardButton("❌ Reject", callback_data="reject"))

    bot.send_message(CHAT_ID, f"🤖 **AI PROPOSAL:**\n\n{last_message}", reply_markup=markup, parse_mode="Markdown")
    print("Waiting for Telegram approval...")

# --- GRAPH CONSTRUCTION ---
def build_graph(tools):
    # Persistence for HITL
    memory = SqliteSaver(sqlite3.connect("/app/data/agent_state.db", check_same_thread=False))

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", ToolNode(tools))

    workflow.set_entry_point("agent")

    # Interrupt before 'action' node
    graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["action"]
    )
    return graph

# --- TELEGRAM CALLBACK HANDLER ---
# This listens for your click and "resumes" the graph
async def run_bot_loop(graph, config):
    @bot.callback_query_handler(func=lambda call: True)
    def callback_query(call):
        if call.data == "approve":
            bot.answer_callback_query(call.id, "Action Approved!")
            # Resume the graph
            asyncio.run(graph.ainvoke(None, config))
        else:
            bot.answer_callback_query(call.id, "Action Cancelled.")

    bot.infinity_polling()


async def main():
    tools = await get_mcp_tools()
    graph = build_graph(tools)
    config = {"configurable": {"thread_id": "sre_session_1"}}

    # Start the initial run
    initial_input = {"messages": [HumanMessage(content="Check for crashing pods and suggest a fix.")]}
    async for event in graph.astream(initial_input, config):
        for node, state in event.items():
            if node == "agent":
                ask_human_via_telegram(state)

    # Start Telegram listener
    await run_bot_loop(graph, config)


if __name__ == "__main__":
    asyncio.run(main())
