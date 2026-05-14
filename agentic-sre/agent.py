import os
import asyncio
import sys
import logging
from typing import Annotated, TypedDict

from ollama import AsyncClient
from mcp import ClientSession
from mcp.client.sse import sse_client
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from telebot.async_telebot import AsyncTeleBot
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
MCP_URL = os.getenv("MCP_URL", "http://mcp-k8s:8080/sse")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL = os.getenv("MODEL", "qwen3.6:27b")
DB_PATH = "agent_state.db"

bot = AsyncTeleBot(TG_TOKEN)
ollama_client = AsyncClient(host=OLLAMA_URL)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The conversation history"]

# --- DYNAMIC MCP TOOL BUILDER ---
async def create_mcp_tools():
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            tools = []
            for t in mcp_tools.tools:
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
async def call_model(state: AgentState):
    messages = []
    for m in state['messages']:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        if isinstance(m, ToolMessage): role = "tool"
        messages.append({"role": role, "content": m.content})

    try:
        response = await ollama_client.chat(model=MODEL, messages=messages)
        content = response['message'].get('content', "")
        return {"messages": [AIMessage(content=content)]}
    except Exception as e:
        logger.error(f"Ollama Error: {e}")
        return {"messages": [AIMessage(content="Model failed to respond.")]}

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
    return builder.compile(checkpointer=checkpointer, interrupt_before=["action"])

# --- TELEGRAM LOGIC ---
async def process_graph_event(event):
    if "messages" in event:
        last_msg = event["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            logger.info(f"AI Message sent to TG: {last_msg.content[:50]}...")
            if any(kw in last_msg.content.lower() for kw in ["propose", "fix", "execute", "plan"]):
                markup = InlineKeyboardMarkup()
                markup.add(InlineKeyboardButton("✅ Approve", callback_data="approve"))
                markup.add(InlineKeyboardButton("❌ Reject", callback_data="reject"))
                await bot.send_message(CHAT_ID, f"🚨 **SRE PROPOSAL:**\n\n{last_msg.content}",
                                     reply_markup=markup, parse_mode="Markdown")
            else:
                await bot.send_message(CHAT_ID, f"ℹ️ {last_msg.content}")

def register_bot_handlers(graph, config):
    @bot.callback_query_handler(func=lambda call: True)
    async def handle_query(call):
        if call.data == "approve":
            await bot.answer_callback_query(call.id, "Executing Action...")
            async for event in graph.astream(None, config, stream_mode="values"):
                await process_graph_event(event)
        else:
            await bot.answer_callback_query(call.id, "Action Rejected.")
            await bot.send_message(CHAT_ID, "Manual Override: Action Aborted.")

# --- MAIN RUNNER ---
async def main():
    logger.info("🛰️ Initializing MCP Connection...")
    try:
        tools = await create_mcp_tools()
        logger.info(f"✅ Loaded {len(tools)} tools.")
    except Exception as e:
        logger.error(f"❌ MCP Connection Error: {e}")
        return

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        graph = build_agent_graph(tools, saver)
        config = {"configurable": {"thread_id": "sre_session_1"}}
        register_bot_handlers(graph, config)

        logger.info("🤖 Agent Online. Monitoring Cluster...")

        # Start initial scan as a background task to keep it from blocking Telegram polling startup
        initial_input = {"messages": [HumanMessage(content="Scan cluster for health issues and propose fixes.")]}

        async def run_initial_scan():
            async for event in graph.astream(initial_input, config, stream_mode="values"):
                await process_graph_event(event)

        # Run both the bot and the scan
        try:
            await asyncio.gather(
                bot.polling(non_stop=True, skip_pending=True, timeout=60),
                run_initial_scan()
            )
        except Exception as e:
            logger.error(f"Runtime Exception: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutdown initiated.")
