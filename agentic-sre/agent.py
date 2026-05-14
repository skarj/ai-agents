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

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- ENVIRONMENT CONFIG ---
MCP_URL = os.getenv("MCP_URL", "http://mcp-k8s:8080/sse")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL = os.getenv("MODEL", "qwen2.5:14b")
DB_PATH = "/app/data/agent_state.db"

# --- GLOBAL CLIENTS ---
bot = AsyncTeleBot(TG_TOKEN)
ollama_client = AsyncClient(host=OLLAMA_URL, timeout=180.0)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The conversation history"]

# --- DYNAMIC MCP TOOL DISCOVERY ---
async def fetch_mcp_tools():
    """Connects to the MCP server and translates Kubernetes tools for the AI."""
    logger.info(f"🔗 Connecting to MCP at {MCP_URL}...")
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            tools_list = []
            ollama_format_tools = []

            for t in mcp_tools.tools:
                # 1. Create the executable function for LangGraph
                def create_tool_fn(t_name=t.name):
                    async def func(args: dict):
                        logger.info(f"🛠️ MCP CALL: {t_name} with {args}")
                        async with sse_client(MCP_URL) as (r, w):
                            async with ClientSession(r, w) as sess:
                                await sess.initialize()
                                res = await sess.call_tool(t_name, args)
                                return str(res.content)
                    return func

                # 2. Add to LangChain StructuredTool list
                tools_list.append(StructuredTool.from_function(
                    coroutine=create_tool_fn(t.name),
                    name=t.name,
                    description=t.description
                ))

                # 3. Format for Ollama's tool-calling API
                ollama_format_tools.append({
                    'type': 'function',
                    'function': {
                        'name': t.name,
                        'description': t.description,
                        'parameters': t.inputSchema
                    }
                })

            return tools_list, ollama_format_tools

# --- AI REASONING NODE ---
async def call_model(state: AgentState, tools_for_ollama: list):
    """Feeds the cluster state to the LLM and retrieves the next action."""
    formatted_messages = []
    for m in state['messages']:
        if isinstance(m, HumanMessage): role = "user"
        elif isinstance(m, ToolMessage): role = "tool"
        else: role = "assistant"

        msg_obj = {"role": role, "content": m.content}
        # Pass tool results back as specific role objects
        if isinstance(m, ToolMessage):
            # Ensure the tool_call_id is preserved so the LLM can link result to request
            msg_obj["tool_call_id"] = getattr(m, 'tool_call_id', None)
        formatted_messages.append(msg_obj)

    logger.info(f"🧠 Asking {MODEL} for next steps...")
    try:
        response = await ollama_client.chat(
            model=MODEL,
            messages=formatted_messages,
            tools=tools_for_ollama
        )

        msg = response['message']
        content = msg.get('content', '')
        tool_calls = msg.get('tool_calls', [])

        return {"messages": [AIMessage(content=content, tool_calls=tool_calls)]}
    except Exception as e:
        logger.error(f"❌ LLM Error: {e}")
        return {"messages": [AIMessage(content=f"Error contacting LLM provider: {str(e)}")]}

# --- GRAPH ORCHESTRATION ---
def build_sre_graph(tools_list, tools_for_ollama, saver):
    workflow = StateGraph(AgentState)

    async def agent_node(state):
        return await call_model(state, tools_for_ollama)

    workflow.add_node("agent", agent_node)
    workflow.add_node("action", ToolNode(tools_list))

    workflow.set_entry_point("agent")

    # Routing logic: Action (Tool) or End (Talk to Human)
    def should_continue(state):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "action"
        return END

    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("action", "agent")

    # HITL: Always stop before the cluster is modified
    return workflow.compile(checkpointer=saver, interrupt_before=["action"])

# --- TELEGRAM COMMUNICATION ---
async def notify_telegram(event):
    """Processes graph updates and sends alerts to Telegram."""
    if "messages" not in event:
        return

    last_msg = event["messages"][-1]

    # 1. AI wants to perform an action
    if isinstance(last_msg, AIMessage):
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            t_names = [tc['function']['name'] for tc in last_msg.tool_calls]
            markup = InlineKeyboardMarkup()
            markup.add(InlineKeyboardButton("✅ Approve Execution", callback_data="approve"))
            markup.add(InlineKeyboardButton("❌ Reject", callback_data="reject"))

            text = f"🚨 **SRE ACTION REQUIRED**\n\n**Proposal:** {last_msg.content}\n\n**Tools to run:** `{t_names}`"
            await bot.send_message(CHAT_ID, text, reply_markup=markup, parse_mode="Markdown")
        elif last_msg.content:
            # AI is just providing information
            await bot.send_message(CHAT_ID, f"ℹ️ **Agent Report:**\n{last_msg.content}")

    # 2. A tool just finished running
    elif isinstance(last_msg, ToolMessage):
        # Truncate long outputs for readability
        short_res = last_msg.content[:800] + ("..." if len(last_msg.content) > 800 else "")
        await bot.send_message(CHAT_ID, f"📦 **Tool Output:**\n```\n{short_res}\n```", parse_mode="Markdown")

# --- MAIN LOOP ---
async def main():
    logger.info("🚀 Starting Agentic SRE Node...")

    try:
        tools_list, tools_for_ollama = await fetch_mcp_tools()
        logger.info(f"✅ Discovered {len(tools_list)} cluster tools.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MCP tools: {e}")
        return

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        graph = build_sre_graph(tools_list, tools_for_ollama, saver)
        config = {"configurable": {"thread_id": "sre_prod_v1"}}

        # Handle button clicks
        @bot.callback_query_handler(func=lambda call: True)
        async def handle_approval(call):
            if call.data == "approve":
                await bot.answer_callback_query(call.id, "Executing...")
                # RESUME the graph: passing None indicates continuing from the interrupt
                async for event in graph.astream(None, config, stream_mode="values"):
                    await notify_telegram(event)
            else:
                await bot.answer_callback_query(call.id, "Aborted.")
                await bot.send_message(CHAT_ID, "🛑 **Manual Override:** Action cancelled.")

        # Initial Scan
        logger.info("🔍 Performing initial cluster health check...")
        prompt = "Review the ai-agents namespace. If any pod is not Running, use your tools to troubleshoot and propose a fix."
        initial_input = {"messages": [HumanMessage(content=prompt)]}

        async for event in graph.astream(initial_input, config, stream_mode="values"):
            await notify_telegram(event)

        logger.info("🤖 System Idle. Waiting for Telegram or Kubernetes events...")
        await bot.polling(non_stop=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested.")
