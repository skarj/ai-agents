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
                def create_tool_fn(t_name=t.name):
                    async def func(**kwargs):
                        actual_args = kwargs.get('v__args', kwargs)
                        logger.info(f"🛠️ MCP CALL: {t_name} with {actual_args}")
                        try:
                            async with sse_client(MCP_URL) as (r, w):
                                async with ClientSession(r, w) as sess:
                                    await sess.initialize()
                                    res = await sess.call_tool(t_name, actual_args)
                                    # Handle structured content from MCP
                                    if hasattr(res, 'content'):
                                        return str(res.content)
                                    return str(res)
                        except Exception as e:
                            error_msg = f"Error executing {t_name}: {str(e)}"
                            logger.error(f"❌ {error_msg}")
                            return error_msg
                    return func

                tools_list.append(StructuredTool.from_function(
                    coroutine=create_tool_fn(t.name),
                    name=t.name,
                    description=t.description
                ))

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

        msg_obj = {"role": role, "content": m.content or ""}
        if isinstance(m, ToolMessage):
            msg_obj["tool_call_id"] = getattr(m, 'tool_call_id', None)
        formatted_messages.append(msg_obj)

    logger.info(f"🧠 Asking {MODEL} for next steps...")
    try:
        response = await ollama_client.chat(
            model=MODEL,
            messages=formatted_messages,
            tools=tools_for_ollama
        )

        msg = response.get('message', {})
        content = msg.get('content', '')
        raw_tool_calls = msg.get('tool_calls', []) or []

        tool_calls = []
        for tc in raw_tool_calls:
            if hasattr(tc, 'function'):
                tool_calls.append({
                    "name": tc.function.name,
                    "args": tc.function.arguments,
                    "id": getattr(tc, 'id', f"call_{tc.function.name}")
                })
            elif isinstance(tc, dict) and 'function' in tc:
                tool_calls.append({
                    "name": tc['function'].get('name'),
                    "args": tc['function'].get('arguments'),
                    "id": tc.get('id', f"call_{tc['function'].get('name')}")
                })

        return {"messages": [AIMessage(content=content, tool_calls=tool_calls)]}
    except Exception as e:
        logger.error(f"❌ LLM Error: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"Error contacting LLM provider: {str(e)}")]}

# --- GRAPH ORCHESTRATION ---
def build_sre_graph(tools_list, tools_for_ollama, saver):
    workflow = StateGraph(AgentState)

    async def agent_node(state):
        return await call_model(state, tools_for_ollama)

    workflow.add_node("agent", agent_node)
    workflow.add_node("action", ToolNode(tools_list))

    workflow.set_entry_point("agent")

    def should_continue(state):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "action"
        return END

    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("action", "agent")

    return workflow.compile(checkpointer=saver, interrupt_before=["action"])

# --- TELEGRAM COMMUNICATION ---
async def notify_telegram(event):
    """Processes graph updates and sends alerts to Telegram."""
    if "messages" not in event:
        return

    last_msg = event["messages"][-1]

    if isinstance(last_msg, AIMessage):
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            t_names = [tc.get('name', 'unknown') for tc in last_msg.tool_calls]
            markup = InlineKeyboardMarkup()
            markup.add(InlineKeyboardButton("✅ Approve Execution", callback_data="approve"))
            markup.add(InlineKeyboardButton("❌ Reject", callback_data="reject"))

            # Content might be empty if the LLM only provided tool calls
            proposal = last_msg.content or "The agent has identified a required tool execution."
            text = f"🚨 **SRE ACTION REQUIRED**\n\n**Proposal:** {proposal}\n\n**Tools to run:** `{t_names}`"
            await bot.send_message(CHAT_ID, text, reply_markup=markup, parse_mode="Markdown")
        elif last_msg.content:
            await bot.send_message(CHAT_ID, f"ℹ️ **Agent Report:**\n{last_msg.content}")

    elif isinstance(last_msg, ToolMessage):
        # Result of execution
        short_res = str(last_msg.content)[:800] + ("..." if len(str(last_msg.content)) > 800 else "")
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

        @bot.callback_query_handler(func=lambda call: True)
        async def handle_approval(call):
            if call.data == "approve":
                await bot.answer_callback_query(call.id, "Executing...")
                async for event in graph.astream(None, config, stream_mode="values"):
                    await notify_telegram(event)
            else:
                await bot.answer_callback_query(call.id, "Aborted.")
                await bot.send_message(CHAT_ID, "🛑 **Manual Override:** Action cancelled.")

        logger.info("🔍 Performing initial cluster health check...")
        prompt = "Review the ai-agents namespace. Identify any pods not in 'Running' state. Propose and execute fixes only after approval."
        initial_input = {"messages": [HumanMessage(content=prompt)]}

        async for event in graph.astream(initial_input, config, stream_mode="values"):
            await notify_telegram(event)

        logger.info("🤖 System Idle. Waiting for Telegram or Kubernetes events...")
        await bot.polling(non_stop=True, interval=3, timeout=30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested.")
