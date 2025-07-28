import json
import uuid
import asyncio
import gradio as gr
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
# ─────────── Hard‑coded secrets (DEV ONLY – remove before publishing) ──────────
load_dotenv()
OPENAI_API_KEY   = os.getenv('OPEN_API_KEY')
NOTION_MCP_TOKEN = os.getenv('NOTION_MCP_TOKEN')
NOTION_VERSION   = os.getenv('NOTION_VERSION')

# ───────────────────────── build LangGraph app once ───────────────────────────
async def build_app():
    notion_cfg = {
        "notion": {
            "command": "npx",
            "args": ["-y", "@notionhq/notion-mcp-server"],
            "transport": "stdio",
            "env": {
                "OPENAPI_MCP_HEADERS": json.dumps(
                    {
                        "Authorization": f"Bearer {NOTION_MCP_TOKEN}",
                        "Notion-Version": NOTION_VERSION,
                    }
                )
            },
        }
    }
    client = MultiServerMCPClient(notion_cfg)
    notion_tools = await client.get_tools()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0,
    ).bind_tools(notion_tools)

    async def agent_node(state: MessagesState):
        msgs = state["messages"]
        ai_msg = await llm.ainvoke(msgs)
        return {"messages": msgs + [ai_msg]}

    tool_node = ToolNode(notion_tools)

    wf = StateGraph(MessagesState)
    wf.add_node("agent", agent_node)
    wf.add_node("tools", tool_node)
    wf.add_edge(START, "agent")

    def need_tool(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    wf.add_conditional_edges("agent", need_tool, {"tools": "tools", END: END})
    wf.add_edge("tools", "agent")

    return wf.compile(checkpointer=MemorySaver())

APP = asyncio.run(build_app())

# ──────────────────────────────── Chat handlers ───────────────────────────────
async def respond(message, ui_history, thread_id):
    """
    Append user+assistant messages to ui_history (list of {role, content}).
    LangGraph memory is handled via thread_id + MemorySaver.
    """
    if ui_history is None:
        ui_history = []

    if not message:
        return "", ui_history  # nothing to do

    # Add user message
    ui_history.append({"role": "user", "content": message})

    # Run graph (previous context auto-loaded by MemorySaver using thread_id)
    input_messages = [HumanMessage(content=message)]
    result = await APP.ainvoke(
        {"messages": input_messages},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Extract last AI message
    ai_response = "No response."
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            ai_response = m.content
            break

    ui_history.append({"role": "assistant", "content": ai_response})
    return "", ui_history

def reset_chat():
    """Clear UI history and start a fresh thread (new memory)."""
    return [], str(uuid.uuid4())

# ─────────────────────────────── Build Gradio UI ──────────────────────────────
with gr.Blocks(css="""
#chatbot_interface {background:#f5f5f5; padding:15px; border-radius:10px;}
""") as demo:
    gr.Markdown("# 🔗 Notion MCP Agent\nChat with your Notion workspace (dev mode).")

    ui_history = gr.State([])                  # list of {"role","content"}
    thread_id = gr.State(str(uuid.uuid4()))    # new thread for MemorySaver

    chatbot = gr.Chatbot(label="Chat", elem_id="chatbot_interface", type="messages")
    msg = gr.Textbox(label="Enter your query:", lines=2, placeholder="Ask something about your Notion data…")
    send_btn = gr.Button("Send")
    reset_btn = gr.Button("Reset Conversation")

    send_btn.click(
        respond,
        inputs=[msg, ui_history, thread_id],
        outputs=[msg, chatbot],
    )
    msg.submit(
        respond,
        inputs=[msg, ui_history, thread_id],
        outputs=[msg, chatbot],
    )

    reset_btn.click(
        reset_chat,
        outputs=[ui_history, thread_id],
    ).then(
    lambda h: gr.update(value=h),   # h will be the (now empty) ui_history
    inputs=ui_history,
    outputs=chatbot)

if __name__ == "__main__":
    demo.queue().launch()
