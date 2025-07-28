# main.py
from __future__ import annotations

import os
import json
import uuid
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    Request,
)
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from jose import jwt, JWTError
from pydantic import BaseModel

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState,
)
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────── Logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("notion‑mcp‑api")

# ─────────────────────────── ENV vars ───────────────────────────
load_dotenv() 
OPENAI_API_KEY   = os.getenv("OPEN_API_KEY")
NOTION_MCP_TOKEN = os.getenv("NOTION_MCP_TOKEN")
NOTION_VERSION   = os.getenv("NOTION_VERSION", "2022-06-28")
JWT_SECRET       = os.getenv("JWT_SECRET")         # set in prod!
ACCESS_EXPIRE_M  = int(os.getenv("JWT_EXP_MIN", 30))

for name, value in {
    "OPEN_API_KEY": OPENAI_API_KEY,
    "NOTION_MCP_TOKEN": NOTION_MCP_TOKEN,
    "JWT_SECRET": JWT_SECRET,
}.items():
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")

# ─────────────────────────── JWT helpers ─────────────────────────
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


def create_jwt(subject: str) -> str:
    """Create a signed JWT for `subject` valid for ACCESS_EXPIRE_M minutes."""
    exp = datetime.utcnow() + timedelta(minutes=ACCESS_EXPIRE_M)
    return jwt.encode({"sub": subject, "exp": exp}, JWT_SECRET, algorithm=ALGORITHM)


def verify_jwt(token: str = Depends(oauth2_scheme)) -> str:
    """FastAPI dependency: return username if token is valid, else 401."""
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload["sub"]
    except (JWTError, KeyError):
        raise cred_exc


# ────────────────────────── User store (demo) ───────────────────

user_cred = {"arya": {"username": "arya", "password": "test123"}}


def authenticate(username: str, password: str) -> bool:
    user = user_cred.get(username)
    return bool(user and user["password"] == password)


# ─────────────────────────── Pydantic models ────────────────────
class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


# ───────────────────── LangGraph builder (async) ─────────────────
async def build_graph():
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

    llm = (
        ChatOpenAI(
            model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0
        ).bind_tools(notion_tools)
    )

    async def agent_node(state: MessagesState):
        messages = state["messages"]
        ai_msg = await llm.ainvoke(messages)
        return {"messages": messages + [ai_msg]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(notion_tools))
    graph.add_edge(START, "agent")

    def choose_next(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph.add_conditional_edges("agent", choose_next, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


# ─────────────────────────── FastAPI app ────────────────────────
app = FastAPI(
    title="Notion‑MCP Agent API",
    version="1.0.0",
    description="JWT‑protected API that chats with your Notion workspace "
    "through LangGraph and the Notion MCP toolset.",
)


@app.on_event("startup")
async def _startup():
    """Build LangGraph once at startup and store in app.state."""
    log.info("Building LangGraph agent …")
    app.state.graph = await build_graph()
    log.info("LangGraph agent ready.")


# ─────────────────────────── Auth route ─────────────────────────
@app.post("/token", response_model=TokenOut, tags=["auth"])
async def login(form: OAuth2PasswordRequestForm = Depends()):
    if not authenticate(form.username, form.password):
        raise HTTPException(status_code=400, detail="Incorrect credentials")
    return {"access_token": create_jwt(form.username)}


# ─────────────────────────── Chat route ─────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest, request: Request, user: str = Depends(verify_jwt)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    graph = request.app.state.graph
    thread_id = req.thread_id or str(uuid.uuid4())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    answer = next(
        (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        "No response.",
    )
    return ChatResponse(answer=answer, thread_id=thread_id)
