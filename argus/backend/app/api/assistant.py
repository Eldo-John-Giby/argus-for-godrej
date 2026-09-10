"""Assistant API routes — supervisor AI assistant chat interface."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.assistant.agent import AssistantAgent

router = APIRouter(prefix="/api/assistant", tags=["assistant"])

# Singleton agent instance
_agent = AssistantAgent()


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[dict]
    grounded: bool


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Send a query to the grounded supervisor assistant.

    The assistant ONLY answers from the events database via tool calling.
    It never invents information.
    """
    result = await _agent.process_query(req.query, db)

    return ChatResponse(
        response=result.text,
        tool_calls=[
            {"tool": tc.tool_name, "arguments": tc.arguments, "result": tc.result}
            for tc in result.tool_calls
        ],
        grounded=result.grounded,
    )


@router.get("/tools")
async def list_tools():
    """List available assistant tools (for UI display)."""
    return [
        {"name": name, "description": tool["description"]}
        for name, tool in _agent.tools.items()
    ]
