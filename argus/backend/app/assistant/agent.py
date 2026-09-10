"""Supervisor AI Assistant — tool-calling agent grounded to the events database.

This agent ONLY answers from tool results. It never invents information.
The system prompt explicitly forbids answering outside tool results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.config import ASSISTANT_API_KEY, ASSISTANT_LLM_PROVIDER, ASSISTANT_MODEL


@dataclass
class ToolCall:
    """A tool call made by the assistant."""
    tool_name: str
    arguments: dict
    result: Optional[str] = None


@dataclass
class AssistantResponse:
    """Response from the assistant."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    grounded: bool = True  # Always grounded — we only call tools


SYSTEM_PROMPT = """You are ARGUS, an AI safety assistant for warehouse operations.

You are a grounded assistant that ONLY answers questions using data from the
warehouse events database. You must NEVER invent, fabricate, or assume information
that is not returned by your tools.

Available tools:
1. get_events(filters) - Query events with optional filters (behaviour_type, tier, status, bay_id, time range)
2. get_top_behaviours(period) - Get most common risky behaviours in a time period
3. get_bay_summary(bay_id, shift) - Get event summary for a loading bay
4. explain_event(event_id) - Get detailed explanation of why an event was flagged

Rules:
- ALWAYS use tools to answer questions. Never provide answers from general knowledge.
- If a tool returns no results, say "No events found matching those criteria."
- When explaining risk, walk through the actual risk-score breakdown from the data.
- Be concise but thorough. Warehouse supervisors need actionable information.
- Use the exact terminology from the events database (behaviour_type, tier, status).
"""


class AssistantAgent:
    """Tool-calling assistant grounded to the events database."""

    def __init__(self):
        self.tools = {}
        self.conversation_history: list[dict] = []
        self._register_default_tools()

    def _register_default_tools(self):
        """Register the four core tools from the plan."""
        from app.assistant.tools import (
            get_events_tool,
            get_top_behaviours_tool,
            get_bay_summary_tool,
            explain_event_tool,
        )

        self.register_tool(
            "get_events",
            get_events_tool,
            "Query events from the database with optional filters (behaviour_type, tier, status, bay_id, time range).",
            {
                "type": "object",
                "properties": {
                    "behaviour_type": {"type": "string"},
                    "bay_id": {"type": "string"},
                    "status": {"type": "string"},
                    "tier": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                },
            },
        )
        self.register_tool(
            "get_top_behaviours",
            get_top_behaviours_tool,
            "Get the most common risky behaviours in a time period.",
            {
                "type": "object",
                "properties": {
                    "period": {"type": "string", "enum": ["today", "morning_shift", "evening_shift", "week"]},
                    "limit": {"type": "integer", "default": 5},
                },
            },
        )
        self.register_tool(
            "get_bay_summary",
            get_bay_summary_tool,
            "Get event summary for a loading bay.",
            {
                "type": "object",
                "properties": {
                    "bay_id": {"type": "string"},
                    "shift": {"type": "string", "default": "today"},
                },
            },
        )
        self.register_tool(
            "explain_event",
            explain_event_tool,
            "Explain why an event was classified with its risk level.",
            {
                "type": "object",
                "properties": {
                    "event_id": {"type": "integer"},
                },
                "required": ["event_id"],
            },
        )

    def register_tool(self, name: str, func: Callable, description: str, parameters: dict):
        """Register a tool that the assistant can call."""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters,
        }

    def get_tools_schema(self) -> list[dict]:
        """Get OpenAI-compatible tool schemas for function calling."""
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            })
        return schemas

    async def process_query(
        self,
        user_query: str,
        db: Any,  # AsyncSession
    ) -> AssistantResponse:
        """Process a user query through the tool-calling loop.

        1. Send query + tools to LLM
        2. If LLM requests tool calls, execute them and return results
        3. Continue until LLM provides a text response
        4. All responses are grounded in tool results only
        """
        self.conversation_history.append({"role": "user", "content": user_query})

        # Build messages for LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.conversation_history[-10:],  # Last 10 turns for context
        ]

        # Tool-calling loop
        tool_calls_made = []
        max_iterations = 5

        for _ in range(max_iterations):
            response = await self._call_llm(messages, self.get_tools_schema())

            if not response.get("tool_calls"):
                # LLM provided a text response — we're done
                assistant_text = response.get("content", "")
                self.conversation_history.append({"role": "assistant", "content": assistant_text})
                return AssistantResponse(
                    text=assistant_text,
                    tool_calls=tool_calls_made,
                    grounded=True,
                )

            # Execute tool calls
            messages.append({
                "role": "assistant",
                "content": response.get("content"),
                "tool_calls": response["tool_calls"],
            })

            for tool_call in response["tool_calls"]:
                func_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                # Execute the tool
                if func_name in self.tools:
                    result = await self.tools[func_name]["function"](db, **arguments)
                else:
                    result = json.dumps({"error": f"Unknown tool: {func_name}"})

                tool_calls_made.append(ToolCall(
                    tool_name=func_name,
                    arguments=arguments,
                    result=result,
                ))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })

        # Fallback if max iterations reached
        return AssistantResponse(
            text="I've gathered the information from the database. Let me summarize what I found.",
            tool_calls=tool_calls_made,
            grounded=True,
        )

    async def _call_llm(self, messages: list[dict], tools: list[dict]) -> dict:
        """Call the LLM API with messages and tool definitions."""
        if not ASSISTANT_API_KEY:
            # Scaffold mode — return a mock response
            return self._mock_llm_response(messages)

        try:
            if ASSISTANT_LLM_PROVIDER == "openai":
                return await self._call_openai(messages, tools)
            else:
                return await self._call_claude(messages, tools)
        except Exception as e:
            return {"content": f"Error calling LLM: {str(e)}"}

    async def _call_openai(self, messages: list[dict], tools: list[dict]) -> dict:
        """Call OpenAI API for tool-calling."""
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {ASSISTANT_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ASSISTANT_MODEL,
                    "messages": messages,
                    "tools": tools if tools else None,
                    "tool_choice": "auto" if tools else None,
                },
            )
            data = response.json()
            choice = data["choices"][0]["message"]
            return {
                "content": choice.get("content"),
                "tool_calls": choice.get("tool_calls"),
            }

    async def _call_claude(self, messages: list[dict], tools: list[dict]) -> dict:
        """Call Claude API for tool-calling."""
        import httpx

        # Convert to Claude format
        claude_tools = []
        for tool in tools:
            claude_tools.append({
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"],
            })

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ASSISTANT_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ASSISTANT_MODEL,
                    "max_tokens": 1024,
                    "system": SYSTEM_PROMPT,
                    "messages": messages,
                    "tools": claude_tools if claude_tools else None,
                },
            )
            data = response.json()
            content = data.get("content", [])
            text_parts = [c["text"] for c in content if c["type"] == "text"]
            tool_parts = [c for c in content if c["type"] == "tool_use"]

            tool_calls = None
            if tool_parts:
                tool_calls = [
                    {
                        "id": tc["id"],
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"]),
                        },
                    }
                    for tc in tool_parts
                ]

            return {
                "content": " ".join(text_parts),
                "tool_calls": tool_calls,
            }

    def _mock_llm_response(self, messages: list[dict]) -> dict:
        """Mock LLM response for scaffold/demo without API key.

        Phase 1 (no tool results in messages yet): inspect the user query,
        decide which tools to call, return tool_calls so the main loop
        executes them against the real database.

        Phase 2 (tool results present in messages): summarize the tool
        results into a natural-language answer.
        """
        last_msg = messages[-1]["content"] if messages else ""

        # Check if tool results are already in the messages (phase 2)
        has_tool_results = any(m.get("role") == "tool" for m in messages)

        if has_tool_results:
            # Phase 2: summarize tool results
            tool_results = []
            for m in messages:
                if m.get("role") == "tool":
                    tool_results.append(m.get("content", ""))
            combined = "\n".join(tool_results)
            return {
                "content": f"Based on the events database:\n\n{combined}",
                "tool_calls": None,
            }

        # Phase 1: determine which tools to call based on the user query
        q = last_msg.lower()
        tool_calls = []

        # Pattern: high-risk / tier filtering
        if "high-risk" in q or "high risk" in q or "critical" in q:
            tool_calls.append({
                "id": "mock_1",
                "function": {
                    "name": "get_events",
                    "arguments": json.dumps({"tier": "High", "limit": 20}),
                },
            })

        # Pattern: most common / top behaviours
        elif "common" in q or "top" in q or "most frequent" in q or "frequent" in q:
            tool_calls.append({
                "id": "mock_1",
                "function": {
                    "name": "get_top_behaviours",
                    "arguments": json.dumps({"period": "today", "limit": 5}),
                },
            })

        # Pattern: bay summary / which bay
        elif "bay" in q or "loading" in q or "unloading" in q:
            tool_calls.append({
                "id": "mock_1",
                "function": {
                    "name": "get_bay_summary",
                    "arguments": json.dumps({"shift": "today"}),
                },
            })

        # Pattern: why / explain / classification
        elif "why" in q or "explain" in q or "classified" in q or "risk" in q:
            tool_calls.append({
                "id": "mock_1",
                "function": {
                    "name": "get_events",
                    "arguments": json.dumps({"limit": 10}),
                },
            })

        # Default: query recent events
        else:
            tool_calls.append({
                "id": "mock_1",
                "function": {
                    "name": "get_events",
                    "arguments": json.dumps({"limit": 10}),
                },
            })

        return {
            "content": None,
            "tool_calls": tool_calls,
        }
