"""Researcher Agent — web search, data retrieval, and synthesis."""
from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.agents.base_agent import BaseSpecializedAgent
from jarvis.config import settings

if TYPE_CHECKING:
    from jarvis.core.memory import MemorySystem
    from jarvis.core.tools import ToolRegistry
    from jarvis.orchestrator.message_bus import MessageBus


_RESEARCHER_SYSTEM = """You are the Researcher — an expert at finding accurate, relevant information.

Your capabilities:
- Web search via web_search tool
- HTTP data retrieval via http_get
- Mathematical calculations via calculator
- Current date/time via get_datetime

Your standards:
- Always search before claiming facts
- Cite your sources (URL or search query used)
- Rate your confidence in each finding
- If information is unavailable, say so explicitly — never fabricate
- Synthesize multiple sources; don't just dump raw results
- Structure findings as: claim → evidence → source → confidence

You are reporting to the Builder agent who will use your findings to produce output.
Be thorough but focused — quality over quantity."""


_ALLOWED_TOOLS = ["web_search", "http_get", "calculator", "get_datetime"]


class ResearchAgent(BaseSpecializedAgent):
    role = "researcher"
    icon = "🔍"
    _role_system = _RESEARCHER_SYSTEM

    def __init__(
        self,
        memory: "MemorySystem",
        tools: "ToolRegistry",
        message_bus: "MessageBus",
    ):
        super().__init__(memory=memory, tools=tools, message_bus=message_bus)

    async def process(self, task: dict, context: dict, run_id: str) -> dict:
        description = task.get("description", task.get("title", ""))
        constraints = task.get("constraints", [])
        memory_context = context.get("memory_context", "")
        prior_results = context.get("prior_results", "")

        self._set_working(description[:60])
        await self.broadcast_status(run_id)

        system = self._build_system_prompt(memory_context)
        if constraints:
            system += f"\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
        if prior_results:
            system += f"\n\nPrevious research context:\n{prior_results[:800]}"

        messages = [{"role": "user", "content": f"Research task:\n{description}"}]
        result, tool_calls = await self._react_loop(
            system=system,
            messages=messages,
            allowed_tools=_ALLOWED_TOOLS,
            max_iter=6,
        )

        confidence = await self.reflect(description, result)

        await self.send_message(
            to="broadcast",
            msg_type="insight",
            content=result[:1500],
            confidence=confidence,
            task_id=task.get("task_id", ""),
            run_id=run_id,
        )

        self._record_confidence(confidence)
        self._set_idle()
        await self.broadcast_status(run_id)

        return {
            "result": result,
            "confidence": confidence,
            "tool_calls": tool_calls,
        }
