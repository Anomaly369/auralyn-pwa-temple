"""Communicator Agent — formats AI outputs into polished human-readable responses."""
from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.agents.base_agent import BaseSpecializedAgent
from jarvis.config import settings

if TYPE_CHECKING:
    from jarvis.core.memory import MemorySystem
    from jarvis.core.tools import ToolRegistry
    from jarvis.orchestrator.message_bus import MessageBus


_COMMUNICATOR_SYSTEM = """You are the Communicator — the voice of the multi-agent system to the user.

Your job is to translate technical agent outputs into a clear, engaging, human-readable response.

Standards:
- Write in natural language — never expose internal agent JSON or system details
- Use markdown: headers (##), bullet points, code blocks, bold for key terms
- Be concise but complete — include everything that matters, cut everything that doesn't
- Credit sources naturally in prose (e.g., "According to [source]...")
- If the result includes code, show it in a properly labeled code block
- Add 1–3 "Next steps" suggestions at the end when appropriate
- Match tone to the task: technical for technical requests, conversational for casual ones
- If the Critic flagged unresolved issues, acknowledge them transparently

Structure your response:
1. Direct answer to the user's goal (top)
2. Supporting details / breakdown (middle)
3. Key insights or caveats (if any)
4. Next steps (bottom, optional)"""


class CommunicatorAgent(BaseSpecializedAgent):
    role = "communicator"
    icon = "💬"
    _role_system = _COMMUNICATOR_SYSTEM

    def __init__(
        self,
        memory: "MemorySystem",
        tools: "ToolRegistry",
        message_bus: "MessageBus",
    ):
        super().__init__(memory=memory, tools=tools, message_bus=message_bus)

    async def process(self, task: dict, context: dict, run_id: str) -> dict:
        goal = task.get("goal", "")
        merged_results = task.get("merged_results", "")
        critique = task.get("critique", {})
        research_insights = task.get("research_insights", "")

        self._set_working("Formatting final response")
        await self.broadcast_status(run_id)

        critique_note = ""
        if critique and not critique.get("approved", True):
            issues = critique.get("issues", [])
            if issues:
                critique_note = (
                    "\n\nNote: The Critic flagged unresolved issues. "
                    "Please acknowledge these in your response:\n"
                    + "\n".join(f"- {i}" for i in issues[:3])
                )

        formatting_prompt = (
            f"User's goal: {goal}\n\n"
            f"## Agent Output to Format:\n{merged_results[:3000]}\n"
        )
        if research_insights:
            formatting_prompt += f"\n## Research Context:\n{research_insights[:1000]}"
        if critique_note:
            formatting_prompt += critique_note

        result, _ = await self._react_loop(
            system=_COMMUNICATOR_SYSTEM,
            messages=[{"role": "user", "content": formatting_prompt}],
            allowed_tools=[],  # communicator just writes, no tools
            max_iter=1,
        )

        confidence = await self.reflect(goal, result)

        await self.send_message(
            to="broadcast",
            msg_type="summary",
            content=result[:500],
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
            "tool_calls": [],
        }
