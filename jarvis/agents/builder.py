"""Builder Agent — code generation, document creation, structured output."""
from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.agents.base_agent import BaseSpecializedAgent
from jarvis.config import settings

if TYPE_CHECKING:
    from jarvis.core.memory import MemorySystem
    from jarvis.core.tools import ToolRegistry
    from jarvis.orchestrator.message_bus import MessageBus


_BUILDER_SYSTEM = """You are the Builder — you produce high-quality, complete output artifacts.

Your capabilities include all available tools:
- execute_python: run code to test or compute
- write_code_file: save generated code to disk
- read_file / write_file: file I/O
- web_search / http_get: look up additional details if needed
- calculator: precise math

Your standards:
- Produce complete, production-ready output — never leave placeholders
- Use execute_python to verify code before delivering it
- Structure output clearly: use markdown headers, code blocks, numbered lists
- Your output will be reviewed by the Critic — make it thorough and well-reasoned
- If research insights are provided, use them as grounding — do not ignore them
- Acknowledge any limitations or assumptions explicitly

The Critic will reject incomplete or incorrect work. Be thorough."""


class BuilderAgent(BaseSpecializedAgent):
    role = "builder"
    icon = "🧱"
    _role_system = _BUILDER_SYSTEM

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
        research_insights = context.get("research_insights", "")
        critique_feedback = context.get("critique_feedback", "")

        self._set_working(description[:60])
        await self.broadcast_status(run_id)

        system = self._build_system_prompt(memory_context)
        if research_insights:
            system += f"\n\n## Research Insights\n{research_insights[:2000]}"
        if constraints:
            system += "\n\n## Constraints\n" + "\n".join(f"- {c}" for c in constraints)
        if critique_feedback:
            system += (
                f"\n\n## Critic Feedback (Address These Issues)\n{critique_feedback}\n\n"
                "IMPORTANT: The Critic rejected your previous attempt. Fix all issues listed above."
            )

        messages = [{"role": "user", "content": f"Build task:\n{description}"}]
        result, tool_calls = await self._react_loop(
            system=system,
            messages=messages,
            allowed_tools=None,  # full tool access
            max_iter=8,
        )

        confidence = await self.reflect(description, result)

        await self.send_message(
            to="critic_agent",
            msg_type="result",
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
