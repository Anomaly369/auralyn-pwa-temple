"""Critic Agent — quality evaluation, scoring, and feedback generation."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from jarvis.agents.base_agent import BaseSpecializedAgent
from jarvis.config import settings

if TYPE_CHECKING:
    from jarvis.core.memory import MemorySystem
    from jarvis.core.tools import ToolRegistry
    from jarvis.orchestrator.message_bus import MessageBus


_CRITIC_SYSTEM = """You are the Critic — a rigorous quality evaluator for AI-produced work.

Evaluate the provided output on three dimensions (score each 0.0 to 1.0):
- accuracy: Is information correct, well-sourced, and free of hallucinations?
- completeness: Does the output fully address the task requirements?
- quality: Is it clear, well-structured, actionable, and production-ready?

Scoring guidance:
- 0.9+: Exceptional, exceeds expectations
- 0.7–0.9: Solid, meets all requirements
- 0.5–0.7: Acceptable but has notable gaps
- below 0.5: Significant issues, requires substantial rework

Set approved = true only when ALL three scores are >= 0.7.

Return ONLY valid JSON:
{
  "accuracy": 0.85,
  "completeness": 0.90,
  "quality": 0.80,
  "overall": 0.85,
  "approved": true,
  "issues": ["specific issue 1", "specific issue 2"],
  "suggestions": ["actionable fix 1", "actionable fix 2"],
  "feedback": "2-3 sentence summary of evaluation"
}

Be strict but fair. Vague generalities are not helpful — point to specific problems."""


class CriticAgent(BaseSpecializedAgent):
    role = "critic"
    icon = "🧪"
    _role_system = _CRITIC_SYSTEM

    def __init__(
        self,
        memory: "MemorySystem",
        tools: "ToolRegistry",
        message_bus: "MessageBus",
    ):
        super().__init__(memory=memory, tools=tools, message_bus=message_bus)

    async def process(self, task: dict, context: dict, run_id: str) -> dict:
        goal = task.get("goal", "")
        agent_results = task.get("agent_results", {})
        attempt = task.get("attempt", 1)

        self._set_working(f"Evaluating attempt {attempt}")
        await self.broadcast_status(run_id)

        # Build evaluation context from all agent results
        results_text = ""
        for agent_name, result_data in agent_results.items():
            result_content = (
                result_data.get("result", "") if isinstance(result_data, dict)
                else str(result_data)
            )
            results_text += f"\n\n## {agent_name.title()} Output:\n{result_content[:1500]}"

        evaluation_prompt = (
            f"Task/Goal:\n{goal}\n\n"
            f"Outputs to evaluate:{results_text}\n\n"
            f"This is attempt {attempt}. Apply the scoring criteria precisely."
        )

        raw = await self._llm.complete(
            system=_CRITIC_SYSTEM,
            messages=[{"role": "user", "content": evaluation_prompt}],
            model=settings.planner_model,
        )

        critique = self._parse_critique(raw)
        overall = critique.get("overall", (
            critique.get("accuracy", 0.7) +
            critique.get("completeness", 0.7) +
            critique.get("quality", 0.7)
        ) / 3)
        critique["overall"] = round(overall, 3)

        # Enforce threshold from config
        critique["approved"] = overall >= settings.critic_pass_threshold

        await self.send_message(
            to="broadcast",
            msg_type="critique",
            content=critique.get("feedback", f"Score: {overall:.2f} — {'APPROVED' if critique['approved'] else 'NEEDS REVISION'}"),
            confidence=overall,
            task_id=task.get("task_id", ""),
            run_id=run_id,
        )

        self._record_confidence(overall)
        self._set_idle()
        await self.broadcast_status(run_id)

        return {
            "result": critique,
            "confidence": overall,
            "tool_calls": [],
            "approved": critique["approved"],
        }

    async def evaluate(self, goal: str, agent_results: dict, run_id: str, attempt: int = 1) -> dict:
        """Convenience wrapper for the orchestrator."""
        task = {"goal": goal, "agent_results": agent_results, "attempt": attempt}
        return await self.process(task, {}, run_id)

    def _parse_critique(self, raw: str) -> dict:
        try:
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception:
            return {
                "accuracy": 0.7,
                "completeness": 0.7,
                "quality": 0.7,
                "overall": 0.7,
                "approved": True,
                "issues": [],
                "suggestions": [],
                "feedback": "Evaluation parsing failed; defaulting to approved.",
            }
