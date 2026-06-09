"""Strategist Agent — goal decomposition and task routing."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from jarvis.agents.base_agent import BaseSpecializedAgent
from jarvis.config import settings

if TYPE_CHECKING:
    from jarvis.core.memory import MemorySystem
    from jarvis.core.tools import ToolRegistry
    from jarvis.orchestrator.message_bus import MessageBus


_STRATEGIST_SYSTEM = """You are the Strategist — the master planner of a multi-agent AI system.

Your sole job is to decompose a user's goal into concrete, assignable subtasks.
Each subtask must be assigned to one of: researcher, builder.

Rules:
- If the goal requires factual information, web data, or research → assign to researcher
- If the goal requires producing code, documents, analysis, or structured output → assign to builder
- Always create at least one subtask
- Research tasks should come before builder tasks when grounding is needed
- Be specific in task descriptions — vague instructions produce poor results

Return ONLY valid JSON, no other text:
{
  "reasoning": "brief explanation of your decomposition strategy",
  "subtasks": [
    {
      "title": "short title",
      "description": "precise, actionable instruction",
      "agent_hint": "researcher" | "builder",
      "depends_on": [],
      "constraints": ["list", "of", "constraints"]
    }
  ]
}"""


_ADJUDICATE_SYSTEM = """You are the Strategist adjudicating a debate between two AI agents.
You will receive a transcript of arguments. Your job is to:
1. Evaluate both positions objectively
2. Pick the stronger, more practical argument
3. Explain why in 2-3 sentences

Return ONLY valid JSON:
{"winner": "pro" | "con", "winning_argument": "...", "rationale": "..."}"""


class StrategistAgent(BaseSpecializedAgent):
    role = "strategist"
    icon = "🧠"
    _role_system = _STRATEGIST_SYSTEM

    def __init__(
        self,
        memory: "MemorySystem",
        tools: "ToolRegistry",
        message_bus: "MessageBus",
    ):
        super().__init__(memory=memory, tools=tools, message_bus=message_bus)

    async def process(self, task: dict, context: dict, run_id: str) -> dict:
        goal = task.get("goal", "")
        memory_context = context.get("memory_context", "")

        self._set_working(f"Decomposing: {goal[:50]}")
        await self.broadcast_status(run_id)

        system = self._build_system_prompt(memory_context)
        messages = [{"role": "user", "content": f"Goal to decompose:\n{goal}"}]

        raw = await self._llm.complete(
            system=system,
            messages=messages,
            model=settings.planner_model,
        )

        subtasks = self._parse_subtasks(goal, raw)
        confidence = await self.reflect(goal, json.dumps(subtasks))

        await self.send_message(
            to="broadcast",
            msg_type="insight",
            content=f"Decomposed into {len(subtasks)} subtasks: " +
                    ", ".join(s["title"] for s in subtasks),
            confidence=confidence,
            task_id=task.get("task_id", ""),
            run_id=run_id,
        )

        self._record_confidence(confidence)
        self._set_idle()
        await self.broadcast_status(run_id)

        return {"subtasks": subtasks, "confidence": confidence, "tool_calls": []}

    async def adjudicate(self, run_id: str, debate_transcript: list[dict]) -> dict:
        """Pick the winning position from a debate transcript."""
        self._set_working("Adjudicating debate")
        await self.broadcast_status(run_id)

        transcript_text = "\n\n".join(
            f"[{m.get('position', 'unknown').upper()}] Round {i+1}:\n{m.get('argument', '')}"
            for i, m in enumerate(debate_transcript)
        )
        raw = await self._llm.complete(
            system=_ADJUDICATE_SYSTEM,
            messages=[{"role": "user", "content": transcript_text}],
            model=settings.planner_model,
        )

        self._set_idle()
        await self.broadcast_status(run_id)

        try:
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception:
            return {
                "winner": "pro",
                "winning_argument": debate_transcript[0].get("argument", ""),
                "rationale": "Defaulted to first position due to parse error.",
            }

    def _parse_subtasks(self, goal: str, raw: str) -> list[dict]:
        try:
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            subtasks = data.get("subtasks", [])
            if isinstance(subtasks, list) and subtasks:
                return subtasks
        except Exception:
            pass
        # Fallback single subtask
        return [{
            "title": "Complete goal",
            "description": goal,
            "agent_hint": "builder",
            "depends_on": [],
            "constraints": [],
        }]
