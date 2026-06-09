"""Task planner — decomposes high-level goals into ordered steps."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from jarvis.config import settings


@dataclass
class Step:
    index: int
    title: str
    description: str
    tool_hint: str | None = None   # suggested tool if any
    depends_on: list[int] = None   # step indices this one waits for
    status: str = "pending"        # pending | running | done | skipped | failed
    result: str = ""

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "description": self.description,
            "tool_hint": self.tool_hint,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result,
        }


@dataclass
class Plan:
    goal: str
    steps: list[Step]
    reasoning: str = ""

    def pending_steps(self) -> list[Step]:
        return [s for s in self.steps if s.status == "pending"]

    def ready_steps(self) -> list[Step]:
        """Steps whose dependencies are all done."""
        done_indices = {s.index for s in self.steps if s.status == "done"}
        return [
            s for s in self.steps
            if s.status == "pending"
            and all(d in done_indices for d in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(s.status in ("done", "skipped") for s in self.steps)

    def has_failure(self) -> bool:
        return any(s.status == "failed" for s in self.steps)

    def summary(self) -> str:
        lines = [f"Goal: {self.goal}", f"Reasoning: {self.reasoning}", ""]
        for s in self.steps:
            icon = {"pending": "○", "running": "►", "done": "✓",
                    "skipped": "–", "failed": "✗"}.get(s.status, "?")
            lines.append(f"  {icon} [{s.index}] {s.title}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "reasoning": self.reasoning,
            "steps": [s.to_dict() for s in self.steps],
        }


# ── TaskPlanner ────────────────────────────────────────────────────────────

class TaskPlanner:
    """Uses the LLM to produce a structured Plan from a natural-language goal."""

    _SYSTEM = """You are a precise task planning assistant.
Given a goal, decompose it into concrete, ordered steps an AI agent can execute.
Each step must be actionable with available tools or by reasoning.

Available tools: web_search, execute_python, read_file, write_file,
list_files, delete_file, calculator, http_get, http_post,
get_datetime, write_code_file.

Return ONLY valid JSON in this exact format:
{
  "reasoning": "brief explanation of the approach",
  "steps": [
    {
      "index": 0,
      "title": "short step title",
      "description": "precise instruction for what to do",
      "tool_hint": "tool_name or null",
      "depends_on": []
    }
  ]
}

Rules:
- Maximum {max_steps} steps.
- Keep steps atomic and testable.
- tool_hint must be a real tool name or null.
- depends_on contains step indices that must complete first.
- If the task is simple (single step), use exactly 1 step.
- Do NOT include markdown fences or any text outside the JSON.
"""

    def __init__(self, llm_client=None):
        self._client = llm_client

    def set_client(self, client):
        self._client = client

    async def plan(self, goal: str, context: str = "") -> Plan:
        if self._client is None:
            return self._fallback_plan(goal)

        system = self._SYSTEM.format(max_steps=settings.max_plan_steps)
        user_msg = f"Goal: {goal}"
        if context:
            user_msg += f"\n\nContext:\n{context}"

        raw = await self._client.complete(
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            model=settings.planner_model,
            max_tokens=2048,
            tools=None,
        )

        return self._parse_plan(goal, raw)

    def _parse_plan(self, goal: str, raw: str) -> Plan:
        raw = raw.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._fallback_plan(goal)

        steps = [
            Step(
                index=s["index"],
                title=s.get("title", f"Step {s['index']}"),
                description=s.get("description", ""),
                tool_hint=s.get("tool_hint"),
                depends_on=s.get("depends_on", []),
            )
            for s in data.get("steps", [])
        ]
        return Plan(goal=goal, steps=steps, reasoning=data.get("reasoning", ""))

    @staticmethod
    def _fallback_plan(goal: str) -> Plan:
        """Single-step fallback when the LLM is unavailable."""
        return Plan(
            goal=goal,
            steps=[Step(index=0, title="Execute goal", description=goal)],
            reasoning="LLM planner unavailable — single-step execution.",
        )
