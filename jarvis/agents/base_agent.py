"""Base class and data models for all specialized agents."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING

import anthropic

from jarvis.config import settings

if TYPE_CHECKING:
    from jarvis.core.memory import MemorySystem
    from jarvis.core.tools import ToolRegistry
    from jarvis.orchestrator.message_bus import MessageBus


# ── Data models ────────────────────────────────────────────────────────────

@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    type: str       # insight | request | result | critique | summary | status
    content: str
    confidence: float
    task_id: str
    run_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentStatus:
    name: str
    role: str
    status: str          # idle | working | waiting | error
    current_task: str
    tasks_completed: int
    avg_confidence: float
    last_active: float


# ── LLM client (shared with JarvisAgent) ──────────────────────────────────

class _AnthropicClient:
    def __init__(self):
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def complete(
        self,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int = settings.max_tokens,
    ) -> str:
        resp = await self._client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        )
        for block in resp.content:
            if block.type == "text":
                return block.text
        return ""

    async def complete_with_tools(
        self,
        system: str,
        messages: list[dict],
        model: str,
        tools: list[dict],
        max_tokens: int = settings.max_tokens,
    ) -> anthropic.types.Message:
        return await self._client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=messages, tools=tools,
        )


# ── Reflection prompt ──────────────────────────────────────────────────────

_REFLECT_SYSTEM = """You are a self-evaluation module for an AI agent.
Given a task description and the result produced, rate the quality of the result
on a scale from 0.0 (complete failure) to 1.0 (perfect).
Return ONLY a JSON object: {"score": 0.85, "reasoning": "brief explanation"}
No other text."""


# ── Base agent ─────────────────────────────────────────────────────────────

class BaseSpecializedAgent:
    """Shared foundation for all specialized agents in the multi-agent system."""

    role: str = "base"
    icon: str = "🤖"
    _role_system: str = "You are a helpful AI assistant."

    def __init__(
        self,
        memory: "MemorySystem",
        tools: "ToolRegistry",
        message_bus: "MessageBus",
    ):
        self.memory = memory
        self.tools = tools
        self.message_bus = message_bus
        self._llm = _AnthropicClient()
        self.name = f"{self.role}_agent"

        # Runtime state
        self.status: str = "idle"
        self.current_task: str = ""
        self.tasks_completed: int = 0
        self._total_confidence: float = 0.0
        self.last_active: float = time.time()

    # ── Public API ─────────────────────────────────────────────────────────

    async def process(self, task: dict, context: dict, run_id: str) -> dict:
        """Override in subclasses. Return {result, confidence, tool_calls}."""
        raise NotImplementedError

    async def get_status(self) -> AgentStatus:
        avg = (
            self._total_confidence / self.tasks_completed
            if self.tasks_completed > 0 else 1.0
        )
        return AgentStatus(
            name=self.name,
            role=self.role,
            status=self.status,
            current_task=self.current_task,
            tasks_completed=self.tasks_completed,
            avg_confidence=round(avg, 3),
            last_active=self.last_active,
        )

    # ── Messaging ──────────────────────────────────────────────────────────

    async def send_message(
        self,
        to: str,
        msg_type: str,
        content: str,
        confidence: float,
        task_id: str,
        run_id: str,
    ):
        msg = AgentMessage(
            from_agent=self.name,
            to_agent=to,
            type=msg_type,
            content=content,
            confidence=confidence,
            task_id=task_id,
            run_id=run_id,
        )
        await self.message_bus.publish(msg, memory=self.memory)

    async def broadcast_status(self, run_id: str):
        status = await self.get_status()
        await self.message_bus.broadcast_event({
            "type": "agent_status",
            "agent": self.name,
            "role": self.role,
            "icon": self.icon,
            "status": status.status,
            "current_task": status.current_task,
            "tasks_completed": status.tasks_completed,
            "avg_confidence": status.avg_confidence,
            "run_id": run_id,
        })

    # ── Self-evaluation ────────────────────────────────────────────────────

    async def reflect(self, task_description: str, result: str) -> float:
        """Ask the LLM to score the quality of result. Returns 0.0–1.0."""
        prompt = (
            f"Task: {task_description[:500]}\n\n"
            f"Result produced:\n{result[:1000]}"
        )
        try:
            raw = await self._llm.complete(
                system=_REFLECT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                model=settings.executor_model,
                max_tokens=128,
            )
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            return max(0.0, min(1.0, float(data.get("score", 0.7))))
        except Exception:
            return 0.7

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_system_prompt(self, context_block: str = "") -> str:
        prompt = self._role_system
        if context_block.strip():
            prompt += f"\n\n{context_block}"
        return prompt

    async def _react_loop(
        self,
        system: str,
        messages: list[dict],
        allowed_tools: list[str] | None = None,
        max_iter: int = 8,
    ) -> tuple[str, list[dict]]:
        """ReAct loop — identical pattern to JarvisAgent._execute_step()."""
        tool_schemas = self.tools.schemas_anthropic()
        if allowed_tools:
            tool_schemas = [t for t in tool_schemas if t["name"] in allowed_tools]

        tool_calls_log: list[dict] = []

        for _ in range(max_iter):
            resp = await self._llm.complete_with_tools(
                system=system,
                messages=messages,
                model=settings.executor_model,
                tools=tool_schemas,
            )
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "end_turn":
                return self._extract_text(resp), tool_calls_log

            if resp.stop_reason == "tool_use":
                tool_results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    result = await self.tools.call(block.name, **block.input)
                    tool_calls_log.append({
                        "tool": block.name,
                        "input": block.input,
                        "result": result[:300],
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

        return "Max iterations reached.", tool_calls_log

    def _set_working(self, task: str):
        self.status = "working"
        self.current_task = task
        self.last_active = time.time()

    def _set_idle(self):
        self.status = "idle"
        self.current_task = ""
        self.tasks_completed += 1
        self.last_active = time.time()

    def _record_confidence(self, confidence: float):
        self._total_confidence += confidence

    @staticmethod
    def _extract_text(resp: anthropic.types.Message) -> str:
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text
        return ""
