"""In-process message bus for structured agent-to-agent communication."""
from __future__ import annotations

import asyncio
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.agents.base_agent import AgentMessage
    from jarvis.core.memory import MemorySystem


class MessageBus:
    """Async pub/sub bus that routes AgentMessages between agents and UI."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._history: list[dict] = []
        self._ws_callbacks: list[Callable] = []

    # ── Subscription ──────────────────────────────────────────────────────

    def subscribe(self, agent_name: str) -> asyncio.Queue:
        """Return (creating if absent) a dedicated asyncio.Queue for an agent."""
        if agent_name not in self._queues:
            self._queues[agent_name] = asyncio.Queue()
        return self._queues[agent_name]

    def register_ws_callback(self, fn: Callable):
        """Register a coroutine that is called on every published message."""
        self._ws_callbacks.append(fn)

    def unregister_ws_callback(self, fn: Callable):
        self._ws_callbacks = [c for c in self._ws_callbacks if c is not fn]

    # ── Publishing ────────────────────────────────────────────────────────

    async def publish(self, msg: "AgentMessage", memory: "MemorySystem | None" = None):
        """Distribute a message to its target queue(s), history, and WS clients."""
        d = msg.to_dict()
        self._history.append(d)

        target = msg.to_agent
        if target == "broadcast":
            for q in self._queues.values():
                await q.put(d)
        elif target in self._queues:
            await self._queues[target].put(d)

        # Persist to DB in background
        if memory is not None:
            try:
                await memory.store_agent_message(d)
            except Exception:
                pass

        # Notify all WebSocket clients
        for cb in list(self._ws_callbacks):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb({"type": "agent_message", **d})
                else:
                    cb({"type": "agent_message", **d})
            except Exception:
                pass

    async def broadcast_event(self, event: dict):
        """Push a non-message UI event (phase_change, run_start, etc.) to WS clients."""
        for cb in list(self._ws_callbacks):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception:
                pass

    # ── History ───────────────────────────────────────────────────────────

    def get_history(self, run_id: str | None = None, limit: int = 200) -> list[dict]:
        msgs = self._history
        if run_id:
            msgs = [m for m in msgs if m.get("run_id") == run_id]
        return msgs[-limit:]

    def clear_history(self):
        self._history.clear()
