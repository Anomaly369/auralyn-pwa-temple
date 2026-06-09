"""AgentRegistry — creates and manages all specialized agent instances."""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from jarvis.agents.strategist import StrategistAgent
from jarvis.agents.researcher import ResearchAgent
from jarvis.agents.builder import BuilderAgent
from jarvis.agents.critic import CriticAgent
from jarvis.agents.communicator import CommunicatorAgent

if TYPE_CHECKING:
    from jarvis.agents.base_agent import BaseSpecializedAgent, AgentStatus
    from jarvis.core.memory import MemorySystem
    from jarvis.core.tools import ToolRegistry
    from jarvis.orchestrator.message_bus import MessageBus


class AgentRegistry:
    """Factory and directory for all specialized agents."""

    def __init__(
        self,
        memory: "MemorySystem",
        tools: "ToolRegistry",
        message_bus: "MessageBus",
    ):
        self._agents: dict[str, "BaseSpecializedAgent"] = {
            "strategist": StrategistAgent(memory, tools, message_bus),
            "researcher": ResearchAgent(memory, tools, message_bus),
            "builder": BuilderAgent(memory, tools, message_bus),
            "critic": CriticAgent(memory, tools, message_bus),
            "communicator": CommunicatorAgent(memory, tools, message_bus),
        }

    def get(self, role: str) -> "BaseSpecializedAgent | None":
        return self._agents.get(role)

    def all_agents(self) -> "list[BaseSpecializedAgent]":
        return list(self._agents.values())

    async def get_statuses(self) -> "list[AgentStatus]":
        return [await a.get_status() for a in self._agents.values()]

    async def get_status_dicts(self) -> list[dict]:
        statuses = await self.get_statuses()
        result = []
        for s, agent in zip(statuses, self._agents.values()):
            d = asdict(s)
            d["icon"] = agent.icon
            result.append(d)
        return result

    async def update_performance(
        self, agent_name: str, confidence: float, memory: "MemorySystem", error: bool = False
    ):
        await memory.upsert_agent_performance(agent_name, confidence, error=error)
