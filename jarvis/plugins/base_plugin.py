"""Plugin base class — extend to add new tool groups to Jarvis."""
from __future__ import annotations

from abc import ABC, abstractmethod
from jarvis.core.tools import ToolDef, ToolRegistry


class BasePlugin(ABC):
    """All plugins inherit from this class."""

    name: str = "unnamed_plugin"
    description: str = ""
    version: str = "0.1.0"

    @abstractmethod
    def tools(self) -> list[ToolDef]:
        """Return the list of ToolDef objects this plugin provides."""
        ...

    def register(self, registry: ToolRegistry):
        for t in self.tools():
            registry.register(t)

    def __repr__(self):
        return f"<Plugin {self.name} v{self.version}>"
