"""Tool registry — every capability the agent can call."""
from __future__ import annotations

import ast
import asyncio
import io
import json
import math
import os
import subprocess
import sys
import textwrap
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

import httpx

from jarvis.config import settings


# ── Tool descriptor ────────────────────────────────────────────────────────

@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Awaitable[str]]

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


# ── Implementations ────────────────────────────────────────────────────────

async def web_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo search — no API key required."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"**{r['title']}**\n{r['href']}\n{r.get('body', '')[:300]}"
                )
        return "\n\n---\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"


async def execute_python(code: str, timeout: int = settings.max_code_execution_time) -> str:
    """Execute Python code in a restricted subprocess and return stdout/stderr."""
    # Basic safety: block obviously dangerous patterns
    _BLOCKED = ["import os", "import sys", "import subprocess", "import shutil",
                 "__import__", "eval(", "exec(", "open(", "os.system",
                 "subprocess.", "socket.", "__builtins__"]
    code_lower = code.lower()
    for pat in _BLOCKED:
        if pat.lower() in code_lower:
            return f"Blocked: '{pat}' is not allowed in sandbox execution."

    wrapper = textwrap.dedent(f"""
import math, json, re, time, random, itertools, functools, collections
from datetime import datetime, date, timedelta

{code}
""")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", wrapper,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode()[:4000]
        err = stderr.decode()[:2000]
        if err:
            return f"STDOUT:\n{out}\n\nSTDERR:\n{err}" if out else f"ERROR:\n{err}"
        return out or "(no output)"
    except asyncio.TimeoutError:
        proc.kill()
        return f"Execution timed out after {timeout}s."
    except Exception as e:
        return f"Execution failed: {e}"


async def read_file(path: str) -> str:
    """Read a file from the workspace."""
    safe = _safe_path(path)
    if safe is None:
        return "Error: path is outside allowed workspace."
    if not safe.exists():
        return f"Error: file not found: {path}"
    try:
        return safe.read_text(encoding="utf-8")[:8000]
    except Exception as e:
        return f"Read error: {e}"


async def write_file(path: str, content: str) -> str:
    """Write/overwrite a file inside the workspace."""
    safe = _safe_path(path)
    if safe is None:
        return "Error: path is outside allowed workspace."
    safe.parent.mkdir(parents=True, exist_ok=True)
    try:
        safe.write_text(content, encoding="utf-8")
        return f"Written {len(content)} characters to {path}."
    except Exception as e:
        return f"Write error: {e}"


async def list_files(directory: str = ".") -> str:
    """List files and directories in the workspace."""
    safe = _safe_path(directory)
    if safe is None:
        return "Error: path is outside allowed workspace."
    if not safe.is_dir():
        return f"Error: not a directory: {directory}"
    try:
        items = sorted(safe.iterdir())
        lines = []
        for p in items:
            tag = "/" if p.is_dir() else ""
            size = f"  ({p.stat().st_size} bytes)" if p.is_file() else ""
            lines.append(f"  {p.name}{tag}{size}")
        return f"Contents of {directory}:\n" + "\n".join(lines)
    except Exception as e:
        return f"List error: {e}"


async def delete_file(path: str) -> str:
    """Delete a file from the workspace."""
    safe = _safe_path(path)
    if safe is None:
        return "Error: path is outside allowed workspace."
    if not safe.exists():
        return f"Error: not found: {path}"
    try:
        safe.unlink()
        return f"Deleted: {path}"
    except Exception as e:
        return f"Delete error: {e}"


async def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression."""
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max,
                           "sum": sum, "len": len, "int": int, "float": float})
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if not (isinstance(node.func, ast.Name) and node.func.id in allowed_names):
                    return "Error: only math functions allowed."
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calc error: {e}"


async def http_get(url: str, headers: dict | None = None) -> str:
    """Perform an HTTP GET request and return the response body (truncated)."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url, headers=headers or {})
        return r.text[:6000]
    except Exception as e:
        return f"HTTP error: {e}"


async def http_post(url: str, body: dict, headers: dict | None = None) -> str:
    """Perform an HTTP POST with a JSON body."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=body, headers=headers or {})
        return r.text[:6000]
    except Exception as e:
        return f"HTTP error: {e}"


async def get_datetime(_: str = "") -> str:
    """Return current date, time, and UTC offset."""
    from datetime import datetime, timezone
    now = datetime.now()
    utc = datetime.now(timezone.utc)
    return (
        f"Local: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"UTC:   {utc.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )


async def write_code_file(filename: str, language: str, code: str) -> str:
    """Write generated source code to a named file in the workspace."""
    return await write_file(filename, code)


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_path(path: str) -> Path | None:
    """Resolve path and verify it is inside workspace_root."""
    root = Path(settings.workspace_root).resolve()
    candidate = (root / path).resolve()
    if not str(candidate).startswith(str(root)):
        return None
    return candidate


# ── Registry ───────────────────────────────────────────────────────────────

class ToolRegistry:
    """Registry of all available tools with lookup and schema export."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._register_builtins()

    def _register_builtins(self):
        defs = [
            ToolDef(
                name="web_search",
                description=(
                    "Search the web using DuckDuckGo. Use this for current events, "
                    "factual lookups, documentation, or any information you don't know."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "max_results": {"type": "integer", "default": 5, "description": "Number of results"},
                    },
                    "required": ["query"],
                },
                fn=web_search,
            ),
            ToolDef(
                name="execute_python",
                description=(
                    "Execute Python code in a sandboxed environment. "
                    "Use for calculations, data processing, generating outputs. "
                    "Standard library modules (math, json, re, datetime, etc.) are available. "
                    "OS/subprocess/file access is blocked — use file tools instead."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                    },
                    "required": ["code"],
                },
                fn=execute_python,
            ),
            ToolDef(
                name="read_file",
                description="Read the contents of a file from the workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path within workspace"},
                    },
                    "required": ["path"],
                },
                fn=read_file,
            ),
            ToolDef(
                name="write_file",
                description="Write text content to a file in the workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                fn=write_file,
            ),
            ToolDef(
                name="list_files",
                description="List files and directories inside the workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "default": ".", "description": "Sub-directory to list"},
                    },
                },
                fn=list_files,
            ),
            ToolDef(
                name="delete_file",
                description="Delete a file from the workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
                fn=delete_file,
            ),
            ToolDef(
                name="calculator",
                description="Evaluate a mathematical expression. Supports math.* functions.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "e.g. 'sqrt(2) * pi'"},
                    },
                    "required": ["expression"],
                },
                fn=calculator,
            ),
            ToolDef(
                name="http_get",
                description="Perform an HTTP GET request to an external URL.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "headers": {"type": "object", "description": "Optional HTTP headers"},
                    },
                    "required": ["url"],
                },
                fn=http_get,
            ),
            ToolDef(
                name="http_post",
                description="Perform an HTTP POST request with a JSON body.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "body": {"type": "object"},
                        "headers": {"type": "object"},
                    },
                    "required": ["url", "body"],
                },
                fn=http_post,
            ),
            ToolDef(
                name="get_datetime",
                description="Get the current local date and time.",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
                fn=get_datetime,
            ),
            ToolDef(
                name="write_code_file",
                description=(
                    "Write generated source code to a file in the workspace. "
                    "Prefer this over write_file when writing code."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "language": {"type": "string", "description": "e.g. python, javascript"},
                        "code": {"type": "string"},
                    },
                    "required": ["filename", "language", "code"],
                },
                fn=write_code_file,
            ),
        ]
        for d in defs:
            self._tools[d.name] = d

    def register(self, tool: ToolDef):
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def all_tools(self) -> list[ToolDef]:
        return list(self._tools.values())

    def schemas_anthropic(self) -> list[dict]:
        return [t.to_anthropic() for t in self._tools.values()]

    def schemas_openai(self) -> list[dict]:
        return [t.to_openai() for t in self._tools.values()]

    async def call(self, name: str, **kwargs) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        try:
            return await tool.fn(**kwargs)
        except Exception as e:
            return f"Tool '{name}' raised: {traceback.format_exc()}"
