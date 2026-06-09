"""Jarvis autonomous agent — Think → Plan → Act → Reflect loop."""
from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from jarvis.config import settings
from jarvis.core.memory import Memory, MemorySystem, Message, Task
from jarvis.core.planner import Plan, Step, TaskPlanner
from jarvis.core.tools import ToolRegistry


# ── Agent response ─────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    content: str
    plan: Plan | None = None
    tool_calls: list[dict] = field(default_factory=list)
    memories_stored: int = 0
    duration_ms: int = 0
    session_id: str = ""
    task_id: int | None = None


# ── LLM client wrapper ─────────────────────────────────────────────────────

class _AnthropicClient:
    """Thin async wrapper around the Anthropic Messages API."""

    def __init__(self):
        import anthropic as _ant
        self._client = _ant.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def complete(self, system, messages, model, max_tokens=settings.max_tokens, tools=None):
        import anthropic as _ant
        kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
        resp = await self._client.messages.create(**kwargs)
        for block in resp.content:
            if block.type == "text":
                return block.text
        return ""

    async def complete_with_tools(self, system, messages, model, tools, max_tokens=settings.max_tokens):
        import anthropic as _ant
        return await self._client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system, messages=messages, tools=tools,
        )

    async def stream(self, system, messages, model, max_tokens=settings.max_tokens):
        import anthropic as _ant
        client = _ant.AsyncAnthropic(api_key=settings.anthropic_api_key)
        async with client.messages.stream(
            model=model, max_tokens=max_tokens, system=system, messages=messages,
        ) as s:
            async for text in s.text_stream:
                yield text


# ── Demo brain (no API key needed) ────────────────────────────────────────

class _DemoBrain:
    """
    Smart rule-based brain that actually uses real tools.
    Activated automatically when no Anthropic API key is set.
    """

    GREETINGS = [
        "Hello! I'm JARVIS — your autonomous AI assistant. I'm running in **demo mode** right now (no API key set). I can still use real tools: web search, run Python code, manage files, and more. Add your `ANTHROPIC_API_KEY` to `.env` to unlock my full intelligence.\n\nWhat can I help you with?",
        "JARVIS online. Running in demo mode — real tools available, limited reasoning. Set `ANTHROPIC_API_KEY` in `.env` for full autonomous capability.",
    ]

    TOOL_TRIGGERS = [
        (r"search (.+)", "web_search", lambda m: {"query": m.group(1)}),
        (r"calculate (.+)", "calculator", lambda m: {"expression": m.group(1)}),
        (r"what.s the time|current time|what time", "get_datetime", lambda m: {}),
        (r"list files?|show files?|what files", "list_files", lambda m: {}),
        (r"run code[:\n]+([\s\S]+)", "execute_python", lambda m: {"code": m.group(1)}),
        (r"python[:\n]+([\s\S]+)", "execute_python", lambda m: {"code": m.group(1)}),
        (r"weather in (.+)", "web_search", lambda m: {"query": f"current weather in {m.group(1)}"}),
        (r"news about (.+)", "web_search", lambda m: {"query": f"latest news {m.group(1)}"}),
        (r"who is (.+)", "web_search", lambda m: {"query": m.group(1)}),
        (r"what is (.+)", "web_search", lambda m: {"query": m.group(1)}),
        (r"how (to|do) (.+)", "web_search", lambda m: {"query": m.group(0)}),
    ]

    RESPONSES = {
        "hello": "Hey! I'm JARVIS. Demo mode is active — I can search the web, run code, and use tools. What do you need?",
        "how are you": "Systems nominal. All 11 tools loaded. Memory system active. What's your command?",
        "what can you do": (
            "In full mode I'm a complete autonomous AI OS. Right now in demo mode I can:\n\n"
            "• **Search the web** — ask me to search anything\n"
            "• **Run Python code** — ask me to calculate or run code\n"
            "• **Manage files** — read/write files in the workspace\n"
            "• **Check the time/date**\n"
            "• **Remember things** — memory system is fully active\n\n"
            "For full autonomous intelligence, add `ANTHROPIC_API_KEY` to `jarvis/.env`.\n"
            "Get a free key at **console.anthropic.com**"
        ),
        "api key": (
            "To unlock my full brain:\n\n"
            "1. Go to **console.anthropic.com**\n"
            "2. Sign up free → API Keys → Create Key\n"
            "3. Open `jarvis/.env` and set:\n```\nANTHROPIC_API_KEY=sk-ant-...\n```\n"
            "4. Restart the server\n\n"
            "Then I can plan, reason, write code, trade, make YouTube videos — everything."
        ),
        "youtube": "YouTube automation plugin is loaded and ready! Once you add your API key I can run the full pipeline: niche research → script → voiceover → thumbnail → video → upload. All autonomous.",
        "trading": "MT5 trading plugin is loaded. With your API key + MT5 connected, I can scan symbols, score setups using RSI + MACD + EMA + Bollinger + Volume confluence, and execute trades automatically.",
        "memory": "Memory system is active right now — I'm storing this conversation in SQLite. Everything you tell me persists across sessions.",
    }

    async def respond(self, user_input: str, tools: ToolRegistry) -> tuple[str, list[dict]]:
        """Return (response_text, tool_calls_log)."""
        lower = user_input.lower().strip()
        tool_calls = []

        # Try tool triggers first
        for pattern, tool_name, arg_fn in self.TOOL_TRIGGERS:
            m = re.search(pattern, lower, re.IGNORECASE)
            if m:
                result = await tools.call(tool_name, **arg_fn(m))
                tool_calls.append({"tool": tool_name, "result": result[:200]})
                return f"**{tool_name}** result:\n\n{result}", tool_calls

        # Keyword responses
        for kw, resp in self.RESPONSES.items():
            if kw in lower:
                return resp, []

        # Generic fallback
        fallbacks = [
            f"I understand you're asking about *{user_input[:60]}*. In demo mode my reasoning is limited — but I can search the web for that! Try: **search {user_input[:40]}**",
            "I'm in demo mode so my reasoning is basic. I CAN use real tools though — try asking me to search something, calculate, or run Python code.",
            f"Got it. Add your Anthropic API key to `jarvis/.env` and I'll give you a full intelligent response to that. For now, want me to **search the web** for: *{user_input[:50]}*?",
        ]
        return random.choice(fallbacks), []

    async def stream_respond(self, user_input: str, tools: ToolRegistry):
        text, _ = await self.respond(user_input, tools)
        for word in text.split(" "):
            yield word + " "
            await __import__("asyncio").sleep(0.02)


# ── Brain factory ──────────────────────────────────────────────────────────

def _make_llm():
    if settings.anthropic_api_key and settings.anthropic_api_key.startswith("sk-"):
        return _AnthropicClient(), False
    return None, True  # demo mode


# ── System prompts ─────────────────────────────────────────────────────────

_JARVIS_SYSTEM = """You are JARVIS — an advanced autonomous AI assistant.

Capabilities:
• Deep reasoning and planning
• Tool use: web search, Python execution, file management, HTTP requests, math
• Long-term memory of users, projects, decisions, and preferences
• Code generation and automation scripting
• Faceless YouTube channel creation and autonomous upload
• MetaTrader 5 algorithmic trading with multi-strategy confluence
• Multi-step task execution

Personality:
• Precise, direct, and efficient
• Proactively asks clarifying questions only when truly needed
• Explains your reasoning concisely before acting
• Always completes tasks fully — never stops halfway
• Address the user confidently and personally

Memory context will be injected below when relevant.
Current session: {session_id}
"""

_REFLECT_SYSTEM = """You are a reflection module for an AI agent.
Given the user's original request, the plan that was executed, and the results,
synthesise a final natural-language response for the user.
Be concise, complete, and human-friendly.
If there were errors, acknowledge them and suggest alternatives.
Do NOT repeat raw tool output verbatim — summarise it."""

_EXTRACT_MEMORY_SYSTEM = """You are a memory extraction module.
Given a conversation turn, extract facts, decisions, preferences, or skills
that should be remembered long-term.
Return a JSON array (may be empty) of:
[{"key": "...", "value": "...", "category": "fact|preference|project|decision|skill"}]
Return ONLY the JSON array, no other text."""


# ── Main agent ─────────────────────────────────────────────────────────────

class JarvisAgent:
    def __init__(self, memory=None, tools=None, planner=None):
        self.memory = memory or MemorySystem()
        self.tools  = tools  or ToolRegistry()
        self.planner = planner or TaskPlanner()
        self._llm, self._demo_mode = _make_llm()
        self._demo_brain = _DemoBrain()
        if not self._demo_mode:
            self.planner.set_client(self._llm)

    # ── Public API ─────────────────────────────────────────────────────────

    async def chat(self, user_input: str, session_id: str | None = None) -> AgentResponse:
        session_id = session_id or str(uuid.uuid4())
        t0 = time.time()

        await self.memory.add_message(Message(role="user", content=user_input, session_id=session_id))
        await self.memory.log_event("chat_request", {"input": user_input[:500]}, session_id)

        if self._demo_mode:
            final_text, tool_calls_log = await self._demo_brain.respond(user_input, self.tools)
        else:
            final_text, tool_calls_log = await self._full_chat(user_input, session_id)

        await self.memory.add_message(Message(
            role="assistant", content=final_text, session_id=session_id,
            metadata={"tool_calls": len(tool_calls_log)},
        ))
        n_mem = await self._maybe_extract_memories(user_input, final_text, session_id)
        duration_ms = int((time.time() - t0) * 1000)
        await self.memory.log_event("chat_response", {"duration_ms": duration_ms}, session_id)
        return AgentResponse(content=final_text, tool_calls=tool_calls_log,
                             memories_stored=n_mem, duration_ms=duration_ms, session_id=session_id)

    async def execute_task(self, goal: str, session_id: str | None = None) -> AgentResponse:
        session_id = session_id or str(uuid.uuid4())

        if self._demo_mode:
            # In demo mode, still execute real tools where possible
            text, tool_calls = await self._demo_brain.respond(
                f"execute task: {goal}", self.tools
            )
            text = (
                f"**Task received:** {goal}\n\n"
                f"I'm in **demo mode** — add your `ANTHROPIC_API_KEY` to `jarvis/.env` "
                f"for full autonomous task execution with planning and multi-step tool use.\n\n"
                + (f"I did run some tools for you:\n\n{text}" if tool_calls else
                   "Once your key is set, I'll decompose this into steps, use tools, and complete it fully.")
            )
            return AgentResponse(content=text, tool_calls=tool_calls, session_id=session_id)

        return await self._full_task(goal, session_id)

    async def stream_chat(self, user_input: str, session_id: str) -> AsyncGenerator[str, None]:
        ctx = await self.memory.build_context_block(session_id, user_input)
        history = await self.memory.get_recent_messages(session_id, n=20)

        if self._demo_mode:
            async for token in self._demo_brain.stream_respond(user_input, self.tools):
                yield token
            return

        import anthropic as _ant
        messages = self._history_to_messages(history, exclude_last=True)
        messages.append({"role": "user", "content": user_input})
        system = _JARVIS_SYSTEM.format(session_id=session_id)
        if ctx.strip():
            system += f"\n\n{ctx}"

        full_text = []
        client = _ant.AsyncAnthropic(api_key=settings.anthropic_api_key)
        async with client.messages.stream(
            model=settings.executor_model, max_tokens=settings.max_tokens,
            system=system, messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                full_text.append(text)
                yield text

        final = "".join(full_text)
        await self.memory.add_message(Message(role="assistant", content=final, session_id=session_id))
        await self._maybe_extract_memories(user_input, final, session_id)

    # ── Full (keyed) implementations ───────────────────────────────────────

    async def _full_chat(self, user_input: str, session_id: str) -> tuple[str, list]:
        ctx = await self.memory.build_context_block(session_id, user_input)
        history = await self.memory.get_recent_messages(session_id, n=20)
        messages = self._history_to_messages(history, exclude_last=True)
        messages.append({"role": "user", "content": user_input})
        system = _JARVIS_SYSTEM.format(session_id=session_id)
        if ctx.strip():
            system += f"\n\n{ctx}"

        tool_calls_log = []
        for _ in range(settings.max_agent_iterations):
            import anthropic as _ant
            resp = await self._llm.complete_with_tools(
                system=system, messages=messages,
                model=settings.executor_model, tools=self.tools.schemas_anthropic(),
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
                    tool_calls_log.append({"tool": block.name, "input": block.input, "result": result[:500]})
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "user", "content": tool_results})
        return "Reached max iterations.", tool_calls_log

    async def _full_task(self, goal: str, session_id: str) -> AgentResponse:
        t0 = time.time()
        await self.memory.log_event("task_start", {"goal": goal[:500]}, session_id)
        ctx = await self.memory.build_context_block(session_id, goal)
        plan = await self.planner.plan(goal, context=ctx)
        task_rec = Task(title=plan.steps[0].title if plan.steps else goal,
                        description=goal, session_id=session_id,
                        steps=[s.to_dict() for s in plan.steps])
        task_id = await self.memory.create_task(task_rec)
        await self.memory.update_task(task_id, status="running")

        all_tool_calls, step_results = [], []
        for step in plan.steps:
            step.status = "running"
            await self.memory.update_task(task_id, steps=[s.to_dict() for s in plan.steps])
            result = await self._execute_step(step, goal, step_results, session_id)
            step.result = result
            step.status = "done"
            step_results.append(f"[Step {step.index}: {step.title}]\n{result}")
            all_tool_calls.extend(getattr(self, "_last_step_tool_calls", []))

        reflection_prompt = (
            f"Original request: {goal}\n\nPlan:\n{plan.summary()}\n\n"
            f"Results:\n" + "\n\n".join(step_results)
        )
        final_answer = await self._llm.complete(
            system=_REFLECT_SYSTEM,
            messages=[{"role": "user", "content": reflection_prompt}],
            model=settings.executor_model,
        )
        await self.memory.update_task(task_id, status="done", result=final_answer)
        n_mem = await self._maybe_extract_memories(goal, final_answer, session_id)
        await self.memory.add_message(Message(role="assistant", content=final_answer,
                                              session_id=session_id, metadata={"task_id": task_id}))
        duration_ms = int((time.time() - t0) * 1000)
        return AgentResponse(content=final_answer, plan=plan, tool_calls=all_tool_calls,
                             memories_stored=n_mem, duration_ms=duration_ms,
                             session_id=session_id, task_id=task_id)

    async def _execute_step(self, step, goal, prior_results, session_id):
        prior = "\n\n".join(prior_results[-3:]) if prior_results else "None"
        system = (f"Execute step {step.index}: {step.description}\nGoal: {goal}\n"
                  f"Prior results:\n{prior}")
        messages = [{"role": "user", "content": f"Execute: {step.description}"}]
        step_calls = []
        for _ in range(8):
            resp = await self._llm.complete_with_tools(
                system=system, messages=messages,
                model=settings.executor_model, tools=self.tools.schemas_anthropic(),
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "end_turn":
                self._last_step_tool_calls = step_calls
                return self._extract_text(resp)
            if resp.stop_reason == "tool_use":
                tool_results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    result = await self.tools.call(block.name, **block.input)
                    step_calls.append({"step": step.index, "tool": block.name, "result": result[:300]})
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "user", "content": tool_results})
        self._last_step_tool_calls = step_calls
        return "Step done."

    async def _maybe_extract_memories(self, user_input, reply, session_id) -> int:
        if self._demo_mode or not self._llm:
            return 0
        try:
            raw = await self._llm.complete(
                system=_EXTRACT_MEMORY_SYSTEM,
                messages=[{"role": "user", "content": f"User: {user_input[:800]}\nAssistant: {reply[:800]}"}],
                model=settings.executor_model, max_tokens=512, tools=None,
            )
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            items = json.loads(raw)
            count = 0
            for item in (items if isinstance(items, list) else [])[:5]:
                if "key" in item and "value" in item:
                    await self.memory.store_memory(Memory(
                        key=item["key"], value=item["value"],
                        category=item.get("category", "fact"), session_id=session_id,
                    ))
                    count += 1
            return count
        except Exception:
            return 0

    @staticmethod
    def _extract_text(resp) -> str:
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    @staticmethod
    def _history_to_messages(history, exclude_last=False):
        items = history[:-1] if (exclude_last and history) else history
        return [{"role": m["role"], "content": m["content"]}
                for m in items if m["role"] in ("user", "assistant")]
