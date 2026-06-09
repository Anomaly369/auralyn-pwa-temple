"""Jarvis autonomous agent — Think → Plan → Act → Reflect loop."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import anthropic

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
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def complete(
        self,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int = settings.max_tokens,
        tools: list[dict] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        resp = await self._client.messages.create(**kwargs)
        # Return text from first text block
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
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )


# ── Main agent ─────────────────────────────────────────────────────────────

_JARVIS_SYSTEM = """You are JARVIS — an advanced autonomous AI assistant.

Capabilities:
• Deep reasoning and planning
• Tool use: web search, Python execution, file management, HTTP requests, math
• Long-term memory of users, projects, decisions, and preferences
• Code generation and automation scripting
• Multi-step task execution

Personality:
• Precise, direct, and efficient
• Proactively asks clarifying questions only when truly needed
• Explains your reasoning concisely before acting
• Always completes tasks fully — never stops halfway

Memory context will be injected below when relevant.

Current session: {session_id}
"""

_REFLECT_SYSTEM = """You are a reflection module for an AI agent.
Given the user's original request, the plan that was executed, and the results,
synthesise a final natural-language response for the user.
Be concise, complete, and human-friendly.
If there were errors, acknowledge them and suggest alternatives.
Do NOT repeat raw tool output verbatim — summarise it.
"""

_EXTRACT_MEMORY_SYSTEM = """You are a memory extraction module.
Given a conversation turn, extract facts, decisions, preferences, or skills
that should be remembered long-term.
Return a JSON array (may be empty) of:
[{"key": "...", "value": "...", "category": "fact|preference|project|decision|skill"}]
Return ONLY the JSON array, no other text."""


class JarvisAgent:
    def __init__(
        self,
        memory: MemorySystem | None = None,
        tools: ToolRegistry | None = None,
        planner: TaskPlanner | None = None,
    ):
        self.memory = memory or MemorySystem()
        self.tools = tools or ToolRegistry()
        self.planner = planner or TaskPlanner()
        self._llm = _AnthropicClient()
        self.planner.set_client(self._llm)

    # ── Public API ─────────────────────────────────────────────────────────

    async def chat(
        self, user_input: str, session_id: str | None = None
    ) -> AgentResponse:
        """Single-turn conversational interaction (no autonomous planning)."""
        session_id = session_id or str(uuid.uuid4())
        t0 = time.time()

        # Persist user message
        await self.memory.add_message(Message(
            role="user", content=user_input, session_id=session_id
        ))
        await self.memory.log_event("chat_request", {"input": user_input[:500]}, session_id)

        # Build context
        ctx = await self.memory.build_context_block(session_id, user_input)
        history = await self.memory.get_recent_messages(session_id, n=20)
        messages = self._history_to_messages(history, exclude_last=True)
        messages.append({"role": "user", "content": user_input})

        system = _JARVIS_SYSTEM.format(session_id=session_id)
        if ctx.strip():
            system += f"\n\n{ctx}"

        # ReAct loop: let the model call tools iteratively
        tool_calls_log = []
        iterations = 0

        while iterations < settings.max_agent_iterations:
            iterations += 1
            resp = await self._llm.complete_with_tools(
                system=system,
                messages=messages,
                model=settings.executor_model,
                tools=self.tools.schemas_anthropic(),
            )

            # Collect all content blocks
            assistant_content = resp.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Check stop reason
            if resp.stop_reason == "end_turn":
                # No tool calls — final answer
                final_text = self._extract_text(resp)
                break

            if resp.stop_reason == "tool_use":
                # Execute all tool calls, then continue
                tool_results = []
                for block in assistant_content:
                    if block.type != "tool_use":
                        continue
                    result = await self.tools.call(block.name, **block.input)
                    tool_calls_log.append({
                        "tool": block.name,
                        "input": block.input,
                        "result": result[:500],
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — break out
            final_text = self._extract_text(resp)
            break
        else:
            final_text = "I reached the maximum number of iterations. Please try a more specific request."

        # Persist assistant reply
        await self.memory.add_message(Message(
            role="assistant", content=final_text, session_id=session_id,
            metadata={"tool_calls": len(tool_calls_log)},
        ))

        # Background: extract and store memories
        n_memories = await self._extract_and_store_memories(
            user_input, final_text, session_id
        )

        duration_ms = int((time.time() - t0) * 1000)
        await self.memory.log_event(
            "chat_response",
            {"duration_ms": duration_ms, "tool_calls": len(tool_calls_log)},
            session_id,
        )

        return AgentResponse(
            content=final_text,
            tool_calls=tool_calls_log,
            memories_stored=n_memories,
            duration_ms=duration_ms,
            session_id=session_id,
        )

    async def execute_task(
        self, goal: str, session_id: str | None = None
    ) -> AgentResponse:
        """Autonomous multi-step task execution via planner."""
        session_id = session_id or str(uuid.uuid4())
        t0 = time.time()

        await self.memory.log_event("task_start", {"goal": goal[:500]}, session_id)

        # Phase 1: Plan
        ctx = await self.memory.build_context_block(session_id, goal)
        plan = await self.planner.plan(goal, context=ctx)

        # Persist task
        task_rec = Task(
            title=plan.steps[0].title if plan.steps else goal,
            description=goal,
            session_id=session_id,
            steps=[s.to_dict() for s in plan.steps],
        )
        task_id = await self.memory.create_task(task_rec)
        await self.memory.update_task(task_id, status="running")

        all_tool_calls: list[dict] = []
        step_results: list[str] = []

        # Phase 2: Execute steps
        for step in plan.steps:
            step.status = "running"
            await self.memory.update_task(
                task_id,
                steps=[s.to_dict() for s in plan.steps],
            )

            result = await self._execute_step(step, goal, step_results, session_id)
            step.result = result
            step.status = "done"
            step_results.append(f"[Step {step.index}: {step.title}]\n{result}")

            all_tool_calls.extend(
                getattr(self, "_last_step_tool_calls", [])
            )

        # Phase 3: Reflect — synthesise final answer
        reflection_prompt = (
            f"Original request: {goal}\n\n"
            f"Plan executed:\n{plan.summary()}\n\n"
            f"Step results:\n" + "\n\n".join(step_results)
        )
        final_answer = await self._llm.complete(
            system=_REFLECT_SYSTEM,
            messages=[{"role": "user", "content": reflection_prompt}],
            model=settings.executor_model,
        )

        # Phase 4: Store outcome
        await self.memory.update_task(task_id, status="done", result=final_answer)
        n_memories = await self._extract_and_store_memories(goal, final_answer, session_id)
        await self.memory.add_message(
            Message(role="assistant", content=final_answer, session_id=session_id,
                    metadata={"task_id": task_id, "plan_steps": len(plan.steps)})
        )

        duration_ms = int((time.time() - t0) * 1000)
        await self.memory.log_event(
            "task_complete",
            {"task_id": task_id, "duration_ms": duration_ms, "steps": len(plan.steps)},
            session_id,
        )

        return AgentResponse(
            content=final_answer,
            plan=plan,
            tool_calls=all_tool_calls,
            memories_stored=n_memories,
            duration_ms=duration_ms,
            session_id=session_id,
            task_id=task_id,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _execute_step(
        self,
        step: Step,
        overall_goal: str,
        prior_results: list[str],
        session_id: str,
    ) -> str:
        """Run a single plan step using the executor model + tools."""
        prior_context = "\n\n".join(prior_results[-3:]) if prior_results else "None yet."

        system = (
            f"You are an execution agent completing step {step.index} of a multi-step task.\n\n"
            f"Overall goal: {overall_goal}\n"
            f"Current step: {step.title}\n"
            f"Instructions: {step.description}\n"
            f"Tool hint: {step.tool_hint or 'use judgment'}\n\n"
            "Previous results:\n" + prior_context
        )

        messages = [{"role": "user", "content": f"Execute step {step.index}: {step.description}"}]
        step_tool_calls: list[dict] = []

        for _ in range(8):
            resp = await self._llm.complete_with_tools(
                system=system,
                messages=messages,
                model=settings.executor_model,
                tools=self.tools.schemas_anthropic(),
            )
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "end_turn":
                self._last_step_tool_calls = step_tool_calls
                return self._extract_text(resp)

            if resp.stop_reason == "tool_use":
                tool_results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    result = await self.tools.call(block.name, **block.input)
                    step_tool_calls.append({
                        "step": step.index,
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

        self._last_step_tool_calls = step_tool_calls
        return "Step completed (max iterations reached)."

    async def _extract_and_store_memories(
        self, user_input: str, assistant_reply: str, session_id: str
    ) -> int:
        """Ask the LLM to extract important facts and persist them."""
        try:
            excerpt = (
                f"User: {user_input[:1000]}\nAssistant: {assistant_reply[:1000]}"
            )
            raw = await self._llm.complete(
                system=_EXTRACT_MEMORY_SYSTEM,
                messages=[{"role": "user", "content": excerpt}],
                model=settings.executor_model,
                max_tokens=512,
                tools=None,
            )
            # Strip markdown if needed
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            items = json.loads(raw)
            if not isinstance(items, list):
                return 0
            count = 0
            for item in items[:5]:
                if "key" in item and "value" in item:
                    await self.memory.store_memory(Memory(
                        key=item["key"],
                        value=item["value"],
                        category=item.get("category", "fact"),
                        session_id=session_id,
                    ))
                    count += 1
            return count
        except Exception:
            return 0

    @staticmethod
    def _extract_text(resp: anthropic.types.Message) -> str:
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    @staticmethod
    def _history_to_messages(history: list[dict], exclude_last: bool = False) -> list[dict]:
        msgs = []
        items = history[:-1] if (exclude_last and history) else history
        for m in items:
            role = m["role"]
            if role in ("user", "assistant"):
                msgs.append({"role": role, "content": m["content"]})
        return msgs

    # ── Streaming variant ─────────────────────────────────────────────────

    async def stream_chat(
        self, user_input: str, session_id: str
    ) -> AsyncGenerator[str, None]:
        """Yield response tokens as they arrive (text only, no tool use)."""
        ctx = await self.memory.build_context_block(session_id, user_input)
        history = await self.memory.get_recent_messages(session_id, n=20)
        messages = self._history_to_messages(history, exclude_last=True)
        messages.append({"role": "user", "content": user_input})

        system = _JARVIS_SYSTEM.format(session_id=session_id)
        if ctx.strip():
            system += f"\n\n{ctx}"

        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        full_text = []

        async with client.messages.stream(
            model=settings.executor_model,
            max_tokens=settings.max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                full_text.append(text)
                yield text

        final = "".join(full_text)
        await self.memory.add_message(
            Message(role="assistant", content=final, session_id=session_id)
        )
        await self._extract_and_store_memories(user_input, final, session_id)
