"""MultiAgentOrchestrator — the central Jarvis Core that drives all agent collaboration."""
from __future__ import annotations

import asyncio
import uuid
import time
from typing import TYPE_CHECKING

from jarvis.config import settings
from jarvis.orchestrator.modes import RunStatus, ExecutionMode
from jarvis.core.memory import Memory

if TYPE_CHECKING:
    from jarvis.agents.registry import AgentRegistry
    from jarvis.core.memory import MemorySystem
    from jarvis.orchestrator.message_bus import MessageBus


class MultiAgentOrchestrator:
    """Routes tasks between agents, maintains global state, enforces execution rules."""

    def __init__(
        self,
        registry: "AgentRegistry",
        memory: "MemorySystem",
        message_bus: "MessageBus",
    ):
        self.registry = registry
        self.memory = memory
        self.bus = message_bus
        self._active_runs: dict[str, asyncio.Task] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    async def start_run(
        self, goal: str, session_id: str, mode: str = ExecutionMode.STANDARD
    ) -> str:
        """Kick off an orchestration run. Returns run_id immediately (non-blocking)."""
        run_id = str(uuid.uuid4())
        await self.memory.create_run(run_id, session_id, goal, mode)

        if mode == ExecutionMode.DEBATE:
            coro = self._run_debate(run_id, goal, session_id)
        elif mode == ExecutionMode.SWARM:
            coro = self._run_swarm(run_id, goal, session_id)
        else:
            coro = self._run_standard(run_id, goal, session_id)

        task = asyncio.create_task(coro)
        self._active_runs[run_id] = task
        task.add_done_callback(lambda t: self._on_task_done(run_id, t))
        return run_id

    async def get_status(self, run_id: str) -> dict | None:
        run = await self.memory.get_run(run_id)
        if not run:
            return None
        agent_statuses = await self.registry.get_status_dicts()
        return {**run, "agents": agent_statuses}

    async def list_runs(self, session_id: str | None = None, limit: int = 20) -> list[dict]:
        return await self.memory.list_runs(session_id=session_id, limit=limit)

    async def get_messages(self, run_id: str) -> list[dict]:
        return await self.memory.get_agent_messages(run_id)

    # ── Standard execution (7-step cycle) ─────────────────────────────────

    async def _run_standard(self, run_id: str, goal: str, session_id: str):
        try:
            await self._broadcast_phase(run_id, RunStatus.PLANNING, "Strategist decomposing goal")

            # Step 1: Build memory context
            memory_context = await self.memory.build_context_block(session_id, goal)

            # Step 2: Strategist decomposes
            strategist = self.registry.get("strategist")
            strategy_result = await strategist.process(
                task={"goal": goal, "task_id": run_id},
                context={"memory_context": memory_context},
                run_id=run_id,
            )
            subtasks = strategy_result.get("subtasks", [])
            if not subtasks:
                subtasks = [{"title": "Complete goal", "description": goal,
                             "agent_hint": "builder", "constraints": []}]

            # Step 3: Parallel execution (research first, then build)
            await self._broadcast_phase(run_id, RunStatus.EXECUTING, "Research + Build in parallel")

            research_tasks = [t for t in subtasks if t.get("agent_hint") == "researcher"]
            build_tasks = [t for t in subtasks if t.get("agent_hint") == "builder"]

            researcher = self.registry.get("researcher")
            builder = self.registry.get("builder")

            # Research phase (parallel across multiple research subtasks)
            research_results = []
            if research_tasks:
                research_coroutines = [
                    researcher.process(
                        task={**t, "task_id": run_id},
                        context={"memory_context": memory_context},
                        run_id=run_id,
                    )
                    for t in research_tasks
                ]
                research_results = list(await asyncio.gather(*research_coroutines))

            # Consolidate research insights for builder
            research_insights = "\n\n".join(
                r.get("result", "") for r in research_results if r.get("result")
            )

            # Build phase (parallel across multiple build subtasks)
            builder_results = []
            if build_tasks:
                build_coroutines = [
                    builder.process(
                        task={**t, "task_id": run_id},
                        context={
                            "memory_context": memory_context,
                            "research_insights": research_insights,
                        },
                        run_id=run_id,
                    )
                    for t in build_tasks
                ]
                builder_results = list(await asyncio.gather(*build_coroutines))

            # Primary result = last builder result or last research result
            primary_result = (
                builder_results[-1] if builder_results
                else (research_results[-1] if research_results
                      else {"result": "No output produced.", "confidence": 0.5})
            )

            # Step 4: Critic evaluation with retry loop
            await self._broadcast_phase(run_id, RunStatus.EVALUATING, "Critic evaluating")

            all_results = {
                "researcher": research_results[-1] if research_results else None,
                "builder": builder_results[-1] if builder_results else None,
            }
            all_results = {k: v for k, v in all_results.items() if v}

            critique = None
            approved = False
            retry_count = 0

            critic = self.registry.get("critic")

            while not approved and retry_count <= settings.max_critic_retries:
                critique_result = await critic.evaluate(
                    goal=goal,
                    agent_results={k: v for k, v in all_results.items()},
                    run_id=run_id,
                    attempt=retry_count + 1,
                )
                critique = critique_result.get("result", {})
                approved = critique_result.get("approved", False)

                if not approved and retry_count < settings.max_critic_retries:
                    retry_count += 1
                    await self._broadcast_phase(
                        run_id, RunStatus.RETRYING,
                        f"Retry {retry_count}/{settings.max_critic_retries} — addressing critique"
                    )
                    feedback = "\n".join(critique.get("suggestions", [])[:5])

                    if build_tasks:
                        retry_result = await builder.process(
                            task={**(build_tasks[-1]), "task_id": run_id},
                            context={
                                "memory_context": memory_context,
                                "research_insights": research_insights,
                                "critique_feedback": feedback,
                            },
                            run_id=run_id,
                        )
                        primary_result = retry_result
                        all_results["builder"] = retry_result
                else:
                    break

            # Step 5: Merge results
            await self._broadcast_phase(run_id, RunStatus.MERGING, "Merging results")
            merged_content = primary_result.get("result", "")

            # Step 6: Store memory
            await self.memory.upsert_agent_performance(
                "orchestrator", critique.get("overall", 0.8) if critique else 0.8
            )
            for role_name, res in all_results.items():
                if res:
                    conf = res.get("confidence", 0.8)
                    await self.memory.upsert_agent_performance(role_name, conf)

            # Persist key insight to long-term memory
            await self.memory.store_memory(Memory(
                key=f"run:{run_id[:8]}",
                value=f"Goal: {goal[:100]} | Result: {merged_content[:200]}",
                category="decision",
                session_id=session_id,
                confidence=critique.get("overall", 0.8) if critique else 0.8,
            ))

            # Step 7: Communicator formats final response
            await self._broadcast_phase(run_id, RunStatus.COMMUNICATING, "Communicator formatting")
            communicator = self.registry.get("communicator")
            comm_result = await communicator.process(
                task={
                    "goal": goal,
                    "merged_results": merged_content,
                    "critique": critique or {},
                    "research_insights": research_insights[:1000],
                    "task_id": run_id,
                },
                context={"memory_context": memory_context},
                run_id=run_id,
            )
            final_response = comm_result.get("result", merged_content)

            await self.memory.update_run(
                run_id,
                status="complete",
                result=final_response,
                agent_results={
                    "researcher": research_results[-1] if research_results else {},
                    "builder": builder_results[-1] if builder_results else {},
                    "critique": critique or {},
                },
            )

            await self.bus.broadcast_event({
                "type": "run_complete",
                "run_id": run_id,
                "result": final_response,
                "critique_score": critique.get("overall", 1.0) if critique else 1.0,
            })

        except Exception as exc:
            await self.memory.update_run(run_id, status="failed", result=str(exc))
            await self.bus.broadcast_event({
                "type": "run_failed",
                "run_id": run_id,
                "error": str(exc),
            })
            raise

    # ── Debate mode ────────────────────────────────────────────────────────

    async def _run_debate(self, run_id: str, topic: str, session_id: str):
        try:
            await self._broadcast_phase(run_id, RunStatus.EXECUTING, "Debate mode starting")
            memory_context = await self.memory.build_context_block(session_id, topic)

            builder = self.registry.get("builder")
            critic = self.registry.get("critic")
            strategist = self.registry.get("strategist")
            communicator = self.registry.get("communicator")

            transcript = []

            # Round 1: Builder argues FOR
            pro_result = await builder.process(
                task={
                    "description": f"Make the strongest possible argument FOR: {topic}\n"
                                   "Present your position, key arguments, and supporting evidence.",
                    "constraints": ["Argue in favor only", "Be persuasive and specific"],
                    "task_id": run_id,
                },
                context={"memory_context": memory_context},
                run_id=run_id,
            )
            transcript.append({"position": "pro", "argument": pro_result.get("result", "")})

            await self.bus.broadcast_event({
                "type": "debate_round",
                "run_id": run_id,
                "round": 1,
                "position": "pro",
                "agent": "builder",
                "content": pro_result.get("result", "")[:300],
            })

            # Round 2: Critic argues AGAINST
            con_result = await critic.process(
                task={
                    "goal": f"Argue AGAINST: {topic}",
                    "agent_results": {"pro_argument": pro_result},
                    "attempt": 1,
                },
                context={"memory_context": memory_context},
                run_id=run_id,
            )
            # For debate, critic produces a counter-argument
            con_text = await self._get_debate_counter(
                topic, pro_result.get("result", ""), memory_context, run_id
            )
            transcript.append({"position": "con", "argument": con_text})

            await self.bus.broadcast_event({
                "type": "debate_round",
                "run_id": run_id,
                "round": 2,
                "position": "con",
                "agent": "critic",
                "content": con_text[:300],
            })

            # Round 3: Builder rebuts
            rebuttal = await builder.process(
                task={
                    "description": f"Rebut this counter-argument about: {topic}\n\nCounter-argument:\n{con_text}",
                    "constraints": ["Address specific objections", "Strengthen your original position"],
                    "task_id": run_id,
                },
                context={
                    "memory_context": memory_context,
                    "critique_feedback": f"Counter-argument to address:\n{con_text[:500]}",
                },
                run_id=run_id,
            )
            transcript.append({"position": "pro_rebuttal", "argument": rebuttal.get("result", "")})

            # Strategist adjudicates
            adjudication = await strategist.adjudicate(run_id, transcript)
            winning_argument = adjudication.get("winning_argument", transcript[0]["argument"])
            rationale = adjudication.get("rationale", "")

            await self.bus.broadcast_event({
                "type": "debate_verdict",
                "run_id": run_id,
                "winner": adjudication.get("winner", "pro"),
                "rationale": rationale,
            })

            # Communicator formats debate summary
            debate_summary = "\n\n".join([
                f"**PRO (Round 1):**\n{transcript[0]['argument'][:500]}",
                f"**CON (Round 2):**\n{transcript[1]['argument'][:500]}",
                f"**PRO REBUTTAL (Round 3):**\n{transcript[2]['argument'][:500]}",
                f"**VERDICT:** {rationale}",
                f"**Winning position:** {adjudication.get('winner', 'pro').upper()}",
                f"**Winning argument:**\n{winning_argument[:500]}",
            ])

            comm_result = await communicator.process(
                task={
                    "goal": f"Summarize this debate about: {topic}",
                    "merged_results": debate_summary,
                    "critique": {},
                    "task_id": run_id,
                },
                context={"memory_context": memory_context},
                run_id=run_id,
            )
            final_response = comm_result.get("result", debate_summary)

            await self.memory.update_run(run_id, status="complete", result=final_response)
            await self.bus.broadcast_event({
                "type": "run_complete",
                "run_id": run_id,
                "result": final_response,
                "mode": "debate",
            })

        except Exception as exc:
            await self.memory.update_run(run_id, status="failed", result=str(exc))
            await self.bus.broadcast_event({"type": "run_failed", "run_id": run_id, "error": str(exc)})
            raise

    async def _get_debate_counter(
        self, topic: str, pro_argument: str, memory_context: str, run_id: str
    ) -> str:
        """Generate a counter-argument for debate mode using the critic's LLM."""
        critic = self.registry.get("critic")
        counter_system = (
            "You are arguing AGAINST the following position. Be rigorous, find flaws, "
            "present evidence against it. Do not agree with the pro position.\n"
            f"Topic: {topic}\n\nPro argument to rebut:\n{pro_argument[:1000]}"
        )
        raw = await critic._llm.complete(
            system=counter_system,
            messages=[{"role": "user", "content": "Present the strongest counter-argument."}],
            model=settings.executor_model,
        )
        return raw

    # ── Swarm mode ─────────────────────────────────────────────────────────

    async def _run_swarm(self, run_id: str, goal: str, session_id: str, count: int = None):
        if count is None:
            count = settings.swarm_size
        try:
            await self._broadcast_phase(
                run_id, RunStatus.EXECUTING,
                f"Swarm mode: {count} agents solving independently"
            )
            memory_context = await self.memory.build_context_block(session_id, goal)
            researcher = self.registry.get("researcher")

            # Run count concurrent researcher instances
            swarm_coroutines = [
                researcher.process(
                    task={"description": goal, "title": goal, "task_id": f"{run_id}_swarm_{i}"},
                    context={"memory_context": memory_context},
                    run_id=run_id,
                )
                for i in range(count)
            ]
            swarm_results = list(await asyncio.gather(*swarm_coroutines, return_exceptions=True))

            valid_results = [
                r for r in swarm_results
                if isinstance(r, dict) and r.get("result")
            ]

            for i, r in enumerate(valid_results):
                await self.bus.broadcast_event({
                    "type": "swarm_vote",
                    "run_id": run_id,
                    "instance": i + 1,
                    "confidence": r.get("confidence", 0.7),
                    "content": r.get("result", "")[:200],
                })

            # Critic picks the best
            critic = self.registry.get("critic")
            best_result = valid_results[0] if valid_results else {"result": "No swarm results.", "confidence": 0.5}

            if len(valid_results) > 1:
                all_results = {
                    f"swarm_instance_{i+1}": r
                    for i, r in enumerate(valid_results)
                }
                critique_result = await critic.evaluate(
                    goal=f"Pick the best answer for: {goal}",
                    agent_results=all_results,
                    run_id=run_id,
                    attempt=1,
                )
                # Use the highest-confidence valid result
                best_result = max(valid_results, key=lambda r: r.get("confidence", 0))

            await self.bus.broadcast_event({
                "type": "swarm_winner",
                "run_id": run_id,
                "confidence": best_result.get("confidence", 0.7),
            })

            # Communicator formats
            communicator = self.registry.get("communicator")
            comm_result = await communicator.process(
                task={
                    "goal": goal,
                    "merged_results": best_result.get("result", ""),
                    "critique": {},
                    "task_id": run_id,
                },
                context={"memory_context": memory_context},
                run_id=run_id,
            )
            final_response = comm_result.get("result", best_result.get("result", ""))

            await self.memory.update_run(run_id, status="complete", result=final_response)
            await self.bus.broadcast_event({
                "type": "run_complete",
                "run_id": run_id,
                "result": final_response,
                "mode": "swarm",
                "instances": count,
            })

        except Exception as exc:
            await self.memory.update_run(run_id, status="failed", result=str(exc))
            await self.bus.broadcast_event({"type": "run_failed", "run_id": run_id, "error": str(exc)})
            raise

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _broadcast_phase(self, run_id: str, status: RunStatus, detail: str):
        await self.memory.update_run(run_id, status=status.value)
        await self.bus.broadcast_event({
            "type": "phase_change",
            "run_id": run_id,
            "phase": status.value,
            "detail": detail,
            "timestamp": time.time(),
        })

    def _on_task_done(self, run_id: str, task: asyncio.Task):
        self._active_runs.pop(run_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            asyncio.create_task(
                self.memory.update_run(run_id, status="failed", result=str(exc))
            )
