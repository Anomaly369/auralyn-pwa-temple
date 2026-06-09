"""FastAPI server — REST + WebSocket endpoints for JARVIS."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks, FastAPI, File, HTTPException,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

from jarvis.config import settings
from jarvis.core.agent import JarvisAgent, AgentResponse
from jarvis.core.memory import MemorySystem
from jarvis.core.tools import ToolRegistry
from jarvis.plugins.youtube_automation import YouTubeAutomationPlugin
from jarvis.agents.registry import AgentRegistry
from jarvis.orchestrator.message_bus import MessageBus
from jarvis.orchestrator.core import MultiAgentOrchestrator


# ── Request / Response schemas ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class TaskRequest(BaseModel):
    goal: str
    session_id: str | None = None


class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 10


class PreferenceRequest(BaseModel):
    key: str
    value: Any


class MultiAgentRunRequest(BaseModel):
    goal: str
    session_id: str | None = None
    mode: str = "standard"


class DebateRequest(BaseModel):
    topic: str
    session_id: str | None = None


class SwarmRequest(BaseModel):
    goal: str
    session_id: str | None = None
    count: int = 3


# ── App factory ────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="JARVIS AI",
        description="Autonomous AI assistant — chat, plan, act, remember.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Shared instances ────────────────────────────────────────────────────

    memory = MemorySystem()
    tools = ToolRegistry()

    # Register all plugins
    YouTubeAutomationPlugin().register(tools)

    from jarvis.plugins.mt5_trading import MT5TradingPlugin
    MT5TradingPlugin().register(tools)

    agent = JarvisAgent(memory=memory, tools=tools)

    # ── Multi-agent system ──────────────────────────────────────────────────

    active_orchestrator_ws: dict[str, WebSocket] = {}

    async def _broadcast_to_clients(event: dict):
        dead = []
        for sid, ws in list(active_orchestrator_ws.items()):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(sid)
        for sid in dead:
            active_orchestrator_ws.pop(sid, None)

    message_bus = MessageBus()
    message_bus.register_ws_callback(_broadcast_to_clients)

    agent_registry = AgentRegistry(memory=memory, tools=tools, message_bus=message_bus)
    orchestrator = MultiAgentOrchestrator(
        registry=agent_registry, memory=memory, message_bus=message_bus
    )

    # ── Startup ─────────────────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        await memory.init()

    # ── Static + Templates ──────────────────────────────────────────────────

    _base = Path(__file__).parent.parent
    templates_dir = _base / "dashboard" / "templates"
    static_dir = _base / "dashboard" / "static"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(templates_dir))

    # ── Dashboard ───────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse(request, "index.html")

    # ── Health ──────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "online", "time": time.time(), "version": "1.0.0"}

    # ── Chat ────────────────────────────────────────────────────────────────

    @app.post("/chat")
    async def chat(req: ChatRequest):
        session_id = req.session_id or str(uuid.uuid4())
        try:
            resp: AgentResponse = await agent.chat(req.message, session_id=session_id)
            return {
                "reply": resp.content,
                "session_id": session_id,
                "tool_calls": resp.tool_calls,
                "memories_stored": resp.memories_stored,
                "duration_ms": resp.duration_ms,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/task")
    async def execute_task(req: TaskRequest, background_tasks: BackgroundTasks):
        session_id = req.session_id or str(uuid.uuid4())
        try:
            resp: AgentResponse = await agent.execute_task(req.goal, session_id=session_id)
            result: dict = {
                "result": resp.content,
                "session_id": session_id,
                "task_id": resp.task_id,
                "plan": resp.plan.to_dict() if resp.plan else None,
                "tool_calls": resp.tool_calls,
                "memories_stored": resp.memories_stored,
                "duration_ms": resp.duration_ms,
            }
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── WebSocket streaming ─────────────────────────────────────────────────

    @app.websocket("/ws/{session_id}")
    async def websocket_chat(websocket: WebSocket, session_id: str):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                msg = payload.get("message", "")
                mode = payload.get("mode", "chat")  # "chat" | "task"

                if mode == "task":
                    resp = await agent.execute_task(msg, session_id=session_id)
                    await websocket.send_json({
                        "type": "task_complete",
                        "content": resp.content,
                        "plan": resp.plan.to_dict() if resp.plan else None,
                        "tool_calls": resp.tool_calls,
                    })
                else:
                    # Stream tokens
                    await websocket.send_json({"type": "start"})
                    async for token in agent.stream_chat(msg, session_id=session_id):
                        await websocket.send_json({"type": "token", "content": token})
                    await websocket.send_json({"type": "end"})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            await websocket.send_json({"type": "error", "content": str(e)})

    # ── Memory endpoints ────────────────────────────────────────────────────

    @app.post("/memory/search")
    async def search_memory(req: MemorySearchRequest):
        results = await memory.search_memories(req.query, limit=req.limit)
        return {"results": results}

    @app.get("/memory/all")
    async def all_memories(limit: int = 100):
        return {"memories": await memory.get_all_memories(limit=limit)}

    @app.delete("/memory/{memory_id}")
    async def forget(memory_id: int):
        await memory.forget_memory(memory_id)
        return {"deleted": memory_id}

    # ── Tasks ────────────────────────────────────────────────────────────────

    @app.get("/tasks")
    async def list_tasks(session_id: str | None = None, limit: int = 50):
        tasks = await memory.list_tasks(session_id=session_id, limit=limit)
        return {"tasks": tasks}

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: int):
        t = await memory.get_task(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="Task not found")
        return t

    # ── Preferences ──────────────────────────────────────────────────────────

    @app.post("/preferences")
    async def set_preference(req: PreferenceRequest):
        await memory.set_preference(req.key, req.value)
        return {"set": req.key}

    @app.get("/preferences")
    async def get_preferences():
        return {"preferences": await memory.all_preferences()}

    # ── Voice ────────────────────────────────────────────────────────────────

    @app.post("/voice/speak")
    async def speak(text: str):
        from jarvis.voice.tts import TTSEngine
        tts = TTSEngine()
        audio = await tts.speak(text)
        if audio is None:
            raise HTTPException(status_code=503, detail="TTS unavailable")
        from fastapi.responses import Response
        return Response(content=audio, media_type="audio/mpeg")

    @app.post("/voice/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        from jarvis.voice.stt import STTEngine
        stt = STTEngine()
        audio_bytes = await file.read()
        ext = Path(file.filename or "audio.wav").suffix.lstrip(".")
        text = await stt.transcribe_bytes(audio_bytes, ext=ext)
        return {"transcript": text}

    # ── Tools list ───────────────────────────────────────────────────────────

    @app.get("/tools")
    async def list_tools():
        return {
            "tools": [
                {"name": t.name, "description": t.description}
                for t in tools.all_tools()
            ]
        }

    # ── Event log ────────────────────────────────────────────────────────────

    @app.get("/logs")
    async def get_logs(event_type: str | None = None, limit: int = 100):
        return {"events": await memory.get_events(event_type=event_type, limit=limit)}

    # ── Multi-agent dashboard ────────────────────────────────────────────────

    @app.get("/agents/dashboard", response_class=HTMLResponse)
    async def multi_agent_dashboard(request: Request):
        return templates.TemplateResponse(request, "multi_agent.html")

    # ── Agent status ─────────────────────────────────────────────────────────

    @app.get("/agents")
    async def list_agents():
        statuses = await agent_registry.get_status_dicts()
        perf = await memory.get_agent_performance()
        perf_map = {p["agent_name"]: p for p in perf}
        for s in statuses:
            p = perf_map.get(s["name"], {})
            tc = p.get("tasks_completed", 0)
            total_conf = p.get("total_confidence", 0.0)
            s["lifetime_tasks"] = tc
            s["lifetime_avg_confidence"] = round(total_conf / tc, 3) if tc > 0 else 1.0
            s["errors"] = p.get("errors", 0)
        return {"agents": statuses}

    @app.get("/agents/performance")
    async def agent_performance():
        perf = await memory.get_agent_performance()
        return {"performance": perf}

    # ── Multi-agent run ──────────────────────────────────────────────────────

    @app.post("/multi-agent/run", status_code=202)
    async def start_multi_agent_run(req: MultiAgentRunRequest):
        session_id = req.session_id or str(uuid.uuid4())
        valid_modes = {"standard", "debate", "swarm"}
        mode = req.mode if req.mode in valid_modes else "standard"
        run_id = await orchestrator.start_run(req.goal, session_id, mode=mode)
        return {"run_id": run_id, "status": "started", "session_id": session_id, "mode": mode}

    @app.get("/multi-agent/status/{run_id}")
    async def get_run_status(run_id: str):
        status = await orchestrator.get_status(run_id)
        if not status:
            raise HTTPException(status_code=404, detail="Run not found")
        return status

    @app.get("/multi-agent/runs")
    async def list_runs(session_id: str | None = None, limit: int = 20):
        runs = await orchestrator.list_runs(session_id=session_id, limit=limit)
        return {"runs": runs}

    @app.get("/multi-agent/messages/{run_id}")
    async def get_run_messages(run_id: str):
        msgs = await orchestrator.get_messages(run_id)
        return {"messages": msgs, "count": len(msgs)}

    @app.get("/multi-agent/messages")
    async def get_all_messages(limit: int = 200):
        msgs = message_bus.get_history(limit=limit)
        return {"messages": msgs, "count": len(msgs)}

    @app.post("/multi-agent/debate", status_code=202)
    async def start_debate(req: DebateRequest):
        session_id = req.session_id or str(uuid.uuid4())
        run_id = await orchestrator.start_run(req.topic, session_id, mode="debate")
        return {"run_id": run_id, "status": "started", "mode": "debate", "session_id": session_id}

    @app.post("/multi-agent/swarm", status_code=202)
    async def start_swarm(req: SwarmRequest):
        session_id = req.session_id or str(uuid.uuid4())
        run_id = await orchestrator.start_run(req.goal, session_id, mode="swarm")
        return {
            "run_id": run_id, "status": "started", "mode": "swarm",
            "swarm_size": req.count, "session_id": session_id,
        }

    # ── Orchestrator WebSocket ───────────────────────────────────────────────

    @app.websocket("/ws/orchestrator/{session_id}")
    async def ws_orchestrator(websocket: WebSocket, session_id: str):
        await websocket.accept()
        active_orchestrator_ws[session_id] = websocket
        try:
            # Send initial agent statuses on connect
            statuses = await agent_registry.get_status_dicts()
            await websocket.send_json({"type": "agent_status_init", "agents": statuses})
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                action = payload.get("action", "")
                if action == "status":
                    statuses = await agent_registry.get_status_dicts()
                    await websocket.send_json({"type": "agent_status", "agents": statuses})
                elif action == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            active_orchestrator_ws.pop(session_id, None)
        except Exception as e:
            active_orchestrator_ws.pop(session_id, None)
            try:
                await websocket.send_json({"type": "error", "content": str(e)})
            except Exception:
                pass

    return app
