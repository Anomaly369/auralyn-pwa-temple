"""
JARVIS — Autonomous AI Operating System
Entry point: CLI, interactive shell, and server launcher.

Usage:
  python -m jarvis.main server          # Launch FastAPI server + dashboard
  python -m jarvis.main chat            # Interactive CLI chat
  python -m jarvis.main task "goal"     # Run a single task autonomously
  python -m jarvis.main voice           # Voice interaction loop
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from jarvis.config import settings
from jarvis.core.agent import JarvisAgent
from jarvis.core.memory import MemorySystem
from jarvis.core.tools import ToolRegistry
from jarvis.plugins.youtube_automation import YouTubeAutomationPlugin
from jarvis.plugins.mt5_trading import MT5TradingPlugin


app = typer.Typer(help="JARVIS — Autonomous AI Operating System")
console = Console()


# ── Shared setup ───────────────────────────────────────────────────────────

def build_agent() -> JarvisAgent:
    memory = MemorySystem()
    tools  = ToolRegistry()

    # Register all plugins
    YouTubeAutomationPlugin().register(tools)
    MT5TradingPlugin().register(tools)

    return JarvisAgent(memory=memory, tools=tools)


# ── CLI commands ───────────────────────────────────────────────────────────

@app.command()
def server(
    host: str = typer.Option(settings.host, help="Bind host"),
    port: int = typer.Option(settings.port, help="Bind port"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
):
    """Launch the FastAPI server and open the dashboard."""
    import uvicorn
    from jarvis.api.server import create_app

    console.print(Panel(
        Text.from_markup(
            f"[bold cyan]⬡ JARVIS[/] is starting\n\n"
            f"Dashboard → [link=http://{host}:{port}]http://{host}:{port}[/link]\n"
            f"API docs  → [link=http://{host}:{port}/docs]http://{host}:{port}/docs[/link]\n\n"
            f"[dim]Press Ctrl+C to stop[/dim]"
        ),
        title="JARVIS AI", border_style="cyan",
    ))

    fastapi_app = create_app()
    uvicorn.run(fastapi_app, host=host, port=port, reload=reload,
                log_level=settings.log_level.lower())


@app.command()
def chat(
    session: str = typer.Option(None, help="Session ID (auto-generated if omitted)"),
):
    """Start an interactive CLI chat session with JARVIS."""
    asyncio.run(_chat_loop(session))


async def _chat_loop(session_id: str | None = None):
    session_id = session_id or f"cli_{uuid.uuid4().hex[:8]}"
    agent = build_agent()
    await agent.memory.init()

    console.print(Panel(
        f"[bold cyan]⬡ JARVIS[/] — Interactive Chat\n"
        f"Session: [dim]{session_id}[/dim]\n"
        f"Type [bold]/task[/bold] to switch to task mode, [bold]/quit[/bold] to exit.",
        border_style="cyan",
    ))

    task_mode = False
    while True:
        try:
            mode_label = "[task]" if task_mode else "[chat]"
            user_input = Prompt.ask(f"\n[bold cyan]{mode_label}[/] You")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input.strip():
            continue
        if user_input.strip() == "/quit":
            break
        if user_input.strip() == "/task":
            task_mode = True
            console.print("[dim]Switched to task mode.[/dim]")
            continue
        if user_input.strip() == "/chat":
            task_mode = False
            console.print("[dim]Switched to chat mode.[/dim]")
            continue
        if user_input.strip().startswith("/tools"):
            for t in agent.tools.all_tools():
                console.print(f"  [cyan]{t.name}[/cyan] — {t.description[:60]}")
            continue

        console.print()
        with console.status("[cyan]Thinking…[/cyan]"):
            if task_mode:
                resp = await agent.execute_task(user_input, session_id=session_id)
            else:
                resp = await agent.chat(user_input, session_id=session_id)

        console.print(Panel(
            Markdown(resp.content),
            title="[bold cyan]JARVIS[/bold cyan]",
            border_style="dim",
        ))

        meta = []
        if resp.tool_calls:
            meta.append(f"[dim]{len(resp.tool_calls)} tool call(s)[/dim]")
        if resp.memories_stored:
            meta.append(f"[dim]{resp.memories_stored} memories stored[/dim]")
        if resp.duration_ms:
            meta.append(f"[dim]{resp.duration_ms}ms[/dim]")
        if meta:
            console.print("  " + "  ·  ".join(meta))


@app.command()
def task(goal: str = typer.Argument(..., help="Task goal to execute autonomously")):
    """Execute a single task autonomously and print the result."""
    asyncio.run(_run_task(goal))


async def _run_task(goal: str):
    session_id = f"task_{uuid.uuid4().hex[:8]}"
    agent = build_agent()
    await agent.memory.init()

    console.print(Panel(
        f"[bold]Goal:[/bold] {goal}\n[dim]Session: {session_id}[/dim]",
        title="⬡ JARVIS Task", border_style="cyan",
    ))

    with console.status("[cyan]Executing…[/cyan]"):
        resp = await agent.execute_task(goal, session_id=session_id)

    if resp.plan:
        console.print("\n[bold]Plan executed:[/bold]")
        for step in resp.plan.steps:
            icon = {"done": "✓", "failed": "✗", "skipped": "–"}.get(step.status, "○")
            console.print(f"  {icon} [{step.index}] {step.title}")

    console.print(Panel(Markdown(resp.content), title="Result", border_style="green"))
    console.print(f"  [dim]{resp.duration_ms}ms · {len(resp.tool_calls)} tool calls · {resp.memories_stored} memories[/dim]")


@app.command()
def voice():
    """Start a voice interaction loop (mic in → TTS out)."""
    asyncio.run(_voice_loop())


async def _voice_loop():
    from jarvis.voice.stt import STTEngine
    from jarvis.voice.tts import TTSEngine
    import tempfile

    session_id = f"voice_{uuid.uuid4().hex[:8]}"
    agent = build_agent()
    await agent.memory.init()
    stt = STTEngine()
    tts = TTSEngine()

    console.print(Panel(
        "[bold cyan]⬡ JARVIS[/] Voice Mode\nPress Enter to speak. Say 'goodbye' to exit.",
        border_style="cyan",
    ))

    while True:
        try:
            input("\n[Press Enter to speak]")
        except (KeyboardInterrupt, EOFError):
            break

        transcript = await stt.record_and_transcribe(duration_seconds=6.0)
        if not transcript or "error" in transcript.lower():
            console.print("[dim]Could not transcribe — try again.[/dim]")
            continue

        console.print(f"[bold]You:[/bold] {transcript}")
        if "goodbye" in transcript.lower():
            console.print("[dim]Goodbye![/dim]")
            break

        with console.status("[cyan]Thinking…[/cyan]"):
            resp = await agent.chat(transcript, session_id=session_id)

        console.print(f"[bold cyan]JARVIS:[/bold cyan] {resp.content[:500]}")

        # Speak the reply
        audio = await tts.speak(resp.content[:800])
        if audio:
            try:
                import sounddevice as sd
                import soundfile as sf
                import io
                data, sr = sf.read(io.BytesIO(audio))
                sd.play(data, sr)
                sd.wait()
            except Exception as e:
                console.print(f"[dim]Audio playback error: {e}[/dim]")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
