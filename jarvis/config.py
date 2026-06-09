"""Central configuration for Jarvis AI System."""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # ── LLM ────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"          # "anthropic" | "openai"
    planner_model: str = "claude-opus-4-8"
    executor_model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096

    # ── Memory / Database ──────────────────────────────────────────────────
    db_path: str = str(BASE_DIR / "data" / "memory.db")

    # ── Voice ──────────────────────────────────────────────────────────────
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL"  # default: Bella
    whisper_model: str = "base"
    tts_provider: str = "elevenlabs"         # "elevenlabs" | "pyttsx3"

    # ── API Server ─────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: str = "change-me-in-production"

    # ── Safety / Sandbox ──────────────────────────────────────────────────
    max_code_execution_time: int = 10        # seconds
    workspace_root: str = str(BASE_DIR / "workspace")

    # ── Logging ────────────────────────────────────────────────────────────
    log_dir: str = str(BASE_DIR / "logs")
    log_level: str = "INFO"

    # ── Agent Behaviour ────────────────────────────────────────────────────
    max_agent_iterations: int = 15
    max_plan_steps: int = 10
    memory_retrieval_limit: int = 8

    # ── Multi-Agent Orchestration ──────────────────────────────────────────
    orchestrator_model: str = "claude-opus-4-8"
    max_critic_retries: int = 3
    critic_pass_threshold: float = 0.7
    swarm_size: int = 3
    debate_rounds: int = 2


settings = Settings()

# Ensure runtime directories exist
for d in [settings.workspace_root, settings.log_dir, str(BASE_DIR / "data")]:
    Path(d).mkdir(parents=True, exist_ok=True)
