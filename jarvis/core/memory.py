"""Long-term memory system backed by SQLite."""
from __future__ import annotations

import json
import time
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import aiosqlite

from jarvis.config import settings


# ── Data models ────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str          # "user" | "assistant" | "system" | "tool"
    content: str
    session_id: str
    metadata: dict     = None
    ts: float          = None
    id: int            = None

    def __post_init__(self):
        if self.ts is None:
            self.ts = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Memory:
    key: str
    value: str
    category: str      # "fact" | "preference" | "project" | "decision" | "skill"
    session_id: str
    confidence: float  = 1.0
    ts: float          = None
    id: int            = None

    def __post_init__(self):
        if self.ts is None:
            self.ts = time.time()


@dataclass
class Task:
    title: str
    description: str
    session_id: str
    status: str        = "pending"   # pending | running | done | failed
    steps: list        = None
    result: str        = ""
    ts_created: float  = None
    ts_updated: float  = None
    id: int            = None

    def __post_init__(self):
        now = time.time()
        if self.ts_created is None:
            self.ts_created = now
        if self.ts_updated is None:
            self.ts_updated = now
        if self.steps is None:
            self.steps = []


# ── Schema ─────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    metadata    TEXT    DEFAULT '{}',
    ts          REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, ts);

CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    confidence  REAL    DEFAULT 1.0,
    ts          REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mem_key     ON memories(key);
CREATE INDEX IF NOT EXISTS idx_mem_cat     ON memories(category);
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    key, value, category, content=memories, content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, key, value, category)
    VALUES (new.id, new.key, new.value, new.category);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, key, value, category)
    VALUES ('delete', old.id, old.key, old.value, old.category);
END;

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    description TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',
    steps       TEXT    DEFAULT '[]',
    result      TEXT    DEFAULT '',
    ts_created  REAL    NOT NULL,
    ts_updated  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS preferences (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    ts          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS event_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    event_type  TEXT NOT NULL,
    payload     TEXT DEFAULT '{}',
    ts          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_log_type    ON event_log(event_type);
CREATE INDEX IF NOT EXISTS idx_log_session ON event_log(session_id);
"""

_MULTI_AGENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    from_agent TEXT NOT NULL,
    to_agent   TEXT NOT NULL,
    type       TEXT NOT NULL,
    content    TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    task_id    TEXT,
    ts         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_amsg_run ON agent_messages(run_id, ts);

CREATE TABLE IF NOT EXISTS agent_performance (
    agent_name        TEXT PRIMARY KEY,
    tasks_completed   INTEGER DEFAULT 0,
    total_confidence  REAL DEFAULT 0.0,
    errors            INTEGER DEFAULT 0,
    last_active       REAL
);

CREATE TABLE IF NOT EXISTS execution_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT UNIQUE NOT NULL,
    session_id    TEXT NOT NULL,
    goal          TEXT NOT NULL,
    mode          TEXT DEFAULT 'standard',
    status        TEXT DEFAULT 'running',
    result        TEXT DEFAULT '',
    agent_results TEXT DEFAULT '{}',
    ts_created    REAL NOT NULL,
    ts_completed  REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON execution_runs(session_id, ts_created);
"""


# ── MemorySystem ───────────────────────────────────────────────────────────

class MemorySystem:
    """Async SQLite-backed memory for conversations, facts, tasks, and prefs."""

    def __init__(self, db_path: str = settings.db_path):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def init(self):
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.executescript(_MULTI_AGENT_SCHEMA)
            await db.commit()

    @asynccontextmanager
    async def _db(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    # ── Messages ───────────────────────────────────────────────────────────

    async def add_message(self, msg: Message) -> int:
        async with self._db() as db:
            cur = await db.execute(
                "INSERT INTO messages(session_id, role, content, metadata, ts) VALUES (?,?,?,?,?)",
                (msg.session_id, msg.role, msg.content, json.dumps(msg.metadata), msg.ts),
            )
            await db.commit()
            return cur.lastrowid

    async def get_history(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        async with self._db() as db:
            rows = await db.execute_fetchall(
                "SELECT * FROM messages WHERE session_id=? ORDER BY ts DESC LIMIT ? OFFSET ?",
                (session_id, limit, offset),
            )
        msgs = [dict(r) for r in rows]
        for m in msgs:
            m["metadata"] = json.loads(m["metadata"])
        return list(reversed(msgs))

    async def get_recent_messages(self, session_id: str, n: int = 20) -> list[dict]:
        return await self.get_history(session_id, limit=n)

    # ── Long-term memories ─────────────────────────────────────────────────

    async def store_memory(self, mem: Memory) -> int:
        async with self._db() as db:
            cur = await db.execute(
                "INSERT INTO memories(key, value, category, session_id, confidence, ts) "
                "VALUES (?,?,?,?,?,?)",
                (mem.key, mem.value, mem.category, mem.session_id, mem.confidence, mem.ts),
            )
            await db.commit()
            return cur.lastrowid

    async def search_memories(self, query: str, limit: int = 8) -> list[dict]:
        """Full-text search across all stored memories."""
        async with self._db() as db:
            rows = await db.execute_fetchall(
                """SELECT m.* FROM memories m
                   JOIN memories_fts f ON m.id = f.rowid
                   WHERE memories_fts MATCH ?
                   ORDER BY m.confidence DESC, m.ts DESC
                   LIMIT ?""",
                (query, limit),
            )
        return [dict(r) for r in rows]

    async def get_memories_by_category(
        self, category: str, limit: int = 20
    ) -> list[dict]:
        async with self._db() as db:
            rows = await db.execute_fetchall(
                "SELECT * FROM memories WHERE category=? ORDER BY ts DESC LIMIT ?",
                (category, limit),
            )
        return [dict(r) for r in rows]

    async def get_all_memories(self, limit: int = 100) -> list[dict]:
        async with self._db() as db:
            rows = await db.execute_fetchall(
                "SELECT * FROM memories ORDER BY ts DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in rows]

    async def forget_memory(self, memory_id: int):
        async with self._db() as db:
            await db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            await db.commit()

    # ── Tasks ──────────────────────────────────────────────────────────────

    async def create_task(self, task: Task) -> int:
        async with self._db() as db:
            cur = await db.execute(
                "INSERT INTO tasks(session_id, title, description, status, steps, result, "
                "ts_created, ts_updated) VALUES (?,?,?,?,?,?,?,?)",
                (
                    task.session_id, task.title, task.description,
                    task.status, json.dumps(task.steps), task.result,
                    task.ts_created, task.ts_updated,
                ),
            )
            await db.commit()
            return cur.lastrowid

    async def update_task(self, task_id: int, **kwargs):
        kwargs["ts_updated"] = time.time()
        if "steps" in kwargs:
            kwargs["steps"] = json.dumps(kwargs["steps"])
        cols = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [task_id]
        async with self._db() as db:
            await db.execute(f"UPDATE tasks SET {cols} WHERE id=?", vals)
            await db.commit()

    async def get_task(self, task_id: int) -> Optional[dict]:
        async with self._db() as db:
            row = await db.execute_fetchall(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            )
        if not row:
            return None
        t = dict(row[0])
        t["steps"] = json.loads(t["steps"])
        return t

    async def list_tasks(self, session_id: str | None = None, limit: int = 50) -> list[dict]:
        async with self._db() as db:
            if session_id:
                rows = await db.execute_fetchall(
                    "SELECT * FROM tasks WHERE session_id=? ORDER BY ts_created DESC LIMIT ?",
                    (session_id, limit),
                )
            else:
                rows = await db.execute_fetchall(
                    "SELECT * FROM tasks ORDER BY ts_created DESC LIMIT ?", (limit,)
                )
        tasks = [dict(r) for r in rows]
        for t in tasks:
            t["steps"] = json.loads(t["steps"])
        return tasks

    # ── Preferences ────────────────────────────────────────────────────────

    async def set_preference(self, key: str, value: Any):
        async with self._db() as db:
            await db.execute(
                "INSERT OR REPLACE INTO preferences(key, value, ts) VALUES (?,?,?)",
                (key, json.dumps(value), time.time()),
            )
            await db.commit()

    async def get_preference(self, key: str, default: Any = None) -> Any:
        async with self._db() as db:
            rows = await db.execute_fetchall(
                "SELECT value FROM preferences WHERE key=?", (key,)
            )
        if not rows:
            return default
        return json.loads(rows[0]["value"])

    async def all_preferences(self) -> dict:
        async with self._db() as db:
            rows = await db.execute_fetchall("SELECT key, value FROM preferences")
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # ── Event log ──────────────────────────────────────────────────────────

    async def log_event(
        self, event_type: str, payload: dict, session_id: str | None = None
    ):
        async with self._db() as db:
            await db.execute(
                "INSERT INTO event_log(session_id, event_type, payload, ts) VALUES (?,?,?,?)",
                (session_id, event_type, json.dumps(payload), time.time()),
            )
            await db.commit()

    async def get_events(
        self, event_type: str | None = None, limit: int = 100
    ) -> list[dict]:
        async with self._db() as db:
            if event_type:
                rows = await db.execute_fetchall(
                    "SELECT * FROM event_log WHERE event_type=? ORDER BY ts DESC LIMIT ?",
                    (event_type, limit),
                )
            else:
                rows = await db.execute_fetchall(
                    "SELECT * FROM event_log ORDER BY ts DESC LIMIT ?", (limit,)
                )
        events = [dict(r) for r in rows]
        for e in events:
            e["payload"] = json.loads(e["payload"])
        return events

    # ── Multi-agent: agent messages ────────────────────────────────────────

    async def store_agent_message(self, msg_dict: dict) -> int:
        async with self._db() as db:
            cur = await db.execute(
                "INSERT INTO agent_messages(run_id, from_agent, to_agent, type, content, "
                "confidence, task_id, ts) VALUES (?,?,?,?,?,?,?,?)",
                (
                    msg_dict["run_id"], msg_dict["from_agent"], msg_dict["to_agent"],
                    msg_dict["type"], msg_dict["content"], msg_dict.get("confidence", 1.0),
                    msg_dict.get("task_id", ""), msg_dict.get("timestamp", time.time()),
                ),
            )
            await db.commit()
            return cur.lastrowid

    async def get_agent_messages(self, run_id: str, limit: int = 200) -> list[dict]:
        async with self._db() as db:
            rows = await db.execute_fetchall(
                "SELECT * FROM agent_messages WHERE run_id=? ORDER BY ts ASC LIMIT ?",
                (run_id, limit),
            )
        return [dict(r) for r in rows]

    # ── Multi-agent: performance ───────────────────────────────────────────

    async def upsert_agent_performance(
        self, agent_name: str, confidence: float, error: bool = False
    ):
        async with self._db() as db:
            await db.execute(
                """INSERT INTO agent_performance(agent_name, tasks_completed, total_confidence, errors, last_active)
                   VALUES (?, 1, ?, ?, ?)
                   ON CONFLICT(agent_name) DO UPDATE SET
                     tasks_completed = tasks_completed + 1,
                     total_confidence = total_confidence + excluded.total_confidence,
                     errors = errors + excluded.errors,
                     last_active = excluded.last_active""",
                (agent_name, confidence, 1 if error else 0, time.time()),
            )
            await db.commit()

    async def get_agent_performance(self) -> list[dict]:
        async with self._db() as db:
            rows = await db.execute_fetchall(
                "SELECT * FROM agent_performance ORDER BY agent_name"
            )
        return [dict(r) for r in rows]

    # ── Multi-agent: execution runs ────────────────────────────────────────

    async def create_run(
        self, run_id: str, session_id: str, goal: str, mode: str
    ) -> int:
        async with self._db() as db:
            cur = await db.execute(
                "INSERT INTO execution_runs(run_id, session_id, goal, mode, status, ts_created) "
                "VALUES (?,?,?,?,'running',?)",
                (run_id, session_id, goal, mode, time.time()),
            )
            await db.commit()
            return cur.lastrowid

    async def update_run(self, run_id: str, **kwargs):
        if "agent_results" in kwargs and isinstance(kwargs["agent_results"], dict):
            kwargs["agent_results"] = json.dumps(kwargs["agent_results"])
        if kwargs.get("status") in ("complete", "failed") and "ts_completed" not in kwargs:
            kwargs["ts_completed"] = time.time()
        cols = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [run_id]
        async with self._db() as db:
            await db.execute(f"UPDATE execution_runs SET {cols} WHERE run_id=?", vals)
            await db.commit()

    async def list_runs(
        self, session_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        async with self._db() as db:
            if session_id:
                rows = await db.execute_fetchall(
                    "SELECT * FROM execution_runs WHERE session_id=? ORDER BY ts_created DESC LIMIT ?",
                    (session_id, limit),
                )
            else:
                rows = await db.execute_fetchall(
                    "SELECT * FROM execution_runs ORDER BY ts_created DESC LIMIT ?", (limit,)
                )
        runs = [dict(r) for r in rows]
        for r in runs:
            if r.get("agent_results"):
                try:
                    r["agent_results"] = json.loads(r["agent_results"])
                except Exception:
                    r["agent_results"] = {}
        return runs

    async def get_run(self, run_id: str) -> Optional[dict]:
        async with self._db() as db:
            rows = await db.execute_fetchall(
                "SELECT * FROM execution_runs WHERE run_id=?", (run_id,)
            )
        if not rows:
            return None
        r = dict(rows[0])
        if r.get("agent_results"):
            try:
                r["agent_results"] = json.loads(r["agent_results"])
            except Exception:
                r["agent_results"] = {}
        return r

    # ── Context builder (for agent prompt injection) ───────────────────────

    async def build_context_block(self, session_id: str, query: str) -> str:
        """Return a formatted string of relevant memories + recent history."""
        relevant = await self.search_memories(query, limit=settings.memory_retrieval_limit)
        prefs = await self.all_preferences()

        lines = ["=== JARVIS MEMORY CONTEXT ==="]
        if prefs:
            lines.append("\n[User Preferences]")
            for k, v in prefs.items():
                lines.append(f"  • {k}: {v}")
        if relevant:
            lines.append("\n[Relevant Memories]")
            for m in relevant:
                lines.append(f"  [{m['category']}] {m['key']}: {m['value']}")
        lines.append("=== END CONTEXT ===")
        return "\n".join(lines)
