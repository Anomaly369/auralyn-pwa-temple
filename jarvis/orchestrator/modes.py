"""Enums for multi-agent execution modes and run statuses."""
from enum import Enum


class ExecutionMode(str, Enum):
    STANDARD = "standard"
    DEBATE = "debate"
    SWARM = "swarm"


class RunStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    RETRYING = "retrying"
    MERGING = "merging"
    COMMUNICATING = "communicating"
    COMPLETE = "complete"
    FAILED = "failed"
