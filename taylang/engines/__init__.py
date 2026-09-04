"""Solver-engine registry for TAY Language."""

from __future__ import annotations

from .taylanus import EngineError, TaylanusEngine, taylanus_status


def create_engine(name: str):
    normalized = str(name).strip().upper()
    if normalized == "TAYLANUS":
        return TaylanusEngine()
    raise EngineError(
        f"Unknown TAY engine: {name}. Available engines: TAYLANUS."
    )


def engine_status() -> dict:
    return {"TAYLANUS": taylanus_status()}


__all__ = [
    "EngineError",
    "TaylanusEngine",
    "create_engine",
    "engine_status",
    "taylanus_status",
]
