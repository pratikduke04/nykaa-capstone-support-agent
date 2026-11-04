"""Shared CrewAI construction settings."""


def no_trace_kwargs() -> dict:
    """Pass to Crew(...). Turns CrewAI execution tracing off (it can upload data), if this version has the option."""
    from crewai import Crew
    return {"tracing": False} if "tracing" in getattr(Crew, "model_fields", {}) else {}