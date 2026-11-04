"""Shared CrewAI construction settings."""


def no_trace_kwargs() -> dict:
    """Pass to Crew(...). Turns CrewAI execution tracing off (it can upload data), if this version has the option."""
    from crewai import Crew
    return {"tracing": False} if "tracing" in getattr(Crew, "model_fields", {}) else {}


def structured_output_kwargs() -> dict:
    """
    Pass to the Composer's Task(...): makes CrewAI validate its output against AnswerSchema (Task 9).
    Set USE_OUTPUT_PYDANTIC=0 to switch this off while debugging (the pipeline still validates in code).
    """
    import os
    from crewai import Task
    from crew.schemas import AnswerSchema
    if os.environ.get("USE_OUTPUT_PYDANTIC", "1") != "1":
        return {}
    return {"output_pydantic": AnswerSchema} if "output_pydantic" in getattr(Task, "model_fields", {}) else {}