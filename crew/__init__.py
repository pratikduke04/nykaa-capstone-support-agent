"""
Importing anything from the `crew` package sets the environment BEFORE crewai is imported,
so the zero-network requirement holds: CrewAI telemetry is switched off and MOCK_LLM is on.
"""
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")   # no execution-trace upload prompt
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("MOCK_LLM", "1")