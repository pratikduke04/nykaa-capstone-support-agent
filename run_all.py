"""
Regenerate every transcript in transcripts/ with one command - the same on Windows, macOS and Linux.

    python run_all.py                 # all steps
    python run_all.py --only 15       # only steps whose transcript name contains "15"

Each step runs as a child process with MOCK_LLM=1, telemetry/tracing switched off and UTF-8 output, so the
transcripts are identical whatever the operating system. Warnings (stderr) go to logs/run_all_stderr.log,
not into the transcripts. Needs the knowledge-base index first, so `rag.index` runs early.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "transcripts"

STEPS = [  # (transcript file, python arguments)
    ("task01_dataset.txt",            ["dataset.py"]),
    ("task02_03_kb_and_chunking.txt", ["-m", "rag.chunking"]),
    ("task03_index.txt",              ["-m", "rag.index"]),
    ("task04_calibration.txt",        ["-m", "rag.calibrate"]),
    ("task04_grounded.txt",           ["-m", "rag.grounded"]),
    ("task05_chunking_eval.txt",      ["-m", "rag.evaluate_chunking"]),
    ("task06_order_status.txt",       ["-m", "crew.order_status"]),
    ("task07_09_crew.txt",            ["-m", "crew.demo_part2", "crew"]),
    ("task08_memory.txt",             ["-m", "crew.demo_part2", "memory"]),
    ("task08_memory_fresh.txt",       ["-m", "crew.demo_part2", "memory_fresh"]),
    ("task10_guardrails.txt",         ["-m", "crew.demo_part2", "guardrails"]),
    ("task11_12_api.txt",             ["-m", "api.demo_api"]),
    ("task13_eval.txt",               ["-m", "eval.run_eval"]),
    ("task14_autogen_review.txt",     ["-m", "review.autogen_review"]),
    ("task15_governance.txt",         ["-m", "governance.demo_governance"]),
    ("task16_cache.txt",              ["-m", "rag.demo_cache"]),
]

ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false",
       "MOCK_LLM": "1", "CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true",
       "CREWAI_TRACING_ENABLED": "false", "CREWAI_DISABLE_TRACKING": "true"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="run only steps whose transcript name contains this text")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    err_log = ROOT / "logs" / "run_all_stderr.log"
    err_log.write_text("", encoding="utf-8")

    failed = []
    for name, py_args in STEPS:
        if args.only and args.only not in name:
            continue
        shown = "python " + " ".join(py_args)
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, *py_args], cwd=ROOT, env=ENV, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        secs = time.perf_counter() - start
        (OUT / name).write_text(f"# command: {shown}   (exit code {proc.returncode}, {secs:.1f}s)\n{proc.stdout}",
                                encoding="utf-8")
        with err_log.open("a", encoding="utf-8") as f:
            f.write(f"===== {shown} =====\n{proc.stderr}\n")
        status = "ok    " if proc.returncode == 0 else "FAILED"
        print(f"[{status}] {name:<32} {shown}   ({secs:.1f}s)")
        if proc.returncode != 0:
            failed.append(name)
            print("          last stderr lines:", *proc.stderr.strip().splitlines()[-3:], sep="\n            ")
    print(f"\n{'All steps passed.' if not failed else 'FAILED: ' + ', '.join(failed)}  Transcripts are in {OUT.name}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())