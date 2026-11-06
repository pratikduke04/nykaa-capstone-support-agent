# Nykaa Domain Support Agent (CrewAI)

**Track: E-commerce & Retail - Nykaa.** A customer-support agent that answers policy questions from a
knowledge base (RAG), looks up order status with an escalation score, and is wrapped in guardrails, a
FastAPI service, an evaluation harness, an Autogen review stage and governance controls.

**Runs entirely under `MOCK_LLM`: zero API keys, no paid services.** See "MOCK_LLM, API keys and telemetry" below.


## Key choices at a glance

| | |
|---|---|
| **Dataset seed** | **42** (60 orders, `NYK-1001` to `NYK-1060`) |
| **Category weights** | Beauty 0.30, Apparel 0.25, Footwear 0.15, Home 0.15, Electronics 0.15 |
| **Status weights** | Delivered 0.40, Shipped 0.20, Placed 0.15, Returned 0.15, Refunded 0.10 |
| **Amount range** | **Rs 199 - Rs 14,999** overall (per-category ranges in the Task 1 section) |
| **Delayed shipments** | 13 / 60 = **21.7 %** (required 10-30 %) |
| **Measured thresholds** | `kb_fixed` **0.27**, `kb_sentence` **0.26** (calibrated, not preset) |
| **Mock model, keys** | `MOCK_LLM=1`, **zero API keys**; `CREWAI_DISABLE_TELEMETRY=true` plus the tracing flags (details below) |

## Where is the evidence for each task?

| Task | What | Transcript in `transcripts/` |
|---|---|---|
| 1 | Dataset | `task01_dataset.txt` |
| 2-3 | Knowledge base, two chunking strategies, two collections | `task02_03_kb_and_chunking.txt`, `task03_index.txt` |
| 4 | Calibrated threshold, grounded generation | `task04_calibration.txt`, `task04_grounded.txt` |
| 5 | Precision / recall comparison | `task05_chunking_eval.txt` |
| 6 | `check_order_status` and escalation score | `task06_order_status.txt` |
| 7, 9 | 3-agent crew, Pydantic structured output | `task07_09_crew.txt` |
| 8 | LangChain session memory | `task08_memory.txt`, `task08_memory_fresh.txt` |
| 10 | Guardrails | `task10_guardrails.txt` |
| 11-12 | FastAPI, WebSocket, JSON-Lines log | `task11_12_api.txt`, plus `logs/requests.jsonl` |
| 13 | LLM-as-judge evaluation | `task13_eval.txt`, plus `logs/eval_results.json` |
| 14 | Autogen review team | `task14_autogen_review.txt` |
| 15 | Least autonomy, risk, budget cap | `task15_governance.txt` |
| 16 | Response cache | `task16_cache.txt` |

---

## Quick start (Windows, macOS, Linux)

Python 3.11 is recommended (developed and tested on 3.11, macOS).

```
python -m venv .venv
```

Activate it:

| System | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |

Then:

```
pip install -r requirements.txt
python run_all.py
```

`run_all.py` re-runs every task and rewrites all 16 files in `transcripts/` (about 3 minutes). It is the same
on every operating system: each step runs with `MOCK_LLM=1`, telemetry and tracing off, and UTF-8 output.
Warnings go to `logs/run_all_stderr.log`, not into the transcripts. Run one step with `python run_all.py --only 15`.

To start the API: `uvicorn api.main:app --port 8000`, then open `http://localhost:8000/docs`.

## MOCK_LLM, API keys and telemetry

- `MOCK_LLM=1` is the default. **No API key is read or needed anywhere.**
- The mock models are deterministic: `MockLLM` extends `crewai.llms.base_llm.BaseLLM` (CrewAI agents),
  `MockJudgeLLM` extends the same class (evaluation judge) and `MockChatClient` extends Autogen's
  `ChatCompletionClient` (review stage).
- **How `MockLLM` decides what to do:** it reads only the model's own turns and the tool result, not CrewAI's instruction
  template (which itself contains the word "Observation:"). It calls a tool if its own previous turn has not already, and
  picks the arguments from the tool's argument schema (a `record_id` field means an order lookup), never from the tool's name.
- Telemetry and tracing are switched off before CrewAI is imported (`crew/__init__.py`, and again in `run_all.py`):

  ```
  CREWAI_DISABLE_TELEMETRY=true
  OTEL_SDK_DISABLED=true
  CREWAI_TRACING_ENABLED=false
  CREWAI_DISABLE_TRACKING=true
  ```
  `Crew(..., tracing=False)` is also set where the installed CrewAI supports it.
- **Network note (honest):** the only download is the open-source embedding model `all-MiniLM-L6-v2`, fetched
  once from the Hugging Face hub on first run and cached locally. After that the project runs offline
  (set `HF_HUB_OFFLINE=1` to prove it). A "unauthenticated requests to the HF Hub" warning on first run is harmless.
- Expected harmless warnings: LangChain deprecation notices for `RunnableWithMessageHistory` and
  `InMemoryChatMessageHistory` (the brief asks for LangChain session memory, so they are left visible).

## Part 1 - Dataset, knowledge base, RAG (Tasks 1-5)

### Task 1 - dataset choices (`dataset.py`, transcript `task01_dataset.txt`)

| Choice | Value |
|---|---|
| Seed | **42** (isolated `random.Random(42)`, fully deterministic) |
| Records | **60** (`NYK-1001` ... `NYK-1060`) |
| Category weights | Beauty 0.30, Apparel 0.25, Footwear 0.15, Home 0.15, Electronics 0.15 |
| Status weights | Delivered 0.40, Shipped 0.20, Placed 0.15, Returned 0.15, Refunded 0.10 |
| Amount range (INR) | per category: Beauty 199-2,999; Apparel 499-3,999; Footwear 799-4,999; Home 299-5,999; Electronics 999-14,999 (overall **Rs 199 - Rs 14,999**) |
| Days since created | 0-30, banded by status (e.g. Placed 0-4, Delivered 3-30) |
| `delayed_shipment` | `True` with probability 0.18 per record |

Price ranges mirror typical Nykaa catalogue price points (Beauty cheap and high-volume, Electronics the priciest),
so order values look like real baskets instead of a flat random draw. Records are never hand-edited.

Realised counts with seed 42: Apparel 17, Electronics 12, Home 6, Footwear 8, Beauty 17; Placed 8, Shipped 10,
Delivered 28, Returned 6, Refunded 8. **Delayed shipments: 13 / 60 = 21.7 %** (required band 10-30 %).

### Task 2 - knowledge base (`kb/`)
12 illustrative policy documents (return windows, COD refunds, delivery SLAs, reverse pickup, warranty,
cancellation, loyalty points, payment failure, size exchange, damaged items, international shipping,
escalation matrix). The text is written for this project and is illustrative, not Nykaa's real policy.

### Task 3 - two chunking strategies, two ChromaDB collections (`rag/chunking.py`, `rag/index.py`)
- `kb_fixed`: 40-word windows, 10-word overlap -> **33 chunks**.
- `kb_sentence`: one chunk per sentence -> **49 chunks**.
- Embeddings: SentenceTransformers `all-MiniLM-L6-v2`; ChromaDB with cosine distance (`similarity = 1 - distance`).

### Task 4 - grounded generation with a calibrated threshold (`rag/calibrate.py`, `rag/grounded.py`)
The threshold is measured, not preset (no 0.5 / 0.6 / 0.7). Top-1 similarity was measured for 7 in-scope and
5 out-of-scope queries (`task04_calibration.txt`); the threshold is the midpoint of the gap, rounded to 2 decimals.

| Collection | Lowest in-scope | Highest out-of-scope | Gap | **Threshold** |
|---|---|---|---|---|
| `kb_fixed` | 0.411 | 0.136 | 0.275 | **0.27** |
| `kb_sentence` | 0.386 | 0.138 | 0.249 | **0.26** |

Below the threshold the agent answers "I don't have enough information ... please contact customer support."

### Task 5 - precision / recall comparison (`rag/evaluate_chunking.py`, `task05_chunking_eval.txt`)
7 queries, top-3 chunks mapped to parent documents and de-duplicated, macro-averaged.

| Collection | Mean precision | Mean recall |
|---|---|---|
| `kb_fixed` | **0.690** | **1.000** |
| `kb_sentence` | 0.667 | 0.952 |

Fixed-size chunks are slightly better on both: each chunk carries enough context to retrieve the right
document, while single sentences lost one of three relevant documents on the multi-topic query (Q7).
The crew's retrieval tool therefore uses `kb_fixed` and expands hits to their parent documents.

## Part 2 - the crew, memory, structured output, guardrails (Tasks 6-10)

- **Task 6 - `check_order_status` (`crew/order_status.py`).** Designed escalation score:
  `open_flag * (0.5 * delayed_shipment + 0.5 * days_since_created / 30)`, where open = Placed, Shipped or Returned.
  Escalate at **>= 0.40**: a delayed open order always escalates (>= 0.50); a non-delayed one needs to be about
  24+ days old. In this dataset 4 of 24 open orders (17 %) escalate. The tool returns found, status,
  order value, escalation score and the recommendation.
- **Task 7 - 3-agent CrewAI crew (`crew/crew_builder.py`)**, run with `crew.kickoff()`: Retrieval Agent
  (only `rag_lookup`), Lookup Agent (only `check_order_status`), Response Composer (no tools).
- **Task 8 - LangChain session memory (`crew/memory.py`)**: `RunnableWithMessageHistory` +
  `InMemoryChatMessageHistory`. "Where is my order NYK-1045?" followed by "Can I still cancel it?" carries the order
  id forward ("... regarding order NYK-1045", `task08_memory.txt`); a fresh session does not (`task08_memory_fresh.txt`).
- **Task 9 - Pydantic structured output (`crew/schemas.py`)**: every response is an `AnswerSchema`
  (intent, answer, grounded, top_similarity, sources, order, escalate), validated in code.
- **Task 10 - guardrails (`crew/guardrails.py`)**: input side masks PII (phone, card numbers / last-4) and blocks
  prompt-injection patterns; output side refuses answers the retrieved context does not support.

## Part 3 - API, logging, evaluation (Tasks 11-13)

### Tasks 11-12 - FastAPI + JSON-Lines logging (`api/`, `task11_12_api.txt`)
| Endpoint | Purpose |
|---|---|
| `POST /ask` | Ask a question (`question`, optional `session_id` for multi-turn memory) |
| `POST /add-document` | Add a policy document (injection scan, PII masked before storing, indexed in both collections) |
| `DELETE /documents/{id}` | Remove an added document (the original 12 are protected: 403) |
| `GET /health` | Liveness |
| `WS /ws/chat?session_id=...` | Multi-turn chat; client disconnects (clean or abrupt) end only that connection |

Every request writes one JSON line to `logs/requests.jsonl` with `@timestamp`, `trace_id` and `latency_ms`
(also returned in the `X-Trace-Id` header). **The request text is logged only after PII masking** (phone numbers, card numbers / last-4 and e-mail addresses), and the
logger masks every string a second time; the demo greps the whole log for the raw phone and card values.
The guardrail transcripts deliberately show *made-up* test values next to their masked form to prove masking works;
`logs/requests.jsonl` and `logs/eval_results.json` contain masked text only.

### Task 13 - LLM-as-judge evaluation (`eval/`, `task13_eval.txt`)
15 queries: one per knowledge-base topic (12), one out-of-scope, one prompt injection, and one order-status
query containing a phone number. Scored 1-5 on Accuracy, Grounding, Completeness and Safety per query.

| Accuracy | Grounding | Completeness | Safety |
|---|---|---|---|
| 5.0 | 5.0 | **4.4** | 5.0 |

Three queries scored 3 or below on Completeness: Q05 (warranty exclusions missing), Q09 (size exchange: the
"unworn" condition and refund fallback missing), Q10 (damaged item: photos and replacement/refund missing).
The composer returns at most 3 sentences and sometimes picks a less relevant one. These are reported, not tuned away.
The run also feeds the judge three deliberately bad answers (invented number, answered an out-of-scope question,
repeated a phone number) and confirms they score 1-2.

## Part 4 - review, governance, cost (Tasks 14-16)

### Task 14 - Autogen review (`review/`, `task14_autogen_review.txt`)
`RoundRobinGroupChat` with a Fact-Checker and a Final-Editor, `max_turns=2`. The Final-Editor uses
`output_content_type=Verdict` (`approved: bool, final_answer: str, reason: str`) and the team is created with
`custom_message_types=[StructuredMessage[Verdict]]`. Demonstrated: a grounded draft is **approved unchanged**; the same
draft with a deliberately injected ungrounded claim ("Refunds are guaranteed within 24 hours for every order.")
is **revised** and the claim removed; an order answer is approved after checking against the order record.

### Task 15 - governance (`governance/`, `task15_governance.txt`)
- **Least autonomy:** only the Lookup Agent holds `check_order_status`. `call_tool_as()` is the single gateway:
  the Retrieval Agent and Composer are denied (and audit-logged) and the tool never runs; in real crew runs the tool
  executes 0 times for a policy question and once for an order question.
- **Risk classification:** a scored register (sensitivity + writes + blast radius + irreversibility + unauthenticated
  exposure; 0-1 Low, 2-3 Medium, 4+ High) with a justification and mitigation per capability, plus a per-request
  classifier. Example: `rag_lookup` Low, `check_order_status` Medium, `add_document` / `delete_document` High.
- **Budget cap:** each request is estimated (characters / 4 + 150 tokens of scaffolding) **before** any agent runs;
  over 400 tokens or Rs 0.20 (a notional planning rate, since the mock model is free) it is rejected. Demonstrated:
  a normal request is allowed (3 LLM calls), a 1,518-character request is rejected with **0 LLM calls and 0 tool calls**.

### Task 16 - response cache (`rag/cache.py`, `task16_cache.txt`)
In-memory cache for the grounded-generation step, keyed by (collection, k, lower-cased and whitespace-collapsed
query). Punctuation is kept in the key because it changes the embedding. Evidence on 12 requests with 4 distinct questions:

| | Before | After |
|---|---|---|
| Embedding + search + generation runs | 12 | **4** |
| Total time | 202 ms | **18 ms** (11.1x faster) |
| Cached answers identical to uncached | - | True |

The cache is cleared when a document is added or deleted, so a cached "I don't know" cannot outlive the
document that now answers it (demonstrated in the same transcript).

## Repository map

| Path | Contents |
|---|---|
| `dataset.py`, `kb/` | Tasks 1-2 |
| `rag/` | chunking, index, calibration, grounded generation, evaluation, cache |
| `crew/` | mock LLM, tools, agents, memory, schemas, guardrails, pipeline |
| `api/` | FastAPI app, models, request log, document management |
| `eval/` | test set, rubric, mock judge, runner |
| `review/` | Autogen review team and mock model client |
| `governance/` | least autonomy, risk, budget, governed entry point |
| `transcripts/` | one transcript per task, regenerated by `run_all.py` |
| `run_all.py` | regenerates every transcript, cross-platform |

## Limitations (what this does and does not show)

- **The mock composer is a heuristic stand-in for a real LLM.** It selects and quotes knowledge-base sentences
  using keyword weighting, so answers are grounded by construction but not always the most helpful (see the three
  Completeness gaps above). The link between "hair dryer" and Electronics is bridged by a hand-written word list.
- **The similarity threshold separates off-topic from on-domain questions only.** An on-domain question the knowledge
  base cannot actually answer can still pass the threshold.
- **The judge is rule-based.** It compares answers with gold facts written from the knowledge base. It does not
  penalise extra cited sources, and Grounding is high partly because the composer quotes the KB verbatim.
  A real LLM judge could assess paraphrases; the judge interface (JSON verdict) is designed so one can replace it.
- **The Autogen Fact-Checker is a heuristic** (sentence must appear in the evidence, order sentences must match the
  order record). It cannot judge paraphrased claims.
- **The review, budget and cache stages are demonstrated as stand-alone stages**, each with its own demo and transcript.
  The live `/ask` endpoint runs input guardrails -> crew -> output guardrail; it is not wired through the Autogen
  review, the budget gate or the cache (`governance.governed.governed_answer` shows the budget and risk gate in front of the pipeline).
- **PII masking covers phone numbers, card numbers / last-4 and e-mail addresses.** It does not detect names or street addresses.
- **Token counts are estimates** (characters / 4 or word counts); there is no real tokenizer or billing.
- **`/add-document` and `DELETE /documents/{id}` are unauthenticated.** They are scored High risk and need
  authentication before any real deployment.
- **Crew runs are serialised** with a lock because tool-call logs are module-level; fine for a demo, not for scale.
- **Knowledge base and orders are synthetic.** The policies are illustrative and do not represent Nykaa's real terms.