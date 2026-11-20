# MEDHA — Multi-Agent Evidence-Driven Architecture

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Groq API](https://img.shields.io/badge/Primary%20LLM-Groq-green.svg)](https://groq.com/)
[![Google Gemini API](https://img.shields.io/badge/Alternative%20LLM-Google%20Gemini-orange.svg)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/tests-offline%20suite-brightgreen.svg)]()
[![Model Context Protocol](https://img.shields.io/badge/MCP-2.3.0-purple.svg)](https://modelcontextprotocol.io/)

**SEED Hackathon 2026** | **Problem Statement 02 — Chain of Command**  
*Topic-Independent, Dynamically Orchestrated Multi-Agent Intelligence with Local Hybrid Retrieval, Ephemeral Execution Storage, and Adversarial Claim Review*

---

## 1. Executive Summary

Traditional LLM workflows and naive multi-agent systems suffer from common failure modes:
1. **Echo Chambers / False Consensus**: Agents with static personas simply agree with one another or uncritically validate the primary prompt.
2. **Ungrounded Confidence**: High token probabilities and self-assessed confidence scores masquerade as factual evidence.
3. **Rigid Predefined Roles**: Fixed agent configurations struggle when encountering novel, multi-disciplinary problems.
4. **All-or-Nothing Rejection**: When an answer contains a single flaw, the entire output is either rejected or accepted with the flaw intact.

**MEDHA** is designed to reduce these issues through a dynamic, evidence-aware multi-provider architecture. It does not guarantee factual correctness; reviewer acceptance is an LLM judgment, not proof.
- **Multi-Provider LLM Architecture**: Uses **Groq** (`openai/gpt-oss-20b`) as its high-speed primary LLM provider while retaining **Google Gemini** (`gemini-3.8-flash`) as a fully supported alternative.
- **Local Hybrid Retrieval**: Parses explicitly supplied PDF, DOCX, text, and structured-text sources, combines SQLite FTS5/BM25 keyword ranking with local FastEmbed vector similarity, and returns bounded excerpts with source locations.
- **Ephemeral-by-Default Storage**: Gives each execution a unique workspace, uses an in-memory retrieval index in Ephemeral Mode, suppresses execution JSON persistence, redacts execution-specific log messages, and verifies cleanup.
- **Dynamic Specialist Creation**: Analyzes each user task to dynamically determine the exact domain expertise required and creates custom specialist agents on the fly.
- **Dependency-Aware Parallel Execution**: Automatically derives DAG dependencies, executing independent specialist subtasks concurrently while routing upstream findings to downstream specialists.
- **Structured Inter-Agent Message Bus**: Agents exchange formal findings, questions, evidence, and challenges across dedicated asynchronous inboxes.
- **Traceable Claim Ledger**: Tracks atomic claims, provenance, references, review status, and history. References are not treated as verified sources until retrieved or supplied with traceable content.
- **Adversarial LLM Review**: A reviewer role challenges unsupported leaps, absolutes, and hidden assumptions. It is an LLM judgment, not an independent institution or proof engine.
- **Targeted Precision Repair**: Routes specific reviewer critiques back to the responsible specialist to narrow claims, declare explicit assumptions, or supply valid derivations.
- **Hybrid Retrieval**: When local documents or allowlisted HTTPS sources are supplied, MEDHA combines SQLite FTS5/BM25 keyword search with local FastEmbed vectors and reciprocal-rank fusion.
- **Privacy-Conscious Storage**: Ephemeral Mode is the default; per-execution retrieval data uses an isolated in-memory SQLite index and is cleaned after answer delivery.
- **Transparent Truthful Execution**: Reports truthful statuses (`SUCCESS`, `PARTIAL_FAILURE`, `FAILED`), explicitly discloses when zero claims were available for verification, and never presents unverified fallback text as truth.
- **Full Model Context Protocol (MCP) Server**: Exposes MEDHA tools over the official MCP Python SDK for seamless integration into external agent environments.
- **Quantitative Benchmarking Suite**: Evaluates MEDHA against a single-agent baseline on reference criteria coverage, claim support rates, repair efficiency, and token economics.

---

## 2. System Architecture

```mermaid
flowchart TD
    User([User Prompt and optional documents]) --> Workspace[Ephemeral execution workspace]
    Workspace --> Ingest[Parse, chunk, and index supplied sources]
    Ingest --> Planner[Task Planner]
    Planner -->|Dynamic Decomposition| Plan[Task Plan & DAG]
    
    subgraph ExecutionEngine [Parallel Execution Engine]
        Plan --> T1[Specialist 1]
        Plan --> T2[Specialist 2]
        T1 & T2 -->|Dependencies| T3[Integrating Specialist]
        Ingest -->|Relevant excerpts + source IDs| T1
        Ingest -->|Relevant excerpts + source IDs| T2
        Ingest -->|Relevant excerpts + source IDs| T3
    end

    subgraph Communication [Message Bus & Shared Context]
        T1 <-->|Findings & Challenges| MsgBus[Message Bus]
        T2 <-->|Evidence Sharing| MsgBus
        T3 <-->|Questions & Notes| SharedCtx[Shared Context]
        T3 <--> MsgBus
    end

    ExecutionEngine --> Extractor[Claim Extractor]
    Extractor --> Ledger[(Evidence Ledger)]

    subgraph Verification [Reliability & Verification Loop]
        Ledger --> Reviewer[Adversarial Reviewer]
        Reviewer -->|Audited Claims| Decision{Decision}
        Decision -->|ACCEPT| Accepted[Accepted Claims]
        Decision -->|REJECT / REVISE| Repair[Targeted Repair Engine]
        Repair -->|Precision Corrections| Reviewer
        Decision -->|Unresolvable| Unverifiable[Preserved Uncertainty]
    end

    Accepted & Unverifiable --> Synthesizer[Final Synthesizer]
    Synthesizer --> FinalAnswer([Answer returned to user])
    FinalAnswer --> Cleanup[Verify workspace cleanup]
```

---

## 3. Project Structure

```
MEDHA/
│
├── main.py                     # Interactive CLI and application entry point
├── requirements.txt            # Providers, schemas, MCP, document parsing, and local embeddings
├── .env                        # Local configuration and credentials (never committed)
├── .env.example                # Example multi-provider environment configuration
├── .gitignore                  # Git ignore rules
│
├── config/
│   ├── settings.py             # Multi-provider environment configuration & validation
│   └── logging_config.py       # Credential-redacting structured logging
│
├── core/
│   ├── llm_client.py           # Provider factory & global singleton manager
│   ├── providers/
│   │   ├── base.py             # BaseLLMClient abstract base class
│   │   ├── groq_client.py      # Groq implementation (AsyncGroq, JSON mode, retry, tokens)
│   │   └── gemini_client.py    # Google Gemini implementation (google.genai, retries)
│   ├── orchestrator.py         # Master workflow controller & execution manager
│   ├── planner.py              # Dynamic task planner & role decomposer
│   ├── agent_factory.py        # Dynamic agent creation & role routing
│   ├── execution_engine.py     # Dependency-aware concurrent executor
│   ├── task_manager.py         # Subtask state machine & dependency DAG
│   └── workflow.py             # Lifecycle stage tracker & audit trail
│
├── agents/
│   ├── base_agent.py           # Abstract agent base class
│   ├── specialist_agent.py     # Dynamic domain specialist implementation
│   ├── research_agent.py       # Literature and state-of-the-art specialist
│   ├── analyst_agent.py        # Quantitative & bottleneck analysis specialist
│   ├── technical_agent.py      # Engineering specification specialist
│   ├── critic_agent.py         # Vulnerability & edge-case specialist
│   ├── reviewer_agent.py       # Independent adversarial claim auditor
│   └── synthesizer_agent.py    # Evidence-grounded final synthesizer
│
├── communication/
│   ├── message.py              # Structured inter-agent message schema
│   ├── message_bus.py          # Asynchronous queue-based message broker
│   └── shared_context.py       # Thread-safe shared knowledge repository
│
├── retrieval/
│   ├── document_parser.py      # PDF, DOCX, HTML, and text extraction
│   ├── chunker.py              # Bounded chunks with source location metadata
│   ├── embedding_engine.py     # Local FastEmbed adapter
│   ├── store.py                # Per-execution SQLite FTS5/vector index
│   ├── reranker.py             # Reciprocal-rank fusion
│   └── retriever.py            # Local and allowlisted HTTPS source ingestion
│
├── storage/
│   └── workspace.py            # Isolated execution workspaces, TTL cleanup, watchdog
│
├── verification/
│   ├── claim_extractor.py      # Discrete claim and assumption extractor
│   ├── evidence_ledger.py      # Centralized claim and evidence ledger
│   ├── claim_validator.py      # Heuristic validator for ungrounded claims
│   ├── adversarial_reviewer.py # Multi-criteria claim audit coordinator
│   └── repair_engine.py        # Iterative targeted claim repair engine
│
├── models/
│   ├── schemas.py              # Enums (TaskStatus, MessageType, ClaimStatus)
│   ├── task_models.py          # SubTask, TaskPlan, TaskResult schemas
│   ├── agent_models.py         # AgentConfig and AgentResponse schemas
│   └── verification_models.py  # Claim, Evidence, ReviewFeedback, Report
│
├── benchmark/
│   ├── baseline.py             # Single-agent direct baseline runner
│   ├── medha_runner.py         # MEDHA workflow benchmark runner
│   ├── evaluator.py            # Comparative evaluation engine
│   ├── metrics.py              # Statistical scoring and pricing metrics
│   ├── benchmark_dataset.json  # Curated multi-domain evaluation cases
│   ├── benchmark_report.json   # Exported quantitative JSON report
│   └── benchmark_results.csv   # Exported task-by-task CSV report
│
├── mcp_server/
│   ├── server.py               # Official Model Context Protocol (MCP) server
│   └── tools.py                # MCP tool implementations & execution cache
│
├── utils/
│   ├── helpers.py              # ID generators, JSON sanitizers, timestamps
│   ├── logger.py               # Structured ExecutionRecord and file logger
│   ├── retry.py                # Exponential backoff, jitter, Retry-After & quota classification
│   └── token_tracker.py        # Thread-safe token consumption accumulator
│
├── tests/
│   ├── test_settings.py        # Multi-provider configuration & validation tests
│   ├── test_llm_client.py      # Groq, Gemini, retries, quota exhaustion, tokens
│   ├── test_agent_factory.py   # Dynamic agent creation and planning tests
│   ├── test_message_bus.py     # Message bus routing and concurrency tests
│   ├── test_verification.py    # Claim extraction, ledger, and auditing tests
│   ├── test_repair.py          # Targeted repair and re-review tests
│   ├── test_orchestrator.py    # Truthful execution status (SUCCESS, PARTIAL, FAILED) tests
│   ├── test_mcp.py             # MCP server tool execution tests
│   └── test_benchmark.py       # Benchmark evaluation and metric tests
│
├── documents/                  # Optional local MCP document root (user data, ignored)
├── data/                       # Optional Persistent Mode execution stores (ignored)
└── logs/                       # Persistent Mode execution traces; Ephemeral Mode skips reports
```

---

## 4. Setup and Installation

### Prerequisites
- Python 3.13+
- Active **Groq API Key** (Primary) and/or **Google Gemini API Key** (Alternative)

### Installation

1. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *(If script execution is restricted on Windows, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
    Retrieval uses PyMuPDF and python-docx for parsing, plus FastEmbed for local CPU embeddings. The first retrieval may download the configured embedding model to the machine's shared model cache. This is not a paid embedding API.

3. **Configure credentials in `.env`**:
   Copy `.env.example` to `.env` and configure your chosen provider:

   **Primary Provider — Groq (Default):**
   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   TEMPERATURE=0.2
   ```

   **Alternative Provider — Google Gemini:**
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-3.8-flash
   TEMPERATURE=0.2
   ```

Set `MEDHA_ALLOWED_SOURCE_DOMAINS` to a comma-separated allowlist before using `--source-url`. Online fetching is disabled for hosts not on that list.

---

## 5. Usage

### 1. Interactive Command Line Interface
Launch the interactive MEDHA assistant:
```powershell
python main.py
```

### 2. Direct Task Execution
Run a single task directly via CLI:
```powershell
python main.py --task "Explain how database indexes help queries."
```
By default this uses Ephemeral Mode. `--full` prints the complete audit; an execution JSON file is written only in Persistent Mode.

Supply local sources explicitly with repeated `--document` arguments:
```powershell
python main.py --task "Summarize the supplied policy." --document .\documents\policy.pdf --document .\documents\appendix.docx
```

Approved HTTPS sources can be supplied with `--source-url`:
```powershell
python main.py --task "Explain the current setting." --source-url https://docs.example.org/guide.html
```
The host must be listed in `MEDHA_ALLOWED_SOURCE_DOMAINS`; only HTTPS is accepted, each redirect is rechecked, and non-public IP destinations are rejected. No source is fetched if neither documents nor URLs are provided.

### Execution Storage

- **Ephemeral Mode (default):** per-execution workspace, in-memory SQLite retrieval index, no execution JSON or MCP job files, content-redacted application logs, and cleanup on success, failure, timeout, or cancellation.
- **Persistent Mode:** opt in with `MEDHA_STORAGE_MODE=persistent` or `--persistent`. Execution records and retrieval indexes may be retained until `MEDHA_PERSISTENT_RETENTION_DAYS` expires.

For MCP, choose `MEDHA_STORAGE_MODE` before starting the server. An ephemeral background result is held in memory while it awaits delivery, then cleared after `get_execution_result`; unclaimed results expire after `MEDHA_MCP_RESULT_TTL` seconds.

Ephemeral Mode is an application-level non-retention policy, not physical erasure. MEDHA cannot delete data already sent to Groq/Gemini, terminal captures, OS swap/crash dumps, backups, or the user's original files. FastEmbed model weights may persist in a shared model cache; they are generic model files, not task content. MEDHA does not encrypt Persistent Mode files at rest.

### Hybrid Knowledge Retrieval

Retrieval is activated only when a task supplies `--document` paths or approved `--source-url` values. MEDHA does not crawl the repository or search the public web automatically.

- Supported local types: PDF, DOCX, TXT, Markdown, reStructuredText, CSV, JSON, and HTML.
- PDF parsing retains page numbers; DOCX parsing retains section headings where available.
- Chunks retain source ID, name, reference, chunk ID, page/heading, source type, and retrieval timestamp.
- SQLite FTS5 provides BM25 keyword ranking. FastEmbed generates local vectors. Reciprocal-rank fusion combines the two rankings; no paid embedding API is used.
- Only retrieved chunks explicitly cited by a specialist using their exact chunk ID are registered as retrieved evidence for review.
- The reviewer receives the actual excerpt and source metadata, not just a URL. An agent-generated URL is still unverified until MEDHA fetches it through the approved-source loader.
- Ephemeral Mode keeps chunks and vectors in an execution-local in-memory SQLite database. Persistent Mode stores an execution-local SQLite index under `data/executions/` and expires it according to the retention setting.
- The local embedding model may be downloaded on first use and cached outside the execution workspace. It is generic model data, not a document or task artifact.

If no relevant excerpts are found, MEDHA tells the specialist that no registered source was available; it does not fabricate a citation. In Strict mode, unsupported claims should remain unverified. In General mode, limited common-knowledge claims may be accepted but are labeled as not externally evidenced.

### 3. Single-Agent Baseline
Run a task using the un-orchestrated single-agent baseline for direct comparison:
```powershell
python main.py --baseline "Design a high-throughput financial ledger processing 100k TPS."
```

### 4. Four-Arm Benchmark Suite
Run the four-arm frozen-plan experiment comparing the single-agent baseline, independent specialists, dependency-aware execution, and full MEDHA:
```powershell
python main.py --benchmark --persistent --benchmark-cases 2 --benchmark-repetitions 1 --experiment-id pilot-01
```
Each explicit experiment ID produces `benchmark/experiments/<experiment-id>.json` and `.csv`. Existing files are never overwritten. `--benchmark-output` selects a directory within `benchmark/`, for example `--benchmark-output experiments/run-01`.
Benchmark startup requires `VERIFICATION_MODE=strict` and the explicit `--persistent` flag. Do not run until the dataset, source packets, scoring process, budget, and pre-pilot authorization checklist are approved.

#### Benchmark Status and Protocol

The evaluator prepares one frozen plan per task block and reuses it for arms B, C, and D; arm A calls the baseline directly. Each run gets a fresh plan copy and independent accounting. Planner preparation cost is recorded once as actual experiment expenditure and included as shared preparation cost in each B/C/D end-to-end comparison figure. The deterministic four-sequence schedule rotates through A-B-D-C, B-C-A-D, C-D-B-A, and D-A-C-B. Planned and actual orders are both exported, along with whether a complete, uninterrupted four-sequence cycle occurred.

Results are exploratory and do not claim statistical significance. Configuration mismatches or unknown required settings remain visible in trial records and exclude incompatible trials from comparative aggregates. Automated completeness, pitfall, uncertainty, and evidence-quality metrics are proxies, not factual correctness. No human factual score is fabricated; factual correctness remains null without an actual blinded human or independent expert score.

Provider-reported immutable model revisions and provider SDK default timeout durations may be unavailable. These are recorded as unavailable compatibility limitations, not assumed equal and not alone grounds to reject an otherwise matching requested configuration. `MODEL_VERSION` is an operator-supplied label, not provider-attested revision metadata. Set `PROVIDER_REQUEST_TIMEOUT_SECONDS` to apply and record an explicit request timeout; blank uses the provider SDK default.

Effective settings are compared by recorded stage: planning, specialist execution, claim extraction, verification/reviewer, repair, synthesis, and deliberation. Reviewer temperature (`0.1`) is compared with reviewer calls, not the global specialist/synthesis temperature (`0.2`). D-only deliberation calls and their remaining-time timeout are recorded as treatment-specific; they are not compared as if B/C should execute deliberation. Differences within a stage shared by B/C/D remain incompatible. Missing stage metadata or dropped provider-call records makes effective configuration indeterminate and excludes comparative aggregates; a known retained mismatch remains incompatible.

Only a nonempty successful output is eligible for answer-quality aggregates. Failed, partial, empty, and interrupted attempts remain in the JSON/CSV outputs with diagnostics and any available answer. Verification metrics for arm A are null/N/A because verification is not executed. Missing values retain eligible counts, attempt counts, and missingness reasons in JSON.

Operational aggregates include all attempts; comparative aggregates apply compatibility and metric eligibility separately. Legacy `aggregate_results` retains operational top-level fields for compatibility and labels them; use nested `operational` and `comparative` objects for unambiguous interpretation. Trial-level timeouts are `TIMED_OUT`; provider-call timeouts are reported separately.

Token counts and total/failed request counters come from the `AccountingSnapshot`; provider-call records provide retry/fallback attribution, per-call settings and status. JSON and CSV use the same snapshot request counters. Retry/fallback counts are marked as lower bounds if call records were dropped. CSV includes compatibility status/limitations, eligibility and missingness fields. Verification reports include the unweighted mean of eligible trial rates and a pooled claim-level rate; unreviewed claims are excluded from the pooled reviewed-claim denominator.

Costs require explicit provider/model input and output pricing and provider-reported input/output usage for a dollar amount. Missing rates, missing usage, estimated-only usage, or mixed usage without a separable input/output split leave the amount null. Legacy two-arm aggregates expose known subtotals and unknown-cost counts; an unknown total is never represented as zero. Estimated and mixed provenance remains visible.

Tests use mocked providers and deterministic fixtures. Do not treat successful mocked tests as evidence of live provider behavior. The benchmark command is an explicit live experiment and should only be run intentionally with valid provider configuration.

For live validation, first confirm provider quota in its console, then run a small two- or three-specialist task. Compare discovered, deduplicated/registered, capacity-rejected, reviewed, and unreviewed claims. Include deliberately unsupported empirical claims and confirm they are not accepted without external evidence. Run the larger attendance-system task only after the smaller run completes successfully. Do not treat a successful mocked test as evidence of live provider behavior.

Known limitations to retain when interpreting results:
- Token estimates are byte-based approximations, not exact provider token counts. When usage is unavailable, MEDHA conservatively charges the reservation estimate; this may overstate consumption.
- Claim-kind classification is heuristic and can miss unusual or misleading phrasing, including performance targets framed as implied measurements. It should be evaluated against a deliberately constructed difficult-claims set.
- Neither limitation has been validated against live provider behavior, and MEDHA cannot observe usage from other applications sharing the provider account.

### 5. Model Context Protocol (MCP) Server
Start MEDHA as a standard MCP server over stdio:
```powershell
python main.py --mcp
```
Exposed Tools:
- `run_medha_task(task: str, max_concurrency: int = 4, max_repair_iterations: int = 2, documents: list[str] = [], source_urls: list[str] = [])`
- `start_medha_task(task: str, ..., documents: list[str] = [], source_urls: list[str] = [])`
- `get_execution_status(execution_id: str)`
- `get_verification_report(execution_id: str)`
- `run_baseline(task: str, model: str = "")`

In Ephemeral Mode, MCP local document paths must resolve under `MEDHA_DOCUMENTS_DIR`. A background result stays in memory awaiting delivery; fetching it clears task-specific result data. Select Persistent Mode before server startup if durable MCP recovery is required.

---

## 6. Truthful Execution Statuses

MEDHA introduces truthful execution reporting to eliminate phantom successes:
- **`SUCCESS`**: No qualifying workflow interruption was recorded. This does not mean every claim was supported or factually true.
- **`PARTIAL_FAILURE`**: Useful work may exist, but a specialist or a critical stage (retrieval, deliberation, extraction, verification, repair, synthesis, or cleanup) was incomplete.
- **`FAILED`**: Zero specialist subtasks completed successfully. The workflow terminates with a transparent failure report detailing subtask errors and explicitly stating that no claims were available for verification. Fallback text is **never** presented as verified truth.

---

## 7. Rate Limiting, Retries, and Quota Troubleshooting

MEDHA's resilient retry engine ([`utils/retry.py`](file:///C:/MEDHA/utils/retry.py)) differentiates between transient rate limits and daily quota limits:

| Error Type | Status Code | Classification | Behavior |
| :--- | :--- | :--- | :--- |
| **Transient Rate Limit (RPM / TPM)** | `429` | Retryable (`LLMRateLimitError`) | Backs off exponentially with jitter and honors `Retry-After` headers for short delays. |
| **Daily / Monthly Quota Limit (RPD)** | `429` | Non-retryable (`LLMQuotaExhaustedError`) | Fails immediately with clear diagnostic guidance. Does **not** retry or hang for 18 hours. |
| **Authentication / Permission** | `401` / `403` | Non-retryable (`LLMAuthError`) | Fails immediately. Check your `GROQ_API_KEY` or `GEMINI_API_KEY` in `.env`. |
| **Invalid / Unavailable Model** | `404` | Non-retryable (`LLMModelNotFoundError`) | Fails immediately. Verify `GROQ_MODEL` or `GEMINI_MODEL` spelling. |
| **Temporary Server Failure** | `500` / `502` / `503` / `504` | Retryable (`LLMServerError`) | Automatically retries up to `MAX_RETRIES=3` with exponential backoff. |
| **Network Timeout** | `408` / `Timeout` | Retryable (`LLMTimeoutError`) | Retries with backoff up to `EXECUTION_TIMEOUT_SECONDS=180.0`. |

---

## 8. Automated Testing

All unit tests use mock responses to ensure zero external API charges during testing.

Run the test suite:
```powershell
.\.venv\Scripts\pytest -v
```
Tests use mocked provider responses. Use the actual pytest summary from the current checkout; a passing mock suite does not establish live API reliability or factual accuracy.

---

## 9. License

Developed for the **SEED Hackathon 2026** — Problem Statement 02 (Chain of Command).
