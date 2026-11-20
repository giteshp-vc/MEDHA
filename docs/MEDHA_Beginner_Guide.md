# MEDHA Beginner Guide

**Project:** MEDHA, Multi-Agent Evidence-Driven Architecture
**Purpose of this guide:** Explain what the repository does, how a task moves through it, and what its privacy and evidence claims mean.

## 1. MEDHA in one minute

MEDHA is a Python application that coordinates calls to a language model. It can ask a planner to divide one user task into smaller subtasks, run specialists for those subtasks, share context between them, collect claims, ask a reviewer to challenge those claims, ask specialists to revise some claims, and then produce a final answer.

MEDHA is **not** a model that was trained by this project. The planner, specialists, reviewer, and synthesizer are different roles and prompts. They usually use the same configured provider and model, so they can share the same blind spots.

The main benefit is workflow structure and visibility: a user can inspect which roles ran, what claims were made, which received support, which remain uncertain, and where the pipeline stopped. The system does not guarantee that an accepted claim is true.

## 2. A task's journey

1. The command-line interface or MCP tool receives a prompt and optional document paths or source URLs.
2. MEDHA creates a unique execution ID and an execution workspace.
3. The planner asks the model for a structured plan containing specialist subtasks and dependencies.
4. The task manager and execution engine schedule ready subtasks. A task with a dependency waits for the required earlier task.
5. The agent factory creates a specialist instance for each role. Some role names map to specialist archetypes; other roles use the general specialist.
6. If the user supplied sources, the retrieval component parses them, creates chunks, indexes them, and selects relevant excerpts for each specialist query.
7. Specialists return a deliverable plus findings, claims, references, assumptions, and open questions.
8. The message bus and shared context pass information between agents. Messages are context, not evidence.
9. MEDHA extracts or adopts claims and stores them in an Evidence Ledger.
10. An adversarial reviewer evaluates claims. Rule-based checks can flag specific issues separately from the reviewer's LLM decision.
11. Claims marked NEEDS_REVISION may be sent back to their originating specialist for a limited repair and re-review.
12. A synthesizer answers the user's task. If synthesis fails, a clearly labeled unverified fallback may be built from a completed specialist deliverable.
13. Ephemeral Mode closes the retrieval store, removes the workspace, verifies that its directory is gone, and returns the answer. Persistent Mode retains records according to its retention policy.

## 3. Main pieces of the code

| Area | Role |
|---|---|
| `main.py` | CLI, argument validation, answer formatting, and benchmark gate. |
| `config/` | Provider, storage, retrieval, and execution settings and logging configuration. |
| `core/orchestrator.py` | Runs the overall lifecycle and assigns truthful status. |
| `core/planner.py` | Creates the dynamic task plan. |
| `core/task_manager.py` | Tracks subtask state and dependencies. |
| `core/execution_engine.py` | Runs ready subtasks with concurrency and deadline limits. |
| `core/agent_factory.py` and `agents/` | Create planner-directed specialist roles, reviewer, and synthesizer. |
| `communication/` | MessageBus, SharedContext, and targeted deliberation. |
| `retrieval/` | Document extraction, chunking, local embeddings, keyword/vector search, and source metadata. |
| `storage/workspace.py` | Unique per-execution workspace, retention, startup cleanup, and watchdog. |
| `verification/` | Claim extraction, ledger, heuristic checks, reviewer coordination, and repair. |
| `models/` | Pydantic contracts for plans, claims, evidence, messages, and results. |
| `utils/` | Retries, token/request accounting, and structured execution logs. |
| `benchmark/` | Baseline, four-arm treatments, evaluator, metrics, and legacy dataset. |
| `mcp_server/` | MCP tools and in-memory/persistent job registry. |

## 4. How retrieval works

Retrieval is activated only when a caller supplies documents or approved URLs. MEDHA does not crawl the repository or search the public web automatically.

Supported local formats are PDF, DOCX, TXT, Markdown, reStructuredText, CSV, JSON, and HTML. PDF chunks retain page numbers. DOCX extraction retains section headings where available. Other formats are rejected rather than silently treated as text.

Each chunk retains a source ID, original source name/reference, source type, chunk ID, page or section location when available, content type, and retrieval timestamp. Chunk sizes and retrieval limits are configurable.

The hybrid index combines:

- SQLite FTS5's built-in BM25 ranking for keyword matches.
- Local FastEmbed vectors and exact cosine similarity for semantic matches.
- Reciprocal-rank fusion to combine the two ranked lists without pretending their raw scores share a scale.

A local file is read from its original path; MEDHA does not delete the user's original file. In Ephemeral Mode, extracted text, chunks, embeddings, and the SQLite FTS/vector index live in the execution-local in-memory store. In Persistent Mode, the index is stored in the workspace on disk and expires according to the retention policy.

An online source is fetched only if its HTTPS hostname is in `MEDHA_ALLOWED_SOURCE_DOMAINS`. Redirect destinations are checked again, and non-public IP destinations are rejected. The default allowlist is empty. Source retrieval is bounded by the configured response-size limit.

A retrieved excerpt is shown to the specialist with its exact chunk ID. The specialist must cite that ID in its structured evidence-reference list for MEDHA to register that excerpt as retrieved evidence for review. A URL or citation invented by a model is not fetched or trusted by itself.

If no relevant chunk is found, the specialist is told that no registered source excerpt was found. If ingestion or retrieval fails, MEDHA records a safe diagnostic and continues without the unavailable source. In strict mode, unsupported claims should remain unsupported or unverifiable.

### Local embedding model

FastEmbed is a free local embedding library. Its model weights may be downloaded the first time retrieval is used and cached in the user's shared model cache. Those weights are generic model files, not user documents or execution embeddings. The cache is outside the per-execution workspace and is not deleted by MEDHA.

## 5. Evidence and verification

A claim is an individual assertion extracted from an agent's response. The ledger records claim text, ID, claim kind, originating role/subtask, provenance, references, assumptions, status, feedback, and status history. Duplicate claim text is merged after normalizing case and whitespace. The ledger has a maximum claim capacity; overflow claims are reported and are not reviewed.

Evidence types distinguish user statements, user-provided artifacts, tool-retrieved sources, benchmark data, reasoning inference, and model domain knowledge. A source qualifies as external only when it has supporting content and traceable provenance. A specialist saying “I found this online” is not equivalent to a recorded retrieval.

The reviewer can return ACCEPT, REJECT, REQUEST_REVISION, or UNVERIFIABLE. Deterministic checks such as the structure/provenance heuristics are stored separately from the original LLM verdict. Current deterministic rules can flag selected unsupported empirical claims and absolutes; they are not a general fact-checker or proof engine.

Repair only applies to NEEDS_REVISION claims. Rejected claims are not rewritten to make them pass. After the configured repair limit, remaining revision claims become UNVERIFIABLE. Claims not reached because of an interruption remain UNREVIEWED.

The support rate is:

`SUPPORTED reviewed claims / all reviewed claims x 100`

Unreviewed claims are excluded. This rate measures the reviewer's decisions, not factual accuracy. A human or expert score against a frozen answer key is needed to evaluate factual accuracy independently.

### Strict and general verification modes

- `VERIFICATION_MODE=strict` requires registered external support or sound deductive necessity for acceptance. This mode is required for benchmark execution.
- `VERIFICATION_MODE=general` permits carefully bounded, widely established textbook knowledge without a registered source. Those claims are explicitly labeled “General knowledge (not externally evidenced).” Numerical, empirical, recent, or contested assertions still need evidence.

General mode improves everyday answers when no corpus is supplied. It is not a web lookup and should not be used for a source-grounded benchmark.

## 6. Ephemeral and Persistent storage

`MEDHA_STORAGE_MODE=ephemeral` is the default. Each execution gets a random-ID workspace under the configured temporary root. The workspace manifest contains lifecycle metadata only, not the prompt. Retrieval data is isolated per execution. On normal completion, failure, exception, timeout, or cancellation, resources are closed, the workspace is removed, and the application verifies that it no longer exists.

At startup, MEDHA sweeps abandoned ephemeral workspaces whose recorded owner process is no longer alive. A watchdog refreshes active workspace leases and checks for abandoned workspaces. Cleanup is idempotent and reports failure instead of claiming deletion if the workspace remains.

`MEDHA_STORAGE_MODE=persistent` is an explicit alternative for research/debugging. Execution JSON, MCP job records, retrieval databases, and workspace data can be retained. Persistent workspace and execution records are removed after the configured retention period. The CLI's `--persistent` flag opts a specific run in; benchmark exports require it. MCP selects mode before server start through the environment.

In Ephemeral Mode, MCP background output is held in memory while waiting for delivery. The job reports `AWAITING_DELIVERY`; after `get_execution_result` returns the answer, MEDHA clears result content. Unclaimed results are cleared after `MEDHA_MCP_RESULT_TTL`.

### What Ephemeral Mode does and does not promise

It prevents MEDHA from writing new task-specific execution reports, retrieval indexes, documents, claims, and agent transcripts to persistent application storage during Ephemeral Mode. File logs are reduced to non-content operational messages. It does not erase historical logs created before the mode was enabled.

It cannot delete information already sent to Groq or Gemini, data saved by the user or terminal, OS swap files, crash dumps, backups, snapshots, or provider-side retention. The user's original document is never deleted. Ephemeral SQLite lives in process memory, but the OS may page process memory to disk. MEDHA does not currently encrypt Persistent Mode indexes at rest. Use OS full-disk encryption and provider privacy controls for stronger protection.

## 7. CLI and MCP usage

```powershell
# Default Ephemeral Mode, no source retrieval
python main.py --task "Explain how database indexes help queries."

# Read local sources and retrieve relevant chunks
python main.py --task "Summarize this policy." --document .\documents\policy.pdf --document .\documents\appendix.docx

# Fetch a source only after its domain is allowlisted
python main.py --task "Explain this documentation." --source-url https://docs.example.org/guide.html

# Print the full audit
python main.py --task "Summarize this policy." --document .\documents\policy.pdf --full

# Explicitly retain a normal task's execution record
python main.py --task "Review this design." --persistent

# Benchmark: strict mode and persistence are mandatory
python main.py --benchmark --persistent --benchmark-cases 2 --experiment-id pilot-01

# Start MCP; storage mode is read from the environment before server startup
python main.py --mcp
```

The MCP task tools accept `documents` and `source_urls`. MCP local paths must be inside `MEDHA_DOCUMENTS_DIR`. This prevents a remote tool caller from using MEDHA as an arbitrary file reader.

## 8. Benchmark compatibility and readiness

The four arms remain unchanged:

| Arm | Treatment |
|---|---|
| A | Direct baseline, without the multi-agent verification/synthesis pipeline. |
| B | Independent specialist execution plus verification and synthesis. |
| C | B plus dependency scheduling, upstream results, and message routing. |
| D | C plus deliberation. |

A single frozen plan is reused for B/C/D. The B-to-C comparison is bundled; it does not isolate a single feature. Order balancing is not a power calculation. The legacy five-case dataset is not an approved pilot dataset. Real task prompts, source packets, human evaluation, cost ceilings, and authorization remain necessary before a benchmark can support comparative claims.

Retrieval is an additional experimental factor. Do not silently enable retrieval for only some arms. For the original benchmark, either disable retrieval in all arms or freeze identical source packets and retrieval configuration for all eligible treatments, then version the protocol separately.

## 9. Installation and dependencies

Install with `pip install -r requirements.txt`. New dependencies include PyMuPDF, python-docx, and FastEmbed. They run locally; no paid embedding API is mandatory. The embedding model download may require network access once. The existing Groq/Gemini provider APIs still require their own credentials for LLM calls.

The current repository does not pin every dependency version with a lockfile. A reproducible release should add a reviewed lockfile or record an exact environment snapshot.
