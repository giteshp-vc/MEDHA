# MEDHA Project Explainer & Q&A Playbook

**Purpose:** Help the project owner explain MEDHA in a demo, review, or Q&A. The short answers come first; use the details when someone asks for more.

## 30-second pitch

MEDHA is a Python application that coordinates LLM-powered roles around one task. It plans specialist work, schedules dependencies, shares context, collects claims, asks a reviewer to challenge them, can retrieve relevant excerpts from supplied sources, and writes an answer with uncertainty visible. Ephemeral Mode is the default and cleans MEDHA's execution workspace. Its reviewer is still an LLM, so the system improves process transparency but does not guarantee truth.

## Two-minute pitch

A user submits a task through the CLI or MCP. The planner asks the configured model to break it into subtasks with roles and dependencies. The execution engine schedules ready work; the factory creates role-directed agents. Those roles normally share the same provider/model, so they are not independent models.

When the user supplies documents or approved HTTPS URLs, MEDHA parses them, chunks the text, preserves source locations, and indexes it in an execution-scoped SQLite store. Search combines FTS5/BM25 keyword ranking with local FastEmbed vector similarity. Specialists see relevant excerpts rather than the whole knowledge repository. An agent-generated URL is not trusted until retrieved and registered.

Specialists return deliverables and candidate claims. MEDHA stores claims and provenance in an Evidence Ledger, asks an adversarial reviewer for verdicts, records deterministic checks separately, and repairs a limited set of claims. A synthesizer answers the task and reports uncertainty. Ephemeral Mode cleans execution-specific application data; Persistent Mode is an explicit opt-in for research and authorized benchmark records.

The main limitation is crucial: an LLM reviewer is not proof, and support rate is not factual accuracy. Source-grounded claims require traceable retrieved/supplied evidence; factual performance requires independent human/expert evaluation.

## Demo script

1. Run a small task with no sources and explain that no web retrieval occurs automatically.
2. Run the same task with `--document` and point to an excerpt's source ID and page/section.
3. Show that the specialist must cite the exact chunk ID before the excerpt is registered for review.
4. Show the separate LLM verdict and deterministic-check records in a Persistent Mode test workspace, or explain that Ephemeral Mode deletes them after delivery.
5. Try a URL outside the allowlist and show that the request is rejected.
6. Show a no-source case: MEDHA preserves uncertainty rather than inventing a citation.

Use mocked providers or prepared test fixtures when API quota is limited. The historical `mock-model` baseline log is a test artifact, not a real performance result.

## Architecture questions

### Why use multiple agents?

To divide a broad task into focused responsibilities, express dependencies, and make intermediate claims inspectable. Tradeoffs include extra requests, latency, duplicated claims, and correlated errors when roles share a model.

### Are the agents independent?

They are separate role instances and prompts, not independent intelligence systems. The current workspace uses Groq `openai/gpt-oss-20b`; switching a reviewer to another model/provider could reduce correlated errors but must be controlled and frozen for benchmark comparisons.

### What does the MessageBus do?

It routes asynchronous messages between registered agent inboxes and records the message history. It supports direct messages, broadcasts, challenges, evidence requests, replies, and handoffs under message/depth limits. Message content is context, not proof.

### What is SharedContext?

A thread-safe place for shared findings, notes, statuses, and communication snapshots. It makes relevant work available to agents but does not validate the contents.

### How are dependencies chosen?

The planner proposes them; TaskManager enforces them. The planner can be wrong, so inspect its DAG for important tasks. Current `MAX_CONCURRENT_AGENTS=1` also conflicts with the planner's instruction to create at least two subtasks; it makes execution sequential and may constrain decomposition.

## Retrieval questions

### Does MEDHA search the web automatically?

No. It retrieves only from supplied local files or explicitly supplied HTTPS URLs whose host is in `MEDHA_ALLOWED_SOURCE_DOMAINS`. The default allowlist is empty; it does not crawl the repository or open model-generated URLs.

### How does hybrid search work?

SQLite FTS5's BM25 ranking finds lexical matches. FastEmbed creates local vectors and exact cosine similarity ranks semantic matches. Reciprocal-rank fusion combines those lists. This implementation avoids a remote embedding API and does not require Faiss or Chroma.

### What file types are supported?

PDF, DOCX, TXT, Markdown, reStructuredText, CSV, JSON, and HTML. PDF page numbers and DOCX headings are preserved where possible. Unsupported formats are rejected.

### How do sources reach the reviewer?

Retrieved excerpts are shown to the specialist with exact chunk IDs. The specialist must cite those IDs in its structured response. Only explicitly cited chunks become registered evidence. The reviewer receives excerpt text and metadata, not just a source title.

### What if retrieval fails or finds nothing?

The task continues without that source where possible. Specialists are told no relevant registered excerpt is available; failures are recorded as safe diagnostics. Strict mode does not turn an unsupported claim into a supported claim.

### Does FastEmbed use a paid API?

No. It runs locally. Its generic model weights may be downloaded once and cached outside the execution workspace. That cache is not task data and is not deleted by workspace cleanup.

## Verification questions

### Is the reviewer independent?

It is an adversarially prompted LLM role, not an external auditor. It can misunderstand or accept a false claim. Deterministic rules catch selected structural/provenance issues but cannot prove general truth.

### What does SUPPORTED mean?

The LLM reviewer returned ACCEPT and deterministic guardrails did not downgrade the claim. In general mode, this can mean accepted as common knowledge without external evidence. It is not a guarantee of truth.

### Is support rate accuracy?

No. It is supported reviewed claims divided by all reviewed claims. Unreviewed claims are excluded. Accuracy needs an independent answer key or blinded human/expert ratings.

### What are deterministic checks?

Rule-based results such as source/provenance and selected wording checks. They are stored separately from the LLM's raw decision. They do not replace evidence review and do not test every claim type.

### What do review statuses mean?

- **SUPPORTED:** accepted under the configured verification mode.
- **REJECTED:** judged false, contradictory, or unacceptable.
- **NEEDS_REVISION:** potentially salvageable but requires narrowing or support.
- **UNVERIFIABLE:** cannot be resolved with available evidence and allowed reasoning.
- **UNREVIEWED:** no verdict was completed, often due to interruption, budget, or ledger capacity.

## Privacy questions

### What does Ephemeral Mode clean?

It removes the per-execution workspace and its retrieval files/indexes, skips execution JSON and MCP job JSON, and masks execution-specific logger messages. The cleanup path reports whether removal was verified. The user's original local file is never deleted.

### Is Ephemeral Mode zero retention?

No. It is an application-level non-retention policy. MEDHA cannot delete prompts or excerpts already sent to Groq/Gemini, terminal captures, OS swap/crash dumps, backups, snapshots, or provider-side data. Historical logs from before Ephemeral Mode are not automatically erased. Model weights can persist in the shared FastEmbed cache. Persistent Mode indexes are not encrypted at rest by MEDHA.

### How does ephemeral MCP background delivery work?

The pipeline-complete job is marked `AWAITING_DELIVERY`, and the final result stays in memory so the MCP caller can fetch it. `get_execution_result` returns and clears it, then the job transitions to its final status. Unclaimed output is cleared after the configured TTL (default 300 seconds). The registry does not write job files in Ephemeral Mode.

### What does Persistent Mode retain?

Execution JSON, MCP records, retrieval databases, and workspace content can remain under logs/data until configured retention cleanup. It is selected with `MEDHA_STORAGE_MODE=persistent` or `--persistent`; for MCP, select the environment before starting the server.

### What happens if deletion fails?

MEDHA marks cleanup as unverified, returns a sanitized diagnostic, and may downgrade status to PARTIAL_FAILURE. It must not claim a workspace was deleted if it remains.

## Benchmark questions

### What are the four arms?

- **A:** Direct single-agent baseline.
- **B:** Independent specialists plus verification and synthesis.
- **C:** B plus dependency scheduling, upstream results, and message routing.
- **D:** C plus deliberation.

B-to-C is a bundled contrast, not an isolated MessageBus test. The arm definitions must not change silently.

### Is the benchmark ready to show MEDHA is better?

No. The legacy five-case dataset is not approved for a pilot. Exact prompts, acceptance keys, source corpora, human scoring, limits, and authorization remain open. The pilot is exploratory, not a significance test.

### Can retrieval be part of the current benchmark?

Only as a separately versioned factor. Retrieval and source packets must be identical across eligible arms or retrieval's effect becomes confounded with the arm. The original treatment definitions stay frozen.

## Quick answers for difficult questions

**“Why use multiple agents if they share one model?”** Roles organize the work, but the current design does not claim independent model opinions. Independent models are a possible future study factor.

**“Can I trust an accepted claim?”** Treat it as a review signal. Check the source excerpt or use an independent test/human reviewer for important claims.

**“What is the strongest current contribution?”** Inspectable orchestration: plans, dependencies, message flow, claim provenance, separate verdicts, status, and cleanup policy.

**“What is the biggest risk?”** False confidence from model-based review when evidence retrieval is absent or insufficient.

**“What is next?”** Validate retrieval against a known-source set; add deterministic checks suited to specific claim classes; freeze pilot tasks and rubric; set cost/time ceilings; run an authorized feasibility pilot; do not make superiority claims without independent evaluation.

## Useful commands

```powershell
# Ephemeral task, no retrieval
python main.py --task "Explain database indexes."

# Local document retrieval
python main.py --task "Summarize this policy." --document .\documents\policy.pdf

# Approved HTTPS retrieval
python main.py --task "Explain this documentation." --source-url https://docs.example.org/guide.html

# Full synthesis and audit
python main.py --task "Summarize this policy." --document .\documents\policy.pdf --full

# Explicitly retain one task's records
python main.py --task "Review this design." --persistent

# Benchmark only after authorization, in Strict mode
python main.py --benchmark --persistent --benchmark-cases 2 --experiment-id pilot-01
```

## Closing statement

MEDHA organizes LLM work, supports source retrieval when configured, and makes claims and uncertainty easier to inspect. It does not make independent models from role prompts, automatically search the web, guarantee accepted claims, or yet provide a validated benchmark superiority result.
# MEDHA Project Explainer & Q&A Playbook

**Purpose:** Help the project owner explain MEDHA in a demo, review, or Q&A. Use the short answers first; use the longer notes when someone asks for detail.

## 30-second explanation

MEDHA is a Python application that coordinates LLM-powered roles around one task. It plans specialist work, schedules dependencies, shares context, collects claims, asks a reviewer to challenge those claims, optionally retrieves relevant excerpts from supplied sources, and writes an answer with uncertainty visible. Ephemeral Mode is the default and cleans MEDHA's execution workspace. The reviewer is still an LLM, so MEDHA improves process transparency but does not guarantee truth.

## Two-minute explanation

A user submits a task through the CLI or MCP. A planner asks the configured model to break it into subtasks with roles and dependencies. An execution engine schedules ready work; a factory creates role-specific agents. Those roles usually share the same underlying provider/model, so they are not independent models.

When documents or approved HTTPS URLs are supplied, MEDHA parses them, chunks the text, preserves source locations, and indexes it in an execution-scoped SQLite store. Search combines FTS5/BM25 keyword ranking with local FastEmbed vector similarity. Specialists see only relevant excerpts, not the whole repository. An agent-generated URL is not trusted until retrieved and registered.

Specialists return deliverables and candidate claims. MEDHA stores claims and provenance in an Evidence Ledger, asks an adversarial reviewer for decisions, applies deterministic checks separately, and repairs only claims marked as needing revision. A synthesizer returns the user-facing answer and audit. In Ephemeral Mode, execution-specific data is not written to reports or job files; the workspace is cleaned. Persistent Mode is an explicit opt-in for research and benchmark records.

The main limitation is important: an LLM reviewer is not proof, and a support rate is not factual accuracy. Source-grounded claims require traceable retrieved/supplied evidence; factual performance claims require independent human/expert evaluation.

## Demo script

1. Start with a simple task and show the short `Direct Answer` output.
2. Run the same task with `--document` and point to a retrieved excerpt's source ID and page/section.
3. Show the execution summary and the separate reviewer/deterministic check records in Persistent Mode, or explain that Ephemeral Mode deletes them after delivery.
4. Demonstrate that a URL outside the configured domain allowlist is rejected.
5. Show that no answer source is invented when retrieval finds nothing.

Use a mock or prepared test fixture for demos when API quota is limited. Do not use the historical `mock-model` baseline log as a real performance result.

## Architecture answers

### Why multiple agents?

The plan divides a broad task into narrower responsibilities. It can make coverage and dependencies more explicit. The tradeoffs are more API calls, more latency, and correlated errors when all roles use the same model.

### Are the agents independent?

They are separate role instances, not separate independent intelligence systems. In the current configuration, they generally share Groq `openai/gpt-oss-20b`. A different reviewer model/provider could reduce correlated errors, but it would add cost/configuration variables and must be fixed across benchmark arms.

### What is the MessageBus?

An asynchronous broker with per-agent inboxes, direct and broadcast messages, IDs, response depth checks, and message budgets. It helps route findings, questions, challenges, and handoffs. Messages are not evidence by themselves.

### What is SharedContext?

A separate thread-safe store for shared findings, notes, agent statuses, and communication records. It makes relevant context available to agents. SharedContext does not independently validate that content.

### What is a dependency?

A planner-declared prerequisite between subtasks. The task manager waits for the dependency to complete before scheduling the dependent task. A bad plan can create unnecessary blocking; review the DAG on important tasks.

## Retrieval answers

### Does MEDHA search the web automatically?

No. Retrieval runs only for sources explicitly supplied as local documents or HTTPS URLs. URLs must use an approved domain allowlist. The repository is not silently crawled.

### How does hybrid search work?

SQLite FTS5 ranks keyword matches using BM25. FastEmbed generates local vectors for semantic similarity. Reciprocal-rank fusion combines the two rankings. Search is bounded by chunk and result limits.

### Which files can I supply?

PDF, DOCX, TXT, Markdown, reStructuredText, CSV, JSON, and HTML. PDF page numbers and DOCX headings are retained where available. Unsupported types fail clearly.

### What is a source ID?

A stable identifier for a document or chunk. Specialists are told to cite exact chunk IDs. MEDHA registers only the retrieved chunks the specialist explicitly cites. The reviewer sees the excerpt and its source metadata.

### What if retrieval fails or finds nothing?

The specialist is told no relevant registered excerpt was available. Retrieval failures produce a safe diagnostic, and the task continues without that source where possible. Strict review should not accept unsupported claims because a URL was mentioned.

### Does FastEmbed call a paid API?

No. It runs locally. The first use may download generic model weights to a shared cache. Those model weights can remain after an execution; they are not execution documents or embeddings.

## Evidence and evaluation answers

### Is the reviewer independent?

It is an adversarially prompted LLM role, not an external auditor. Its output is a review signal. It can still misunderstand, miss facts, or accept a false claim.

### What does SUPPORTED mean?

The reviewer accepted a claim under the active verification mode and deterministic guardrails did not downgrade it. In General mode, a claim can be labeled accepted as general knowledge without external evidence. It does not mean an independent fact-check proved the claim.

### Is support rate accuracy?

No. Support rate is supported reviewed claims divided by all reviewed claims. Unreviewed claims are excluded. Accuracy needs an independent answer key or blinded human/expert ratings.

### What is deterministic validation?

Rule-based checks such as schema validation, selected claim-kind rules, unsupported-superlative checks, and wording patterns for empirical claims. These results are stored separately from LLM verdicts. They do not understand every claim or prove all facts.

### What are the review statuses?

- **SUPPORTED:** reviewer accepted under current mode.
- **REJECTED:** reviewer judged false/contradictory or unacceptable.
- **NEEDS_REVISION:** reviewer believes it may be salvageable but needs narrowing/support.
- **UNVERIFIABLE:** available sources and allowed reasoning cannot resolve it.
- **UNREVIEWED:** no verdict was completed, often due to interruption or capacity limits.

## Privacy and storage answers

### What does Ephemeral Mode delete?

MEDHA creates a unique workspace, uses an in-memory SQLite retrieval index, suppresses execution JSON persistence, redacts execution-specific application log messages, closes the index, and removes the workspace on success, failure, exception, timeout, or cancellation. Cleanup is verified. Startup cleanup removes abandoned workspaces whose owner process is gone; a watchdog refreshes active leases.

### Is it zero retention?

No. It is application-level non-retention, not a guarantee of physical erasure. MEDHA cannot delete prompts or retrieved excerpts already sent to Groq/Gemini, terminal captures, OS swap/crash dumps, backups, snapshots, or the user's source documents. It also cannot guarantee provider-side retention behavior. FastEmbed model weights may persist in a shared model cache. Persistent Mode indexes are not encrypted at rest by MEDHA.

### How does MCP background work avoid retaining output?

In Ephemeral Mode the job reports `AWAITING_DELIVERY` after computation and holds the result in memory until the client fetches it. `get_execution_result` clears the result and moves the job to its final status. If it is never fetched, the result expires after `MEDHA_MCP_RESULT_TTL` (default 300 seconds). No MCP job JSON is written in Ephemeral Mode.

### Why is there Persistent Mode?

Research and benchmarking sometimes require saved execution records. Persistent Mode must be explicitly selected. It retains records under `logs/` and execution retrieval data under `data/executions/`, with configured retention. Benchmark CLI additionally requires `--persistent` and `VERIFICATION_MODE=strict`.

### What happens if cleanup fails?

The result reports `cleanup_verified=false`, includes a sanitized cleanup category, and becomes `PARTIAL_FAILURE` where appropriate. MEDHA must not claim deletion if the workspace still exists.

## Benchmark answers

### What are the four arms?

- **A:** Direct baseline.
- **B:** Independent specialists plus verification and synthesis.
- **C:** B plus dependency scheduling, upstream results, and message routing.
- **D:** C plus deliberation.

B-to-C is a bundled contrast, not an isolated message-bus test. The arm definitions should not be silently changed.

### Is the benchmark ready to prove MEDHA is better?

No. It is an exploratory feasibility design, not a significance test. The legacy five-case dataset is not approved as pilot data; exact tasks, source packets, human scoring, cost ceilings, and authorization remain open. The evaluator's automated quality metrics are proxies, not factual correctness.

### Can retrieval be added to the existing benchmark?

Yes, but it is a separate experimental factor. Keep retrieval and source packets identical across eligible arms or define a separately versioned protocol. Do not turn retrieval on for only one treatment and attribute its effect to the original arm design.

## Quick reference: active workspace configuration

The non-secret values last read from this workspace included Groq `openai/gpt-oss-20b`, reasoning effort `low`, concurrency `1`, max calls `40`, task token budget `120000`, output cap `4096`, Groq TPM cap `8000`, review batch size `2`, claim cap `15`, repair limit `3` per iteration and `2` iterations, deliberation timeout `90s` and token cap `8192`, and `VERIFICATION_MODE=general`. Settings can change through `.env`; never put API keys in presentations, reports, or source control.

## Useful commands

```powershell
# Ephemeral task, no source retrieval
python main.py --task "Explain database indexes."

# Local source retrieval
python main.py --task "Summarize this policy." --document .\documents\policy.pdf

# Approved online retrieval
python main.py --task "Explain this documentation." --source-url https://docs.example.org/guide.html

# Full synthesis/audit output
python main.py --task "Summarize this policy." --document .\documents\policy.pdf --full

# Explicitly retain one task's records
python main.py --task "Review this design." --persistent

# Benchmark, only after approval and strict mode
python main.py --benchmark --persistent --benchmark-cases 2 --experiment-id pilot-01
```

## Short closing statement

MEDHA's defensible contribution is orchestrated, inspectable LLM work with traceable source excerpts and privacy-conscious local storage. Its limitations are equally important: agents share a model by default, the reviewer is an LLM, retrieval requires supplied/approved sources, support rate is not accuracy, and the benchmark is not yet a valid basis for superiority claims.
