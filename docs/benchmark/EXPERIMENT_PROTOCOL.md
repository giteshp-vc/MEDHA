# MEDHA Four-Arm Experimental Protocol

- **Protocol version:** 5.2G-draft-1
- **Status:** DRAFT / NOT AUTHORIZED FOR EXECUTION
- **Project:** MEDHA, SEED Hackathon 2026, Problem Statement 02: Chain of Command

This protocol defines a feasibility pilot only. It does not authorize a live run. Provider/model identity, active settings, pilot dataset, monetary ceiling, and final human-scoring rules must be frozen and approved before execution.

## Source and Configuration Freeze

The source state observed on 2026-11-06 is recorded in `EXPERIMENT_MANIFEST_TEMPLATE.json`. The working tree is not clean: tracked Phase 5.2F changes are present, along with two untracked historical smoke outputs. The active `.env` values were not read or disclosed. The manifest contains the tracked-diff hash and selected source hashes; the historical smoke files are excluded from the source patch hash and must not be changed.

There is no dependency lockfile. `requirements.txt` is hashed. Create a final source freeze only after the remaining uncommitted code changes are reviewed, committed by the owner, and the manifest is regenerated. Do not treat this draft's current diff hash as an immutable release identifier.

## Arms and Contrasts

| Dimension | A: Baseline | B: Independent | C: Dependency-aware | D: Full MEDHA |
|---|---|---|---|---|
| Direct single-agent answer | Yes | No | No | No |
| Frozen specialist plan | No | Shared setup only | Shared plan | Shared plan |
| Dependency scheduling | No | No | Yes | Yes |
| Upstream result context | No | No | Yes | Yes |
| Message/SharedContext routing | No | No | Yes | Yes |
| Deliberation | No | No | No | Yes |
| Common verification, repair, synthesis | No | Yes | Yes | Yes |

A-to-B is a broad comparison of direct answering against the multi-agent pipeline. B-to-C is a **bundled contrast** of dependency scheduling, upstream context, and message routing; it does not isolate MessageBus alone. C-to-D is the incremental contrast for deliberation, subject to the same frozen plan and common post-processing.

B/C/D must use identical serialized plan contents and hash within a task block, with fresh mutable plan instances. Verification and synthesis policies are shared across B/C/D. Reviewer temperature is intentionally 0.1; specialist, repair, and synthesis defaults are 0.2. Compare effective configuration by corresponding stage, not against one global temperature. D's deliberation timeout is treatment-specific and is not required for B/C.

## Proposed Pilot Tasks

The current `benchmark/benchmark_dataset.json` has five technical/general-knowledge cases with lexical reference criteria. It has not been independently validated for this protocol and has no frozen evidence corpus. Do not silently reuse it as the pilot dataset. Use the eight task-block slots in `TASK_DATASET_SPECIFICATION.md`; actual prompts, task-specific acceptance rules, source corpora, and hashes remain OPEN until reviewed and frozen.

Proposed size: **8 task blocks, one repetition per block, 32 planned arm trials**. The block-major schedule assigns each of the four Williams orders twice. The task order must be predeclared and category-stratified across sequence slots. This is a feasibility pilot for workflow, failure, latency, request, telemetry, and scoring variance checks. It is not a power calculation and does not establish significance.

All arms receive the exact same user prompt. No arm-specific rewriting is permitted. Every B/C/D arm in a block reuses that block's single frozen plan. Freeze and hash supplied source material for research-style tasks; avoid current-fact questions without a supplied corpus.

## Outcome Hierarchy

**Primary pilot outcome (proposed):** blinded human factual/constraint-correctness rating, ordinal 1–5, on each attempted answer using the rubric in `HUMAN_EVALUATION_RUBRIC.md`. Report the mean and full distribution descriptively, plus the number rated. Empty or failed attempted outputs receive the rubric's lowest anchor (1: no usable/correct answer); partial outputs are scored as presented. A trial never started has no score and is reported as unstarted/missing. The pilot does not test superiority or significance.

| Outcome | Definition and denominator | Eligibility / missingness | Role |
|---|---|---|---|
| Blinded correctness | Human 1–5 score; mean over scored attempted trials, with count and distribution | Missing reviewer score stays missing; failed/empty attempted answer gets anchor 1; unstarted is missing | Primary feasibility outcome; not confirmatory |
| Task completion | Successful completed attempts / all attempted trials | Include failures, partial failures, timeouts, cancellations; report unstarted separately | Operational secondary |
| Completeness, pitfall avoidance, uncertainty, lexical evidence proxy | Existing automated heuristics | Successful nonempty outputs only; missing denominator stays null | Exploratory only; never factual accuracy |
| Evidence support / unsupported claims | Reviewer/ledger statuses | Report trial means and pooled counts over reviewed claims; exclude unreviewed claims from pooled denominator; A is N/A | Internal verification, not independent factual scoring |
| Latency | End-to-end trial duration; planning reported separately | All attempted trials; report missing count and schedule position | Operational secondary |
| Requests, retries, tokens | Accounting snapshot for total/failed request and token counters; call records for retry/fallback/stage detail | Report usage provenance and lower-bound flags when records are truncated | Resource secondary |
| Dollar cost | Explicit provider/model prices × separable provider-reported input/output usage | Unknown, incomplete, estimated-only, or inseparable mixed usage remains null | OPEN until pricing is verified |

Do not infer factual correctness from lexical matching, reviewer acceptance alone, or agent agreement. Human scores are external to the current evaluator and must be keyed by blinded response ID, not arm ID.

## Blinding and Scoring

An independent coordinator exports answer-only packets with randomized opaque IDs. Remove arm, execution, agent, model, stage, plan, and provider metadata. Give raters the frozen task, acceptance criteria, supplied evidence corpus, and rubric only. Raters score independently before arm mapping is restored. Use two raters per answer; report weighted Cohen's kappa (ordinal dimensions) with confidence intervals and raw agreement. A third blinded adjudicator resolves any dimension disagreement of two or more points or a pass/fail disagreement. If style or content reveals an arm, record the blinding breach and do not claim perfect blinding. This human process is proposed, not implemented in MEDHA.

## Schedule, Retry, and Failure Rules

The four orders are A-B-D-C, B-C-A-D, C-D-B-A, and D-A-C-B. Record planned and actual order. Counterbalancing balances order, not statistical power. One repetition per task block is proposed; eight blocks provide two assignments per sequence. Do not call this statistically powered.

Provider retries are the SDK/client retry policy and are part of the same arm trial; each transport attempt remains in telemetry. Do not rerun a failed arm as a replacement. Preserve the original failed trial and any later explicitly authorized rerun as a separate trial ID. A missing arm makes that block incomplete for paired/comparative summaries; retain all operational records and mark the missing arm reason.

Stop before the next arm/block on provider quota/billing exhaustion, an unexpected known configuration mismatch, nonzero dropped provider-call records, missing accounting snapshot, or a request/token ceiling being reached. Preserve completed trials. Do not silently continue with a partially compatible or truncated experiment. The evaluator does not provide durable mid-run checkpoint/resume; this is a pilot risk and requires an operator to retain logs and stop safely.

## Resource and Cost Policy

The accounting snapshot is authoritative for input/output/total and estimated tokens, plus total/failed requests. Provider-call records provide retry/fallback and per-call configuration/status; they are not added to snapshot totals. Retry/fallback counts are lower bounds if records were dropped. Planning is incurred once per task block. Report actual experiment expenditure as planner setup once plus each arm's execution; report B/C/D end-to-end comparisons with the shared plan cost attached once per comparison, clearly labeled as allocated shared setup, not repeated actual charges.

Current code defaults include 25 provider calls per task execution, 50,000 local task tokens, 12,000 input tokens/call, 2,048 output tokens/call, 180-second planned specialist execution deadline, and 20-second deliberation timeout. Active `.env` overrides were not read; freeze actual values locally before authorization. At defaults, a block can consume up to five per-execution call budgets (one plan preparation plus four arms), or 125 calls; this is a theoretical per-block ceiling, not an experiment-wide enforced cap. The code has no global experiment request cap or wall-clock cap. Active monetary pricing and maximum spend are OPEN; no cost should be inferred.

The optional `PROVIDER_REQUEST_TIMEOUT_SECONDS` can be set explicitly; blank uses the provider SDK default. SDK default timeout duration and immutable model revision may be unavailable. Record them as unavailable limitations, not equalities. Never log credentials.

## Exclusions and Analysis

Comparative quality requires (1) a compatible stage-aware configuration, (2) complete stage/call evidence with no unaccounted telemetry drops, and (3) metric-specific eligibility. Operational aggregates include all attempted trials regardless of compatibility. A missing planned arm is not a failed answer, but it makes the block incomplete; report it as unstarted/missing. A completed timeout is `TIMED_OUT`, not `UNKNOWN` or success. Preserve partial outputs and failure status.

Report unweighted means of eligible trial verification rates and pooled claim-level rates separately. Pooled reviewed denominator is accepted + rejected + needs-revision + unverifiable; unreviewed is excluded. No verification for A is null/N/A, never zero.

## Authorization

This protocol is a draft and **does not authorize execution**. Before any pilot, complete `PRE_PILOT_CHECKLIST.md`, resolve all OPEN fields in the manifest, freeze a reviewed dataset and source corpus, review the dirty worktree, set an approved budget/stop rule, and obtain explicit written pilot authorization. `python main.py --benchmark` is live-provider execution and must not be run before that authorization.
