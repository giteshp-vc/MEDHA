# Phase 5.2G — Pre-Pilot Freeze and Experimental Protocol

- **Date:** 2026-11-06
- **Decision:** Protocol prepared; **pilot NOT authorized**
- **Benchmark executed:** No
- **Live provider calls:** None

## Repository Freeze Audit

- Branch: `vaibhavi-medha`
- HEAD: `8949ab200f03dc3ca4c4926fac621c1a68774fae`
- Tracked worktree: modified, not clean. Tracked changes are documented by SHA-256 `F7524CE0EBB643194FF3C5A23A4297880BE82FF3DD4BE54F981A9894B61724D2` (`git diff --binary`); this hash covers the tracked diff only.
- Untracked user data: `benchmark/outputs/smoke/medha-smoke-20261105.json` and `.csv`. They were not changed and are excluded from source hashes.
- Runtime: Python 3.13.16 via `C:\MEDHA\.venv\Scripts\python.exe`.
- Dependencies: pytest 9.2.0; pytest-asyncio 1.4.1.post1; Pydantic 2.14.2; Groq 1.7.0; google-genai 2.34.0; MCP 2.3.0; httpx 0.28.1.
- No dependency lockfile was found; `requirements.txt` SHA-256 is recorded in the manifest.
- Active provider/model/operator label were not read from `.env`; they remain OPEN in the manifest. The code supports Groq/Gemini and stores operator model revision separately from provider-response revision.
- Existing offline test history for this source state: 260 focused tests and 297 full-suite tests passed on 2026-11-06 with provider keys unset. No benchmark was run.

## Treatment and Outcome Protocol

The protocol defines A as direct baseline; B as independent specialists with common verification/repair/synthesis; C as dependency-aware execution plus upstream context and message routing; D as C plus deliberation. B-to-C is a bundled contrast, not MessageBus-only. C-to-D is the deliberation increment. The implementation uses one frozen plan per task block for B/C/D and isolated mutable plan copies.

Proposed pilot: eight task-block templates, one repetition/block, 32 planned arm trials. The deterministic four Williams sequences occur twice. This is a feasibility pilot only, not a power or significance design. The existing five-case dataset is a legacy candidate, not independently validated or approved; exact pilot prompts, task keys, and source corpora remain OPEN. See `TASK_DATASET_SPECIFICATION.md`.

Proposed primary outcome: blinded human factual/constraint-correctness score, 1–5, using task-specific frozen acceptance criteria. Report mean, distribution, denominator, missingness, and inter-rater agreement. Failed/empty attempted outputs receive the rubric's bottom anchor; unstarted trials remain missing. Human scoring is not integrated into MEDHA.

Automated completeness, pitfall, uncertainty, and evidence-language metrics are exploratory proxies, not factual correctness. Verification is separate: report unweighted eligible trial-rate means and pooled claim counts/rates over accepted, rejected, needs-revision, and unverifiable claims; unreviewed claims are excluded. Arm A verification remains N/A.

## Resource and Compatibility Policy

Use `AccountingSnapshot` for tokens and total/failed request counters; provider-call records supply per-call settings and retry/fallback detail. Dropped records make effective configuration incomplete; retries/fallbacks from records are lower bounds. Shared plan cost is incurred once and reported separately from B/C/D end-to-end comparison allocation. Actual pricing, active provider/model, explicit timeout, monetary ceiling, experiment-wide request cap, and maximum experiment duration remain OPEN. Current code has per-execution limits but no global experiment request/wall-clock cap.

Effective configuration comparisons are stage-aware. Reviewer `0.1` is compared with reviewers; specialist/synthesis `0.2` with their corresponding stages. D-only deliberation timeout is treatment-specific. Known common-stage differences reject comparison; missing stages, snapshots, or dropped-call completeness produce indeterminate compatibility. Missing immutable provider revision and provider-default timeout duration are explicit limitations, not fabricated equality.

## Validation and Limitations

The final protocol artifacts are templates and guidance, not a validated task dataset or run authorization. Validate their JSON structure locally; do not run `--benchmark` without explicit authorization. Before pilot, complete every item in `PRE_PILOT_CHECKLIST.md`, obtain a clean/reproducible source freeze, resolve OPEN settings and limits, and independently freeze task prompts, corpora, rubric, and exclusions.

The source worktree is intentionally not committed or pushed. Existing smoke artifacts remain untouched. No live API/network call or four-arm experiment was made.
