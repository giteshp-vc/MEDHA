# Task Dataset Specification

**Status: OPEN — no validated pilot dataset is supplied by this document.** The current repository contains five legacy benchmark cases; they are not frozen or approved for this pilot. Their prompts rely on domain facts and lexical criteria, and no evidence corpus or human reference scoring has been validated for them. Do not claim those cases have passed pilot validation.

## Proposed Size and Selection

Use **8 task blocks** as a manageable feasibility pilot. At one repetition per block this yields 32 planned arm trials. The four deterministic Williams sequences occur twice across the eight block slots. This supports schedule-balance checks and operational feasibility, not significance or statistical power. Eight blocks span diverse coordination demands; task order must be frozen and category-stratified across sequence slots before the run.

Do not select tasks based on observed arm performance. All arms receive identical task text, constraints, and (where applicable) source corpus. B/C/D share one generated frozen plan per block. Any task using external/current facts must include a frozen, licensed/approved evidence packet and hash; otherwise exclude it from the pilot.

## Candidate Block Slots

These are category templates, not authored or validated prompts. Each prompt and scoring key remains OPEN for independent review.

| ID | Category | Difficulty | Coordination profile | Objective/evidence requirement | Status |
|---|---|---|---|---|---|
| PILOT-T01 | Analytical reasoning | Medium | Include at least one independent subtask and one explicit dependency | Deterministic premises and a checkable conclusion | OPEN |
| PILOT-T02 | Technical problem solving | Medium | Include a solution path where coordination may add overhead | Frozen constraints and a reference acceptance checklist | OPEN |
| PILOT-T03 | Multi-step planning | Medium | Named artifacts create explicit dependencies | Checklist with ordering, constraints, and completion conditions | OPEN |
| PILOT-T04 | Research-style synthesis | Medium | Independent source summaries plus dependent synthesis | Supplied, frozen source corpus; every factual criterion maps to a source | OPEN |
| PILOT-T05 | Contradiction detection | Medium | Independent extraction followed by comparison | Synthetic statements with a known contradiction key | OPEN |
| PILOT-T06 | Evidence evaluation | Medium | Independent evidence appraisal and dependent recommendation | Frozen corpus with known support, contradiction, and insufficiency labels | OPEN |
| PILOT-T07 | Dependency-heavy problem | High | Multiple explicit upstream artifacts and integration | Task-specific dependency DAG and independently reviewed acceptance key | OPEN |
| PILOT-T08 | Low-coordination control | Low/medium | Mostly independent or single-part work to measure overhead | Objective checklist; avoid requiring current external facts | OPEN |

## Required Record Per Block

```json
{
  "id": "PILOT-T01",
  "category": "OPEN",
  "difficulty": "OPEN",
  "prompt": "OPEN: freeze exact user-facing task text",
  "expected_deliverable": "OPEN",
  "acceptance_criteria": [],
  "known_pitfalls": [],
  "evidence_requirements": "OPEN",
  "external_evidence_required": null,
  "objective_check_available": null,
  "source_corpus_path": null,
  "source_corpus_sha256": null,
  "independent_review_status": "OPEN"
}
```

## Freeze and Validation Rules

Before pilot authorization:

1. Fill every field and freeze exact UTF-8 prompt text; no arm-specific rewriting.
2. Use clear scope, expected output format, constraints, and independently checkable criteria.
3. For research-style tasks, include the evidence packet and per-criterion source mapping; no live browsing during trials.
4. Review the dependency graph against the task itself. Do not invent dependencies merely to favor arm C/D.
5. Have reviewers not involved in MEDHA development assess ambiguity, difficulty, and answerability.
6. Hash the canonical dataset and each evidence corpus; store hashes in the experiment manifest.
7. Reject tasks requiring facts that are neither in the prompt nor in the frozen corpus.
8. Archive the final dataset under an explicitly versioned path without overwriting prior datasets.

Until these steps are complete, dataset version/hash and actual prompts remain OPEN and no pilot is authorized.
