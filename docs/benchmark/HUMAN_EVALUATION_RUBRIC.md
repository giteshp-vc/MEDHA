# Blinded Human Evaluation Rubric

- **Rubric version:** human-rubric-draft-1
- **Status:** Proposed; raters and task-specific answer keys are not yet appointed or validated.

## Primary Outcome

For the feasibility pilot, the proposed primary outcome is the **blinded factual/constraint-correctness score (1–5)** per attempted answer, averaged descriptively with its denominator and score distribution. The task-specific scoring key must be frozen before execution. This is a pilot outcome, not a significance test or claim of factual superiority.

An attempted run with no usable answer (provider failure, empty answer, or no deliverable) receives score 1 under the explicit bottom anchor; it is not treated as missing. Partial outputs are scored as presented. A scheduled trial never started has no answer score; report it as unstarted/missing and retain its operational schedule status. A reviewer score not returned remains missing, never zero.

## Dimensions and Anchors

Use a 1–5 ordinal scale. Raters must cite the task criterion/source evidence that supports each dimension score.

| Score | Factual/constraint correctness | Completeness | Reasoning validity | Evidence support | Uncertainty handling | Task compliance |
|---|---|---|---|---|---|---|
| 1 | No usable answer, or materially false/contradictory claims dominate | Most required elements absent | Central reasoning invalid or absent | Assertions materially unsupported or evidence misrepresented | Uncertainty concealed or assertions overstated | Does not address the requested task/format |
| 2 | Major errors; only a small portion is correct | Several critical criteria missing | Major unsupported inferential steps | Evidence mostly irrelevant, weak, or not traceable | Important uncertainty omitted | Major constraints/format violated |
| 3 | Mixed correctness; some material gaps remain | Core request partly met; notable omissions | Plausible structure with meaningful gaps | Some claims supported; support is incomplete | Some uncertainty acknowledged but inconsistently | Partial compliance with at least one material violation |
| 4 | Substantially correct; no known material contradiction | All critical criteria met; minor omissions only | Mostly valid reasoning; minor gaps do not change conclusion | Important claims trace to supplied evidence; limitations noted | Material uncertainties are appropriately qualified | Meets task and format requirements with minor deviations |
| 5 | Correct within the frozen task facts/constraints; no material error | All criteria met at the requested depth | Coherent, valid, and appropriately bounded | Claims and conclusions are traceable to relevant evidence; no unsupported leap | Calibrated, precise limitations/assumptions | Fully satisfies all frozen task constraints and format |

Scores are not interchangeable with automated lexical proxies. Factual correctness is task- and corpus-specific. A score of 5 means correct under the frozen evidence/acceptance rules, not universal truth.

## Rater Instructions

- Score each response independently before discussion.
- Read the frozen task, acceptance criteria, evidence packet (if any), and this rubric only.
- Do not use model/provider knowledge or web search not supplied in the frozen packet.
- Separate unsupported empirical claims from reasonable inference; agent consensus is not evidence.
- Score partial failures using the output actually retained. An empty/no-answer attempt is score 1 for every answer dimension and counted as an attempted outcome.
- If a criterion cannot be judged from the task/corpus, mark that dimension `NOT_SCORABLE` with a reason; do not guess.
- Keep an evidence note per score: criterion/source IDs, error category, and concise rationale.

## Blinding Procedure

A coordinator assigns random opaque response IDs and randomizes order independently for each rater. Remove arm, trial, execution, agent, stage, provider, model, token, plan, and verification metadata. Provide only prompt, task-specific rubric key, frozen sources, and answer text. Keep the key mapping and rater identities in a separate access-controlled file. Raters do not receive aggregate results before scoring is locked.

Writing style may reveal the system (e.g. structured synthesis or explicit verification section). Record suspected arm guesses and confidence after scoring. If origin is guessed, mark that rating as potentially unblinded and report a sensitivity analysis excluding suspected unblinds; do not claim perfect blinding.

## Independent Scoring and Adjudication

Use at least two independent raters per answer. Report per-dimension score distributions, number of raters, missing ratings, raw agreement, and weighted Cohen's kappa with confidence intervals for ordinal scores. If a third rater is used, specify the adjudication rule before unblinding.

Refer an answer to a third blinded adjudicator when raters differ by two or more scale points on any primary dimension, disagree on a task-specific pass/fail criterion, or identify a possible critical factual error. Preserve original ratings and adjudicated ratings separately. Do not overwrite disagreement records.

The current benchmark application does not implement blinded human scoring; scores must remain in a separate, versioned scoring artifact until a reviewed integration is approved.
