# Pre-Pilot Authorization Checklist

**Current state: NOT AUTHORIZED.** Every required item must be checked and signed by the experiment owner before any live provider-backed pilot. This checklist is a procedural gate; the CLI itself does not enforce written authorization.

## Source and Configuration

- [ ] Record the final Git commit and verify the source worktree is clean or archive and hash the exact uncommitted patch.
- [ ] Review all uncommitted changes; distinguish source, tests, documentation, historical outputs, and generated files.
- [ ] Freeze Python and dependency versions; add/identify an approved lockfile or record the exact `requirements.txt` hash.
- [ ] In a secure local environment, verify active provider/model IDs and stage-specific temperatures/limits without copying secrets into artifacts.
- [ ] Confirm operator model label is not misrepresented as a provider-attested immutable revision.
- [ ] Freeze explicit request timeout or document provider SDK default; document planned-run execution deadline and deliberation limit separately.
- [ ] Freeze retry configuration, per-execution call/token budgets, maximum concurrency, repair limit, and deliberation settings.
- [ ] Obtain an explicit per-experiment provider-request ceiling, wall-clock ceiling, and monetary ceiling or mark monetary cost unavailable. Verify that stop limits can actually be enforced; the current code has per-execution limits but no global experiment cap/deadline.

## Dataset and Scoring

- [ ] Select and independently review exact task blocks; do not assume the existing five-case benchmark dataset is pilot-ready.
- [ ] Freeze task prompts, acceptance criteria, known pitfalls, difficulty, and task order; hash canonical dataset.
- [ ] Freeze and hash source corpora for research/evidence tasks; prohibit live browsing during arms.
- [ ] Ensure tasks include both coordination-benefit cases and low-coordination controls.
- [ ] Freeze the human rubric, raters, blinding/randomization, adjudication, agreement reporting, and missing-score policy.
- [ ] Confirm task-specific human correctness criteria are defensible; do not use lexical scores as factual accuracy.

## Treatment, Schedule, and Telemetry

- [ ] Verify the four-arm matrix from the current stage-aware implementation and treatment snapshots.
- [ ] Verify B/C/D reuse each block's frozen plan and independent plan copies.
- [ ] Verify common-stage configuration compatibility; D-only deliberation settings must remain treatment-specific.
- [ ] Confirm no stage/call metadata is missing and no provider-call records were dropped; otherwise do not authorize a comparative run.
- [ ] Freeze task-block order and repetition count. A complete four-sequence cycle is counterbalancing, not proof of statistical significance.
- [ ] Freeze failure, timeout, retry, interruption, missing-arm, exclusion, and rerun rules; original failures must remain in records.
- [ ] Validate JSON/CSV consistency, null costs, request/failure counters, compatibility reasons, and lower-bound flags on synthetic data.
- [ ] Confirm planner/setup cost is counted once in actual expenditure and separately allocated in end-to-end comparisons.

## Security and Authorization

- [ ] Confirm no API keys, authorization headers, secrets, or raw provider exceptions appear in reports/fixtures.
- [ ] Confirm output directory, unique experiment ID, no-overwrite behavior, and backup/retention procedure.
- [ ] Confirm all focused/offline tests pass with provider credentials unset.
- [ ] Confirm pilot dataset and evidence rights/handling are approved.
- [ ] Record owner approval and date: ____________________
- [ ] Record explicit written pilot authorization reference: ____________________
- [ ] Only after all prior checks are complete, authorize `python main.py --benchmark ...`.

Any unchecked item means **DO NOT RUN**. A mocked test pass is not live-provider validation or pilot authorization.
