"""Four-arm evaluator built on the existing frozen-plan and generic policy APIs."""

from __future__ import annotations

import asyncio
import csv
from collections import Counter
import importlib.metadata
import inspect
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from benchmark.experiment import (
    BenchmarkArmResult,
    BenchmarkConfigurationSnapshot,
    BenchmarkEvaluationResult,
    BenchmarkExperiment,
    BenchmarkSchedule,
    BenchmarkTaskBlock,
    BenchmarkTrial,
    MetricMeasurement,
    balanced_sequence,
    canonical_hash,
)
from benchmark.treatments import BenchmarkArm, get_treatment_preset
from config import settings
from models.frozen_plan import FrozenTaskPlan
from models.provider_call import AccountingContext, AccountingSnapshot, ProviderCallRecord
from utils.helpers import utcnow_iso

_SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MISSING = object()
_REQUIRED_CONFIGURATION = (
    "provider",
    "model",
    "temperature",
    "input_token_limit",
    "output_token_limit",
    "task_token_budget",
    "call_budget",
)
_CSV_FIELDS = (
    "experiment_id", "task_block_id", "task_id", "trial_id", "execution_id", "arm_id",
    "repetition", "planned_order", "actual_order", "plan_hash", "status", "success",
    "quality_eligible", "start_timestamp", "end_timestamp", "duration_seconds", "error_category",
    "answer_completeness", "task_completion", "pitfall_avoidance", "uncertainty_handling",
    "evidence_quality", "factual_correctness", "verification_support_rate",
    "verification_unsupported_rate", "verification_reviewed_claim_count",
    "verification_accepted_claims_count", "verification_rejected_claims_count",
    "verification_needs_revision_claims_count", "verification_unverifiable_claims_count",
    "verification_unreviewed_claims_count", "answer_completeness_eligible_count",
    "answer_completeness_missing_count", "task_completion_eligible_count",
    "task_completion_missing_count", "pitfall_avoidance_eligible_count",
    "pitfall_avoidance_missing_count", "uncertainty_handling_eligible_count",
    "uncertainty_handling_missing_count", "evidence_quality_eligible_count",
    "evidence_quality_missing_count", "factual_correctness_eligible_count",
    "factual_correctness_missing_count", "verification_eligible_count",
    "verification_missing_count", "metric_attempt_count", "metric_missing_count",
    "input_tokens", "output_tokens", "total_tokens", "provider_reported_total_tokens",
    "estimated_total_tokens", "usage_source",
    "provider_request_count", "failed_provider_request_count", "retry_count",
        "provider_failure_count", "fallback_request_count", "provider_timeout_count",
        "provider_timeout_count_is_lower_bound", "timed_out_count",
        "retry_counts_are_lower_bounds", "fallback_counts_are_lower_bounds",
        "provider_failure_count_is_lower_bound", "provider_request_count_source",
        "failed_provider_request_count_source", "provider_request_count_is_lower_bound",
        "failed_provider_request_count_is_lower_bound",
        "usage_complete", "usage_completeness", "compatibility_status", "compatibility_limitations",
    "compatibility_exclusion_reason", "metric_eligibility", "metric_missingness_reasons",
    "telemetry_complete", "dropped_provider_call_records", "plan_preparation_cost_usd",
    "trial_execution_cost_usd", "end_to_end_cost_usd", "cost_provenance",
    "configuration_compatible", "answer",
)


def _safe_error_category(error: BaseException | str | None) -> Optional[str]:
    if error is None:
        return None
    if isinstance(error, BaseException):
        return type(error).__name__
    return "runner_reported_error"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _accounting_snapshot(value: Any) -> Optional[AccountingSnapshot]:
    accounting = _read_attr(value, "accounting")
    if accounting is None:
        return None
    if isinstance(accounting, AccountingSnapshot):
        return accounting
    try:
        return AccountingSnapshot.model_validate(accounting)
    except (TypeError, ValueError):
        return None


def _provider_records(value: Any, accounting: Optional[AccountingSnapshot]) -> list[ProviderCallRecord]:
    records = accounting.provider_calls if accounting is not None else _read_attr(value, "provider_call_records", [])
    result = []
    for record in records or []:
        try:
            result.append(record if isinstance(record, ProviderCallRecord) else ProviderCallRecord.model_validate(record))
        except (TypeError, ValueError):
            continue
    return result


def _cost_for(
    token_usage: dict[str, Any],
    provider_configuration: dict[str, Any],
    pricing: dict[str, dict[str, float]],
) -> dict[str, Any]:
    provider = provider_configuration.get("provider")
    model = provider_configuration.get("model")
    price = pricing.get(f"{provider}:{model}") or pricing.get(str(provider))
    usage_source = str(token_usage.get("usage_source") or "unknown").lower()
    input_tokens = token_usage.get("prompt_tokens")
    output_tokens = token_usage.get("candidate_tokens")
    if price is None:
        return {"amount_usd": None, "status": "unknown", "usage_source": usage_source,
                "reason": "No provider/model pricing was configured."}
    if input_tokens is None or output_tokens is None:
        return {"amount_usd": None, "status": "unknown", "usage_source": usage_source,
                "reason": "Separate input and output token usage is unavailable."}
    if "input_per_million_usd" not in price or "output_per_million_usd" not in price:
        return {"amount_usd": None, "status": "unknown", "usage_source": usage_source,
                "reason": "Pricing must explicitly define input and output rates."}
    if usage_source in {"estimated", "mixed"}:
        return {
            "amount_usd": None,
            "status": usage_source,
            "usage_source": usage_source,
            "estimated_total_tokens": token_usage.get("estimated_tokens_without_usage"),
            "reason": (
                "Estimated token usage lacks an input/output split; a dollar amount cannot be priced."
                if usage_source == "estimated"
                else "Mixed usage includes estimated tokens without an input/output split; a complete dollar amount cannot be priced."
            ),
        }
    if usage_source != "provider":
        return {
            "amount_usd": None,
            "status": "unknown",
            "usage_source": usage_source,
            "reason": "Token usage provenance is unknown.",
        }
    amount = (
        input_tokens * price["input_per_million_usd"]
        + output_tokens * price["output_per_million_usd"]
    ) / 1_000_000
    status = "actual"
    return {
        "amount_usd": round(amount, 10) if status != "unknown" else None,
        "status": status,
        "usage_source": usage_source,
        "reason": "Explicit provider/model pricing applied to accounting-snapshot token usage."
        if status != "unknown" else "Token usage provenance is unknown.",
    }


def _sum_costs(costs: Iterable[dict[str, Any]], *, shared_once: bool = False) -> dict[str, Any]:
    selected = list(costs)
    provenance_counts = dict(Counter(
        str(cost.get("status", "unknown")) for cost in selected
    ))
    known_amounts = [
        float(cost["amount_usd"])
        for cost in selected
        if cost.get("amount_usd") is not None
    ]
    unknown_count = len(selected) - len(known_amounts)
    known_subtotal = round(sum(known_amounts), 10) if known_amounts else None
    unknown = [cost for cost in selected if cost.get("amount_usd") is None]
    if unknown:
        return {
            "amount_usd": None,
            "status": "unknown",
            "known_subtotal_usd": known_subtotal,
            "known_cost_count": len(known_amounts),
            "unknown_cost_count": unknown_count,
            "provenance_counts": provenance_counts,
            "reason": "At least one component cost is unknown.",
            "shared_preparation_counted_once": shared_once,
        }
    statuses = set(provenance_counts)
    status = next(iter(statuses)) if len(statuses) == 1 else "mixed" if statuses else "unknown"
    return {
        "amount_usd": known_subtotal,
        "known_subtotal_usd": known_subtotal,
        "known_cost_count": len(known_amounts),
        "unknown_cost_count": 0,
        "provenance_counts": provenance_counts,
        "status": status,
        "reason": "Sum of explicitly priced accounting snapshots.",
        "shared_preparation_counted_once": shared_once,
    }


def _version_snapshot() -> tuple[Optional[str], Optional[str], Optional[bool], dict[str, Optional[str]]]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = os.getenv("GIT_COMMIT") or None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout
        worktree_dirty: Optional[bool] = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        worktree_dirty = None
    versions = {}
    for package in ("pydantic", "groq", "google-genai", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return os.getenv("MEDHA_VERSION"), commit, worktree_dirty, versions


class FourArmRunnerRegistry:
    """Routes A to the direct baseline and B/C/D to shared planned orchestration."""

    def __init__(self, baseline_runner: Any, medha_runner: Any):
        self.baseline_runner = baseline_runner
        self.medha_runner = medha_runner
        self.orchestrator = getattr(medha_runner, "orchestrator", None)

    async def prepare(self, task: str, context: AccountingContext) -> FrozenTaskPlan:
        if self.orchestrator is None or not callable(getattr(self.orchestrator, "prepare_frozen_plan", None)):
            raise TypeError("Four-arm evaluation requires a MEDHA runner with prepare_frozen_plan().")
        return await self.orchestrator.prepare_frozen_plan(task, accounting_context=context)

    async def execute(
        self,
        arm: BenchmarkArm,
        task: str,
        context: AccountingContext,
        frozen_plan: Optional[FrozenTaskPlan],
    ) -> Any:
        if arm is BenchmarkArm.BASELINE:
            run = self.baseline_runner.run
            try:
                accepts_context = "accounting_context" in inspect.signature(run).parameters
            except (TypeError, ValueError):
                accepts_context = False
            if accepts_context:
                return await run(task, accounting_context=context)
            return await run(task)
        if frozen_plan is None or self.orchestrator is None:
            raise RuntimeError("A validated frozen plan is required for planned benchmark arms.")
        frozen_plan.load_plan(task)
        preset = get_treatment_preset(arm)
        return await self.orchestrator.run_planned_task(
            task,
            frozen_plan,
            preset.policy,
            accounting_context=context,
        )


class FourArmBenchmark:
    def __init__(self, evaluator: Any):
        self.evaluator = evaluator
        self.registry = FourArmRunnerRegistry(evaluator.baseline_runner, evaluator.medha_runner)

    async def run(
        self,
        *,
        max_cases: Optional[int] = None,
        repetitions: int = 1,
        experiment_id: Optional[str] = None,
        output_directory: Optional[Path] = None,
    ) -> BenchmarkEvaluationResult:
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if max_cases is not None and max_cases < 1:
            raise ValueError("max_cases must be positive when supplied.")
        experiment_id = experiment_id or f"exp_{uuid.uuid4().hex[:12]}"
        if not _SAFE_EXPERIMENT_ID.fullmatch(experiment_id) or ".." in experiment_id:
            raise ValueError("experiment_id must contain only letters, digits, dot, underscore, or hyphen.")

        output_root = Path(output_directory or (self.evaluator.dataset_path.parent / "experiments")).resolve()
        json_path = output_root / f"{experiment_id}.json"
        csv_path = output_root / f"{experiment_id}.csv"
        if json_path.exists() or csv_path.exists():
            raise FileExistsError(f"Benchmark output already exists for experiment {experiment_id}.")

        cases = self.evaluator.load_dataset()
        if max_cases is not None:
            cases = cases[:max_cases]
        dataset_hash = canonical_hash(cases)
        raw_configurations = {
            "BASELINE": self._runner_configuration(self.evaluator.baseline_runner),
            "MEDHA": self._runner_configuration(self.evaluator.medha_runner),
        }
        differences = self._configuration_differences(raw_configurations)
        compatibility_assessments, compatibility_limitations = self._configuration_assessments(
            raw_configurations
        )
        application_version, git_commit, git_worktree_dirty, dependencies = _version_snapshot()
        snapshots = {}
        for arm in BenchmarkArm:
            preset = get_treatment_preset(arm)
            raw = raw_configurations["BASELINE" if arm is BenchmarkArm.BASELINE else "MEDHA"]
            snapshots[arm.value] = BenchmarkConfigurationSnapshot(
                application_version=application_version,
                git_commit=git_commit,
                git_worktree_dirty=git_worktree_dirty,
                python_version=sys.version.split()[0],
                dependency_versions=dependencies,
                provider_configuration=raw,
                pricing_configuration=self.evaluator.pricing,
                runtime_configuration={
                    "max_retries": settings.MAX_RETRIES,
                    "initial_retry_delay": settings.INITIAL_RETRY_DELAY,
                    "max_retry_delay": settings.MAX_RETRY_DELAY,
                    "provider_request_timeout_seconds": settings.PROVIDER_REQUEST_TIMEOUT_SECONDS,
                    "execution_timeout_seconds": settings.EXECUTION_TIMEOUT_SECONDS,
                    "max_concurrent_agents": settings.MAX_CONCURRENT_AGENTS,
                    "max_repair_iterations": settings.MAX_REPAIR_ITERATIONS,
                    "max_llm_calls_per_task": settings.MAX_LLM_CALLS_PER_TASK,
                    "max_claims_per_task": settings.MAX_CLAIMS_PER_TASK,
                    "review_batch_size": settings.REVIEW_BATCH_SIZE,
                    "max_repair_claims_per_iteration": settings.MAX_REPAIR_CLAIMS_PER_ITERATION,
                    "max_tokens_per_task": settings.MAX_TOKENS_PER_TASK,
                    "max_input_tokens_per_call": settings.MAX_INPUT_TOKENS_PER_CALL,
                    "max_output_tokens_per_call": settings.MAX_OUTPUT_TOKENS_PER_CALL,
                    "claim_extraction_token_reserve": settings.CLAIM_EXTRACTION_TOKEN_RESERVE,
                    "verification_token_reserve": settings.VERIFICATION_TOKEN_RESERVE,
                    "synthesis_token_reserve": settings.SYNTHESIS_TOKEN_RESERVE,
                    "max_deliberation_rounds": settings.MAX_DELIBERATION_ROUNDS,
                    "max_exchanges_per_claim": settings.MAX_EXCHANGES_PER_CLAIM,
                    "max_deliberation_messages": settings.MAX_DELIBERATION_MESSAGES,
                    "max_deliberation_messages_per_agent": settings.MAX_DELIBERATION_MESSAGES_PER_AGENT,
                    "max_deliberation_tokens": settings.MAX_DELIBERATION_TOKENS,
                    "deliberation_timeout_seconds": settings.DELIBERATION_TIMEOUT_SECONDS,
                },
                treatment_configuration={
                    "uses_plan": preset.uses_plan,
                    "specialists_enabled": preset.specialists_enabled,
                    "policy": preset.policy.model_dump(mode="json"),
                },
                planner_configuration=raw if preset.uses_plan else {},
                verification_configuration={
                    "enabled": preset.policy.verification_enabled,
                    "max_claims_per_task": settings.MAX_CLAIMS_PER_TASK,
                    "review_batch_size": settings.REVIEW_BATCH_SIZE,
                    "max_repair_claims_per_iteration": settings.MAX_REPAIR_CLAIMS_PER_ITERATION,
                    "max_repair_iterations": settings.MAX_REPAIR_ITERATIONS,
                },
                synthesis_configuration={
                    "enabled": preset.policy.synthesis_enabled,
                    "token_reserve": settings.SYNTHESIS_TOKEN_RESERVE,
                },
                scoring_version="medha-heuristic-v1",
                dataset_hash=dataset_hash,
                compatible=not differences,
                compatibility_status=(
                    "INCOMPATIBLE" if differences
                    else "COMPATIBLE_WITH_LIMITATIONS" if compatibility_limitations
                    else "COMPATIBLE"
                ),
                compatibility_assessments=compatibility_assessments,
                compatibility_limitations=compatibility_limitations,
                compatibility_differences=differences,
            )

        experiment = BenchmarkExperiment(
            experiment_id=experiment_id,
            created_at=_timestamp(),
            repetitions=repetitions,
            dataset_hash=dataset_hash,
            configuration_snapshots=snapshots,
        )
        interrupted = False
        for block_index, case in enumerate(cases):
            task_id = str(case.get("id", f"case_{block_index + 1}"))
            task = str(case.get("prompt", case.get("task", "")))
            block_id = f"block_{block_index + 1:04d}"
            block = BenchmarkTaskBlock(
                task_block_id=block_id,
                task_id=task_id,
                task_text=task,
                dataset_case_id=str(case.get("id")) if case.get("id") is not None else None,
                dataset_hash=canonical_hash(case),
            )
            preparation_started = time.perf_counter()
            preparation_error: Optional[BaseException] = None
            try:
                frozen_plan = await self.registry.prepare(
                    task,
                    AccountingContext(experiment_id=experiment_id, task_block_id=block_id),
                )
                frozen_plan.load_plan(task)
                block.frozen_plan = frozen_plan.model_dump(mode="json")
                block.plan_hash = frozen_plan.plan_hash
                block.plan_preparation_accounting = frozen_plan.preparation_accounting
            except asyncio.CancelledError as error:
                preparation_error = error
                block.plan_preparation_accounting = getattr(error, "preparation_accounting", None)
                interrupted = True
                frozen_plan = None
            except Exception as error:
                preparation_error = error
                block.plan_preparation_accounting = getattr(error, "preparation_accounting", None)
                frozen_plan = None
            block.plan_preparation_duration_seconds = round(time.perf_counter() - preparation_started, 6)
            block.plan_preparation_error_category = _safe_error_category(preparation_error)
            plan_cost = self._snapshot_cost(
                block.plan_preparation_accounting,
                raw_configurations["MEDHA"],
            ) if block.plan_preparation_accounting else self._unknown_cost("Planning accounting unavailable.")

            for repetition_index in range(repetitions):
                planned_order = list(balanced_sequence(block_index, repetition_index, repetitions))
                schedule = BenchmarkSchedule(
                    task_block_id=block_id,
                    repetition=repetition_index + 1,
                    sequence_index=(block_index * repetitions + repetition_index) % 4,
                    planned_order=planned_order,
                )
                block.schedules.append(schedule)
                if interrupted:
                    schedule.interrupted = True
                    break
                for arm_name in planned_order:
                    arm = BenchmarkArm(arm_name)
                    if arm is not BenchmarkArm.BASELINE and frozen_plan is None:
                        continue
                    schedule.actual_order.append(arm.value)
                    trial_id = f"{experiment_id}:{block_id}:r{repetition_index + 1}:{arm.value.lower()}"
                    trial = await self._run_trial(
                        case=case,
                        task=task,
                        task_id=task_id,
                        block=block,
                        arm=arm,
                        trial_id=trial_id,
                        repetition=repetition_index + 1,
                        planned_order=planned_order,
                        actual_order=schedule.actual_order,
                        frozen_plan=frozen_plan,
                        plan_cost=plan_cost,
                        raw_configurations=raw_configurations,
                        compatibility_differences=differences,
                        experiment_id=experiment_id,
                    )
                    block.trials.append(trial)
                    if trial.result.status == "INTERRUPTED":
                        schedule.interrupted = True
                        interrupted = True
                        break
                for trial in block.trials:
                    if trial.repetition == repetition_index + 1:
                        trial.actual_order = list(schedule.actual_order)
                        trial.result.actual_order = list(schedule.actual_order)
                if interrupted:
                    break
            experiment.task_blocks.append(block)
            if interrupted:
                break

        schedules = [schedule for block in experiment.task_blocks for schedule in block.schedules]
        sequence_indices = [schedule.sequence_index for schedule in schedules]
        sequence_counts = [sequence_indices.count(index) for index in range(4)]
        experiment.schedule_sequences_used = sorted(set(sequence_indices))
        experiment.schedule_cycle_complete = bool(
            sequence_indices
            and len(sequence_indices) % 4 == 0
            and len(set(sequence_counts)) == 1
            and all(not schedule.interrupted and schedule.actual_order == schedule.planned_order for schedule in schedules)
        )
        self._apply_effective_configuration_compatibility(experiment, differences)
        operational_aggregates, comparative_aggregates, aggregate = self._aggregate(experiment)
        actual_costs = []
        for block in experiment.task_blocks:
            if block.plan_preparation_accounting is not None:
                actual_costs.append(self._snapshot_cost(
                    block.plan_preparation_accounting, raw_configurations["MEDHA"]
                ))
            actual_costs.extend(trial.result.trial_execution_cost for trial in block.trials)
        result = BenchmarkEvaluationResult(
            experiment=experiment,
            aggregate_results=aggregate,
            operational_aggregates=operational_aggregates,
            comparative_aggregates=comparative_aggregates,
            actual_experiment_cost=_sum_costs(actual_costs, shared_once=True),
            export_paths={"json": str(json_path), "csv": str(csv_path)},
        )
        output_root.mkdir(parents=True, exist_ok=True)
        self._write_json(json_path, result)
        self._write_csv(csv_path, experiment)
        return result

    @staticmethod
    def _apply_effective_configuration_compatibility(
        experiment: BenchmarkExperiment,
        initial_differences: dict[str, Any],
    ) -> None:
        profile_fields = (
            "provider", "model", "temperature", "input_token_limit", "output_token_limit",
            "timeout_source", "timeout_seconds",
        )
        known_stages = {
            "baseline", "planning", "specialist_execution", "claim_extraction",
            "verification", "repair", "synthesis", "deliberation",
        }
        planned_arms = (
            BenchmarkArm.INDEPENDENT_PARALLEL.value,
            BenchmarkArm.DEPENDENCY_AWARE.value,
            BenchmarkArm.FULL_MEDHA.value,
        )
        differences = dict(initial_differences)
        assessments = dict(
            experiment.configuration_snapshots["BASELINE"].compatibility_assessments
        )
        limitations = list(
            experiment.configuration_snapshots["BASELINE"].compatibility_limitations
        )
        profiles: dict[str, list[dict[str, Any]]] = {arm.value: [] for arm in BenchmarkArm}
        profiles_by_stage: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        revisions: set[str] = set()
        response_models: set[str] = set()
        provider_call_observation_count = 0
        revision_observation_count = 0
        response_model_observation_count = 0
        blocking_unavailable = False
        dropped_by_arm = {arm.value: 0 for arm in BenchmarkArm}

        def make_profile(call: ProviderCallRecord) -> dict[str, Any]:
            profile = {
                "provider": call.provider,
                "model": call.model,
                "temperature": call.temperature,
                "input_token_limit": call.input_token_limit,
                "output_token_limit": call.output_token_limit,
                "timeout_source": call.timeout_source or "unavailable",
            }
            if profile["timeout_source"] == "explicit_request":
                profile["timeout_seconds"] = call.timeout_seconds
            return profile

        def record_call(arm_name: str, call: ProviderCallRecord, *, planner: bool = False) -> None:
            nonlocal blocking_unavailable
            nonlocal provider_call_observation_count
            nonlocal revision_observation_count
            nonlocal response_model_observation_count
            provider_call_observation_count += 1
            stage = str(call.stage or "").strip().casefold()
            assessment_prefix = "planner" if planner else arm_name
            if not stage or stage not in known_stages:
                assessments[f"{assessment_prefix}.provider_call_stage"] = {
                    "status": "unavailable",
                    "stage": call.stage,
                }
                limitations.append(f"{assessment_prefix} provider-call stage metadata is missing or unknown.")
                blocking_unavailable = True
                return

            profile = make_profile(call)
            profile["stage"] = stage
            if not planner:
                profiles_by_stage.setdefault(stage, {}).setdefault(arm_name, {})[
                    canonical_hash(profile)
                ] = profile
                profiles[arm_name].append(profile)

            declared = experiment.configuration_snapshots[
                BenchmarkArm.BASELINE.value if arm_name == BenchmarkArm.BASELINE.value else
                BenchmarkArm.FULL_MEDHA.value
            ].provider_configuration
            required_fields = ("provider", "model", "input_token_limit")
            missing_fields = []
            for name in required_fields:
                configured = declared.get(name)
                effective = getattr(call, name)
                if configured in (None, "", "unknown") or effective in (None, "", "unknown"):
                    missing_fields.append(name)
                elif configured != effective:
                    differences[f"{assessment_prefix}.{stage}.{name}"] = {
                        "configured": configured,
                        "effective": effective,
                    }
            configured_output = declared.get("output_token_limit")
            if call.output_token_limit is None or configured_output in (None, "", "unknown"):
                missing_fields.append("output_token_limit")
            elif call.output_token_limit > configured_output:
                differences[f"{assessment_prefix}.{stage}.output_token_limit"] = {
                    "configured_maximum": configured_output,
                    "effective": call.output_token_limit,
                }
            if call.temperature is None:
                missing_fields.append("temperature")
            if profile["timeout_source"] == "unavailable":
                missing_fields.append("timeout_source")
            elif profile["timeout_source"] == "explicit_request" and call.timeout_seconds is None:
                missing_fields.append("timeout_seconds")
            elif profile["timeout_source"] == "provider_sdk_default":
                limitations.append(f"{assessment_prefix}.{stage} uses an SDK-default timeout duration.")

            if missing_fields:
                assessments[f"{assessment_prefix}.{stage}.effective_configuration"] = {
                    "status": "unavailable",
                    "fields": sorted(set(missing_fields)),
                }
                limitations.append(f"{assessment_prefix}.{stage} effective configuration is incomplete.")
                blocking_unavailable = True
            elif not any(key.startswith(f"{assessment_prefix}.{stage}.") for key in differences):
                assessments[f"{assessment_prefix}.{stage}.effective_configuration"] = {
                    "status": "known",
                }

            if call.model_version_source == "provider_response" and call.model_version:
                revisions.add(call.model_version)
                revision_observation_count += 1
            else:
                limitations.append(f"{assessment_prefix}.{stage} provider revision is unavailable.")
            if call.response_model:
                response_models.add(call.response_model)
                response_model_observation_count += 1

        for arm in BenchmarkArm:
            arm_name = arm.value
            arm_trials = [
                trial.result
                for block in experiment.task_blocks
                for trial in block.trials
                if trial.arm_id == arm_name
            ]
            if not arm_trials:
                assessments[f"{arm_name}.effective_provider_calls"] = {"status": "unavailable"}
                limitations.append(f"{arm_name} has no provider-call records to verify effective configuration.")
                blocking_unavailable = True
            for result in arm_trials:
                if result.accounting is None:
                    assessments[f"{arm_name}.effective_provider_calls"] = {
                        "status": "unavailable",
                        "reason": "accounting snapshot/drop counter unavailable",
                    }
                    limitations.append(
                        f"{arm_name} provider-call completeness cannot be verified without an accounting snapshot."
                    )
                    blocking_unavailable = True
                accounting_dropped = (
                    result.accounting.dropped_provider_call_records
                    if result.accounting is not None
                    else 0
                )
                dropped = max(result.dropped_provider_call_records, accounting_dropped)
                dropped_by_arm[arm_name] += dropped
                for call in result.provider_call_records:
                    record_call(arm_name, call)
            if arm_trials and not any(result.provider_call_records for result in arm_trials):
                assessments[f"{arm_name}.effective_provider_calls"] = {"status": "unavailable"}
                limitations.append(f"{arm_name} has no retained provider-call records for effective configuration.")
                blocking_unavailable = True
            if dropped_by_arm[arm_name]:
                assessments[f"{arm_name}.effective_provider_calls"] = {
                    "status": "unavailable",
                    "reason": "provider-call records were dropped",
                    "dropped_record_count": dropped_by_arm[arm_name],
                }
                limitations.append(
                    f"{arm_name} effective configuration is incomplete because provider-call records were dropped."
                )
                blocking_unavailable = True

        planner_snapshots = [
            block.plan_preparation_accounting
            for block in experiment.task_blocks
            if block.plan_preparation_accounting is not None
        ]
        planner_calls = []
        planner_dropped = 0
        for snapshot in planner_snapshots:
            planner_dropped += snapshot.dropped_provider_call_records
            planner_calls.extend(snapshot.provider_calls)
        if not planner_calls:
            if planner_snapshots:
                assessments["planner.effective_provider_calls"] = {"status": "not_applicable"}
                limitations.append("Planning accounting snapshots contain no provider-call records.")
            else:
                assessments["planner.effective_provider_calls"] = {"status": "unavailable"}
                limitations.append("Planner accounting snapshots are unavailable.")
                blocking_unavailable = True
        else:
            for call in planner_calls:
                record_call(BenchmarkArm.FULL_MEDHA.value, call, planner=True)
            assessments["planner.effective_provider_calls"] = {
                "status": "unavailable" if planner_dropped else "known",
                "dropped_record_count": planner_dropped,
            }
        if planner_dropped:
            limitations.append("Planner provider-call records were dropped; planning telemetry is incomplete.")
            blocking_unavailable = True

        stage_differences = {}
        common_stage_assessments = {}
        observed_stages = set(profiles_by_stage)
        for stage in sorted(observed_stages):
            stage_arms = profiles_by_stage.get(stage, {})
            participating = [arm for arm in planned_arms if arm in stage_arms]
            if stage == "deliberation" and participating == [BenchmarkArm.FULL_MEDHA.value]:
                common_stage_assessments[stage] = {
                    "status": "treatment_specific",
                    BenchmarkArm.INDEPENDENT_PARALLEL.value: "not_applicable",
                    BenchmarkArm.DEPENDENCY_AWARE.value: "not_applicable",
                    BenchmarkArm.FULL_MEDHA.value: sorted(stage_arms[BenchmarkArm.FULL_MEDHA.value].values(), key=canonical_hash),
                }
                assessments[f"stage.{stage}"] = common_stage_assessments[stage]
                continue
            if not participating:
                common_stage_assessments[stage] = {
                    "status": "not_applicable",
                    "observed_arms": participating,
                }
                assessments[f"stage.{stage}"] = common_stage_assessments[stage]
                continue
            stage_profiles = {
                arm: sorted(stage_arms[arm].values(), key=canonical_hash)
                for arm in participating
            }
            if len(participating) != len(planned_arms):
                common_stage_assessments[stage] = {
                    "status": "unavailable",
                    "observed_arms": participating,
                    "profiles": stage_profiles,
                }
                limitations.append(f"Stage {stage} was not observed in every planned arm.")
                blocking_unavailable = True
            elif len({canonical_hash(profiles_for_arm) for profiles_for_arm in stage_profiles.values()}) > 1:
                common_stage_assessments[stage] = {
                    "status": "known_different",
                    "profiles": stage_profiles,
                }
                stage_differences[stage] = stage_profiles
            else:
                common_stage_assessments[stage] = {
                    "status": "known_equal",
                    "profiles": stage_profiles,
                }
            assessments[f"stage.{stage}"] = common_stage_assessments[stage]

        if stage_differences:
            differences["shared_stage_provider_call_profiles"] = stage_differences
        if len(revisions) > 1:
            differences["provider_model_revision"] = sorted(revisions)
            assessments["provider_model_revision"] = {
                "status": "known_different",
                "values": sorted(revisions),
            }
        elif revisions and revision_observation_count == provider_call_observation_count and not any(dropped_by_arm.values()) and not planner_dropped:
            assessments["provider_model_revision"] = {
                "status": "known_equal",
                "value": next(iter(revisions)),
            }
        else:
            assessments["provider_model_revision"] = {
                "status": "unavailable",
                "value": next(iter(revisions), None),
                "non_blocking": True,
            }
        if len(response_models) > 1:
            differences["provider_response_model"] = sorted(response_models)
            assessments["provider_response_model"] = {
                "status": "known_different",
                "values": sorted(response_models),
            }
        elif response_models and response_model_observation_count == provider_call_observation_count and not any(dropped_by_arm.values()) and not planner_dropped:
            assessments["provider_response_model"] = {
                "status": "known_equal",
                "value": next(iter(response_models)),
            }
        else:
            assessments["provider_response_model"] = {
                "status": "unavailable",
                "value": next(iter(response_models), None),
                "non_blocking": True,
            }

        limitations = list(dict.fromkeys(limitations))
        compatible = not differences and not blocking_unavailable
        compatibility_status = (
            "INCOMPATIBLE" if differences
            else "INDETERMINATE" if blocking_unavailable
            else "COMPATIBLE_WITH_LIMITATIONS" if limitations
            else "COMPATIBLE"
        )
        for arm in BenchmarkArm:
            current = experiment.configuration_snapshots[arm.value]
            experiment.configuration_snapshots[arm.value] = current.model_copy(update={
                "effective_provider_calls": profiles[arm.value],
                "compatibility_status": compatibility_status,
                "compatibility_assessments": assessments,
                "compatibility_limitations": limitations,
                "planner_configuration": {
                    **current.planner_configuration,
                    "effective_provider_calls": [
                        call.model_dump(mode="json") for call in planner_calls
                    ] if arm is not BenchmarkArm.BASELINE else [],
                },
                "compatible": compatible,
                "compatibility_differences": differences,
            })
        for block in experiment.task_blocks:
            for trial in block.trials:
                current = trial.result.configuration
                if current is not None:
                    trial.result.configuration = current.model_copy(update={
                        "effective_provider_calls": profiles[trial.arm_id],
                        "compatible": compatible,
                        "compatibility_status": compatibility_status,
                        "compatibility_assessments": assessments,
                        "compatibility_limitations": limitations,
                        "compatibility_differences": differences,
                    })

    @staticmethod
    def _runner_configuration(runner: Any) -> dict[str, Any]:
        method = getattr(runner, "benchmark_configuration", None)
        candidate = method() if callable(method) else {}
        return dict(candidate) if isinstance(candidate, dict) else {}

    @staticmethod
    def _configuration_differences(configurations: dict[str, dict[str, Any]]) -> dict[str, Any]:
        baseline = configurations.get("BASELINE", {})
        medha = configurations.get("MEDHA", {})
        differences = {}
        for name in _REQUIRED_CONFIGURATION:
            left = baseline.get(name)
            right = medha.get(name)
            if left in (None, "", "unknown") or right in (None, "", "unknown"):
                differences[name] = {"baseline": left, "medha": right, "reason": "required value unknown"}
            elif left != right:
                differences[name] = {"baseline": left, "medha": right}
        baseline_revision = baseline.get("model_version")
        medha_revision = medha.get("model_version")
        if baseline_revision not in (None, "", "unknown") and medha_revision not in (None, "", "unknown"):
            if baseline_revision != medha_revision:
                differences["operator_configured_model_version"] = {
                    "baseline": baseline_revision,
                    "medha": medha_revision,
                }
        baseline_timeout = baseline.get("provider_request_timeout_seconds", _MISSING)
        medha_timeout = medha.get("provider_request_timeout_seconds", _MISSING)
        if baseline_timeout is not _MISSING and medha_timeout is not _MISSING:
            if baseline_timeout != medha_timeout:
                differences["provider_request_timeout_seconds"] = {
                    "baseline": baseline_timeout,
                    "medha": medha_timeout,
                }
        return differences

    @staticmethod
    def _configuration_assessments(
        configurations: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], list[str]]:
        baseline = configurations.get("BASELINE", {})
        medha = configurations.get("MEDHA", {})
        assessments = {}
        limitations = []
        assessments["plan_preparation_for_baseline"] = {"status": "not_applicable"}
        for name in _REQUIRED_CONFIGURATION:
            left = baseline.get(name)
            right = medha.get(name)
            if left in (None, "", "unknown") or right in (None, "", "unknown"):
                assessments[name] = {"status": "unavailable", "baseline": left, "medha": right}
            else:
                assessments[name] = {
                    "status": "known_equal" if left == right else "known_different",
                    "baseline": left,
                    "medha": right,
                }

        baseline_revision = baseline.get("model_version")
        medha_revision = medha.get("model_version")
        if baseline_revision in (None, "", "unknown") or medha_revision in (None, "", "unknown"):
            assessments["provider_model_revision"] = {
                "status": "unavailable",
                "baseline_operator_label": baseline_revision,
                "medha_operator_label": medha_revision,
                "provider_attested": False,
            }
            limitations.append(
                "Provider immutable model revision is unavailable; configured model identifiers may resolve through mutable aliases."
            )
        else:
            assessments["operator_configured_model_version"] = {
                "status": "known_equal" if baseline_revision == medha_revision else "known_different",
                "provider_attested": False,
            }

        baseline_timeout = baseline.get("provider_request_timeout_seconds", _MISSING)
        medha_timeout = medha.get("provider_request_timeout_seconds", _MISSING)
        if baseline_timeout is _MISSING or medha_timeout is _MISSING:
            assessments["request_timeout_mode"] = {"status": "unavailable"}
            assessments["request_timeout_seconds"] = {"status": "unavailable"}
            limitations.append("Explicit request-timeout configuration metadata is unavailable.")
        elif baseline_timeout is None and medha_timeout is None:
            assessments["request_timeout_mode"] = {
                "status": "known_equal",
                "value": "provider_sdk_default",
            }
            assessments["request_timeout_seconds"] = {
                "status": "unavailable",
                "reason": "Provider SDK default duration is not exposed by the client configuration.",
            }
            limitations.append("Request timeout duration is provider-managed and unavailable.")
        else:
            assessments["request_timeout_seconds"] = {
                "status": "known_equal" if baseline_timeout == medha_timeout else "known_different",
                "baseline": baseline_timeout,
                "medha": medha_timeout,
            }
        return assessments, limitations

    @staticmethod
    def _unknown_cost(reason: str) -> dict[str, Any]:
        return {"amount_usd": None, "status": "unknown", "usage_source": "unknown", "reason": reason}

    def _snapshot_cost(self, snapshot: AccountingSnapshot, configuration: dict[str, Any]) -> dict[str, Any]:
        return _cost_for(snapshot.token_usage, configuration, self.evaluator.pricing)

    @staticmethod
    def _provider_request_counts(
        accounting: Optional[AccountingSnapshot],
        provider_calls: list[ProviderCallRecord],
        dropped_records: int = 0,
    ) -> dict[str, Any]:
        usage = accounting.token_usage if accounting is not None else {}
        dropped = max(
            dropped_records,
            accounting.dropped_provider_call_records if accounting is not None else 0,
        )

        def count(name: str, record_value: int) -> tuple[Optional[int], str, bool]:
            snapshot_value = usage.get(name)
            if snapshot_value is not None:
                return int(snapshot_value), "accounting_snapshot", False
            if provider_calls:
                return int(record_value), "provider_call_records", bool(dropped or accounting is None)
            return None, "unavailable", False

        requests, request_source, requests_lower_bound = count(
            "total_requests", len(provider_calls)
        )
        failures, failure_source, failures_lower_bound = count(
            "failed_requests",
            sum(str(call.status).upper() != "SUCCEEDED" for call in provider_calls),
        )
        return {
            "total_requests": requests,
            "failed_requests": failures,
            "request_count_source": request_source,
            "failure_count_source": failure_source,
            "request_count_is_lower_bound": requests_lower_bound,
            "failure_count_is_lower_bound": failures_lower_bound,
        }

    async def _run_trial(
        self,
        *,
        case: dict[str, Any],
        task: str,
        task_id: str,
        block: BenchmarkTaskBlock,
        arm: BenchmarkArm,
        trial_id: str,
        repetition: int,
        planned_order: list[str],
        actual_order: list[str],
        frozen_plan: Optional[FrozenTaskPlan],
        plan_cost: dict[str, Any],
        raw_configurations: dict[str, dict[str, Any]],
        compatibility_differences: dict[str, Any],
        experiment_id: str,
    ) -> BenchmarkTrial:
        started = _timestamp()
        start_time = time.perf_counter()
        context = AccountingContext(
            experiment_id=experiment_id,
            task_block_id=block.task_block_id,
            trial_id=trial_id,
            treatment_id=arm.value,
        )
        result_value = None
        error: Optional[BaseException] = None
        accounting: Optional[AccountingSnapshot] = None
        try:
            result_value = await self.registry.execute(arm, task, context, frozen_plan)
        except asyncio.CancelledError as caught:
            error = caught
            accounting = _accounting_snapshot({
                "accounting": getattr(caught, "accounting_snapshot", None)
            })
        except Exception as caught:
            error = caught
        duration = round(time.perf_counter() - start_time, 6)
        ended = _timestamp()
        if accounting is None:
            accounting = _accounting_snapshot(result_value)
        provider_calls = _provider_records(result_value, accounting)
        answer = str(
            _read_attr(result_value, "answer", "")
            if arm is BenchmarkArm.BASELINE
            else _read_attr(result_value, "final_answer", "")
        ) if result_value is not None else ""
        status_value = _read_attr(result_value, "status") if result_value is not None else None
        if arm is BenchmarkArm.BASELINE and result_value is not None:
            success_flag = _read_attr(result_value, "success", bool(answer.strip())) is True
            reported_error = _read_attr(result_value, "error")
            if success_flag and not reported_error and answer.strip():
                status = "SUCCESS"
            elif isinstance(reported_error, str) and "timeout" in reported_error.casefold():
                status = "TIMED_OUT"
            else:
                status = "FAILED"
        else:
            status = str(getattr(status_value, "value", status_value) or "FAILED").upper()
            if status not in {
                "SUCCESS", "PARTIAL_FAILURE", "FAILED", "TIMED_OUT", "CANCELLED", "INTERRUPTED"
            }:
                status = "UNKNOWN"
            reported_error = _read_attr(result_value, "error") if result_value is not None else None
        if isinstance(error, asyncio.CancelledError):
            status = "INTERRUPTED"
        elif error is not None:
            status = "FAILED"
        elif status == "SUCCESS" and not answer.strip():
            status = "EMPTY_OUTPUT"
        success = status == "SUCCESS"
        quality_eligible = success and bool(answer.strip())
        verification_report = _read_attr(result_value, "verification_report", {}) if result_value is not None else {}
        verification_report = verification_report if isinstance(verification_report, dict) else {}
        if arm is BenchmarkArm.BASELINE:
            verification = {
                "support_rate": None,
                "unsupported_rate": None,
                "repair_success_rate": None,
                "total_claims": None,
                "accepted_claims_count": None,
                "rejected_claims_count": None,
                "needs_revision_claims_count": None,
                "unverifiable_claims_count": None,
                "unreviewed_claims_count": None,
                "reviewed_claim_count": None,
            }
        else:
            accepted = verification_report.get("accepted_claims")
            rejected = verification_report.get("rejected_claims")
            needs_revision = verification_report.get("needs_revision_claims")
            unverifiable = verification_report.get("unverifiable_claims")
            unreviewed = verification_report.get("unreviewed_claims")
            accepted_count = len(accepted) if isinstance(accepted, list) else None
            rejected_count = len(rejected) if isinstance(rejected, list) else None
            needs_revision_count = len(needs_revision) if isinstance(needs_revision, list) else None
            unverifiable_count = len(unverifiable) if isinstance(unverifiable, list) else None
            unreviewed_count = len(unreviewed) if isinstance(unreviewed, list) else None
            reviewed_count = (
                accepted_count + rejected_count + needs_revision_count + unverifiable_count
                if all(value is not None for value in (
                    accepted_count, rejected_count, needs_revision_count, unverifiable_count
                ))
                else None
            )
            repair_history = verification_report.get("repair_history")
            verification = {
                "support_rate": verification_report.get("support_rate"),
                "unsupported_rate": verification_report.get("unsupported_rate"),
                "repair_success_rate": verification_report.get("repair_success_rate"),
                "total_claims": verification_report.get("total_claims"),
                "accepted_claims_count": accepted_count,
                "rejected_claims_count": rejected_count,
                "needs_revision_claims_count": needs_revision_count,
                "unverifiable_claims_count": unverifiable_count,
                "unreviewed_claims_count": unreviewed_count,
                "reviewed_claim_count": reviewed_count,
                "repair_attempt_count": len(repair_history) if isinstance(repair_history, list) else None,
            }
        token_usage = accounting.token_usage if accounting is not None else _read_attr(result_value, "token_usage", {})
        token_usage = token_usage if isinstance(token_usage, dict) else {}
        raw_config = raw_configurations["BASELINE" if arm is BenchmarkArm.BASELINE else "MEDHA"]
        trial_cost = self._snapshot_cost(accounting, raw_config) if accounting is not None else self._unknown_cost(
            "Trial accounting snapshot unavailable."
        )
        if arm is BenchmarkArm.BASELINE:
            preparation_cost = {"amount_usd": 0.0, "status": "not_applicable", "usage_source": "none"}
            end_to_end = trial_cost
        else:
            preparation_cost = plan_cost
            end_to_end = _sum_costs((plan_cost, trial_cost), shared_once=True)
        evaluated = self.evaluator.evaluate_answer(
            task_id=task_id,
            domain=str(case.get("domain", "unknown")),
            mode=arm.value,
            prompt=task,
            answer=answer,
            duration=duration,
            token_usage=token_usage,
            reference_criteria=list(case.get("reference_criteria", [])),
            known_pitfalls=list(case.get("known_pitfalls", [])),
            verification_stats=verification,
            error="runner_reported_error" if (error is not None or reported_error) else None,
            experiment_id=experiment_id,
            trial_id=trial_id,
            trial_index=repetition,
            execution_order=planned_order,
            timestamp=started,
            configuration=raw_config,
            human_scores=None,
            completion_status=status,
        )
        measurements = {}
        for name, value in evaluated.answer_metrics.items():
            eligible_count = int(quality_eligible and value is not None)
            measurements[name] = MetricMeasurement(
                value=value,
                eligible_sample_count=eligible_count,
                attempt_count=1,
                missing_count=int(eligible_count == 0),
                missingness_reason=(
                    "output ineligible" if not quality_eligible else "metric unavailable"
                ) if eligible_count == 0 else None,
            )
        verification_metrics = {}
        verification_available = (
            arm is not BenchmarkArm.BASELINE
            and bool(verification_report)
        )
        for name in ("support_rate", "unsupported_rate", "repair_success_rate"):
            value = verification.get(name) if verification_available else None
            verification_metrics[name] = value
            if not verification_available:
                missing_reason = "verification not executed" if arm is BenchmarkArm.BASELINE else "verification report unavailable"
            elif value is None and verification.get("reviewed_claim_count") == 0:
                missing_reason = "no reviewed claims"
            else:
                missing_reason = "verification metric unavailable" if value is None else None
            measurements[f"verification_{name}"] = MetricMeasurement(
                value=value,
                eligible_sample_count=int(verification_available and value is not None),
                attempt_count=1,
                missing_count=int(value is None),
                missingness_reason=missing_reason,
            )
        dropped_records = accounting.dropped_provider_call_records if accounting is not None else 0
        request_counts = self._provider_request_counts(accounting, provider_calls, dropped_records)
        request_count = request_counts["total_requests"]
        has_resource_telemetry = accounting is not None or bool(provider_calls)
        retries = sum(1 for call in provider_calls if call.retry_number > 0 and not call.is_fallback)
        failures = request_counts["failed_requests"]
        provider_timeouts = sum(
            str(call.status).upper() == "TIMED_OUT" for call in provider_calls
        )
        execution_metrics = {
            "attempt_count": 1,
            "successful_completion": float(success),
            "failure_or_interruption": float(not success),
            "latency_seconds": duration,
            "provider_request_count": request_count,
            "retry_count": retries if has_resource_telemetry else None,
            "provider_failure_count": failures if has_resource_telemetry else None,
            "provider_timeout_count": provider_timeouts if has_resource_telemetry else None,
            "retry_counts_are_lower_bounds": bool(dropped_records or accounting is None),
            "fallback_counts_are_lower_bounds": bool(dropped_records or accounting is None),
        }
        snapshot = BenchmarkConfigurationSnapshot(
            application_version=None,
            git_commit=None,
            python_version=sys.version.split()[0],
            provider_configuration=raw_config,
            effective_provider_calls=[call.model_dump(mode="json") for call in provider_calls],
            treatment_configuration=get_treatment_preset(arm).model_dump(mode="json"),
            dataset_hash=block.dataset_hash,
            plan_hash=block.plan_hash,
            compatible=not compatibility_differences,
            compatibility_differences=compatibility_differences,
        )
        arm_result = BenchmarkArmResult(
            experiment_id=experiment_id,
            task_block_id=block.task_block_id,
            trial_id=trial_id,
            execution_id=(accounting.context.execution_id if accounting else _read_attr(result_value, "execution_id")),
            arm_id=arm.value,
            plan_hash=block.plan_hash if arm is not BenchmarkArm.BASELINE else None,
            task_id=task_id,
            repetition=repetition,
            planned_order=planned_order,
            actual_order=list(actual_order),
            start_timestamp=started,
            end_timestamp=ended,
            duration_seconds=duration,
            status=status,
            success=success,
            quality_eligible=quality_eligible,
            error_category=_safe_error_category(error or reported_error),
            answer=answer,
            partial_answer=answer if not success and answer else None,
            metrics=measurements,
            execution_metrics=execution_metrics,
            operational_telemetry=request_counts,
            verification_metrics=verification_metrics,
            provider_call_records=provider_calls,
            accounting=accounting,
            cost=end_to_end,
            plan_preparation_cost=preparation_cost,
            trial_execution_cost=trial_cost,
            end_to_end_cost=end_to_end,
            telemetry_complete=accounting is not None and accounting.dropped_provider_call_records == 0,
            dropped_provider_call_records=(accounting.dropped_provider_call_records if accounting else 0),
            configuration=snapshot,
        )
        return BenchmarkTrial(
            trial_id=trial_id,
            task_block_id=block.task_block_id,
            repetition=repetition,
            arm_id=arm.value,
            planned_order=planned_order,
            actual_order=list(actual_order),
            result=arm_result,
        )

    def _aggregate(
        self,
        experiment: BenchmarkExperiment,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        results = [trial.result for block in experiment.task_blocks for trial in block.trials]
        operational_aggregates = {}
        comparative_aggregates = {}
        aggregate = {}
        for arm in BenchmarkArm:
            arm_results = [result for result in results if result.arm_id == arm.value]
            compatible_results = [
                result for result in arm_results
                if result.configuration is not None and result.configuration.compatible
            ]
            operational = self._aggregate_operational(arm_results)
            comparative = self._aggregate_comparative(arm_results, compatible_results, arm)
            operational_aggregates[arm.value] = operational
            comparative_aggregates[arm.value] = comparative
            comparative_verification = comparative["verification_metrics"]
            aggregate[arm.value] = {
                **operational,
                "operational": operational,
                "incompatible_attempt_count": comparative["incompatible_attempt_count"],
                "compatible_attempt_count": comparative["compatible_attempt_count"],
                "provider_request_count": operational["provider_request_count"]["value"],
                "provider_failure_count": operational["failed_provider_request_count"]["value"],
                "retry_count": operational["retry_count"]["value"],
                "input_tokens": operational["tokens"]["input_tokens"]["value"],
                "output_tokens": operational["tokens"]["output_tokens"]["value"],
                "total_tokens": operational["tokens"]["total_tokens"]["value"],
                "average_latency_seconds": operational["latency"]["average_seconds"],
                "comparative_success_rate": comparative["success_rate"],
                "metrics": comparative["answer_quality_metrics"],
                "verification_metrics": {
                    "support_rate": comparative_verification.get("verification_support_rate", {}).get("value"),
                    "unsupported_rate": comparative_verification.get("verification_unsupported_rate", {}).get("value"),
                    "repair_success_rate": comparative_verification.get("verification_repair_success_rate", {}).get("value"),
                },
                "comparative": comparative,
                "legacy_field_semantics": {
                    "success_rate": "Operational rate over all attempted trials; use comparative.success_rate for compatible comparisons.",
                    "metrics": "Comparative answer-quality metrics filtered by compatibility and metric eligibility.",
                    "verification_metrics": "Comparative verification results; null when verification was not executed or eligible.",
                },
            }
        return operational_aggregates, comparative_aggregates, aggregate

    @staticmethod
    def _aggregate_operational(arm_results: list[BenchmarkArmResult]) -> dict[str, Any]:
        attempt_count = len(arm_results)
        successful = sum(result.success for result in arm_results)
        statuses: dict[str, int] = {}
        for result in arm_results:
            statuses[result.status] = statuses.get(result.status, 0) + 1

        def summarize(values: list[Optional[int]], reason: str) -> dict[str, Any]:
            available = [value for value in values if value is not None]
            return {
                "value": sum(available) if available else None,
                "attempt_count": attempt_count,
                "available_attempt_count": len(available),
                "missing_count": attempt_count - len(available),
                "missingness_reason": reason if len(available) < attempt_count else None,
            }

        request_counts = [
            FourArmBenchmark._provider_request_counts(
                result.accounting,
                result.provider_call_records,
                result.dropped_provider_call_records,
            )
            for result in arm_results
        ]
        retries = [
            sum(call.retry_number > 0 and not call.is_fallback for call in result.provider_call_records)
            if result.accounting is not None or result.provider_call_records
            else None
            for result in arm_results
        ]
        fallback_requests = [
            sum(call.is_fallback for call in result.provider_call_records)
            if result.accounting is not None or result.provider_call_records
            else None
            for result in arm_results
        ]
        token_keys = {
            "input_tokens": "prompt_tokens",
            "output_tokens": "candidate_tokens",
            "total_tokens": "accounted_total_tokens",
            "provider_reported_total_tokens": "total_tokens",
            "estimated_tokens": "estimated_tokens_without_usage",
        }
        token_aggregates = {}
        for name, key in token_keys.items():
            token_aggregates[name] = summarize([
                int(result.accounting.token_usage.get(key, 0) or 0)
                if result.accounting is not None and key in result.accounting.token_usage
                else None
                for result in arm_results
            ], "accounting snapshot unavailable")
        snapshot_values = [result.accounting for result in arm_results]
        dropped_records = summarize([
            snapshot.dropped_provider_call_records if snapshot is not None else None
            for snapshot in snapshot_values
        ], "dropped-record count unavailable without accounting snapshot")
        usage_sources: dict[str, int] = {}
        for result in arm_results:
            source = (
                str(result.accounting.token_usage.get("usage_source", "unknown"))
                if result.accounting is not None
                else "unavailable"
            )
            usage_sources[source] = usage_sources.get(source, 0) + 1

        durations = [result.duration_seconds for result in arm_results if result.duration_seconds is not None]
        def summarize_costs(field_name: str) -> dict[str, Any]:
            costs = [getattr(result, field_name) for result in arm_results]
            known_costs = [cost["amount_usd"] for cost in costs if cost.get("amount_usd") is not None]
            unknown_cost_count = len(costs) - len(known_costs)
            provenance_counts: dict[str, int] = {}
            for cost in costs:
                provenance = str(cost.get("status", "unknown"))
                provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
            return {
                "amount_usd": sum(known_costs) if costs and not unknown_cost_count else None,
                "known_subtotal_usd": sum(known_costs) if known_costs else None,
                "attempt_count": attempt_count,
                "known_cost_count": len(known_costs),
                "unknown_cost_count": unknown_cost_count,
                "provenance_counts": provenance_counts,
                "uncertainty_reason": "one or more trial costs are unknown" if unknown_cost_count else None,
            }

        any_dropped_records = any(
            max(
                result.dropped_provider_call_records,
                result.accounting.dropped_provider_call_records if result.accounting else 0,
            ) > 0
            for result in arm_results
        )

        return {
            "attempt_count": attempt_count,
            "compatible_attempt_count": sum(
                result.configuration is not None and result.configuration.compatible
                for result in arm_results
            ),
            "incompatible_attempt_count": sum(
                result.configuration is None or not result.configuration.compatible
                for result in arm_results
            ),
            "successful_completion_count": successful,
            "failure_or_interruption_count": attempt_count - successful,
            "partial_failure_count": statuses.get("PARTIAL_FAILURE", 0),
            "failed_count": statuses.get("FAILED", 0) + statuses.get("EXCEPTION", 0),
            "timed_out_count": statuses.get("TIMED_OUT", 0),
            "cancelled_count": statuses.get("CANCELLED", 0),
            "interrupted_count": statuses.get("INTERRUPTED", 0),
            "empty_output_count": statuses.get("EMPTY_OUTPUT", 0),
            "status_counts": statuses,
            "success_rate": successful / attempt_count if attempt_count else None,
            "latency": {
                "average_seconds": round(sum(durations) / len(durations), 6) if durations else None,
                "minimum_seconds": min(durations) if durations else None,
                "maximum_seconds": max(durations) if durations else None,
                "attempt_count": attempt_count,
                "available_attempt_count": len(durations),
                "missing_count": attempt_count - len(durations),
            },
            "provider_request_count": summarize(
                [item["total_requests"] for item in request_counts],
                "request count unavailable from snapshot or complete records",
            ),
            "provider_request_count_sources": dict(Counter(
                item["request_count_source"] for item in request_counts
            )),
            "provider_request_count_is_lower_bound": any(
                item["request_count_is_lower_bound"] for item in request_counts
            ),
            "failed_provider_request_count": summarize(
                [item["failed_requests"] for item in request_counts],
                "failed request count unavailable from snapshot or complete records",
            ),
            "failed_provider_request_count_sources": dict(Counter(
                item["failure_count_source"] for item in request_counts
            )),
            "failed_provider_request_count_is_lower_bound": any(
                item["failure_count_is_lower_bound"] for item in request_counts
            ),
            "retry_count": summarize(retries, "provider-call records unavailable"),
            "fallback_request_count": summarize(fallback_requests, "provider-call records unavailable"),
            "retry_and_fallback_counts_are_lower_bounds": any(
                max(
                    result.dropped_provider_call_records,
                    result.accounting.dropped_provider_call_records if result.accounting else 0,
                ) > 0 for result in arm_results
            ),
            "provider_timeout_count": summarize([
                sum(str(call.status).upper() == "TIMED_OUT" for call in result.provider_call_records)
                if result.accounting is not None or result.provider_call_records
                else None
                for result in arm_results
            ], "provider-call records unavailable"),
            "provider_timeout_count_is_lower_bound": any(
                max(
                    result.dropped_provider_call_records,
                    result.accounting.dropped_provider_call_records if result.accounting else 0,
                ) > 0 for result in arm_results
            ),
            "provider_call_duration_seconds": summarize([
                sum(call.duration_seconds or 0.0 for call in result.provider_call_records)
                if result.accounting is not None or result.provider_call_records
                else None
                for result in arm_results
            ], "provider-call records unavailable"),
            "tokens": token_aggregates,
            "usage_source_counts": usage_sources,
            "usage_completeness": {
                "complete_attempt_count": sum(result.telemetry_complete for result in arm_results),
                "incomplete_attempt_count": sum(not result.telemetry_complete for result in arm_results),
                "provider_usage_requests": sum(
                    int(result.accounting.token_usage.get("provider_usage_requests", 0) or 0)
                    for result in arm_results if result.accounting is not None
                ) if all(result.accounting is not None for result in arm_results) and arm_results else None,
                "estimated_usage_requests": sum(
                    int(result.accounting.token_usage.get("estimated_usage_requests", 0) or 0)
                    for result in arm_results if result.accounting is not None
                ) if all(result.accounting is not None for result in arm_results) and arm_results else None,
            },
            "dropped_provider_call_records": dropped_records,
            "telemetry_complete_attempt_count": sum(result.telemetry_complete for result in arm_results),
            "telemetry_incomplete_attempt_count": sum(not result.telemetry_complete for result in arm_results),
            "any_provider_call_records_dropped": any_dropped_records,
            "trial_execution_cost": summarize_costs("trial_execution_cost"),
            "end_to_end_comparison_cost": summarize_costs("end_to_end_cost"),
            "quality_eligible_attempt_count": sum(result.quality_eligible for result in arm_results),
        }

    @staticmethod
    def _aggregate_comparative(
        arm_results: list[BenchmarkArmResult],
        compatible_results: list[BenchmarkArmResult],
        arm: BenchmarkArm,
    ) -> dict[str, Any]:
        answer_metric_names = {
            name for result in arm_results for name in result.metrics
            if name not in {"task_completion", "unsupported_claim_rate"}
            and not name.startswith("verification_")
        }
        verification_metric_names = {
            name for result in arm_results for name in result.metrics
            if name.startswith("verification_") or name == "unsupported_claim_rate"
        }

        def aggregate_measurements(metric_name: str, *, verification: bool) -> dict[str, Any]:
            values = []
            reasons: dict[str, int] = {}
            for result in arm_results:
                if result not in compatible_results:
                    reason = "configuration_incompatible"
                else:
                    measurement = result.metrics.get(metric_name)
                    if measurement is None:
                        reason = "metric_not_recorded"
                    elif not verification and not result.quality_eligible:
                        reason = measurement.missingness_reason or "output_ineligible"
                    elif measurement.value is None:
                        reason = measurement.missingness_reason or "metric_unavailable"
                    elif measurement.eligible_sample_count < 1:
                        reason = measurement.missingness_reason or "not_metric_eligible"
                    else:
                        values.append(float(measurement.value))
                        continue
                reasons[reason] = reasons.get(reason, 0) + 1
            return {
                "value": round(sum(values) / len(values), 6) if values else None,
                "eligible_sample_count": len(values),
                "attempt_count": len(arm_results),
                "compatible_attempt_count": len(compatible_results),
                "missing_count": len(arm_results) - len(values),
                "missingness_reasons": reasons,
            }

        answer_quality = {
            name: aggregate_measurements(name, verification=False)
            for name in sorted(answer_metric_names)
        }
        verification = {
            name: aggregate_measurements(name, verification=True)
            for name in sorted(verification_metric_names)
        }
        verification_trials = [
            result for result in compatible_results
            if result.arm_id != BenchmarkArm.BASELINE.value
            and result.verification_metrics.get("reviewed_claim_count") is not None
        ]
        pooled_counts = {
            name: sum(
                int(result.verification_metrics.get(f"{name}_count") or 0)
                for result in verification_trials
            )
            for name in (
                "accepted_claims", "rejected_claims", "needs_revision_claims",
                "unverifiable_claims", "unreviewed_claims",
            )
        }
        pooled_reviewed = sum(
            int(result.verification_metrics.get("reviewed_claim_count") or 0)
            for result in verification_trials
        )
        pooled_support_rate = (
            round(pooled_counts["accepted_claims"] / pooled_reviewed * 100, 6)
            if pooled_reviewed else None
        )
        pooled_unsupported_rate = (
            round(
                (pooled_counts["rejected_claims"] + pooled_counts["unverifiable_claims"])
                / pooled_reviewed * 100,
                6,
            )
            if pooled_reviewed else None
        )
        mean_trial_verification_rates = {
            name: verification.get(f"verification_{name}", {}).get("value")
            for name in ("support_rate", "unsupported_rate", "repair_success_rate")
        }
        successes = sum(result.success for result in compatible_results)
        return {
            "attempt_count": len(arm_results),
            "compatible_attempt_count": len(compatible_results),
            "incompatible_attempt_count": len(arm_results) - len(compatible_results),
            "successful_completion_count": successes,
            "success_rate": successes / len(compatible_results) if compatible_results else None,
            "failure_or_interruption_count": len(compatible_results) - successes,
            "quality_eligible_count": sum(result.quality_eligible for result in compatible_results),
            "answer_quality_metrics": answer_quality,
            "verification_aggregation": {
                "mean_trial_rates": mean_trial_verification_rates,
                "mean_trial_denominators": {
                    name: verification.get(f"verification_{name}", {}).get("eligible_sample_count", 0)
                    for name in ("support_rate", "unsupported_rate", "repair_success_rate")
                },
                "pooled_claim_level": {
                    "accepted_claims": pooled_counts["accepted_claims"],
                    "rejected_claims": pooled_counts["rejected_claims"],
                    "needs_revision_claims": pooled_counts["needs_revision_claims"],
                    "unverifiable_claims": pooled_counts["unverifiable_claims"],
                    "unreviewed_claims": pooled_counts["unreviewed_claims"],
                    "reviewed_claim_denominator": pooled_reviewed,
                    "support_rate": pooled_support_rate,
                    "unsupported_rate": pooled_unsupported_rate,
                    "trial_count": len(verification_trials),
                    "missing_count": len(compatible_results) - len(verification_trials),
                },
            },
            "verification_metrics": verification if arm is not BenchmarkArm.BASELINE else {
                name: {
                    "value": None,
                    "eligible_sample_count": 0,
                    "attempt_count": len(arm_results),
                    "compatible_attempt_count": len(compatible_results),
                    "missing_count": len(arm_results),
                    "missingness_reasons": {"verification_not_executed": len(arm_results)},
                }
                for name in sorted(verification_metric_names)
            },
        }

    @staticmethod
    def _write_json(path: Path, result: BenchmarkEvaluationResult) -> None:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result.model_dump(mode="json"), stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")

    @staticmethod
    def _write_csv(path: Path, experiment: BenchmarkExperiment) -> None:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for block in experiment.task_blocks:
                for trial in block.trials:
                    result = trial.result
                    usage = result.accounting.token_usage if result.accounting else {}
                    metrics = result.metrics
                    calls = result.provider_call_records
                    request_counts = FourArmBenchmark._provider_request_counts(
                        result.accounting,
                        calls,
                        result.dropped_provider_call_records,
                    )
                    compatibility = result.configuration
                    metric_eligibility = {
                        name: item.eligible_sample_count for name, item in metrics.items()
                    }
                    metric_missingness = {
                        name: item.missingness_reason
                        for name, item in metrics.items()
                        if item.missingness_reason is not None
                    }
                    row = {
                        "experiment_id": result.experiment_id,
                        "task_block_id": result.task_block_id,
                        "task_id": result.task_id,
                        "trial_id": result.trial_id,
                        "execution_id": result.execution_id,
                        "arm_id": result.arm_id,
                        "repetition": result.repetition,
                        "planned_order": ">".join(result.planned_order),
                        "actual_order": ">".join(result.actual_order),
                        "plan_hash": result.plan_hash,
                        "status": result.status,
                        "success": result.success,
                        "quality_eligible": result.quality_eligible,
                        "start_timestamp": result.start_timestamp,
                        "end_timestamp": result.end_timestamp,
                        "duration_seconds": result.duration_seconds,
                        "error_category": result.error_category,
                        "verification_support_rate": result.verification_metrics.get("support_rate"),
                        "verification_unsupported_rate": result.verification_metrics.get("unsupported_rate"),
                        "verification_reviewed_claim_count": result.verification_metrics.get("reviewed_claim_count"),
                        "verification_accepted_claims_count": result.verification_metrics.get("accepted_claims_count"),
                        "verification_rejected_claims_count": result.verification_metrics.get("rejected_claims_count"),
                        "verification_needs_revision_claims_count": result.verification_metrics.get("needs_revision_claims_count"),
                        "verification_unverifiable_claims_count": result.verification_metrics.get("unverifiable_claims_count"),
                        "verification_unreviewed_claims_count": result.verification_metrics.get("unreviewed_claims_count"),
                        "metric_eligible_samples": sum(item.eligible_sample_count for item in metrics.values()),
                        "metric_attempt_count": sum(item.attempt_count for item in metrics.values()),
                        "metric_missing_count": sum(item.missing_count for item in metrics.values()),
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("candidate_tokens"),
                        "total_tokens": usage.get("accounted_total_tokens", usage.get("total_tokens")),
                        "provider_reported_total_tokens": usage.get("total_tokens"),
                        "estimated_total_tokens": usage.get("estimated_tokens_without_usage"),
                        "usage_source": usage.get("usage_source", "unknown"),
                        "provider_request_count": request_counts["total_requests"],
                        "failed_provider_request_count": request_counts["failed_requests"],
                        "provider_request_count_source": request_counts["request_count_source"],
                        "failed_provider_request_count_source": request_counts["failure_count_source"],
                        "provider_request_count_is_lower_bound": request_counts["request_count_is_lower_bound"],
                        "failed_provider_request_count_is_lower_bound": request_counts["failure_count_is_lower_bound"],
                        "retry_count": sum(
                            call.retry_number > 0 and not call.is_fallback for call in calls
                        ) if result.accounting or calls else None,
                        "fallback_request_count": sum(call.is_fallback for call in calls) if result.accounting or calls else None,
                        "provider_failure_count": request_counts["failed_requests"],
                        "provider_failure_count_is_lower_bound": request_counts["failure_count_is_lower_bound"],
                        "provider_timeout_count": sum(str(call.status).upper() == "TIMED_OUT" for call in calls) if result.accounting or calls else None,
                        "provider_timeout_count_is_lower_bound": bool(
                            result.accounting is None
                            or max(
                                result.dropped_provider_call_records,
                                result.accounting.dropped_provider_call_records,
                            )
                        ),
                        "timed_out_count": int(result.status == "TIMED_OUT"),
                        "retry_counts_are_lower_bounds": bool(
                            result.accounting is None
                            or max(
                                result.dropped_provider_call_records,
                                result.accounting.dropped_provider_call_records,
                            )
                        ),
                        "fallback_counts_are_lower_bounds": bool(
                            result.accounting is None
                            or max(
                                result.dropped_provider_call_records,
                                result.accounting.dropped_provider_call_records,
                            )
                        ),
                        "usage_complete": result.telemetry_complete,
                        "usage_completeness": json.dumps({
                            "provider_usage_requests": usage.get("provider_usage_requests"),
                            "estimated_usage_requests": usage.get("estimated_usage_requests"),
                            "accounting_available": result.accounting is not None,
                        }, sort_keys=True),
                        "compatibility_status": compatibility.compatibility_status if compatibility else "INDETERMINATE",
                        "compatibility_limitations": json.dumps(
                            compatibility.compatibility_limitations if compatibility else [],
                            ensure_ascii=False,
                        ),
                        "compatibility_exclusion_reason": json.dumps(
                            (
                                compatibility.compatibility_differences
                                if compatibility and compatibility.compatibility_differences
                                else compatibility.compatibility_limitations
                                if compatibility and not compatibility.compatible
                                else []
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "metric_eligibility": json.dumps(metric_eligibility, sort_keys=True),
                        "metric_missingness_reasons": json.dumps(metric_missingness, sort_keys=True),
                        "telemetry_complete": result.telemetry_complete,
                        "dropped_provider_call_records": result.dropped_provider_call_records,
                        "plan_preparation_cost_usd": result.plan_preparation_cost.get("amount_usd"),
                        "trial_execution_cost_usd": result.trial_execution_cost.get("amount_usd"),
                        "end_to_end_cost_usd": result.end_to_end_cost.get("amount_usd"),
                        "cost_provenance": result.end_to_end_cost.get("status"),
                        "configuration_compatible": result.configuration.compatible if result.configuration else False,
                        "answer": result.answer,
                    }
                    for name in (
                        "answer_completeness", "task_completion", "pitfall_avoidance",
                        "uncertainty_handling", "evidence_quality", "factual_correctness",
                    ):
                        row[name] = metrics.get(name).value if metrics.get(name) else None
                    for name in (
                        "answer_completeness", "task_completion", "pitfall_avoidance",
                        "uncertainty_handling", "evidence_quality", "factual_correctness",
                    ):
                        measurement = metrics.get(name)
                        row[f"{name}_eligible_count"] = (
                            measurement.eligible_sample_count if measurement else 0
                        )
                        row[f"{name}_missing_count"] = measurement.missing_count if measurement else 1
                    verification_measurements = [
                        metrics.get(f"verification_{name}")
                        for name in ("support_rate", "unsupported_rate", "repair_success_rate")
                    ]
                    row["verification_eligible_count"] = sum(
                        item.eligible_sample_count for item in verification_measurements if item
                    )
                    row["verification_missing_count"] = sum(
                        item.missing_count for item in verification_measurements if item
                    )
                    writer.writerow({key: "null" if value is None else value for key, value in row.items()})
