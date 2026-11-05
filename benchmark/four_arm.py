"""Four-arm evaluator built on the existing frozen-plan and generic policy APIs."""

from __future__ import annotations

import asyncio
import csv
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
    "verification_unsupported_rate", "answer_completeness_eligible_count",
    "answer_completeness_missing_count", "task_completion_eligible_count",
    "task_completion_missing_count", "pitfall_avoidance_eligible_count",
    "pitfall_avoidance_missing_count", "uncertainty_handling_eligible_count",
    "uncertainty_handling_missing_count", "evidence_quality_eligible_count",
    "evidence_quality_missing_count", "factual_correctness_eligible_count",
    "factual_correctness_missing_count", "verification_eligible_count",
    "verification_missing_count", "metric_attempt_count", "metric_missing_count",
    "input_tokens", "output_tokens", "total_tokens", "provider_reported_total_tokens",
    "estimated_total_tokens", "usage_source",
    "provider_request_count", "retry_count", "provider_failure_count", "telemetry_complete",
    "dropped_provider_call_records", "plan_preparation_cost_usd", "trial_execution_cost_usd",
    "end_to_end_cost_usd", "cost_provenance", "configuration_compatible", "answer",
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
    unknown = [cost for cost in selected if cost.get("amount_usd") is None]
    if unknown:
        return {
            "amount_usd": None,
            "status": "unknown",
            "reason": "At least one component cost is unknown.",
            "shared_preparation_counted_once": shared_once,
        }
    statuses = {str(cost.get("status", "unknown")) for cost in selected}
    status = next(iter(statuses)) if len(statuses) == 1 else "mixed" if statuses else "unknown"
    return {
        "amount_usd": round(sum(float(cost["amount_usd"]) for cost in selected), 10),
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
        )
        profiles: dict[str, list[dict[str, Any]]] = {}
        differences = dict(initial_differences)
        assessments = dict(
            experiment.configuration_snapshots["BASELINE"].compatibility_assessments
        )
        limitations = list(
            experiment.configuration_snapshots["BASELINE"].compatibility_limitations
        )
        profiles_by_stage: dict[str, dict[str, list[dict[str, Any]]]] = {}
        revisions_by_stage: dict[str, dict[str, set[str]]] = {}
        response_models_by_stage: dict[str, dict[str, set[str]]] = {}
        provider_call_coverage: dict[str, int] = {}
        revision_observation_count = 0
        response_model_observation_count = 0
        declared_fields = {
            "provider": "provider",
            "model": "model",
            "model_version": "model_version",
            "temperature": "temperature",
            "input_token_limit": "input_token_limit",
            "output_token_limit": "output_token_limit",
        }
        for arm in BenchmarkArm:
            calls = [
                call
                for block in experiment.task_blocks
                for trial in block.trials
                if trial.arm_id == arm.value
                for call in trial.result.provider_call_records
            ]
            normalized_profiles = []
            for call in calls:
                profile = {name: getattr(call, name) for name in profile_fields}
                timeout_source = getattr(call, "timeout_source", None)
                profile["timeout_source"] = timeout_source or "unavailable"
                if timeout_source == "explicit_request":
                    profile["timeout_seconds"] = call.timeout_seconds
                normalized_profiles.append(profile)
            unique = {
                canonical_hash(profile): profile
                for profile in normalized_profiles
            }
            profiles[arm.value] = [unique[key] for key in sorted(unique)]
            provider_call_coverage[arm.value] = len(calls)
            for call in calls:
                declared = experiment.configuration_snapshots[arm.value].provider_configuration
                for call_field, config_field in declared_fields.items():
                    if call_field == "output_token_limit":
                        continue
                    configured = declared.get(config_field)
                    effective = getattr(call, call_field)
                    assessment_key = f"{arm.value}.effective_call.{config_field}"
                    if configured in (None, "", "unknown") or effective in (None, "", "unknown"):
                        assessments[assessment_key] = {
                            "status": "unavailable",
                            "configured": configured,
                            "effective": effective,
                        }
                        limitations.append(f"{arm.value} effective {config_field} metadata is unavailable.")
                    elif effective != configured:
                        assessments[assessment_key] = {
                            "status": "known_different",
                            "configured": configured,
                            "effective": effective,
                        }
                        differences[f"{arm.value}.effective_call.{config_field}"] = {
                            "configured": configured,
                            "effective": effective,
                        }
                    else:
                        assessments[assessment_key] = {
                            "status": "known_equal",
                            "value": effective,
                        }
                configured_output_limit = declared.get("output_token_limit")
                if (
                    configured_output_limit not in (None, "", "unknown")
                    and call.output_token_limit is not None
                    and call.output_token_limit > configured_output_limit
                ):
                    assessments[f"{arm.value}.effective_call.output_token_limit"] = {
                        "status": "known_different",
                        "configured_maximum": configured_output_limit,
                        "effective": call.output_token_limit,
                    }
                    differences[f"{arm.value}.effective_call.output_token_limit"] = {
                        "configured_maximum": configured_output_limit,
                        "effective": call.output_token_limit,
                    }
                elif call.output_token_limit is not None:
                    assessments[f"{arm.value}.effective_call.output_token_limit"] = {
                        "status": "known_equal",
                        "comparison": "effective value is within the configured maximum",
                        "configured_maximum": configured_output_limit,
                        "effective": call.output_token_limit,
                    }
                profile = {name: getattr(call, name) for name in profile_fields}
                profile["timeout_source"] = getattr(call, "timeout_source", None) or "unavailable"
                if profile["timeout_source"] == "explicit_request":
                    profile["timeout_seconds"] = call.timeout_seconds
                profile_hash = canonical_hash(profile)
                profiles_by_stage.setdefault(str(call.stage or "unknown"), {}).setdefault(arm.value, {})[
                    profile_hash
                ] = profile
                stage = str(call.stage or "unknown")
                if getattr(call, "model_version_source", "unavailable") == "provider_response" and call.model_version:
                    revisions_by_stage.setdefault(stage, {}).setdefault(arm.value, set()).add(call.model_version)
                    revision_observation_count += 1
                if call.response_model:
                    response_models_by_stage.setdefault(stage, {}).setdefault(arm.value, set()).add(call.response_model)
                    response_model_observation_count += 1
                if getattr(call, "timeout_source", None) == "provider_sdk_default":
                    limitations.append(
                        "Provider SDK default request-timeout duration is unavailable."
                    )
                elif getattr(call, "timeout_source", None) not in ("explicit_request", "provider_sdk_default"):
                    limitations.append("Effective request-timeout mode is unavailable for one or more calls.")
                if getattr(call, "model_version_source", "unavailable") != "provider_response" or not call.model_version:
                    limitations.append(
                        "Provider response did not expose an immutable model revision for one or more calls."
                    )
            missing = [name for item in profiles[arm.value] for name in profile_fields
                       if item.get(name) in (None, "", "unknown")]
            if not profiles[arm.value]:
                limitations.append(f"{arm.value} has no provider-call records to verify effective configuration.")
                assessments[f"{arm.value}.effective_provider_calls"] = {"status": "unavailable"}
            elif missing:
                assessments[f"{arm.value}.effective_provider_calls"] = {
                    "status": "unavailable",
                    "reason": "required effective call fields unknown",
                    "fields": sorted(set(missing)),
                }
                limitations.append(f"{arm.value} effective provider-call settings are partially unavailable.")
            else:
                assessments[f"{arm.value}.effective_provider_calls"] = {"status": "known"}
        planned_arms = {
            BenchmarkArm.INDEPENDENT_PARALLEL.value,
            BenchmarkArm.DEPENDENCY_AWARE.value,
            BenchmarkArm.FULL_MEDHA.value,
        }
        stage_differences = {}
        for stage, arms in profiles_by_stage.items():
            if not planned_arms.issubset(arms):
                continue
            arm_profiles = {
                arm: sorted(arms[arm].values(), key=canonical_hash)
                for arm in sorted(planned_arms)
            }
            if len({canonical_hash(value) for value in arm_profiles.values()}) != 1:
                stage_differences[stage] = arm_profiles
        if stage_differences:
            differences["shared_stage_provider_call_profiles"] = stage_differences

        for metadata_name, profiles_for_metadata, observation_count in (
            ("provider_model_revision", revisions_by_stage, revision_observation_count),
            ("provider_response_model", response_models_by_stage, response_model_observation_count),
        ):
            observed = [
                value
                for stage_profiles in profiles_for_metadata.values()
                for arm_values in stage_profiles.values()
                for value in arm_values
            ]
            if observed and len(set(observed)) > 1:
                differences[metadata_name] = sorted(set(observed))
                assessments[metadata_name] = {"status": "known_different", "values": sorted(set(observed))}
            elif observed and observation_count == sum(provider_call_coverage.values()):
                assessments[metadata_name] = {"status": "known_equal", "value": observed[0]}
            else:
                assessments[metadata_name] = {"status": "unavailable", "non_blocking": True}
                limitations.append(f"{metadata_name} was not provider-reported for every observed call.")

        timeout_sources_by_arm = {
            arm: {profile.get("timeout_source") for profile in arm_profiles}
            for arm, arm_profiles in profiles.items()
        }
        timeout_sources = set().union(*timeout_sources_by_arm.values()) if timeout_sources_by_arm else set()
        all_arms_have_timeout_observations = all(
            provider_call_coverage.get(arm, 0) > 0
            and len(timeout_sources_by_arm.get(arm, set())) == 1
            for arm in (candidate.value for candidate in BenchmarkArm)
        )
        if len(timeout_sources) > 1:
            differences["effective_request_timeout_mode"] = sorted(timeout_sources)
            assessments["request_timeout_mode"] = {
                "status": "known_different",
                "values": sorted(timeout_sources),
            }
        elif timeout_sources == {"explicit_request"} and all_arms_have_timeout_observations:
            explicit_values = {
                profile.get("timeout_seconds")
                for arm_profiles in profiles.values()
                for profile in arm_profiles
            }
            if len(explicit_values) > 1:
                differences["effective_request_timeout_seconds"] = sorted(explicit_values, key=str)
                assessments["request_timeout_seconds"] = {
                    "status": "known_different",
                    "values": sorted(explicit_values, key=str),
                }
            else:
                assessments["request_timeout_seconds"] = {
                    "status": "known_equal",
                    "value": next(iter(explicit_values), None),
                }
        elif timeout_sources == {"provider_sdk_default"} and all_arms_have_timeout_observations:
            assessments["request_timeout_mode"] = {"status": "known_equal", "value": "provider_sdk_default"}
            assessments["request_timeout_seconds"] = {"status": "unavailable", "non_blocking": True}
        elif not timeout_sources:
            assessments["request_timeout_seconds"] = {"status": "unavailable", "non_blocking": True}
            limitations.append("No effective request-timeout observations were recorded.")
        else:
            assessments["request_timeout_seconds"] = {"status": "unavailable", "non_blocking": True}
            limitations.append("Request timeout duration could not be established for every provider call.")

        planner_calls = [
            call.model_dump(mode="json")
            for block in experiment.task_blocks
            if block.plan_preparation_accounting is not None
            for call in block.plan_preparation_accounting.provider_calls
        ]
        if not planner_calls:
            limitations.append("Planner effective provider-call settings are unavailable.")
            assessments["planner.effective_provider_calls"] = {"status": "unavailable"}
        else:
            missing_planner_fields = sorted({
                name
                for call in planner_calls
                for name in profile_fields
                if call.get(name) in (None, "", "unknown")
            })
            if missing_planner_fields:
                assessments["planner.effective_provider_calls"] = {
                    "status": "unavailable",
                    "reason": "required effective call fields unknown",
                    "fields": missing_planner_fields,
                }
                limitations.append("Planner effective provider-call settings are partially unavailable.")
            else:
                assessments["planner.effective_provider_calls"] = {"status": "known"}
            if any(call.get("model_version_source") != "provider_response" or not call.get("model_version") for call in planner_calls):
                limitations.append("Planner provider response did not expose an immutable model revision.")
            if any(call.get("timeout_source") == "provider_sdk_default" for call in planner_calls):
                limitations.append("Planner request timeout duration is provider-managed and unavailable.")

        compatible = not differences
        limitations = list(dict.fromkeys(limitations))
        compatibility_status = (
            "INCOMPATIBLE" if differences
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
                    "effective_provider_calls": planner_calls if arm is not BenchmarkArm.BASELINE else [],
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
            status = "SUCCESS" if success_flag and not reported_error and answer.strip() else "FAILED"
        else:
            status = str(getattr(status_value, "value", status_value) or "FAILED").upper()
            if status not in {"SUCCESS", "PARTIAL_FAILURE", "FAILED", "CANCELLED", "INTERRUPTED"}:
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
            }
        else:
            accepted = verification_report.get("accepted_claims")
            rejected = verification_report.get("rejected_claims")
            verification = {
                "support_rate": verification_report.get("support_rate"),
                "unsupported_rate": verification_report.get("unsupported_rate"),
                "repair_success_rate": verification_report.get("repair_success_rate"),
                "total_claims": verification_report.get("total_claims"),
                "accepted_claims_count": len(accepted) if isinstance(accepted, list) else None,
                "rejected_claims_count": len(rejected) if isinstance(rejected, list) else None,
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
        verification_available = arm is not BenchmarkArm.BASELINE
        for name in ("support_rate", "unsupported_rate", "repair_success_rate"):
            value = verification.get(name) if verification_available else None
            verification_metrics[name] = value
            measurements[f"verification_{name}"] = MetricMeasurement(
                value=value,
                eligible_sample_count=int(verification_available and value is not None),
                attempt_count=1,
                missing_count=int(value is None),
                missingness_reason=(
                    "verification not executed" if not verification_available
                    else "verification metric unavailable"
                ) if value is None else None,
            )
        request_count = len(provider_calls) if accounting is not None else None
        retries = sum(1 for call in provider_calls if call.retry_number > 0 and not call.is_fallback)
        failures = sum(1 for call in provider_calls if str(call.status) != "SUCCEEDED")
        execution_metrics = {
            "attempt_count": 1,
            "successful_completion": float(success),
            "failure_or_interruption": float(not success),
            "latency_seconds": duration,
            "provider_request_count": request_count,
            "retry_count": retries if accounting is not None else None,
            "provider_failure_count": failures if accounting is not None else None,
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

        request_counts = []
        failed_requests = []
        for result in arm_results:
            usage = result.accounting.token_usage if result.accounting is not None else {}
            has_call_telemetry = result.accounting is not None or bool(result.provider_call_records)
            request_counts.append(
                int(usage["total_requests"])
                if "total_requests" in usage
                else len(result.provider_call_records) if has_call_telemetry else None
            )
            failed_requests.append(
                int(usage["failed_requests"])
                if "failed_requests" in usage
                else sum(str(call.status).upper() != "SUCCEEDED" for call in result.provider_call_records)
                if has_call_telemetry else None
            )
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
            "provider_request_count": summarize(request_counts, "provider-call records unavailable"),
            "failed_provider_request_count": summarize(failed_requests, "provider-call records unavailable"),
            "retry_count": summarize(retries, "provider-call records unavailable"),
            "fallback_request_count": summarize(fallback_requests, "provider-call records unavailable"),
            "retry_and_fallback_counts_are_lower_bounds": any(
                result.dropped_provider_call_records > 0 for result in arm_results
            ),
            "provider_call_duration_seconds": summarize([
                sum(call.duration_seconds or 0.0 for call in result.provider_call_records)
                if result.accounting is not None or result.provider_call_records
                else None
                for result in arm_results
            ], "provider-call records unavailable"),
            "tokens": token_aggregates,
            "usage_source_counts": usage_sources,
            "dropped_provider_call_records": dropped_records,
            "telemetry_complete_attempt_count": sum(result.telemetry_complete for result in arm_results),
            "telemetry_incomplete_attempt_count": sum(not result.telemetry_complete for result in arm_results),
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
                        "metric_eligible_samples": sum(item.eligible_sample_count for item in metrics.values()),
                        "metric_attempt_count": sum(item.attempt_count for item in metrics.values()),
                        "metric_missing_count": sum(item.missing_count for item in metrics.values()),
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("candidate_tokens"),
                        "total_tokens": usage.get("accounted_total_tokens", usage.get("total_tokens")),
                        "provider_reported_total_tokens": usage.get("total_tokens"),
                        "estimated_total_tokens": usage.get("estimated_tokens_without_usage"),
                        "usage_source": usage.get("usage_source", "unknown"),
                        "provider_request_count": len(calls) if result.accounting else None,
                        "retry_count": sum(
                            call.retry_number > 0 and not call.is_fallback for call in calls
                        ) if result.accounting else None,
                        "provider_failure_count": sum(str(call.status) != "SUCCEEDED" for call in calls) if result.accounting else None,
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
