import asyncio
import contextvars
import math
import uuid
from contextlib import asynccontextmanager
from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterator, Optional, Tuple

from config.settings import (
    CLAIM_EXTRACTION_TOKEN_RESERVE,
    MAX_INPUT_TOKENS_PER_CALL,
    MAX_OUTPUT_TOKENS_PER_CALL,
    MAX_TOKENS_PER_TASK,
    SYNTHESIS_TOKEN_RESERVE,
    VERIFICATION_TOKEN_RESERVE,
)
from models.provider_call import (
    AccountingSnapshot,
    AccountingContext,
    ProviderCallRecord,
    ProviderCallStatus,
    ProviderUsageSource,
)

MAX_RETAINED_PROVIDER_CALLS = 4096

_active_call_slot: contextvars.ContextVar[bool] = contextvars.ContextVar("_active_call_slot", default=False)
_active_task_budget: contextvars.ContextVar[Optional["_TaskTokenBudget"]] = contextvars.ContextVar(
    "_active_task_token_budget", default=None
)
_active_operation: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_active_token_operation", default="general"
)
_active_token_reservation: contextvars.ContextVar[Optional[Tuple["_TaskTokenBudget", str]]] = (
    contextvars.ContextVar("_active_token_reservation", default=None)
)
_active_tracker_overrides: contextvars.ContextVar[Optional[Dict[int, "TokenTracker"]]] = (
    contextvars.ContextVar("_active_tracker_overrides", default=None)
)
_active_accounting_scope: contextvars.ContextVar[Optional["AccountingScope"]] = (
    contextvars.ContextVar("_active_accounting_scope", default=None)
)
_active_call_attribution: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("_active_call_attribution", default=None)
)


@dataclass
class AccountingScope:
    """One context-local usage/call-record boundary, optionally backed by an isolated tracker."""

    tracker: "TokenTracker"
    context: AccountingContext
    started_at: float = field(default_factory=perf_counter)
    records: Any = field(default_factory=lambda: deque(maxlen=MAX_RETAINED_PROVIDER_CALLS))
    dropped_provider_call_records: int = 0

    @property
    def provider_calls(self) -> list[ProviderCallRecord]:
        return list(self.records)

    def snapshot(self) -> AccountingSnapshot:
        return AccountingSnapshot(
            context=self.context,
            duration_seconds=round(perf_counter() - self.started_at, 6),
            token_usage=self.tracker.get_summary_sync(),
            provider_calls=self.provider_calls,
            dropped_provider_call_records=self.dropped_provider_call_records,
        )


class ProviderCallRecorder:
    """Mutable in-flight handle used only to fill a serializable ProviderCallRecord."""

    def __init__(self, record: ProviderCallRecord, started_at: float):
        self.record = record
        self.started_at = started_at
        self.enabled = True

    def set_provider_usage(
        self,
        prompt_tokens: Optional[int],
        candidate_tokens: Optional[int],
        total_tokens: Optional[int],
    ) -> None:
        if not self.enabled:
            return
        self.record.prompt_tokens = prompt_tokens
        self.record.candidate_tokens = candidate_tokens
        self.record.total_tokens = total_tokens
        if any(value is not None for value in (prompt_tokens, candidate_tokens, total_tokens)):
            self.record.usage_source = (
                ProviderUsageSource.MIXED
                if self.record.estimated_candidate_tokens is not None
                else ProviderUsageSource.PROVIDER_REPORTED
            )

    def set_estimated_candidate_tokens(self, candidate_tokens: Optional[int]) -> None:
        if not self.enabled or candidate_tokens is None:
            return
        self.record.estimated_candidate_tokens = candidate_tokens
        estimated_prompt_tokens = self.record.estimated_prompt_tokens
        self.record.estimated_total_tokens = (
            (estimated_prompt_tokens or 0) + candidate_tokens
            if estimated_prompt_tokens is not None
            else candidate_tokens
        )
        if self.record.usage_source == ProviderUsageSource.PROVIDER_REPORTED:
            self.record.usage_source = ProviderUsageSource.MIXED
        elif self.record.usage_source == ProviderUsageSource.UNAVAILABLE:
            self.record.usage_source = ProviderUsageSource.LOCALLY_ESTIMATED

    def set_response_metadata(self, response: Any) -> None:
        if not self.enabled:
            return
        response_model = getattr(response, "model", None)
        if isinstance(response_model, str) and response_model.strip():
            self.record.response_model = response_model.strip()
        response_revision = getattr(response, "model_version", None)
        if isinstance(response_revision, str) and response_revision.strip():
            self.record.model_version = response_revision.strip()
            self.record.model_version_source = "provider_response"

    def finish(self, status: ProviderCallStatus, error: Optional[BaseException] = None) -> None:
        if not self.enabled:
            return
        self.record.duration_seconds = round(perf_counter() - self.started_at, 6)
        self.record.status = status
        if error is not None:
            classified_error = error
            if isinstance(error, Exception):
                try:
                    from utils.retry import classify_error

                    _, classified_error = classify_error(error)
                except Exception:
                    classified_error = error
            error_name = type(classified_error).__name__
            status_code = getattr(classified_error, "status_code", None)
            if status_code is None:
                status_code = getattr(classified_error, "code", None)
            self.record.error_category = error_name
            self.record.sanitized_error = (
                f"{error_name} (status {status_code})"
                if isinstance(status_code, int)
                else error_name
            )


@dataclass
class _TaskTokenBudget:
    tracker_id: int
    max_tokens: int
    essential_reserves: Dict[str, int]
    used_tokens: int = 0
    used_by_category: Dict[str, int] = field(default_factory=dict)
    reservations: Dict[str, Tuple[int, str]] = field(default_factory=dict)
    released_categories: set[str] = field(default_factory=set)
    violations: list[str] = field(default_factory=list)
    estimated_tokens_without_usage: int = 0
    usage_diagnostics: list[str] = field(default_factory=list)


class TokenTracker:
    """Thread-safe and async-safe token usage, request accumulator, and concurrency-safe call budget manager."""

    def __init__(
        self,
        name: str = "global",
        max_tokens_per_task: int = MAX_TOKENS_PER_TASK,
        max_input_tokens_per_call: int = MAX_INPUT_TOKENS_PER_CALL,
        max_output_tokens_per_call: int = MAX_OUTPUT_TOKENS_PER_CALL,
        essential_token_reserves: Optional[Dict[str, int]] = None,
    ):
        self.name = name
        self.max_tokens_per_task = max_tokens_per_task
        self.max_input_tokens_per_call = max_input_tokens_per_call
        self.max_output_tokens_per_call = max_output_tokens_per_call
        self.essential_token_reserves = (
            {
                "claim_extraction": CLAIM_EXTRACTION_TOKEN_RESERVE,
                "verification": VERIFICATION_TOKEN_RESERVE,
                "synthesis": SYNTHESIS_TOKEN_RESERVE,
            }
            if essential_token_reserves is None
            else dict(essential_token_reserves)
        )
        self._prompt_tokens: int = 0
        self._candidate_tokens: int = 0
        self._total_tokens: int = 0
        self._estimated_tokens_without_usage: int = 0
        self._provider_usage_requests: int = 0
        self._estimated_usage_requests: int = 0
        self._cached_tokens: int = 0
        self._request_count: int = 0
        self._failed_request_count: int = 0
        self._in_flight_requests: int = 0
        self._lock = asyncio.Lock()
        self._provider_call_records = deque(maxlen=MAX_RETAINED_PROVIDER_CALLS)
        self._dropped_provider_call_records = 0

    def _effective_tracker(self) -> "TokenTracker":
        overrides = _active_tracker_overrides.get() or {}
        return overrides.get(id(self), self)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Conservatively estimate tokens from UTF-8 bytes; this is not a provider tokenizer."""
        return math.ceil(len(text.encode("utf-8")) / 3) if text else 0

    @asynccontextmanager
    async def accounting_session(
        self,
        context: Optional[AccountingContext] = None,
        isolate_usage: bool = True,
    ):
        """Create a task-local call-record scope and optionally isolate all tracker counters."""
        parent = self._effective_tracker()
        child = (
            TokenTracker(
                name=f"{parent.name}:accounting_scope",
                max_tokens_per_task=parent.max_tokens_per_task,
                max_input_tokens_per_call=parent.max_input_tokens_per_call,
                max_output_tokens_per_call=parent.max_output_tokens_per_call,
                essential_token_reserves=parent.essential_token_reserves,
            )
            if isolate_usage
            else parent
        )
        scope = AccountingScope(tracker=child, context=context or AccountingContext())
        overrides = dict(_active_tracker_overrides.get() or {})
        if isolate_usage:
            overrides[id(self)] = child
        override_token = _active_tracker_overrides.set(overrides)
        scope_token = _active_accounting_scope.set(scope)
        previous_attribution = dict(_active_call_attribution.get() or {})
        previous_attribution.update(scope.context.model_dump(exclude_none=True))
        attribution_token = _active_call_attribution.set(previous_attribution)
        try:
            yield scope
        finally:
            _active_call_attribution.reset(attribution_token)
            _active_accounting_scope.reset(scope_token)
            _active_tracker_overrides.reset(override_token)

    @contextmanager
    def call_context(self, **attribution: Any):
        """Temporarily attach execution/agent attribution to provider calls in this task."""
        merged = dict(_active_call_attribution.get() or {})
        merged.update({key: value for key, value in attribution.items() if value is not None})
        token = _active_call_attribution.set(merged)
        try:
            yield
        finally:
            _active_call_attribution.reset(token)

    @asynccontextmanager
    async def provider_call(
        self,
        *,
        attempt_number: int,
        retry_number: int = 0,
        is_fallback: bool = False,
        operator_model_version: Optional[str] = None,
        model_identifier_policy: str = "provider_may_resolve_mutable_alias",
        provider: str,
        model: Optional[str],
        model_version: Optional[str],
        temperature: Optional[float],
        input_token_limit: Optional[int],
        output_token_limit: Optional[int],
        timeout_seconds: Optional[float],
        estimated_prompt_tokens: Optional[int] = None,
        estimated_candidate_tokens: Optional[int] = None,
    ):
        """Record one provider transport attempt without allowing telemetry to affect its result."""
        effective = self._effective_tracker()
        if effective is not self:
            recorder = None
            try:
                async with effective.provider_call(
                    attempt_number=attempt_number,
                    retry_number=retry_number,
                    is_fallback=is_fallback,
                    operator_model_version=operator_model_version,
                    model_identifier_policy=model_identifier_policy,
                    provider=provider,
                    model=model,
                    model_version=model_version,
                    temperature=temperature,
                    input_token_limit=input_token_limit,
                    output_token_limit=output_token_limit,
                    timeout_seconds=timeout_seconds,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    estimated_candidate_tokens=estimated_candidate_tokens,
                ) as recorder:
                    yield recorder
            finally:
                if recorder is not None:
                    try:
                        if len(self._provider_call_records) == MAX_RETAINED_PROVIDER_CALLS:
                            self._dropped_provider_call_records += 1
                        self._provider_call_records.append(recorder.record)
                    except Exception:
                        pass
            return

        attribution = dict(_active_call_attribution.get() or {})
        scope = _active_accounting_scope.get()
        try:
            record = ProviderCallRecord(
                attempt_number=attempt_number,
                retry_number=retry_number,
                is_fallback=is_fallback,
                operator_model_version=operator_model_version,
                model_identifier_policy=model_identifier_policy,
                model_version_source="unavailable",
                timeout_source=("explicit_request" if timeout_seconds is not None else "provider_sdk_default"),
                provider=provider,
                model=model,
                model_version=model_version,
                temperature=temperature,
                input_token_limit=input_token_limit,
                output_token_limit=output_token_limit,
                timeout_seconds=timeout_seconds,
                estimated_prompt_tokens=estimated_prompt_tokens,
                estimated_candidate_tokens=estimated_candidate_tokens,
                usage_source=(
                    ProviderUsageSource.LOCALLY_ESTIMATED
                    if estimated_prompt_tokens is not None or estimated_candidate_tokens is not None
                    else ProviderUsageSource.UNAVAILABLE
                ),
                experiment_id=attribution.get("experiment_id"),
                task_block_id=attribution.get("task_block_id"),
                trial_id=attribution.get("trial_id"),
                treatment_id=attribution.get("treatment_id"),
                execution_id=attribution.get("execution_id"),
                stage=attribution.get("stage") or {
                    "specialist": "specialist_execution",
                }.get(_active_operation.get(), _active_operation.get()),
                agent_id=attribution.get("agent_id"),
                agent_role=attribution.get("agent_role"),
            )
            recorder = ProviderCallRecorder(record, perf_counter())
            if scope is not None:
                if len(scope.records) == MAX_RETAINED_PROVIDER_CALLS:
                    scope.dropped_provider_call_records += 1
                scope.records.append(record)
            else:
                if len(self._provider_call_records) == MAX_RETAINED_PROVIDER_CALLS:
                    self._dropped_provider_call_records += 1
                self._provider_call_records.append(record)
        except Exception:
            # Telemetry setup is best-effort; it must never prevent the provider request.
            recorder = None

        try:
            yield recorder
        except asyncio.CancelledError as error:
            self._finish_provider_recorder(recorder, ProviderCallStatus.CANCELLED, error)
            raise
        except BaseException as error:
            status = (
                ProviderCallStatus.TIMED_OUT
                if isinstance(error, (asyncio.TimeoutError, TimeoutError))
                else ProviderCallStatus.FAILED
            )
            self._finish_provider_recorder(recorder, status, error)
            raise
        else:
            self._finish_provider_recorder(recorder, ProviderCallStatus.SUCCEEDED)

    @staticmethod
    def _finish_provider_recorder(
        recorder: Optional[ProviderCallRecorder],
        status: ProviderCallStatus,
        error: Optional[BaseException] = None,
    ) -> None:
        if recorder is None:
            return
        try:
            recorder.finish(status, error)
        except Exception:
            # Telemetry failures must not alter provider outcomes or turn a successful call into failure.
            try:
                recorder.record.status = status
                recorder.record.duration_seconds = round(perf_counter() - recorder.started_at, 6)
                if error is not None:
                    recorder.record.error_category = type(error).__name__
                    recorder.record.sanitized_error = type(error).__name__
            except Exception:
                return

    def get_provider_call_records(
        self,
        execution_id: Optional[str] = None,
    ) -> list[ProviderCallRecord]:
        effective = self._effective_tracker()
        if effective is not self:
            return effective.get_provider_call_records(execution_id)
        scope = _active_accounting_scope.get()
        if scope is not None and scope.tracker is self:
            records = list(scope.records)
        else:
            records = list(self._provider_call_records)
        if execution_id is not None:
            records = [record for record in records if record.execution_id == execution_id]
        return records

    def get_provider_call_records_dropped(self) -> int:
        effective = self._effective_tracker()
        if effective is not self:
            return effective.get_provider_call_records_dropped()
        scope = _active_accounting_scope.get()
        if scope is not None and scope.tracker is self:
            return scope.dropped_provider_call_records
        return self._dropped_provider_call_records

    def get_active_accounting_snapshot(self) -> Optional[AccountingSnapshot]:
        effective = self._effective_tracker()
        scope = _active_accounting_scope.get()
        if scope is None or scope.tracker is not effective:
            return None
        try:
            return scope.snapshot()
        except Exception:
            return None

    @asynccontextmanager
    async def task_budget(
        self,
        max_tokens: Optional[int] = None,
        essential_reserves: Optional[Dict[str, int]] = None,
    ):
        effective = self._effective_tracker()
        if effective is not self:
            async with effective.task_budget(max_tokens, essential_reserves) as budget:
                yield budget
            return

        limit = self.max_tokens_per_task if max_tokens is None else max_tokens
        reserves = dict(
            self.essential_token_reserves if essential_reserves is None else essential_reserves
        )
        if limit < 1 or any(value < 0 for value in reserves.values()) or sum(reserves.values()) >= limit:
            raise ValueError("Task token budget must exceed its non-negative essential reserves.")

        budget = _TaskTokenBudget(
            tracker_id=id(self),
            max_tokens=limit,
            essential_reserves=reserves,
        )
        token = _active_task_budget.set(budget)
        try:
            yield budget
        finally:
            _active_task_budget.reset(token)

    @contextmanager
    def operation(self, category: str, release_reserve_on_exit: bool = True) -> Iterator[None]:
        token = _active_operation.set(category)
        try:
            yield
        finally:
            _active_operation.reset(token)
            if release_reserve_on_exit:
                self.release_operation_reserve(category)

    def release_operation_reserve(self, category: str) -> None:
        effective = self._effective_tracker()
        if effective is not self:
            effective.release_operation_reserve(category)
            return
        budget = _active_task_budget.get()
        if budget is not None and budget.tracker_id == id(self):
            budget.released_categories.add(category)

    @asynccontextmanager
    async def track_call(
        self,
        max_calls: int = 0,
        estimated_input_tokens: int = 0,
        max_output_tokens: int = 0,
    ):
        """Atomically reserve and verify a call slot under concurrency, transitioning it on completion.

        Ensures simultaneous agents cannot exceed MAX_LLM_CALLS_PER_TASK due to a race condition,
        and prevents double-counting between in-flight and completed/failed counts.
        """
        from utils.retry import LLMBudgetExceededError, LLMTokenBudgetExceededError

        effective = self._effective_tracker()
        if effective is not self:
            async with effective.track_call(max_calls, estimated_input_tokens, max_output_tokens):
                yield
            return

        if estimated_input_tokens > self.max_input_tokens_per_call:
            raise LLMTokenBudgetExceededError(
                f"Local input estimate {estimated_input_tokens} exceeds "
                f"MAX_INPUT_TOKENS_PER_CALL={self.max_input_tokens_per_call}; no provider request was sent."
            )
        if max_output_tokens > self.max_output_tokens_per_call:
            raise LLMTokenBudgetExceededError(
                f"Requested output limit {max_output_tokens} exceeds "
                f"MAX_OUTPUT_TOKENS_PER_CALL={self.max_output_tokens_per_call}; no provider request was sent."
            )

        operation = _active_operation.get()
        budget = _active_task_budget.get()
        reservation_id: Optional[str] = None
        estimated_total = estimated_input_tokens + max_output_tokens

        async with self._lock:
            total_allocated = self._in_flight_requests + self._request_count + self._failed_request_count
            if max_calls > 0 and total_allocated >= max_calls:
                raise LLMBudgetExceededError(
                    f"Central LLM call budget exceeded: {total_allocated}/{max_calls} requests executed or in-flight."
                )

            if budget is not None and budget.tracker_id == id(self):
                pending_by_category: Dict[str, int] = {}
                pending_tokens = 0
                for reserved_tokens, category in budget.reservations.values():
                    pending_tokens += reserved_tokens
                    pending_by_category[category] = pending_by_category.get(category, 0) + reserved_tokens

                used_and_pending = budget.used_tokens + pending_tokens
                available_total = budget.max_tokens - used_and_pending
                reserve_remaining = {
                    category: max(
                        0,
                        reserve
                        - budget.used_by_category.get(category, 0)
                        - pending_by_category.get(category, 0),
                    )
                    for category, reserve in budget.essential_reserves.items()
                    if category not in budget.released_categories
                }
                protected_reserve = sum(
                    remaining for category, remaining in reserve_remaining.items() if category != operation
                )
                available_for_operation = max(0, available_total - protected_reserve)
                if estimated_total > available_for_operation:
                    raise LLMTokenBudgetExceededError(
                        f"Local task token budget cannot reserve {estimated_total} estimated tokens for "
                        f"{operation}; {available_for_operation} available after essential reserves. "
                        "This estimate is application-local and does not report the provider's daily quota."
                    )

                reservation_id = uuid.uuid4().hex
                budget.reservations[reservation_id] = (estimated_total, operation)
            self._in_flight_requests += 1

        token = _active_call_slot.set(True)
        reservation_token = _active_token_reservation.set(
            (budget, reservation_id)
            if budget is not None and reservation_id is not None
            else None
        )
        try:
            yield
        finally:
            async with self._lock:
                if budget is not None and reservation_id is not None:
                    unreconciled = budget.reservations.pop(reservation_id, None)
                    if unreconciled is not None:
                        reserved_tokens, category = unreconciled
                        budget.used_tokens += reserved_tokens
                        budget.used_by_category[category] = (
                            budget.used_by_category.get(category, 0) + reserved_tokens
                        )
                        budget.estimated_tokens_without_usage += reserved_tokens
                        self._estimated_tokens_without_usage += reserved_tokens
                        self._estimated_usage_requests += 1
                        budget.usage_diagnostics.append(
                            f"No provider usage metadata was received for a dispatched {category} call; "
                            f"conservatively charged its {reserved_tokens}-token estimate."
                        )
                if _active_call_slot.get():
                    if self._in_flight_requests > 0:
                        self._in_flight_requests -= 1
                    _active_call_slot.set(False)
            _active_call_slot.reset(token)
            _active_token_reservation.reset(reservation_token)

    async def add_usage(
        self,
        prompt_tokens: int = 0,
        candidate_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
        usage_available: bool = True,
    ) -> None:
        """Accumulate token counts from a successful model response.
        
        Atomically transitions an in-flight request to completed without double-counting.
        """
        effective = self._effective_tracker()
        if effective is not self:
            await effective.add_usage(
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                usage_available=usage_available,
            )
            return
        budget_violation: Optional[str] = None
        async with self._lock:
            self._prompt_tokens += max(0, prompt_tokens)
            self._candidate_tokens += max(0, candidate_tokens)
            # If total_tokens is provided by API, use it, else sum
            if total_tokens > 0:
                self._total_tokens += total_tokens
            else:
                self._total_tokens += max(0, prompt_tokens) + max(0, candidate_tokens)
            self._cached_tokens += max(0, cached_tokens)

            actual_tokens = total_tokens if total_tokens > 0 else prompt_tokens + candidate_tokens
            active_reservation = _active_token_reservation.get()
            if active_reservation is not None:
                budget, reservation_id = active_reservation
                reservation = budget.reservations.pop(reservation_id, None)
                if reservation is not None:
                    reserved_tokens, category = reservation
                    usage_reconciled = usage_available and actual_tokens > 0
                    budget_tokens = actual_tokens if usage_reconciled else reserved_tokens
                    if not usage_reconciled:
                        self._estimated_tokens_without_usage += reserved_tokens
                        self._estimated_usage_requests += 1
                        budget.estimated_tokens_without_usage += reserved_tokens
                        budget.usage_diagnostics.append(
                            f"Provider usage metadata was missing or empty for a {category} response; "
                            f"conservatively charged its {reserved_tokens}-token estimate."
                        )

                    pending_other_tokens = 0
                    pending_other_by_category: Dict[str, int] = {}
                    for other_id, (other_tokens, other_category) in budget.reservations.items():
                        if other_id == reservation_id:
                            continue
                        pending_other_tokens += other_tokens
                        pending_other_by_category[other_category] = (
                            pending_other_by_category.get(other_category, 0) + other_tokens
                        )
                    protected_other_reserves = sum(
                        max(
                            0,
                            reserve
                            - budget.used_by_category.get(other_category, 0)
                            - pending_other_by_category.get(other_category, 0),
                        )
                        for other_category, reserve in budget.essential_reserves.items()
                        if other_category != category
                        and other_category not in budget.released_categories
                    )
                    allowed_actual = max(
                        0,
                        budget.max_tokens - budget.used_tokens - pending_other_tokens
                        - protected_other_reserves,
                    )
                    if budget_tokens > allowed_actual:
                        overage = budget_tokens - allowed_actual
                        budget_violation = (
                            f"Actual provider usage ({budget_tokens}) exceeded the reserved local budget "
                            f"available to {category} by {overage} token(s); local task budget is "
                            f"{budget.max_tokens}. Provider daily quota is not observable here."
                        )
                        budget.violations.append(budget_violation)

                    budget.used_tokens += budget_tokens
                    budget.used_by_category[category] = (
                        budget.used_by_category.get(category, 0) + budget_tokens
                    )

            # Atomically transition in-flight reservation to completed
            if _active_call_slot.get():
                if self._in_flight_requests > 0:
                    self._in_flight_requests -= 1
                _active_call_slot.set(False)

            self._request_count += 1
            if usage_available and actual_tokens > 0:
                self._provider_usage_requests += 1

        if budget_violation is not None:
            from utils.retry import LLMTokenBudgetExceededError

            raise LLMTokenBudgetExceededError(budget_violation)

    def add_usage_sync(
        self,
        prompt_tokens: int = 0,
        candidate_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        """Synchronous version for non-async callers."""
        effective = self._effective_tracker()
        if effective is not self:
            effective.add_usage_sync(prompt_tokens, candidate_tokens, total_tokens, cached_tokens)
            return
        self._prompt_tokens += max(0, prompt_tokens)
        self._candidate_tokens += max(0, candidate_tokens)
        if total_tokens > 0:
            self._total_tokens += total_tokens
        else:
            self._total_tokens += max(0, prompt_tokens) + max(0, candidate_tokens)
        self._cached_tokens += max(0, cached_tokens)
        self._request_count += 1

    async def record_failure(self) -> None:
        """Record a failed API request, transitioning in-flight slot if active."""
        effective = self._effective_tracker()
        if effective is not self:
            await effective.record_failure()
            return
        async with self._lock:
            if _active_call_slot.get():
                if self._in_flight_requests > 0:
                    self._in_flight_requests -= 1
                _active_call_slot.set(False)
            self._failed_request_count += 1

    def record_failure_sync(self) -> None:
        """Record a failed API request synchronously."""
        effective = self._effective_tracker()
        if effective is not self:
            effective.record_failure_sync()
            return
        self._failed_request_count += 1

    async def get_summary(self) -> Dict[str, Any]:
        """Return the current token usage summary."""
        effective = self._effective_tracker()
        if effective is not self:
            return await effective.get_summary()
        async with self._lock:
            return self._build_dict()

    def get_summary_sync(self) -> Dict[str, Any]:
        """Return current summary synchronously."""
        effective = self._effective_tracker()
        if effective is not self:
            return effective.get_summary_sync()
        return self._build_dict()

    def _build_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "prompt_tokens": self._prompt_tokens,
            "candidate_tokens": self._candidate_tokens,
            "total_tokens": self._total_tokens,
            "estimated_tokens_without_usage": self._estimated_tokens_without_usage,
            "accounted_total_tokens": self._total_tokens + self._estimated_tokens_without_usage,
            "usage_source": (
                "provider" if self._provider_usage_requests and not self._estimated_usage_requests
                else "estimated" if self._estimated_usage_requests and not self._provider_usage_requests
                else "mixed" if self._provider_usage_requests or self._estimated_usage_requests
                else "unknown"
            ),
            "provider_usage_requests": self._provider_usage_requests,
            "estimated_usage_requests": self._estimated_usage_requests,
            "cached_tokens": self._cached_tokens,
            "successful_requests": self._request_count,
            "failed_requests": self._failed_request_count,
            "in_flight_requests": self._in_flight_requests,
            "total_requests": self._request_count + self._failed_request_count,
        }
        budget = _active_task_budget.get()
        if budget is not None and budget.tracker_id == id(self):
            pending_tokens = sum(value for value, _ in budget.reservations.values())
            result["local_task_token_budget"] = {
                "limit_tokens": budget.max_tokens,
                "accounted_tokens": budget.used_tokens,
                "reserved_tokens": pending_tokens,
                "remaining_tokens": max(0, budget.max_tokens - budget.used_tokens - pending_tokens),
                "protected_essential_reserves": {
                    category: max(
                        0,
                        reserve
                        - budget.used_by_category.get(category, 0)
                        - sum(
                            reserved
                            for reserved, reserved_category in budget.reservations.values()
                            if reserved_category == category
                        ),
                    )
                    for category, reserve in budget.essential_reserves.items()
                    if category not in budget.released_categories
                },
                "budget_violations": list(budget.violations),
                "estimated_tokens_without_usage": budget.estimated_tokens_without_usage,
                "usage_diagnostics": list(budget.usage_diagnostics),
                "provider_daily_quota_known": False,
            }
        return result

    async def reset(self) -> None:
        """Reset all tracked counters."""
        effective = self._effective_tracker()
        if effective is not self:
            await effective.reset()
            return
        async with self._lock:
            self._prompt_tokens = 0
            self._candidate_tokens = 0
            self._total_tokens = 0
            self._estimated_tokens_without_usage = 0
            self._provider_usage_requests = 0
            self._estimated_usage_requests = 0
            self._cached_tokens = 0
            self._request_count = 0
            self._failed_request_count = 0
            self._in_flight_requests = 0

    def check_call_budget(self, max_calls: int) -> None:
        """Synchronous budget check (accounting for completed and in-flight requests)."""
        effective = self._effective_tracker()
        if effective is not self:
            effective.check_call_budget(max_calls)
            return
        if max_calls > 0:
            current_calls = self._in_flight_requests + self._request_count + self._failed_request_count
            if current_calls >= max_calls:
                from utils.retry import LLMBudgetExceededError

                raise LLMBudgetExceededError(
                    f"Central LLM call budget exceeded: {current_calls}/{max_calls} requests executed or in-flight."
                )
