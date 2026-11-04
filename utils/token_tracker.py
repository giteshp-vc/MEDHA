import asyncio
import contextvars
import math
import uuid
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple

from config.settings import (
    CLAIM_EXTRACTION_TOKEN_RESERVE,
    MAX_INPUT_TOKENS_PER_CALL,
    MAX_OUTPUT_TOKENS_PER_CALL,
    MAX_TOKENS_PER_TASK,
    SYNTHESIS_TOKEN_RESERVE,
    VERIFICATION_TOKEN_RESERVE,
)

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
        self._cached_tokens: int = 0
        self._request_count: int = 0
        self._failed_request_count: int = 0
        self._in_flight_requests: int = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Conservatively estimate tokens from UTF-8 bytes; this is not a provider tokenizer."""
        return math.ceil(len(text.encode("utf-8")) / 3) if text else 0

    @asynccontextmanager
    async def task_budget(
        self,
        max_tokens: Optional[int] = None,
        essential_reserves: Optional[Dict[str, int]] = None,
    ):
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
        async with self._lock:
            if _active_call_slot.get():
                if self._in_flight_requests > 0:
                    self._in_flight_requests -= 1
                _active_call_slot.set(False)
            self._failed_request_count += 1

    def record_failure_sync(self) -> None:
        """Record a failed API request synchronously."""
        self._failed_request_count += 1

    async def get_summary(self) -> Dict[str, Any]:
        """Return the current token usage summary."""
        async with self._lock:
            return self._build_dict()

    def get_summary_sync(self) -> Dict[str, Any]:
        """Return current summary synchronously."""
        return self._build_dict()

    def _build_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "prompt_tokens": self._prompt_tokens,
            "candidate_tokens": self._candidate_tokens,
            "total_tokens": self._total_tokens,
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
        async with self._lock:
            self._prompt_tokens = 0
            self._candidate_tokens = 0
            self._total_tokens = 0
            self._cached_tokens = 0
            self._request_count = 0
            self._failed_request_count = 0
            self._in_flight_requests = 0

    def check_call_budget(self, max_calls: int) -> None:
        """Synchronous budget check (accounting for completed and in-flight requests)."""
        if max_calls > 0:
            current_calls = self._in_flight_requests + self._request_count + self._failed_request_count
            if current_calls >= max_calls:
                from utils.retry import LLMBudgetExceededError

                raise LLMBudgetExceededError(
                    f"Central LLM call budget exceeded: {current_calls}/{max_calls} requests executed or in-flight."
                )
