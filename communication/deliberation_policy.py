import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config.settings import (
    MAX_DELIBERATION_MESSAGES,
    MAX_DELIBERATION_MESSAGES_PER_AGENT,
    MAX_DELIBERATION_ROUNDS,
    MAX_EXCHANGES_PER_CLAIM,
)
from models.deliberation_models import (
    DeliberationReport,
    DeliberationTerminationReason,
    DeliberationTrigger,
    DeliberationTriggerType,
)
from models.schemas import ClaimKind, TaskStatus
from models.task_models import TaskPlan, TaskResult
from models.verification_models import Claim
from utils.logger import get_logger
from verification.claim_extractor import classify_claim_kind

logger = get_logger("deliberation_policy")

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|%|qps|tps|gbps)?\b", re.IGNORECASE)
_NEGATION_TERMS = {"not", "no", "never", "without", "cannot", "can't", "isn't", "doesn't", "fails"}
_STOP_WORDS = {
    "a", "an", "and", "are", "be", "by", "for", "from", "in", "is", "of", "on",
    "or", "that", "the", "to", "with", "this", "will", "would", "should", "can",
}


def temporary_claim_reference(execution_id: str, subtask_id: str, index: int) -> str:
    """Return the stable pre-ledger reference used by deliberation triggers."""
    return f"claim:{execution_id}:{subtask_id}:{index}"


class DeliberationPolicy:
    """Deterministically identify relevant, bounded deliberation triggers.

    This class does not publish messages, invoke agents, call an LLM, or change
    verification state. Temporary claim references map to ledger claims later by
    execution ID, subtask ID, and original claim index.
    """

    def __init__(
        self,
        max_rounds: int = MAX_DELIBERATION_ROUNDS,
        max_exchanges_per_claim: int = MAX_EXCHANGES_PER_CLAIM,
        max_messages_per_execution: int = MAX_DELIBERATION_MESSAGES,
        max_messages_per_agent: int = MAX_DELIBERATION_MESSAGES_PER_AGENT,
    ):
        if min(
            max_rounds,
            max_exchanges_per_claim,
            max_messages_per_execution,
            max_messages_per_agent,
        ) < 1:
            raise ValueError("Deliberation limits must be positive.")
        self.max_rounds = max_rounds
        self.max_exchanges_per_claim = max_exchanges_per_claim
        self.max_messages_per_execution = max_messages_per_execution
        self.max_messages_per_agent = max_messages_per_agent

    def evaluate(
        self,
        execution_id: str,
        plan: TaskPlan,
        task_results: Sequence[TaskResult],
        *,
        claims: Optional[Sequence[Claim]] = None,
        round_number: int = 1,
        existing_triggers: Optional[Sequence[DeliberationTrigger]] = None,
        current_message_count: int = 0,
        exchanges_by_claim: Optional[Dict[str, int]] = None,
    ) -> DeliberationReport:
        """Evaluate one bounded policy pass and return serializable triggers."""
        triggers = self.evaluate_triggers(
            execution_id,
            plan,
            task_results,
            claims=claims,
            round_number=round_number,
            existing_triggers=existing_triggers,
            current_message_count=current_message_count,
            exchanges_by_claim=exchanges_by_claim,
        )
        termination = (
            DeliberationTerminationReason.NO_RELEVANT_TRIGGER
            if not triggers
            else None
        )
        return DeliberationReport(
            execution_id=execution_id,
            triggered=bool(triggers),
            triggers=triggers,
            termination_reason=termination,
            total_messages=current_message_count + len(triggers),
        )

    def evaluate_triggers(
        self,
        execution_id: str,
        plan: TaskPlan,
        task_results: Sequence[TaskResult],
        *,
        claims: Optional[Sequence[Claim]] = None,
        round_number: int = 1,
        existing_triggers: Optional[Sequence[DeliberationTrigger]] = None,
        current_message_count: int = 0,
        exchanges_by_claim: Optional[Dict[str, int]] = None,
    ) -> List[DeliberationTrigger]:
        if round_number > self.max_rounds or current_message_count >= self.max_messages_per_execution:
            return []

        completed = [
            result for result in task_results
            if result.status == TaskStatus.COMPLETED and not result.error
        ]
        if len(completed) < 1:
            return []

        result_by_subtask = {result.subtask_id: result for result in completed}
        subtask_by_id = {subtask.id: subtask for subtask in plan.subtasks}
        existing_keys = {
            self._trigger_key(trigger.trigger_type, trigger.claim_reference, trigger.source_agent_id, trigger.target_agent_id)
            for trigger in (existing_triggers or [])
        }
        exchanges_by_claim = exchanges_by_claim or {}
        triggers: List[DeliberationTrigger] = []
        agent_counts: Counter[str] = Counter()

        def add_trigger(
            trigger_type: DeliberationTriggerType,
            source: TaskResult,
            target: TaskResult,
            claim_reference: Optional[str],
            reason: str,
            claim_text: Optional[str] = None,
        ) -> None:
            if len(triggers) + current_message_count >= self.max_messages_per_execution:
                return
            if agent_counts[source.agent_id] >= self.max_messages_per_agent:
                return
            if claim_reference and exchanges_by_claim.get(claim_reference, 0) >= self.max_exchanges_per_claim:
                return
            key = self._trigger_key(trigger_type, claim_reference, source.agent_id, target.agent_id)
            if key in existing_keys or any(
                self._trigger_key(item.trigger_type, item.claim_reference, item.source_agent_id, item.target_agent_id) == key
                for item in triggers
            ):
                return
            source_subtask = subtask_by_id.get(source.subtask_id)
            target_subtask = subtask_by_id.get(target.subtask_id)
            if source_subtask and target.subtask_id in source_subtask.dependencies:
                target_rationale = f"Target is a declared dependency ({target.subtask_id}) of the source subtask."
            elif target_subtask and source.subtask_id in target_subtask.dependencies:
                target_rationale = f"Target subtask ({target.subtask_id}) declares the source as a dependency."
            else:
                overlap = self._overlap(
                    claim_text or reason,
                    target.role + " " + (target_subtask.objective if target_subtask else "")
                    + " " + " ".join(target.findings),
                )
                target_rationale = f"Target selected by relevant scope overlap ({overlap} shared terms)."
            triggers.append(
                DeliberationTrigger(
                    execution_id=execution_id,
                    trigger_type=trigger_type,
                    claim_reference=claim_reference,
                    source_agent_id=source.agent_id,
                    target_agent_id=target.agent_id,
                    source_subtask_id=source.subtask_id,
                    target_subtask_id=target.subtask_id,
                    reason=reason,
                    target_selection_rationale=target_rationale,
                    related_claim_text=claim_text,
                    round_number=round_number,
                )
            )
            agent_counts[source.agent_id] += 1

        claim_entries = self._claim_entries(execution_id, completed, claims)

        # Related contradictory claims only: overlap plus numeric or polarity conflict.
        for left_index, left in enumerate(claim_entries):
            for right in claim_entries[left_index + 1:]:
                if left["result"].agent_id == right["result"].agent_id:
                    continue
                if not self._related(left["text"], right["text"]):
                    continue
                if not self._contradictory(left["text"], right["text"]):
                    continue
                add_trigger(
                    DeliberationTriggerType.CONFLICTING_CLAIMS,
                    left["result"],
                    right["result"],
                    left["reference"],
                    "Related claims have conflicting numeric or polarity assertions.",
                    left["text"],
                )

        # A clarification can contradict a claim without itself becoming a claim.
        response_results = [
            result for result in completed if ":response:" in result.subtask_id
        ]
        for response_result in response_results:
            for clarification in response_result.findings:
                for entry in claim_entries:
                    other_result = entry["result"]
                    if other_result.agent_id == response_result.agent_id:
                        continue
                    if self._related(clarification, entry["text"]) and self._contradictory(
                        clarification, entry["text"]
                    ):
                        add_trigger(
                            DeliberationTriggerType.CONFLICTING_CLAIMS,
                            response_result,
                            other_result,
                            entry["reference"],
                            "A deliberation clarification conflicts with an existing claim.",
                            clarification,
                        )

        # Evidence and assumption triggers are targeted to a relevant dependent peer.
        for entry in claim_entries:
            result = entry["result"]
            kind = entry["kind"]
            target = self._relevant_target(result, completed, plan, entry["text"])
            if target is None:
                continue
            has_verified_external_evidence = any(
                isinstance(item, dict)
                and item.get("verified_external") is True
                and item.get("source_type") in {
                    "TOOL_RETRIEVED",
                    "BENCHMARK_DATA",
                    "USER_PROVIDED_ARTIFACT",
                }
                for item in result.evidence_items
            )
            if kind == ClaimKind.EMPIRICAL_RESULT and not has_verified_external_evidence:
                add_trigger(
                    DeliberationTriggerType.UNSUPPORTED_EMPIRICAL_CLAIM,
                    result,
                    target,
                    entry["reference"],
                    "Empirical-looking claim has no specialist evidence item available for inspection.",
                    entry["text"],
                )

        for result in completed:
            target = self._relevant_target(result, completed, plan, " ".join(result.assumptions))
            if target is not None:
                target_subtask = subtask_by_id.get(target.subtask_id)
                target_scope = (
                    f"{target_subtask.objective} {target_subtask.instructions}"
                    if target_subtask is not None
                    else target.role
                )
                for assumption in result.assumptions:
                    if self._related(assumption, target_scope + " " + target.role):
                        add_trigger(
                            DeliberationTriggerType.UNSUPPORTED_ASSUMPTION,
                            result,
                            target,
                            None,
                            "Material assumption overlaps a related specialist objective without independent support.",
                            assumption,
                        )

            for question in result.unresolved_questions:
                question_target = self._relevant_target(result, completed, plan, question)
                if question_target is not None:
                    add_trigger(
                        DeliberationTriggerType.CRITICAL_UNRESOLVED_QUESTION,
                        result,
                        question_target,
                        None,
                        "Unresolved question is relevant to another completed specialist's scope.",
                        question,
                    )

        # A dependency gap is only emitted when both sides completed but the upstream output is empty.
        for subtask in plan.subtasks:
            dependent = result_by_subtask.get(subtask.id)
            if dependent is None:
                continue
            for dependency_id in subtask.dependencies:
                upstream = result_by_subtask.get(dependency_id)
                if upstream is not None and not (upstream.output.strip() or upstream.findings or upstream.claims):
                    add_trigger(
                        DeliberationTriggerType.DEPENDENCY_INFORMATION_GAP,
                        dependent,
                        upstream,
                        None,
                        f"Dependency {dependency_id} completed without a usable deliverable.",
                        None,
                    )

        return triggers

    @staticmethod
    def _trigger_key(
        trigger_type: DeliberationTriggerType,
        claim_reference: Optional[str],
        source_agent_id: str,
        target_agent_id: str,
    ) -> Tuple[str, Optional[str], str, str]:
        return (str(trigger_type), claim_reference, source_agent_id, target_agent_id)

    @classmethod
    def _claim_entries(
        cls,
        execution_id: str,
        results: Sequence[TaskResult],
        claims: Optional[Sequence[Claim]],
    ) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        if claims:
            for index, claim in enumerate(claims):
                result = next((item for item in results if item.subtask_id == claim.subtask_id), None)
                if result is None:
                    continue
                entries.append({
                    "result": result,
                    "text": claim.claim_text,
                    "kind": claim.claim_kind,
                    "reference": claim.claim_id,
                })
            return entries

        for result in results:
            for index, text in enumerate(result.claims):
                entries.append({
                    "result": result,
                    "text": text,
                    "kind": classify_claim_kind(text, result.assumptions),
                    "reference": temporary_claim_reference(execution_id, result.subtask_id, index),
                })
        return entries

    @classmethod
    def _relevant_target(
        cls,
        source: TaskResult,
        results: Sequence[TaskResult],
        plan: TaskPlan,
        text: str,
    ) -> Optional[TaskResult]:
        source_subtask = next((item for item in plan.subtasks if item.id == source.subtask_id), None)
        dependencies = set(source_subtask.dependencies if source_subtask else [])
        candidates = [result for result in results if result.agent_id != source.agent_id]
        dependent_candidates = [
            result for result in candidates
            if result.subtask_id in dependencies
            or source.subtask_id in {
                item.id for item in plan.subtasks if result.subtask_id in item.dependencies
            }
        ]
        ranked = dependent_candidates or candidates
        ranked = sorted(
            ranked,
            key=lambda result: cls._overlap(text, result.output + " " + " ".join(result.findings) + " " + result.role),
            reverse=True,
        )
        if not ranked or cls._overlap(text, ranked[0].output + " " + " ".join(ranked[0].findings) + " " + ranked[0].role) == 0:
            return None
        return ranked[0]

    @classmethod
    def _related(cls, left: str, right: str) -> bool:
        return cls._overlap(left, right) >= 2

    @classmethod
    def _overlap(cls, left: str, right: str) -> int:
        left_tokens = cls._tokens(left)
        right_tokens = cls._tokens(right)
        return len(left_tokens & right_tokens)

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {
            token.casefold()
            for token in _TOKEN_PATTERN.findall(text)
            if token.casefold() not in _STOP_WORDS and len(token) > 2
        }

    @classmethod
    def _contradictory(cls, left: str, right: str) -> bool:
        left_numbers = set(_NUMBER_PATTERN.findall(left.casefold()))
        right_numbers = set(_NUMBER_PATTERN.findall(right.casefold()))
        left_negative = bool(cls._tokens(left) & _NEGATION_TERMS)
        right_negative = bool(cls._tokens(right) & _NEGATION_TERMS)
        return bool(left_numbers and right_numbers and left_numbers != right_numbers) or left_negative != right_negative


__all__ = ["DeliberationPolicy", "temporary_claim_reference"]
