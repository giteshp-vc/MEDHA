import asyncio
import json
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence

from communication.message import Message
from communication.message_bus import MessageBus
from communication.deliberation_policy import DeliberationPolicy, temporary_claim_reference
from communication.shared_context import SharedContext
from config.settings import (
    DELIBERATION_TIMEOUT_SECONDS,
    MAX_DELIBERATION_MESSAGES,
    MAX_DELIBERATION_MESSAGES_PER_AGENT,
    MAX_DELIBERATION_ROUNDS,
    MAX_EXCHANGES_PER_CLAIM,
    MAX_DELIBERATION_TOKENS,
)
from core.agent_factory import AgentFactory
from models.deliberation_models import (
    DeliberationExchange,
    DeliberationReport,
    DeliberationResponsePayload,
    DeliberationResponseType,
    DeliberationRound,
    DeliberationTerminationReason,
    DeliberationTrigger,
    DeliberationTriggerType,
    SuppressedDeliberationTrigger,
)
from models.schemas import MessageType, TaskStatus
from models.task_models import TaskPlan, TaskResult
from utils.retry import LLMBudgetExceededError, LLMTokenBudgetExceededError
from utils.token_tracker import TokenTracker


_RESPONSE_MESSAGE_TYPES = {
    DeliberationResponseType.CLARIFICATION: MessageType.CLARIFICATION,
    DeliberationResponseType.EVIDENCE_RESPONSE: MessageType.EVIDENCE_RESPONSE,
    DeliberationResponseType.AGREEMENT: MessageType.AGREEMENT,
    DeliberationResponseType.DISAGREEMENT: MessageType.DISAGREEMENT,
    DeliberationResponseType.HANDOFF: MessageType.HANDOFF,
}
_REQUEST_MESSAGE_TYPES = {
    DeliberationTriggerType.CONFLICTING_CLAIMS: MessageType.CHALLENGE,
    DeliberationTriggerType.UNSUPPORTED_EMPIRICAL_CLAIM: MessageType.EVIDENCE_REQUEST,
    DeliberationTriggerType.UNSUPPORTED_ASSUMPTION: MessageType.CLARIFICATION,
    DeliberationTriggerType.CRITICAL_UNRESOLVED_QUESTION: MessageType.CLARIFICATION,
    DeliberationTriggerType.DEPENDENCY_INFORMATION_GAP: MessageType.EVIDENCE_REQUEST,
}
_RELEVANCE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_RELEVANCE_STOP_WORDS = {
    "and", "are", "for", "from", "has", "have", "how", "into", "is", "of", "the",
    "this", "that", "to", "was", "were", "what", "when", "where", "which", "with",
}


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


_EMPTY_CLAIM_REFERENCES = {"", "none", "null", "n/a", "na"}


def _normalize_claim_reference(value: Optional[str]) -> str:
    return str(value or "").strip().strip("[]()<>`'\"").strip().lower()


def _same_claim_reference(response_reference: Optional[str], trigger_reference: Optional[str]) -> bool:
    """Tolerate model echo drift; an absent/'none' response reference never conflicts."""
    response_value = _normalize_claim_reference(response_reference)
    if response_value in _EMPTY_CLAIM_REFERENCES:
        return True
    return response_value == _normalize_claim_reference(trigger_reference)


class DeliberationCoordinator:
    """Run bounded targeted specialist responses without rerunning specialist tasks."""

    def __init__(
        self,
        execution_id: str,
        task_results: Sequence[TaskResult],
        plan: TaskPlan,
        agent_factory: AgentFactory,
        message_bus: MessageBus,
        shared_context: SharedContext,
        token_tracker: TokenTracker,
        policy: Optional[DeliberationPolicy] = None,
        max_rounds: int = MAX_DELIBERATION_ROUNDS,
        max_exchanges_per_claim: int = MAX_EXCHANGES_PER_CLAIM,
        max_messages: int = MAX_DELIBERATION_MESSAGES,
        max_messages_per_agent: int = MAX_DELIBERATION_MESSAGES_PER_AGENT,
        max_tokens: int = MAX_DELIBERATION_TOKENS,
        timeout_seconds: float = DELIBERATION_TIMEOUT_SECONDS,
    ):
        if min(
            max_rounds,
            max_exchanges_per_claim,
            max_messages,
            max_messages_per_agent,
            max_tokens,
        ) < 1:
            raise ValueError("Deliberation coordinator limits must be positive.")
        if timeout_seconds <= 0:
            raise ValueError("Deliberation timeout must be positive.")
        self.execution_id = execution_id
        self.task_results = list(task_results)
        self.plan = plan
        self.agent_factory = agent_factory
        self.message_bus = message_bus
        self.shared_context = shared_context
        self.token_tracker = token_tracker
        self.policy = policy or DeliberationPolicy(
            max_rounds=max_rounds,
            max_exchanges_per_claim=max_exchanges_per_claim,
            max_messages_per_execution=max_messages,
            max_messages_per_agent=max_messages_per_agent,
        )
        self.max_rounds = max_rounds
        self.max_exchanges_per_claim = max_exchanges_per_claim
        self.max_messages = max_messages
        self.max_messages_per_agent = max_messages_per_agent
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._tokens_used = 0
        self._token_source = "unknown"
        self._started_at = 0.0
        self._termination_hint: Optional[DeliberationTerminationReason] = None
        self._agent_message_counts: Counter[str] = Counter()
        self._exchanges_by_claim: Counter[str] = Counter()
        self._seen_correlations: set[tuple[str, Optional[str], str, str]] = set()
        self._processed_trigger_keys: set[tuple[str, Optional[str], str, str]] = set()
        self._has_run = False

    async def run(self) -> DeliberationReport:
        """Evaluate triggers and execute bounded response exchanges sequentially."""
        summary = await self.token_tracker.get_summary()
        if "local_task_token_budget" in summary:
            return await self._run_with_active_budget()
        async with self.token_tracker.task_budget():
            return await self._run_with_active_budget()

    async def _run_with_active_budget(self) -> DeliberationReport:
        """Execute deliberation within the caller's or coordinator-owned task budget."""
        if self._has_run:
            raise RuntimeError("A DeliberationCoordinator instance can only run once.")
        self._has_run = True
        self._started_at = time.perf_counter()
        report = DeliberationReport(execution_id=self.execution_id)
        try:
            failed_results = [
                result for result in self.task_results
                if _enum_value(result.status) != TaskStatus.COMPLETED.value or result.error
            ]
            if failed_results:
                report.termination_reason = DeliberationTerminationReason.AGENT_FAILURE
                report.diagnostics.extend(
                    f"Deliberation skipped because subtask {result.subtask_id} did not complete: "
                    f"{result.error or result.status}"
                    for result in failed_results
                )
                return report

            triggers = self.policy.evaluate_triggers(
                self.execution_id,
                self.plan,
                self.task_results,
                round_number=1,
            )
            if not triggers:
                report.termination_reason = DeliberationTerminationReason.NO_RELEVANT_TRIGGER
                return report

            report.triggered = True
            report.triggers.extend(triggers)
            self._processed_trigger_keys.update(self._trigger_key(trigger) for trigger in triggers)
            pending = list(triggers)
            for round_number in range(1, self.max_rounds + 1):
                if not pending or self._termination_hint is not None:
                    break
                if self._timed_out():
                    self._termination_hint = DeliberationTerminationReason.TIMEOUT
                    break
                pending = await self._run_round(report, pending, round_number)

            if self._termination_hint is not None:
                report.termination_reason = self._termination_hint
            elif pending and len(report.rounds) >= self.max_rounds:
                report.termination_reason = DeliberationTerminationReason.MAX_ROUNDS_REACHED
            elif report.unresolved_disagreements:
                report.termination_reason = DeliberationTerminationReason.UNRESOLVED_DISAGREEMENT
            elif report.unresolved_questions:
                report.termination_reason = DeliberationTerminationReason.UNRESOLVED_QUESTIONS
            else:
                report.termination_reason = DeliberationTerminationReason.RESOLVED
        except asyncio.CancelledError:
            report.termination_reason = DeliberationTerminationReason.TASK_CANCELLED
            report.diagnostics.append("Deliberation task was cancelled.")
            return report
        except asyncio.TimeoutError:
            report.termination_reason = DeliberationTerminationReason.TIMEOUT
            report.diagnostics.append("Deliberation response execution timed out.")
        finally:
            report.total_messages = sum(
                1
                for message in self.message_bus.get_history(task_id=self.execution_id)
                if message.metadata.get("deliberation_exchange")
            )
            report.total_tokens = self._tokens_used
            if self._token_source == "estimated":
                report.diagnostics.append("Deliberation token usage used conservative local estimates.")
        return report

    async def _run_round(
        self,
        report: DeliberationReport,
        triggers: Sequence[DeliberationTrigger],
        round_number: int,
    ) -> List[DeliberationTrigger]:
        round_record = DeliberationRound(
            execution_id=self.execution_id,
            round_number=round_number,
        )
        report.rounds.append(round_record)
        messages_before = self._message_count()
        followups: List[DeliberationTrigger] = []

        for trigger in triggers:
            if self._termination_hint is not None:
                round_record.termination_reason = self._termination_hint
                break
            if self._timed_out():
                round_record.termination_reason = DeliberationTerminationReason.TIMEOUT
                self._termination_hint = DeliberationTerminationReason.TIMEOUT
                break
            if self._message_count() >= self.max_messages:
                round_record.termination_reason = DeliberationTerminationReason.MESSAGE_BUDGET_EXHAUSTED
                self._termination_hint = DeliberationTerminationReason.MESSAGE_BUDGET_EXHAUSTED
                break
            if (
                self._agent_message_counts[trigger.source_agent_id] >= self.max_messages_per_agent
                or self._agent_message_counts[trigger.target_agent_id] >= self.max_messages_per_agent
            ):
                self._termination_hint = DeliberationTerminationReason.MESSAGE_BUDGET_EXHAUSTED
                continue
            if trigger.claim_reference and self._exchanges_by_claim[trigger.claim_reference] >= self.max_exchanges_per_claim:
                self._termination_hint = DeliberationTerminationReason.MAX_EXCHANGES_REACHED
                round_record.termination_reason = DeliberationTerminationReason.MAX_EXCHANGES_REACHED
                continue

            exchange = await self._execute_exchange(trigger, report)
            round_record.exchange_ids.append(exchange.exchange_id)
            report.exchanges.append(exchange)
            if trigger.claim_reference:
                self._exchanges_by_claim[trigger.claim_reference] += 1
            if (
                trigger.trigger_type == DeliberationTriggerType.CRITICAL_UNRESOLVED_QUESTION
                and trigger.related_claim_text
                and exchange.response_message_id
                and not exchange.unresolved_questions
            ):
                report.unresolved_questions = [
                    question for question in report.unresolved_questions
                    if question != trigger.related_claim_text
                ]
            if (
                round_number < self.max_rounds
                and exchange.response_message_id
                and (
                    exchange.introduced_claims
                    or exchange.clarifications
                    or exchange.assumptions
                    or exchange.evidence_references
                    or exchange.unresolved_questions
                )
            ):
                followups.extend(
                    self._reevaluate_response(trigger, exchange, report, round_number + 1)
                )
            elif (
                round_number >= self.max_rounds
                and exchange.response_message_id
                and exchange.unresolved_questions
            ):
                round_record.termination_reason = DeliberationTerminationReason.MAX_ROUNDS_REACHED
                self._termination_hint = DeliberationTerminationReason.MAX_ROUNDS_REACHED

        round_record.message_count = self._message_count() - messages_before
        round_record.token_count = self._tokens_used
        return followups

    async def _execute_exchange(
        self,
        trigger: DeliberationTrigger,
        report: DeliberationReport,
    ) -> DeliberationExchange:
        exchange = DeliberationExchange(
            exchange_id=trigger.exchange_id,
            execution_id=self.execution_id,
            correlation_id=trigger.correlation_id,
            claim_reference=trigger.claim_reference,
            trigger_type=trigger.trigger_type,
            round_number=trigger.round_number,
            initiator_agent_id=trigger.source_agent_id,
            target_agent_id=trigger.target_agent_id,
        )
        if any(
            message.metadata.get("deliberation_exchange")
            and message.metadata.get("exchange_id") == trigger.exchange_id
            for message in self.message_bus.get_history(task_id=self.execution_id)
        ):
            exchange.diagnostic = "Duplicate deliberation exchange was suppressed."
            report.diagnostics.append(exchange.diagnostic)
            return exchange
        correlation_key = (
            trigger.correlation_id,
            trigger.claim_reference,
            trigger.source_agent_id,
            trigger.target_agent_id,
        )
        if correlation_key in self._seen_correlations:
            exchange.diagnostic = "Repeated correlation/claim exchange was suppressed."
            report.diagnostics.append(exchange.diagnostic)
            return exchange
        self._seen_correlations.add(correlation_key)
        if self._message_count() + 2 > self.max_messages:
            exchange.diagnostic = "Deliberation message budget cannot fit a request/response pair."
            report.diagnostics.append(exchange.diagnostic)
            self._termination_hint = DeliberationTerminationReason.MESSAGE_BUDGET_EXHAUSTED
            return exchange
        if (
            self._agent_message_counts[trigger.source_agent_id] + 1 > self.max_messages_per_agent
            or self._agent_message_counts[trigger.target_agent_id] + 1 > self.max_messages_per_agent
        ):
            exchange.diagnostic = "Per-agent deliberation message budget cannot fit a request/response pair."
            report.diagnostics.append(exchange.diagnostic)
            self._termination_hint = DeliberationTerminationReason.MESSAGE_BUDGET_EXHAUSTED
            return exchange
        source_agent = self.agent_factory.get_agent(trigger.source_agent_id)
        target_agent = self.agent_factory.get_agent(trigger.target_agent_id)
        if source_agent is None or target_agent is None:
            exchange.diagnostic = "Source or target agent identity was not found in AgentFactory."
            report.diagnostics.append(exchange.diagnostic)
            self._termination_hint = DeliberationTerminationReason.AGENT_FAILURE
            return exchange

        request = Message(
            sender_id=source_agent.agent_id,
            recipient_id=target_agent.agent_id,
            task_id=self.execution_id,
            execution_id=self.execution_id,
            subtask_id=trigger.source_subtask_id,
            message_type=_REQUEST_MESSAGE_TYPES[trigger.trigger_type],
            content=self._request_content(trigger),
            correlation_id=trigger.correlation_id,
            reply_to_message_id=trigger.reply_to_message_id,
            claim_references=[trigger.claim_reference] if trigger.claim_reference else [],
            metadata={
                "deliberation_exchange": True,
                "exchange_id": trigger.exchange_id,
                "trigger_type": _enum_value(trigger.trigger_type),
                "round_number": trigger.round_number,
            },
        )
        try:
            await self.message_bus.publish(request)
            self._agent_message_counts[source_agent.agent_id] += 1
            exchange.request_message_id = request.message_id
            await self.shared_context.add_communication(request.model_dump(mode="json"))
        except Exception as error:
            exchange.diagnostic = f"Request publication failed: {type(error).__name__}: {error}"
            report.diagnostics.append(exchange.diagnostic)
            if "message limit reached" in str(error).lower():
                self._termination_hint = DeliberationTerminationReason.MESSAGE_BUDGET_EXHAUSTED
            else:
                self._termination_hint = DeliberationTerminationReason.AGENT_FAILURE
            return exchange

        remaining_seconds = max(0.01, self.timeout_seconds - (time.perf_counter() - self._started_at))
        prompt = self._response_prompt(trigger, request)
        system_instruction = (
            "Respond only to the targeted deliberation request. Preserve uncertainty. "
            "Do not claim verification status and do not invent evidence."
        )
        schema_text = json.dumps(DeliberationResponsePayload.model_json_schema(), sort_keys=True)
        input_estimate = self.token_tracker.estimate_tokens(prompt + system_instruction + schema_text)
        before = await self.token_tracker.get_summary()
        remaining_task_tokens = self._remaining_task_tokens(before)
        remaining_tokens = min(self.max_tokens - self._tokens_used, remaining_task_tokens)
        output_limit = min(self.token_tracker.max_output_tokens_per_call, remaining_tokens - input_estimate)
        if input_estimate > self.token_tracker.max_input_tokens_per_call or output_limit < 1:
            exchange.diagnostic = (
                f"Deliberation token budget exhausted before response dispatch "
                f"(estimated_input={input_estimate}, remaining_task={remaining_task_tokens}, "
                f"remaining_deliberation={self.max_tokens - self._tokens_used})."
            )
            report.diagnostics.append(exchange.diagnostic)
            self._termination_hint = DeliberationTerminationReason.TOKEN_BUDGET_EXHAUSTED
            return exchange

        estimate_if_unreconciled = True
        try:
            with self.token_tracker.operation("deliberation"):
                with self.token_tracker.call_context(
                    execution_id=self.execution_id,
                    agent_id=target_agent.agent_id,
                    agent_role=target_agent.role,
                    stage="deliberation",
                ):
                    response = await asyncio.wait_for(
                        target_agent.client.generate_structured(
                            prompt=prompt,
                            response_schema=DeliberationResponsePayload,
                            system_instruction=system_instruction,
                            model=target_agent.model_name,
                            temperature=target_agent.temperature,
                            timeout_seconds=remaining_seconds,
                            max_tokens=output_limit,
                        ),
                        timeout=remaining_seconds,
                    )
        except asyncio.CancelledError:
            exchange.diagnostic = "Targeted response was cancelled."
            report.diagnostics.append(exchange.diagnostic)
            raise
        except asyncio.TimeoutError as error:
            exchange.diagnostic = f"Targeted response timed out: {error}"
            report.diagnostics.append(exchange.diagnostic)
            self._termination_hint = DeliberationTerminationReason.TIMEOUT
            return exchange
        except (LLMBudgetExceededError, LLMTokenBudgetExceededError) as error:
            exchange.diagnostic = f"Deliberation token budget exhausted: {error}"
            report.diagnostics.append(exchange.diagnostic)
            self._termination_hint = DeliberationTerminationReason.TOKEN_BUDGET_EXHAUSTED
            estimate_if_unreconciled = False
            return exchange
        except Exception as error:
            exchange.diagnostic = f"Targeted response failed: {type(error).__name__}: {error}"
            report.diagnostics.append(exchange.diagnostic)
            self._termination_hint = DeliberationTerminationReason.AGENT_FAILURE
            return exchange
        finally:
            after = await self.token_tracker.get_summary()
            self._reconcile_tokens(
                before,
                after,
                input_estimate + output_limit,
                estimate_if_unreconciled=estimate_if_unreconciled,
            )

        if self._tokens_used > self.max_tokens:
            diagnostic = (
                f"Actual or conservatively estimated deliberation usage ({self._tokens_used}) exceeded "
                f"MAX_DELIBERATION_TOKENS={self.max_tokens}; no further responses will be requested."
            )
            report.diagnostics.append(diagnostic)
            self._termination_hint = DeliberationTerminationReason.TOKEN_BUDGET_EXHAUSTED

        validation_error = self._validate_response(
            trigger,
            response,
            target_agent.agent_id,
            source_agent.agent_id,
            request.message_id,
        )
        if validation_error:
            exchange.diagnostic = validation_error
            report.diagnostics.append(validation_error)
            self._termination_hint = DeliberationTerminationReason.AGENT_FAILURE
            return exchange
        response = response.model_copy(update={"claim_reference": trigger.claim_reference})

        response_message_type = _RESPONSE_MESSAGE_TYPES.get(
            DeliberationResponseType(response.response_type)
        )
        if response_message_type is None:
            exchange.diagnostic = f"Unsupported deliberation response type: {response.response_type}"
            report.diagnostics.append(exchange.diagnostic)
            return exchange
        response_message = Message(
            sender_id=target_agent.agent_id,
            recipient_id=source_agent.agent_id,
            task_id=self.execution_id,
            execution_id=self.execution_id,
            subtask_id=trigger.target_subtask_id,
            message_type=response_message_type,
            content=response.content,
            correlation_id=trigger.correlation_id,
            reply_to_message_id=request.message_id,
            claim_references=[response.claim_reference or trigger.claim_reference] if (response.claim_reference or trigger.claim_reference) else [],
            evidence_references=response.evidence_references,
            metadata={
                "deliberation_exchange": True,
                "exchange_id": trigger.exchange_id,
                "round_number": trigger.round_number,
                "useful_new_information": response.useful_new_information,
                "disagreement": response.disagreement,
            },
        )
        if self._message_count() >= self.max_messages:
            exchange.diagnostic = "Deliberation message budget exhausted before response publication."
            report.diagnostics.append(exchange.diagnostic)
            self._termination_hint = DeliberationTerminationReason.MESSAGE_BUDGET_EXHAUSTED
            return exchange
        try:
            await self.message_bus.publish(response_message)
            self._agent_message_counts[target_agent.agent_id] += 1
            await self.shared_context.add_communication(response_message.model_dump(mode="json"))
        except Exception as error:
            exchange.diagnostic = f"Response publication failed: {type(error).__name__}: {error}"
            report.diagnostics.append(exchange.diagnostic)
            if "message limit reached" in str(error).lower():
                self._termination_hint = DeliberationTerminationReason.MESSAGE_BUDGET_EXHAUSTED
            else:
                self._termination_hint = DeliberationTerminationReason.AGENT_FAILURE
            return exchange

        exchange.response_message_id = response_message.message_id
        exchange.response_type = response.response_type
        exchange.response_content = response.content
        exchange.useful_new_information = response.useful_new_information
        exchange.introduced_claims = list(response.new_claims)
        exchange.clarifications = list(response.clarifications)
        exchange.assumptions = list(response.assumptions)
        response_subtask_id = (
            f"{trigger.target_subtask_id or 'unknown'}:response:{response_message.message_id}"
        )
        exchange.introduced_claim_references = [
            temporary_claim_reference(self.execution_id, response_subtask_id, index)
            for index in range(len(exchange.introduced_claims))
        ]
        exchange.evidence_references = list(response.evidence_references)
        exchange.unresolved_questions = list(response.unresolved_questions)
        exchange.disagreement = response.disagreement or response.response_type == DeliberationResponseType.DISAGREEMENT
        if exchange.disagreement:
            report.unresolved_disagreements.append(response.content)
        report.unresolved_questions.extend(response.unresolved_questions)
        if exchange.evidence_references:
            report.diagnostics.append(
                f"Evidence references from response {response_message.message_id} remain unverified: "
                f"{', '.join(exchange.evidence_references)}"
            )
        return exchange

    def _reevaluate_response(
        self,
        parent_trigger: DeliberationTrigger,
        exchange: DeliberationExchange,
        report: DeliberationReport,
        round_number: int,
    ) -> List[DeliberationTrigger]:
        """Run the deterministic policy on original results plus one response delta."""
        target_agent = self.agent_factory.get_agent(parent_trigger.target_agent_id)
        if target_agent is None or exchange.response_message_id is None:
            return []

        response_subtask_id = (
            f"{parent_trigger.target_subtask_id or 'unknown'}:response:"
            f"{exchange.response_message_id}"
        )
        introduced_claims = exchange.introduced_claims
        response_result = TaskResult(
            task_id=self.execution_id,
            subtask_id=response_subtask_id,
            agent_id=parent_trigger.target_agent_id,
            role=target_agent.role,
            status=TaskStatus.COMPLETED,
            output=exchange.response_content or "",
            findings=exchange.clarifications,
            claims=introduced_claims,
            evidence_items=[
                {
                    "reference": reference,
                    "source": "deliberation_response_unverified",
                    "verified_external": False,
                }
                for reference in exchange.evidence_references
            ],
            assumptions=exchange.assumptions,
            unresolved_questions=exchange.unresolved_questions,
        )
        affected_results = self._affected_results(parent_trigger, exchange)
        candidates = self.policy.evaluate_triggers(
            self.execution_id,
            self.plan,
            [response_result, *affected_results],
            round_number=round_number,
            # The coordinator records suppressed candidates itself to preserve reasons.
            current_message_count=0,
            exchanges_by_claim=None,
        )
        accepted: List[DeliberationTrigger] = []
        for policy_candidate in candidates:
            candidate = policy_candidate.model_copy(deep=True)
            candidate.originating_response_message_id = exchange.response_message_id
            candidate.correlation_id = parent_trigger.correlation_id
            candidate.reply_to_message_id = exchange.response_message_id
            reason = self._suppression_reason(candidate, round_number)
            if reason:
                report.suppressed_triggers.append(
                    SuppressedDeliberationTrigger(
                        execution_id=self.execution_id,
                        originating_response_message_id=exchange.response_message_id,
                        trigger_type=candidate.trigger_type,
                        claim_reference=candidate.claim_reference,
                        source_agent_id=candidate.source_agent_id,
                        target_agent_id=candidate.target_agent_id,
                        reason=reason,
                        target_selection_rationale=candidate.target_selection_rationale,
                    )
                )
                continue
            key = self._trigger_key(candidate)
            self._processed_trigger_keys.add(key)
            report.triggers.append(candidate)
            accepted.append(candidate)
        return accepted

    def _affected_results(
        self,
        trigger: DeliberationTrigger,
        exchange: DeliberationExchange,
    ) -> List[TaskResult]:
        """Select only the exchange participants and relevant dependency/scope neighbors."""
        focus_subtasks = {
            subtask_id for subtask_id in (trigger.source_subtask_id, trigger.target_subtask_id)
            if subtask_id
        }
        delta_text = " ".join(
            [
                trigger.related_claim_text or "",
                exchange.response_content or "",
                *exchange.introduced_claims,
                *exchange.clarifications,
                *exchange.assumptions,
                *exchange.unresolved_questions,
                *exchange.evidence_references,
            ]
        )
        delta_tokens = self._relevance_tokens(delta_text)
        subtask_map = {subtask.id: subtask for subtask in self.plan.subtasks}
        affected: List[TaskResult] = []
        for result in self.task_results:
            if result.agent_id in (trigger.source_agent_id, trigger.target_agent_id):
                affected.append(result)
                continue

            subtask = subtask_map.get(result.subtask_id)
            related_by_dependency = bool(
                subtask
                and set(subtask.dependencies) & focus_subtasks
            ) or any(
                result.subtask_id in focus_subtask.dependencies
                for focus_subtask in self.plan.subtasks
                if focus_subtask.id in focus_subtasks
            )
            content = " ".join(
                [
                    result.role,
                    subtask.objective if subtask else "",
                    subtask.instructions if subtask else "",
                    result.output,
                    *result.claims,
                    *result.findings,
                    *result.assumptions,
                    *result.unresolved_questions,
                ]
            )
            overlap = len(delta_tokens & self._relevance_tokens(content))
            if related_by_dependency or overlap >= 2:
                affected.append(result)
        return affected

    @staticmethod
    def _relevance_tokens(text: str) -> set[str]:
        return {
            token.casefold()
            for token in _RELEVANCE_TOKEN_PATTERN.findall(text)
            if len(token) > 2 and token.casefold() not in _RELEVANCE_STOP_WORDS
        }

    def _suppression_reason(
        self,
        trigger: DeliberationTrigger,
        round_number: int,
    ) -> Optional[str]:
        key = self._trigger_key(trigger)
        if key in self._processed_trigger_keys:
            return "Duplicate trigger for the same type, claim, source, and target was already processed."
        correlation = (
            trigger.correlation_id,
            trigger.claim_reference,
            trigger.source_agent_id,
            trigger.target_agent_id,
        )
        if correlation in self._seen_correlations:
            return "Repeated correlation cycle was suppressed."
        if self._termination_hint == DeliberationTerminationReason.TOKEN_BUDGET_EXHAUSTED:
            return "Deliberation token budget was exhausted before this trigger could run."
        if self._termination_hint == DeliberationTerminationReason.TIMEOUT or self._timed_out():
            return "Deliberation time budget was exhausted before this trigger could run."
        if self._tokens_used >= self.max_tokens:
            return "Maximum deliberation token budget was exhausted."
        task_budget = self.token_tracker.get_summary_sync().get("local_task_token_budget")
        if isinstance(task_budget, dict) and self._remaining_task_tokens(
            self.token_tracker.get_summary_sync()
        ) <= 0:
            return "Local task token budget was exhausted before this trigger could run."
        if round_number > self.max_rounds:
            return "Maximum deliberation round count was reached."
        if self._message_count() + 2 > self.max_messages:
            return "Global deliberation message budget cannot fit another request/response pair."
        if (
            self._agent_message_counts[trigger.source_agent_id] + 1 > self.max_messages_per_agent
            or self._agent_message_counts[trigger.target_agent_id] + 1 > self.max_messages_per_agent
        ):
            return "Per-agent deliberation message budget is exhausted."
        if trigger.claim_reference and self._exchanges_by_claim[trigger.claim_reference] >= self.max_exchanges_per_claim:
            return "Maximum exchanges for this claim were reached."
        return None

    @staticmethod
    def _trigger_key(trigger: DeliberationTrigger) -> tuple[str, Optional[str], str, str]:
        return (
            _enum_value(trigger.trigger_type),
            trigger.claim_reference,
            trigger.source_agent_id,
            trigger.target_agent_id,
        )

    @staticmethod
    def _request_content(trigger: DeliberationTrigger) -> str:
        return (
            f"{_enum_value(trigger.trigger_type)}: {trigger.reason} "
            f"Claim or finding: {trigger.related_claim_text or 'See the relevant execution context.'}"
        )

    def _response_prompt(self, trigger: DeliberationTrigger, request: Message) -> str:
        relevant_result = next(
            (result for result in self.task_results if result.agent_id == trigger.target_agent_id),
            None,
        )
        context = relevant_result.output[:1000] if relevant_result else "No target deliverable was available."
        source_result = next(
            (result for result in self.task_results if result.agent_id == trigger.source_agent_id),
            None,
        )
        evidence = []
        for result in (source_result, relevant_result):
            if result is None:
                continue
            evidence.extend(
                item.get("reference", str(item)) if isinstance(item, dict) else str(item)
                for item in result.evidence_items
            )
        evidence = list(dict.fromkeys(evidence))
        return (
            f"Execution ID: {self.execution_id}\n"
            f"Exchange ID: {trigger.exchange_id}\n"
            f"Correlation ID: {trigger.correlation_id}\n"
            f"Reply to message ID: {request.message_id}\n"
            f"Your agent ID (use as sender_agent_id): {trigger.target_agent_id}\n"
            f"Requesting agent ID (use as recipient_agent_id): {trigger.source_agent_id}\n"
            f"Trigger: {_enum_value(trigger.trigger_type)}\n"
            f"Claim reference: {trigger.claim_reference or 'none'}\n"
            f"Evidence references: {', '.join(evidence) or 'none'}\n"
            f"Request: {self._request_content(trigger)}\n"
            f"Target specialist context:\n{context}\n\n"
            "Return a concise structured response. Only provide useful clarification, explicitly referenced "
            "evidence, a disagreement, or a handoff. Consensus is not evidence."
        )

    def _remaining_task_tokens(self, summary: Dict[str, object]) -> int:
        budget = summary.get("local_task_token_budget")
        if isinstance(budget, dict):
            remaining = int(budget.get("remaining_tokens", 0) or 0)
            protected = budget.get("protected_essential_reserves", {})
            if isinstance(protected, dict):
                remaining -= sum(int(value or 0) for value in protected.values())
            return max(0, remaining)

        accounted = int(summary.get("accounted_total_tokens", summary.get("total_tokens", 0)) or 0)
        return max(0, self.token_tracker.max_tokens_per_task - accounted)

    @staticmethod
    def _validate_response(
        trigger: DeliberationTrigger,
        response: DeliberationResponsePayload,
        expected_sender: str,
        expected_recipient: str,
        request_message_id: str,
    ) -> Optional[str]:
        if response.execution_id != trigger.execution_id:
            return "Response execution identity did not match the trigger."
        if response.exchange_id != trigger.exchange_id:
            return "Response exchange identity did not match the trigger."
        if response.correlation_id != trigger.correlation_id:
            return "Response correlation identity did not match the trigger."
        if response.sender_agent_id != expected_sender:
            return "Response sender identity did not match the target agent instance."
        if response.recipient_agent_id != expected_recipient:
            return "Response recipient identity did not match the initiating agent."
        if response.reply_to_message_id and response.reply_to_message_id != request_message_id:
            return "Response reply target did not match the originating request message."
        if not _same_claim_reference(response.claim_reference, trigger.claim_reference):
            return "Response claim reference did not match the trigger."
        return None

    def _reconcile_tokens(
        self,
        before: Dict[str, object],
        after: Dict[str, object],
        estimate: int,
        estimate_if_unreconciled: bool = True,
    ) -> None:
        before_total = int(before.get("accounted_total_tokens", before.get("total_tokens", 0)) or 0)
        after_total = int(after.get("accounted_total_tokens", after.get("total_tokens", 0)) or 0)
        delta = max(0, after_total - before_total)
        if delta:
            self._tokens_used += delta
            provider_delta = int(after.get("provider_usage_requests", 0) or 0) - int(
                before.get("provider_usage_requests", 0) or 0
            )
            estimated_delta = int(after.get("estimated_usage_requests", 0) or 0) - int(
                before.get("estimated_usage_requests", 0) or 0
            )
            observed_source = (
                "mixed" if provider_delta and estimated_delta
                else "estimated" if estimated_delta
                else "provider" if provider_delta
                else "unknown"
            )
            if self._token_source == "unknown":
                self._token_source = observed_source
            elif observed_source != "unknown" and self._token_source != observed_source:
                self._token_source = "mixed"
        elif estimate_if_unreconciled:
            self._tokens_used += estimate
            self._token_source = "mixed" if self._token_source == "provider" else "estimated"

    def _message_count(self) -> int:
        return sum(
            1
            for message in self.message_bus.get_history(task_id=self.execution_id)
            if message.metadata.get("deliberation_exchange")
        )

    def _timed_out(self) -> bool:
        return (time.perf_counter() - self._started_at) >= self.timeout_seconds


__all__ = ["DeliberationCoordinator"]
