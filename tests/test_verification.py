from unittest.mock import AsyncMock, MagicMock
import pytest

from core.llm_client import GeminiClient
from models.schemas import ClaimKind, ClaimStatus, EvidenceSourceType, VerificationDecision
from models.task_models import TaskResult
from models.verification_models import Claim, Evidence, ReviewFeedback
from verification.adversarial_reviewer import AdversarialReviewer
from verification.claim_extractor import ClaimExtractor
from verification.claim_validator import ClaimValidator
from verification.evidence_ledger import EvidenceLedger


def _mock_llm_response(text: str):
    resp = MagicMock()
    resp.text = text
    resp.parsed = None
    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 20
    usage.total_token_count = 30
    usage.cached_content_token_count = 0
    resp.usage_metadata = usage
    return resp


@pytest.mark.asyncio
async def test_claim_extractor_from_result():
    """Verify ClaimExtractor converts pre-identified claims in TaskResult."""
    result = TaskResult(
        subtask_id="T1",
        agent_id="agent_1",
        role="Systems Architect",
        output="Complete architecture report.",
        claims=["Postgres handles 10,000 writes/sec with sharding", "Redis caching reduces DB read load by 80%"],
        assumptions=["Standard SSD NVMe drives used"],
    )
    extractor = ClaimExtractor()
    claims = await extractor.extract_claims_from_result(result, task_id="task_123")

    assert len(claims) == 2
    assert claims[0].originating_agent == "agent_1"
    assert claims[0].originating_role == "Systems Architect"
    assert "10,000 writes/sec" in claims[0].claim_text
    assert claims[0].status == ClaimStatus.PENDING


@pytest.mark.asyncio
async def test_claim_extractor_registers_all_preidentified_claims_without_llm():
    client = MagicMock()
    client.generate_structured = AsyncMock()
    result = TaskResult(
        subtask_id="T4",
        agent_id="agent_reliability",
        role="Reliability Specialist",
        output="Detailed reliability analysis.",
        claims=[f"Reliability assertion number {index} is stated." for index in range(10)],
    )

    claims = await ClaimExtractor(llm_client=client).extract_claims_from_result(result, task_id="exec_test")

    assert len(claims) == 10
    assert all(claim.originating_agent == "agent_reliability" for claim in claims)
    assert all(claim.subtask_id == "T4" for claim in claims)
    assert all(claim.source_provenance[0]["role"] == "Reliability Specialist" for claim in claims)
    client.generate_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_extraction_failure_is_returned_as_diagnostic_without_fabrication():
    from utils.retry import LLMQuotaExhaustedError

    client = MagicMock()
    client.generate_structured = AsyncMock(side_effect=LLMQuotaExhaustedError("provider quota exhausted"))
    result = TaskResult(
        subtask_id="T2",
        agent_id="agent_privacy",
        role="Privacy Specialist",
        output="The report contains prose but no pre-identified claims.",
    )

    outcome = await ClaimExtractor(llm_client=client).extract_claims_with_diagnostics(result)

    assert outcome.claims == []
    assert outcome.source == "failed"
    assert isinstance(outcome.error, LLMQuotaExhaustedError)


@pytest.mark.asyncio
async def test_claims_discovered_before_extraction_error_are_preserved():
    client = MagicMock()
    client.generate_structured = AsyncMock(
        return_value=MagicMock(
            claims=["First discovered assertion is preserved.", None],
            assumptions=[],
        )
    )
    result = TaskResult(
        subtask_id="T5",
        agent_id="agent_test",
        role="Test Specialist",
        output="Source report",
    )

    outcome = await ClaimExtractor(llm_client=client).extract_claims_with_diagnostics(result)

    assert len(outcome.claims) == 1
    assert outcome.claims[0].claim_text == "First discovered assertion is preserved."
    assert outcome.error is not None


@pytest.mark.asyncio
async def test_ledger_deduplicates_claims_and_merges_specialist_provenance():
    ledger = EvidenceLedger(task_id="dedupe_test")
    first = Claim(
        claim_id="c1",
        claim_text="Target latency should remain below 50 ms.",
        originating_agent="agent_arch",
        originating_role="Architect",
        subtask_id="T1",
    )
    duplicate = Claim(
        claim_id="c2",
        claim_text="  target latency should remain below 50 ms. ",
        originating_agent="agent_reliability",
        originating_role="Reliability Specialist",
        subtask_id="T4",
    )

    registered_ids = await ledger.register_claims_batch([first, duplicate])
    claims = await ledger.get_all_claims()

    assert registered_ids == ["c1", "c1"]
    assert len(claims) == 1
    assert {source["subtask_id"] for source in claims[0].source_provenance} == {"T1", "T4"}


@pytest.mark.asyncio
async def test_evidence_ledger_lifecycle():
    """Verify registering, linking, and reporting on claims and evidence in EvidenceLedger."""
    ledger = EvidenceLedger(task_id="task_123")

    claim = Claim(
        claim_id="c1",
        claim_text="AES-256 is quantum-resistant against known Grover attacks within 128-bit security margin",
        originating_agent="sec_agent",
        originating_role="Security Specialist",
    )
    await ledger.register_claim(claim)

    evidence = Evidence(
        evidence_id="e1",
        source_type=EvidenceSourceType.DOMAIN_KNOWLEDGE,
        source_reference="NIST Post-Quantum Cryptography Guidelines",
        supporting_content="Grover's algorithm halves key length; AES-256 retains 128 bits of security.",
        originating_agent="sec_agent",
        associated_claim_ids=["c1"],
    )
    await ledger.register_evidence(evidence)

    # Check evidence association
    linked_ev = await ledger.get_evidence_for_claim("c1")
    assert len(linked_ev) == 1
    assert linked_ev[0].evidence_id == "e1"

    # Update claim status
    await ledger.update_claim_status("c1", ClaimStatus.SUPPORTED, reviewer_feedback="Verified via NIST standards")

    # Generate report
    report = await ledger.generate_report()
    assert report.total_claims == 1
    assert len(report.accepted_claims) == 1
    # Defined formula: Supported / (Supported + Rejected + Needs Revision + Unverifiable) * 100
    assert report.support_rate == 100.0


def test_claim_validator_flags_superlatives():
    """Verify ClaimValidator flags ungrounded absolutes."""
    c_bad = Claim(
        claim_id="c_bad",
        claim_text="Our system is 100% secure and will achieve zero latency under any load.",
        originating_agent="a1",
    )
    is_valid, issues = ClaimValidator.validate_claim_structure(c_bad)
    assert is_valid is False
    assert any("100% secure" in issue for issue in issues)
    assert any("zero latency" in issue for issue in issues)

    c_good = Claim(
        claim_id="c_good",
        claim_text="Database read replicas scale throughput to approximately 25,000 reads per second.",
        originating_agent="a1",
    )
    is_valid_good, issues_good = ClaimValidator.validate_claim_structure(c_good)
    assert is_valid_good is True
    assert len(issues_good) == 0


def test_claim_validator_flags_fabricated_empirical_evidence():
    """Verify ClaimValidator flags empirical results stated without evidence artifacts."""
    c_unbacked = Claim(
        claim_id="c_bench",
        claim_text="Our benchmarks proved that Redis delivers 0.5ms response times across all shards.",
        originating_agent="a1",
        evidence_references=[],  # No evidence attached!
    )
    is_valid, issues = ClaimValidator.validate_claim_structure(c_unbacked)
    assert is_valid is False
    assert any("benchmarks proved" in issue or "empirical test result" in issue for issue in issues)


def test_claim_extractor_distinguishes_targets_assumptions_recommendations_and_results():
    from verification.claim_extractor import classify_claim_kind

    assert classify_claim_kind("Target latency should be below 50 ms.") == ClaimKind.PROPOSED_TARGET
    assert classify_claim_kind("Assumption: the network is reliable.") == ClaimKind.ASSUMPTION
    assert classify_claim_kind("We recommend local edge processing.") == ClaimKind.RECOMMENDATION
    assert classify_claim_kind("We measured 12 ms p95 latency in a pilot.") == ClaimKind.EMPIRICAL_RESULT


def test_claim_kind_prevents_empirical_acceptance_without_external_artifact():
    empirical_claim = Claim(
        claim_id="c_uptime",
        claim_text="The system achieved 99.9% uptime.",
        originating_agent="agent_reliability",
        claim_kind=ClaimKind.EMPIRICAL_RESULT,
    )

    is_valid, issues = ClaimValidator.validate_claim_structure(empirical_claim, evidence_items=[])

    assert is_valid is False
    assert any("no external evidence" in issue for issue in issues)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim_kind",
    [ClaimKind.PROPOSED_TARGET, ClaimKind.ASSUMPTION, ClaimKind.RECOMMENDATION],
)
async def test_reviewer_does_not_accept_non_factual_claim_categories(claim_kind):
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_mock_llm_response(
            '{"reviews":[{"claim_id":"typed_claim","decision":"ACCEPT",'
            '"reasoning":"Looks reasonable.","is_supported_by_evidence":true,'
            '"is_evidence_relevant":true,"logically_valid":true}]}'
        )
    )
    claim = Claim(
        claim_id="typed_claim",
        claim_text="Target uptime should be 99.9%.",
        originating_agent="agent_reliability",
        claim_kind=claim_kind,
    )
    ledger = EvidenceLedger(task_id="typed_claim_test")
    await ledger.register_claim(claim)

    feedback = await AdversarialReviewer(
        llm_client=GeminiClient(api_key="mock_key", client=mock_client)
    ).review_claims(ledger=ledger, task_context="Review performance claims")

    assert feedback[0].decision == VerificationDecision.REQUEST_REVISION
    assert "cannot be accepted as an established factual result" in feedback[0].reasoning


@pytest.mark.asyncio
async def test_support_rate_calculation_reviewed_vs_unreviewed():
    """Verify support rate excludes unreviewed/pending claims and never divides by zero."""
    ledger = EvidenceLedger(task_id="rate_test", max_claims=30)

    # 13 supported claims
    for i in range(1, 14):
        cid = f"c{i}"
        await ledger.register_claim(Claim(claim_id=cid, claim_text=f"Valid claim {i}", originating_agent="a1"))
        await ledger.update_claim_status(cid, ClaimStatus.SUPPORTED)

    # 2 pending (unreviewed) claims
    for i in range(14, 16):
        cid = f"c{i}"
        await ledger.register_claim(Claim(claim_id=cid, claim_text=f"Pending claim {i}", originating_agent="a1"))

    report = await ledger.generate_report()
    assert report.total_claims == 15
    assert len(report.accepted_claims) == 13
    # 13 supported out of 13 REVIEWED claims = 100.0%, NOT 0.0% or 86.7%!
    assert report.support_rate == 100.0

    # Add 1 rejected claim
    await ledger.register_claim(Claim(claim_id="c_rej", claim_text="Invalid claim", originating_agent="a1"))
    await ledger.update_claim_status("c_rej", ClaimStatus.REJECTED)

    report2 = await ledger.generate_report()
    # 13 supported / 14 reviewed * 100 = 92.9%
    assert report2.support_rate == 92.9


@pytest.mark.asyncio
async def test_zero_reviewed_claims_have_undefined_support_rate():
    ledger = EvidenceLedger(task_id="no_reviews", max_claims=10)
    await ledger.register_claim(
        Claim(claim_id="pending", claim_text="A claim awaiting review", originating_agent="a1")
    )

    report = await ledger.generate_report()

    assert report.support_rate is None
    assert report.unsupported_rate is None
    assert len(report.unreviewed_claims) == 1
    assert "No claims were reviewed" in report.summary_message


@pytest.mark.asyncio
async def test_adversarial_reviewer_batch_audit_success():
    """Verify AdversarialReviewer audits multiple claims in a single batch call."""
    mock_client = MagicMock()
    review_batch_json = """{
        "reviews": [
            {
                "claim_id": "c1",
                "decision": "ACCEPT",
                "reasoning": "Backed by empirical evidence.",
                "critique": "",
                "suggested_fix": null,
                "is_supported_by_evidence": true,
                "is_evidence_relevant": true,
                "has_contradictions": false,
                "assumptions_presented_as_facts": false,
                "logically_valid": true
            },
            {
                "claim_id": "c2",
                "decision": "REJECT",
                "reasoning": "Direct contradiction with CAP theorem.",
                "critique": "Physically impossible consistency and availability during partition",
                "suggested_fix": null,
                "is_supported_by_evidence": false,
                "is_evidence_relevant": false,
                "has_contradictions": true,
                "assumptions_presented_as_facts": true,
                "logically_valid": false
            }
        ]
    }"""
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(review_batch_json))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    ledger = EvidenceLedger(task_id="task_batch")
    c1 = Claim(claim_id="c1", claim_text="Cache-aside pattern improves read latency.", originating_agent="a1")
    c2 = Claim(claim_id="c2", claim_text="Our distributed database provides strong consistency with zero latency across WAN.", originating_agent="a2")
    await ledger.register_claim(c1)
    await ledger.register_claim(c2)

    reviewer = AdversarialReviewer(llm_client=gemini_client, batch_size=5)
    feedback_list = await reviewer.review_claims(ledger=ledger, task_context="Architecture audit")

    assert len(feedback_list) == 2
    # Verify exact claim mapping
    assert feedback_list[0].claim_id == "c1"
    assert feedback_list[0].decision == VerificationDecision.ACCEPT
    assert feedback_list[1].claim_id == "c2"
    assert feedback_list[1].decision == VerificationDecision.REJECT

    updated_c1 = await ledger.get_claim("c1")
    assert updated_c1.status == ClaimStatus.SUPPORTED
    updated_c2 = await ledger.get_claim("c2")
    assert updated_c2.status == ClaimStatus.REJECTED


@pytest.mark.asyncio
async def test_batch_reviewer_failure_never_accepts_claims():
    """Verify that a malformed batch response or failure falls back safely and NEVER silently accepts claims."""
    mock_client = MagicMock()
    # Malformed response missing claim evaluations
    bad_batch_json = """{"reviews": []}"""
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(bad_batch_json))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    ledger = EvidenceLedger(task_id="task_fail_safe")
    c1 = Claim(claim_id="c_unverified", claim_text="Critical architecture assertion.", originating_agent="a1")
    await ledger.register_claim(c1)

    reviewer = AdversarialReviewer(llm_client=gemini_client, batch_size=5)
    feedback_list = await reviewer.review_claims(ledger=ledger, task_context="Audit")

    assert len(feedback_list) == 1
    # Under failure, claim MUST NOT be ACCEPTED
    assert feedback_list[0].decision != VerificationDecision.ACCEPT
    assert feedback_list[0].decision in (VerificationDecision.UNVERIFIABLE, VerificationDecision.REQUEST_REVISION)

    claim_in_ledger = await ledger.get_claim("c_unverified")
    assert claim_in_ledger.status != ClaimStatus.SUPPORTED


@pytest.mark.asyncio
async def test_unverifiable_vs_unreviewed_claims_in_support_rate():
    """Verify that unreviewed claims (interrupted audits) are excluded from the denominator,

    while unverifiable claims (reviewed but lacking evidence) are included.
    """
    ledger = EvidenceLedger(task_id="rate_distinction", max_claims=30)

    # 10 supported claims
    for i in range(1, 11):
        cid = f"supp_{i}"
        await ledger.register_claim(Claim(claim_id=cid, claim_text=f"Solid claim {i}", originating_agent="a1"))
        await ledger.update_claim_status(cid, ClaimStatus.SUPPORTED)

    # 2 unverifiable claims (audited by reviewer, found to lack provable evidence)
    for i in range(1, 3):
        cid = f"unver_{i}"
        await ledger.register_claim(Claim(claim_id=cid, claim_text=f"Unverifiable claim {i}", originating_agent="a1"))
        await ledger.update_claim_status(cid, ClaimStatus.UNVERIFIABLE, reviewer_feedback="Lacks verifiable external facts.")

    # 4 unreviewed claims (never audited due to quota/budget exhaustion interruption)
    for i in range(1, 5):
        cid = f"unrev_{i}"
        await ledger.register_claim(Claim(claim_id=cid, claim_text=f"Unreviewed claim {i}", originating_agent="a1"))
        await ledger.update_claim_status(cid, ClaimStatus.UNREVIEWED, reviewer_feedback="Audit interrupted before verification.")

    report = await ledger.generate_report()

    # Total claims registered
    assert report.total_claims == 16
    assert len(report.accepted_claims) == 10
    assert len(report.unverifiable_claims) == 2
    assert len(report.unreviewed_claims) == 4

    # Denominator must ONLY include reviewed claims: 10 supported + 2 unverifiable = 12 reviewed claims
    # Support rate = 10 / 12 = 83.3%
    # If unreviewed claims were erroneously counted as reviewed or UNVERIFIABLE,
    # the denominator would be 16, yielding 10 / 16 = 62.5%.
    assert report.support_rate == 83.3
    # Unsupported rate: (0 rejected + 2 unverifiable) / 12 reviewed = 16.7%
    assert report.unsupported_rate == 16.7
    assert "10 of 12 reviewed claims supported (83.3%)" in report.summary_message
    assert "4 claim(s) unreviewed due to audit interruption" in report.summary_message


def test_evidence_independence_internal_reasoning_vs_external_data():
    """Verify that model reasoning is distinguishable from external evidence and empirical claims require external artifacts."""
    ev_reasoning = Evidence(
        evidence_id="ev_model_deduction",
        source_type=EvidenceSourceType.REASONING_INFERENCE,
        source_reference="Agent analytical reasoning",
        supporting_content="Deduction that Kafka partitions scale linearly based on key hashing.",
        originating_agent="Specialist_1",
    )
    assert ev_reasoning.is_external is False

    ev_domain = Evidence(
        evidence_id="ev_internal_knowledge",
        source_type=EvidenceSourceType.DOMAIN_KNOWLEDGE,
        source_reference="General LLM training data",
        supporting_content="Kafka partition concepts.",
        originating_agent="Specialist_1",
    )
    assert ev_domain.is_external is False

    ev_benchmark = Evidence(
        evidence_id="ev_real_test",
        source_type=EvidenceSourceType.BENCHMARK_DATA,
        source_reference="load_test_results.json",
        supporting_content="Observed p99 latency of 1.2ms under 50k QPS load.",
        originating_agent="BenchmarkRunner",
    )
    assert ev_benchmark.is_external is True

    # Empirical claim: asserts benchmarks demonstrated throughput
    empirical_claim = Claim(
        claim_id="c_empirical",
        claim_text="Our benchmarks demonstrated that Redis achieves 80,000 requests per second.",
        originating_agent="Specialist_1",
        evidence_references=[ev_reasoning.evidence_id],
    )

    # When validated with only internal model reasoning, it MUST be flagged
    is_valid, issues = ClaimValidator.validate_claim_structure(empirical_claim, evidence_items=[ev_reasoning])
    assert is_valid is False
    assert any("asserts empirical test result" in issue for issue in issues)

    # When validated with real benchmark data, it passes structural provenance
    empirical_claim.evidence_references = [ev_benchmark.evidence_id]
    is_valid_with_ext, issues_ext = ClaimValidator.validate_claim_structure(empirical_claim, evidence_items=[ev_benchmark])
    assert is_valid_with_ext is True
    assert len(issues_ext) == 0


@pytest.mark.asyncio
async def test_reviewer_acceptance_does_not_override_empirical_evidence_requirement():
    """Verify that if LLM reviewer falsely ACCEPTs an empirical claim without external evidence,

    AdversarialReviewer's validation layer downgrades the verdict to REQUEST_REVISION.
    """
    mock_client = MagicMock()
    # LLM reviewer attempts to ACCEPT an empirical claim without real evidence
    batch_json = """{
        "reviews": [
            {
                "claim_id": "c_bench",
                "decision": "ACCEPT",
                "reasoning": "The model thinks this benchmark claim looks plausible.",
                "critique": "",
                "suggested_fix": null,
                "is_supported_by_evidence": true,
                "is_evidence_relevant": true,
                "has_contradictions": false,
                "assumptions_presented_as_facts": false,
                "logically_valid": true
            }
        ]
    }"""
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(batch_json))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    ledger = EvidenceLedger(task_id="task_empirical_check")
    # Empirical claim citing only an internal reasoning evidence
    ev = Evidence(
        evidence_id="ev_reasoning_only",
        source_type=EvidenceSourceType.REASONING_INFERENCE,
        source_reference="Internal reasoning",
        supporting_content="Calculated estimate",
        originating_agent="Specialist_1",
    )
    await ledger.register_evidence(ev)

    claim = Claim(
        claim_id="c_bench",
        claim_text="We measured write speed of 500 MB/s across all storage nodes.",
        originating_agent="Specialist_1",
        evidence_references=[ev.evidence_id],
    )
    await ledger.register_claim(claim)

    reviewer = AdversarialReviewer(llm_client=gemini_client, batch_size=5)
    feedbacks = await reviewer.review_claims(ledger=ledger, task_context="Audit")

    assert len(feedbacks) == 1
    # Even though LLM returned ACCEPT, heuristic provenance protection must downgrade it!
    assert feedbacks[0].decision == VerificationDecision.REQUEST_REVISION
    assert "Heuristic Flag" in feedbacks[0].reasoning
    assert "asserts empirical test result" in feedbacks[0].reasoning

    # Ledger state must also reflect NEEDS_REVISION, NOT SUPPORTED
    updated_claim = await ledger.get_claim("c_bench")
    assert updated_claim.status == ClaimStatus.NEEDS_REVISION


def test_plain_user_assertion_cannot_validate_empirical_claim():
    """Verify that a plain user statement/assertion cannot independently validate an empirical claim."""
    user_statement = Evidence(
        evidence_id="ev_user_stmt",
        source_type=EvidenceSourceType.USER_INPUT,
        source_reference="User prompt: 'Our benchmark showed Redis throughput is 100k rps'",
        supporting_content="User states Redis achieves 100,000 rps in their private testing environment.",
        originating_agent="User",
    )
    assert user_statement.is_user_assertion is True
    assert user_statement.is_external is False

    empirical_claim = Claim(
        claim_id="c_redis_emp",
        claim_text="Our benchmarks showed Redis throughput reached 100,000 rps.",
        originating_agent="Specialist_1",
        evidence_references=[user_statement.evidence_id],
    )

    is_valid, issues = ClaimValidator.validate_claim_structure(
        empirical_claim, evidence_items=[user_statement]
    )
    assert is_valid is False
    assert any("asserts empirical test result" in issue for issue in issues)


def test_uploaded_benchmark_report_qualifies_as_evidence():
    """Verify that an uploaded, inspectable benchmark report qualifies as external evidence."""
    benchmark_artifact = Evidence(
        evidence_id="ev_upload_bench",
        source_type=EvidenceSourceType.USER_PROVIDED_ARTIFACT,
        artifact_type="benchmark_report",
        source_identifier="benchmarks/redis_load_2026.json",
        source_reference="Uploaded benchmark output file redis_load_2026.json",
        supporting_content="Measured throughput: 102,400 rps at p99 latency 1.1ms across 4 shards.",
        retrieval_timestamp="2026-11-04T12:00:00Z",
        evidence_location="results.summary.throughput",
        originating_agent="User",
    )
    assert benchmark_artifact.is_user_assertion is False
    assert benchmark_artifact.is_external is True

    empirical_claim = Claim(
        claim_id="c_redis_valid",
        claim_text="Our benchmarks showed Redis throughput reached 100,000 rps.",
        originating_agent="Specialist_1",
        evidence_references=[benchmark_artifact.evidence_id],
    )

    is_valid, issues = ClaimValidator.validate_claim_structure(
        empirical_claim, evidence_items=[benchmark_artifact]
    )
    assert is_valid is True
    assert len(issues) == 0


def test_model_generated_inference_cannot_qualify_as_external_evidence():
    """Verify that model-generated inferences cannot qualify as external evidence for empirical claims."""
    model_inference = Evidence(
        evidence_id="ev_model_calc",
        source_type=EvidenceSourceType.REASONING_INFERENCE,
        source_reference="Specialist analytical derivation",
        supporting_content="Deduced from network bandwidth of 10Gbps that throughput could achieve 80k rps.",
        originating_agent="DatabaseArchitect",
    )
    assert model_inference.is_user_assertion is False
    assert model_inference.is_external is False

    empirical_claim = Claim(
        claim_id="c_model_emp",
        claim_text="We measured throughput of 80,000 rps under production stress tests.",
        originating_agent="DatabaseArchitect",
        evidence_references=[model_inference.evidence_id],
    )

    is_valid, issues = ClaimValidator.validate_claim_structure(
        empirical_claim, evidence_items=[model_inference]
    )
    assert is_valid is False
    assert any("asserts empirical test result" in issue for issue in issues)


def test_traceable_tool_retrieved_source_qualifies():
    """Verify that traceable tool-retrieved sources qualify as external evidence."""
    tool_evidence = Evidence(
        evidence_id="ev_tool_perf",
        source_type=EvidenceSourceType.TOOL_RETRIEVED,
        artifact_type="prometheus_query_result",
        source_identifier="tool://prometheus_query/lat_p99",
        source_reference="Prometheus metric query: histogram_quantile(0.99, rate(http_duration_bucket[5m]))",
        supporting_content="p99 write latency observed at 4.2ms across 12 node clusters.",
        retrieval_timestamp="2026-11-04T12:05:00Z",
        evidence_location="data.result[0].value",
        originating_agent="MetricsSpecialist",
    )
    assert tool_evidence.is_user_assertion is False
    assert tool_evidence.is_external is True

    empirical_claim = Claim(
        claim_id="c_tool_emp",
        claim_text="We measured latency of 4.2ms across the database cluster.",
        originating_agent="MetricsSpecialist",
        evidence_references=[tool_evidence.evidence_id],
    )

    is_valid, issues = ClaimValidator.validate_claim_structure(
        empirical_claim, evidence_items=[tool_evidence]
    )
    assert is_valid is True
    assert len(issues) == 0


@pytest.mark.asyncio
async def test_reviewer_acceptance_cannot_override_missing_evidence():
    """Verify that an LLM reviewer returning ACCEPT cannot override missing or non-qualifying evidence."""
    mock_client = MagicMock()
    batch_json = """{
        "reviews": [
            {
                "claim_id": "c_bench_user_say",
                "decision": "ACCEPT",
                "reasoning": "The reviewer accepts this claim based on the user's assertion.",
                "critique": "",
                "suggested_fix": null,
                "is_supported_by_evidence": true,
                "is_evidence_relevant": true,
                "has_contradictions": false,
                "assumptions_presented_as_facts": false,
                "logically_valid": true
            }
        ]
    }"""
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(batch_json))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    ledger = EvidenceLedger(task_id="override_test")
    user_statement = Evidence(
        evidence_id="ev_user_unbacked",
        source_type=EvidenceSourceType.USER_ASSERTION,
        source_reference="User prompt comment",
        supporting_content="User remarked that the pilot test showed 500 MB/s throughput.",
        originating_agent="User",
    )
    await ledger.register_evidence(user_statement)

    claim = Claim(
        claim_id="c_bench_user_say",
        claim_text="Our pilot study proved write throughput exceeded 500 MB/s.",
        originating_agent="Specialist_1",
        evidence_references=[user_statement.evidence_id],
    )
    await ledger.register_claim(claim)

    reviewer = AdversarialReviewer(llm_client=gemini_client, batch_size=5)
    feedbacks = await reviewer.review_claims(ledger=ledger, task_context="Audit")

    assert len(feedbacks) == 1
    # Downgrade ACCEPT to REQUEST_REVISION because evidence is only a user assertion
    assert feedbacks[0].decision == VerificationDecision.REQUEST_REVISION
    assert "Heuristic Flag" in feedbacks[0].reasoning
    assert "asserts empirical test result" in feedbacks[0].reasoning

    updated_claim = await ledger.get_claim("c_bench_user_say")
    assert updated_claim.status == ClaimStatus.NEEDS_REVISION
    assert updated_claim.status != ClaimStatus.SUPPORTED
