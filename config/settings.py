import os
import math
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Locate the MEDHA project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the root .env file
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

# LLM Configuration — Groq is the default primary provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Groq Configuration (Primary)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# Reasoning models only (e.g. gpt-oss): 'low' leaves more of the output limit for the answer. Blank = provider default.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "").strip().lower()

# Google Gemini Configuration (Optional Alternative)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")
_PROVIDER_REQUEST_TIMEOUT_RAW = os.getenv("PROVIDER_REQUEST_TIMEOUT_SECONDS", "").strip()
PROVIDER_REQUEST_TIMEOUT_SECONDS = (
    float(_PROVIDER_REQUEST_TIMEOUT_RAW) if _PROVIDER_REQUEST_TIMEOUT_RAW else None
)

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Retry & Rate Limit Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
INITIAL_RETRY_DELAY = float(os.getenv("INITIAL_RETRY_DELAY", "1.0"))
MAX_RETRY_DELAY = float(os.getenv("MAX_RETRY_DELAY", "10.0"))

# Multi-Agent Execution Configuration
MAX_CONCURRENT_AGENTS = int(os.getenv("MAX_CONCURRENT_AGENTS", "4"))
MAX_REPAIR_ITERATIONS = int(os.getenv("MAX_REPAIR_ITERATIONS", "2"))
EXECUTION_TIMEOUT_SECONDS = float(os.getenv("EXECUTION_TIMEOUT_SECONDS", "180.0"))

# Verification & LLM Budget Configuration
MAX_CLAIMS_PER_TASK = int(os.getenv("MAX_CLAIMS_PER_TASK", "50"))
REVIEW_BATCH_SIZE = int(os.getenv("REVIEW_BATCH_SIZE", "5"))
MAX_REPAIR_CLAIMS_PER_ITERATION = int(os.getenv("MAX_REPAIR_CLAIMS_PER_ITERATION", "3"))
MAX_LLM_CALLS_PER_TASK = int(os.getenv("MAX_LLM_CALLS_PER_TASK", "25"))
MAX_OUTPUT_TOKENS_PER_CALL = int(os.getenv("MAX_OUTPUT_TOKENS_PER_CALL", "2048"))
MAX_INPUT_TOKENS_PER_CALL = int(os.getenv("MAX_INPUT_TOKENS_PER_CALL", "12000"))
# Optional provider tokens-per-minute cap (e.g. 8000 on Groq free tier); 0 disables output-limit fitting.
PROVIDER_TOKENS_PER_MINUTE = int(os.getenv("PROVIDER_TOKENS_PER_MINUTE", "0"))
MAX_TOKENS_PER_TASK = int(os.getenv("MAX_TOKENS_PER_TASK", "50000"))
CLAIM_EXTRACTION_TOKEN_RESERVE = int(os.getenv("CLAIM_EXTRACTION_TOKEN_RESERVE", "4096"))
VERIFICATION_TOKEN_RESERVE = int(os.getenv("VERIFICATION_TOKEN_RESERVE", "12288"))
SYNTHESIS_TOKEN_RESERVE = int(os.getenv("SYNTHESIS_TOKEN_RESERVE", "12288"))

# 'strict' requires registered evidence; 'general' also accepts well-established textbook knowledge.
VERIFICATION_MODE = os.getenv("VERIFICATION_MODE", "strict").strip().lower()

# MCP background execution
MCP_MAX_ACTIVE_JOBS = int(os.getenv("MCP_MAX_ACTIVE_JOBS", "20"))
MCP_MAX_RECORDS = int(os.getenv("MCP_MAX_RECORDS", "100"))
MEDHA_MCP_RESULT_TTL = int(os.getenv("MEDHA_MCP_RESULT_TTL", "300"))

# Phase 1 policy-controlled deliberation limits
MAX_DELIBERATION_ROUNDS = int(os.getenv("MAX_DELIBERATION_ROUNDS", "2"))
MAX_EXCHANGES_PER_CLAIM = int(os.getenv("MAX_EXCHANGES_PER_CLAIM", "2"))
MAX_DELIBERATION_MESSAGES = int(os.getenv("MAX_DELIBERATION_MESSAGES", "30"))
MAX_DELIBERATION_MESSAGES_PER_AGENT = int(os.getenv("MAX_DELIBERATION_MESSAGES_PER_AGENT", "8"))
MAX_DELIBERATION_TOKENS = int(os.getenv("MAX_DELIBERATION_TOKENS", "8192"))
DELIBERATION_TIMEOUT_SECONDS = float(os.getenv("DELIBERATION_TIMEOUT_SECONDS", "90.0"))

# Application & Logging Configuration
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGS_DIR = BASE_DIR / "logs"
BENCHMARK_DIR = BASE_DIR / "benchmark"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


# Ephemeral mode is the default. Persistent execution artifacts require an explicit mode choice.
MEDHA_STORAGE_MODE = os.getenv("MEDHA_STORAGE_MODE", "ephemeral").strip().lower()
MEDHA_AUTO_CLEANUP = _env_bool("MEDHA_AUTO_CLEANUP", True)
MEDHA_CLEANUP_ON_FAILURE = _env_bool("MEDHA_CLEANUP_ON_FAILURE", True)
MEDHA_WORKSPACE_TTL = int(os.getenv("MEDHA_WORKSPACE_TTL", "3600"))
MEDHA_PERSISTENT_RETENTION_DAYS = int(os.getenv("MEDHA_PERSISTENT_RETENTION_DAYS", "30"))
def _configured_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    selected = Path(raw).expanduser() if raw else default
    if not selected.is_absolute():
        selected = BASE_DIR / selected
    return selected.resolve()


MEDHA_EPHEMERAL_ROOT = _configured_path(
    "MEDHA_EPHEMERAL_ROOT", Path(tempfile.gettempdir()) / "medha" / "executions"
)
MEDHA_PERSISTENT_ROOT = _configured_path("MEDHA_PERSISTENT_ROOT", BASE_DIR / "data" / "executions")
MEDHA_DOCUMENTS_DIR = _configured_path("MEDHA_DOCUMENTS_DIR", BASE_DIR / "documents")

# Retrieval is activated only when a caller supplies documents or approved URLs.
MEDHA_RETRIEVAL_ENABLED = _env_bool("MEDHA_RETRIEVAL_ENABLED", True)
MEDHA_HYBRID_SEARCH = _env_bool("MEDHA_HYBRID_SEARCH", True)
MEDHA_EMBEDDING_PROVIDER = os.getenv("MEDHA_EMBEDDING_PROVIDER", "fastembed").strip().lower()
MEDHA_EMBEDDING_MODEL = os.getenv("MEDHA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
MEDHA_MAX_RETRIEVED_CHUNKS = int(os.getenv("MEDHA_MAX_RETRIEVED_CHUNKS", "5"))
MEDHA_CHUNK_SIZE_CHARS = int(os.getenv("MEDHA_CHUNK_SIZE_CHARS", "1400"))
MEDHA_CHUNK_OVERLAP_CHARS = int(os.getenv("MEDHA_CHUNK_OVERLAP_CHARS", "180"))
MEDHA_MAX_DOCUMENT_BYTES = int(os.getenv("MEDHA_MAX_DOCUMENT_BYTES", str(25 * 1024 * 1024)))
MEDHA_MAX_SOURCE_RESPONSE_BYTES = int(os.getenv("MEDHA_MAX_SOURCE_RESPONSE_BYTES", str(10 * 1024 * 1024)))
MEDHA_ALLOWED_SOURCE_DOMAINS = tuple(
    domain.strip().lower()
    for domain in os.getenv("MEDHA_ALLOWED_SOURCE_DOMAINS", "").split(",")
    if domain.strip()
)

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def validate_settings() -> bool:
    """Validate essential MEDHA configuration based on the active provider."""
    if VERIFICATION_MODE not in ("strict", "general"):
        raise ValueError("VERIFICATION_MODE must be 'strict' or 'general'.")

    if MEDHA_STORAGE_MODE not in ("ephemeral", "persistent"):
        raise ValueError("MEDHA_STORAGE_MODE must be 'ephemeral' or 'persistent'.")
    if MEDHA_STORAGE_MODE == "ephemeral" and (not MEDHA_AUTO_CLEANUP or not MEDHA_CLEANUP_ON_FAILURE):
        raise ValueError("Ephemeral mode requires automatic cleanup on success and failure.")
    if MEDHA_WORKSPACE_TTL < 1 or MEDHA_PERSISTENT_RETENTION_DAYS < 1:
        raise ValueError("Workspace TTL and persistent retention must be positive.")
    if MEDHA_RETRIEVAL_ENABLED and MEDHA_EMBEDDING_PROVIDER not in ("fastembed", "none"):
        raise ValueError("MEDHA_EMBEDDING_PROVIDER must be 'fastembed' or 'none'.")
    if MEDHA_RETRIEVAL_ENABLED and MEDHA_HYBRID_SEARCH and MEDHA_EMBEDDING_PROVIDER == "none":
        raise ValueError("Hybrid retrieval requires a local embedding provider.")
    if MEDHA_MAX_RETRIEVED_CHUNKS < 1:
        raise ValueError("MEDHA_MAX_RETRIEVED_CHUNKS must be at least 1.")
    if MEDHA_CHUNK_SIZE_CHARS < 100 or not 0 <= MEDHA_CHUNK_OVERLAP_CHARS < MEDHA_CHUNK_SIZE_CHARS:
        raise ValueError("Chunk size must be >= 100 and overlap must be non-negative and smaller than chunk size.")
    if MEDHA_MAX_DOCUMENT_BYTES < 1 or MEDHA_MAX_SOURCE_RESPONSE_BYTES < 1:
        raise ValueError("Document and source response byte limits must be positive.")

    provider = (LLM_PROVIDER or "").lower().strip()
    if provider not in ("groq", "gemini"):
        raise ValueError(
            f"Unsupported LLM provider: '{LLM_PROVIDER}'. Supported providers are 'groq' and 'gemini'."
        )

    if provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing from .env")
        if not GROQ_MODEL:
            raise ValueError("GROQ_MODEL is not configured.")
    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing from .env")
        if not GEMINI_MODEL:
            raise ValueError("GEMINI_MODEL is not configured.")

    if MAX_CONCURRENT_AGENTS < 1:
        raise ValueError("MAX_CONCURRENT_AGENTS must be at least 1.")

    if MAX_REPAIR_ITERATIONS < 0:
        raise ValueError("MAX_REPAIR_ITERATIONS cannot be negative.")

    if not math.isfinite(EXECUTION_TIMEOUT_SECONDS) or EXECUTION_TIMEOUT_SECONDS <= 0:
        raise ValueError("EXECUTION_TIMEOUT_SECONDS must be a finite positive number.")
    if PROVIDER_REQUEST_TIMEOUT_SECONDS is not None and (
        not math.isfinite(PROVIDER_REQUEST_TIMEOUT_SECONDS)
        or PROVIDER_REQUEST_TIMEOUT_SECONDS <= 0
    ):
        raise ValueError("PROVIDER_REQUEST_TIMEOUT_SECONDS must be a finite positive number when set.")

    if MAX_CLAIMS_PER_TASK < 1:
        raise ValueError("MAX_CLAIMS_PER_TASK must be at least 1.")

    if REVIEW_BATCH_SIZE < 1:
        raise ValueError("REVIEW_BATCH_SIZE must be at least 1.")

    if MAX_REPAIR_CLAIMS_PER_ITERATION < 1:
        raise ValueError("MAX_REPAIR_CLAIMS_PER_ITERATION must be at least 1.")

    if MAX_LLM_CALLS_PER_TASK < 1:
        raise ValueError("MAX_LLM_CALLS_PER_TASK must be at least 1.")

    if MAX_OUTPUT_TOKENS_PER_CALL < 1:
        raise ValueError("MAX_OUTPUT_TOKENS_PER_CALL must be at least 1.")

    if MAX_INPUT_TOKENS_PER_CALL < 1:
        raise ValueError("MAX_INPUT_TOKENS_PER_CALL must be at least 1.")

    if PROVIDER_TOKENS_PER_MINUTE < 0:
        raise ValueError("PROVIDER_TOKENS_PER_MINUTE cannot be negative.")

    if GROQ_REASONING_EFFORT not in ("", "none", "default", "low", "medium", "high"):
        raise ValueError("GROQ_REASONING_EFFORT must be blank or one of: none, default, low, medium, high.")

    if MAX_TOKENS_PER_TASK < 1:
        raise ValueError("MAX_TOKENS_PER_TASK must be at least 1.")

    essential_token_reserve = (
        CLAIM_EXTRACTION_TOKEN_RESERVE + VERIFICATION_TOKEN_RESERVE + SYNTHESIS_TOKEN_RESERVE
    )
    if min(
        CLAIM_EXTRACTION_TOKEN_RESERVE,
        VERIFICATION_TOKEN_RESERVE,
        SYNTHESIS_TOKEN_RESERVE,
    ) < 0:
        raise ValueError("Essential token reserves cannot be negative.")
    if essential_token_reserve >= MAX_TOKENS_PER_TASK:
        raise ValueError("Essential token reserves must be less than MAX_TOKENS_PER_TASK.")
    if MAX_INPUT_TOKENS_PER_CALL + MAX_OUTPUT_TOKENS_PER_CALL > MAX_TOKENS_PER_TASK:
        raise ValueError("Per-call input and output limits cannot exceed MAX_TOKENS_PER_TASK in total.")
    if MCP_MAX_ACTIVE_JOBS < 1:
        raise ValueError("MCP_MAX_ACTIVE_JOBS must be at least 1.")
    if MCP_MAX_RECORDS < 1:
        raise ValueError("MCP_MAX_RECORDS must be at least 1.")
    if MEDHA_MCP_RESULT_TTL < 1:
        raise ValueError("MEDHA_MCP_RESULT_TTL must be at least 1 second.")
    if MAX_DELIBERATION_ROUNDS < 1:
        raise ValueError("MAX_DELIBERATION_ROUNDS must be at least 1.")
    if MAX_EXCHANGES_PER_CLAIM < 1:
        raise ValueError("MAX_EXCHANGES_PER_CLAIM must be at least 1.")
    if MAX_DELIBERATION_MESSAGES < 1:
        raise ValueError("MAX_DELIBERATION_MESSAGES must be at least 1.")
    if MAX_DELIBERATION_MESSAGES_PER_AGENT < 1:
        raise ValueError("MAX_DELIBERATION_MESSAGES_PER_AGENT must be at least 1.")
    if MAX_DELIBERATION_TOKENS < 1:
        raise ValueError("MAX_DELIBERATION_TOKENS must be at least 1.")
    if DELIBERATION_TIMEOUT_SECONDS <= 0:
        raise ValueError("DELIBERATION_TIMEOUT_SECONDS must be positive.")

    return True
