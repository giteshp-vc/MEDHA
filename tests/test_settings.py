import pytest

import config.settings as settings


def test_settings_load_defaults():
    """Verify default settings values are correctly loaded with Groq as default."""
    assert settings.LLM_PROVIDER in ("groq", "gemini")
    assert settings.GROQ_MODEL == "openai/gpt-oss-20b"
    assert settings.GEMINI_MODEL is not None
    assert settings.MAX_CONCURRENT_AGENTS >= 1
    assert settings.MAX_REPAIR_ITERATIONS >= 0
    assert settings.LOGS_DIR.exists()
    assert settings.MAX_TOKENS_PER_TASK > 0
    assert settings.MAX_INPUT_TOKENS_PER_CALL > 0
    assert settings.MAX_OUTPUT_TOKENS_PER_CALL > 0
    assert settings.CLAIM_EXTRACTION_TOKEN_RESERVE >= 0
    assert settings.VERIFICATION_TOKEN_RESERVE >= 0
    assert settings.SYNTHESIS_TOKEN_RESERVE >= 0


def test_validate_settings_groq_success(monkeypatch):
    """Verify settings validation succeeds when Groq is configured."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "dummy_groq_key_123")
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-20b")
    assert settings.validate_settings() is True


def test_validate_settings_groq_missing_api_key(monkeypatch):
    """Verify validation fails when GROQ_API_KEY is missing for Groq provider."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    with pytest.raises(ValueError, match="GROQ_API_KEY is missing"):
        settings.validate_settings()


def test_validate_settings_groq_missing_model(monkeypatch):
    """Verify validation fails when GROQ_MODEL is missing."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "dummy_groq_key_123")
    monkeypatch.setattr(settings, "GROQ_MODEL", "")
    with pytest.raises(ValueError, match="GROQ_MODEL is not configured"):
        settings.validate_settings()


def test_validate_settings_gemini_success(monkeypatch):
    """Verify settings validation succeeds when Gemini is selected and configured."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "dummy_gemini_key_123")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-3.8-flash")
    assert settings.validate_settings() is True


def test_validate_settings_gemini_missing_api_key(monkeypatch):
    """Verify validation fails when GEMINI_API_KEY is missing for Gemini provider."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is missing"):
        settings.validate_settings()


def test_validate_settings_gemini_missing_model(monkeypatch):
    """Verify validation fails when GEMINI_MODEL is missing."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "dummy_gemini_key_123")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "")
    with pytest.raises(ValueError, match="GEMINI_MODEL is not configured"):
        settings.validate_settings()


def test_validate_settings_unsupported_provider(monkeypatch):
    """Verify validation fails when an unsupported LLM provider is configured."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "unsupported_provider")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        settings.validate_settings()


def test_validate_settings_invalid_concurrency(monkeypatch):
    """Verify validation fails when MAX_CONCURRENT_AGENTS < 1."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "dummy_groq_key_123")
    monkeypatch.setattr(settings, "MAX_CONCURRENT_AGENTS", 0)
    with pytest.raises(ValueError, match="MAX_CONCURRENT_AGENTS must be at least 1"):
        settings.validate_settings()


def test_validate_settings_invalid_repair_iterations(monkeypatch):
    """Verify validation fails when MAX_REPAIR_ITERATIONS < 0."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "dummy_groq_key_123")
    monkeypatch.setattr(settings, "MAX_REPAIR_ITERATIONS", -1)
    with pytest.raises(ValueError, match="MAX_REPAIR_ITERATIONS cannot be negative"):
        settings.validate_settings()
