from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_storage_mode: ContextVar[Optional[str]] = ContextVar("medha_storage_mode", default=None)


def current_storage_mode() -> Optional[str]:
    selected = _storage_mode.get()
    if selected is not None:
        return selected
    from config import settings

    return settings.MEDHA_STORAGE_MODE


def active_storage_mode() -> Optional[str]:
    """Return only a context override, without falling back to the application default."""
    return _storage_mode.get()


def is_ephemeral_context() -> bool:
    return current_storage_mode() == "ephemeral"


@contextmanager
def storage_mode_context(mode: str) -> Iterator[None]:
    normalized = (mode or "").strip().lower()
    if normalized not in {"ephemeral", "persistent"}:
        raise ValueError("Storage mode must be 'ephemeral' or 'persistent'.")
    token = _storage_mode.set(normalized)
    try:
        yield
    finally:
        _storage_mode.reset(token)
