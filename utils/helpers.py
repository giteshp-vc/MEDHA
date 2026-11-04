import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict


def generate_id(prefix: str = "id") -> str:
    """Generate a clean, readable unique identifier with a given prefix."""
    short_uuid = uuid.uuid4().hex[:10]
    return f"{prefix}_{short_uuid}"


def utcnow_iso() -> str:
    """Return the current UTC timestamp formatted as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def sanitize_json_text(raw_text: str) -> str:
    """Extract the contents of a Markdown code fence when one is present."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """Parse a JSON object, including one embedded in prose or a Markdown fence."""
    clean_text = sanitize_json_text(raw_text)
    if not clean_text:
        raise ValueError("Empty response: no JSON object was provided.")

    decoder = json.JSONDecoder()
    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as error:
        original_error = error
    else:
        original_error = ValueError("The JSON response must be an object.")

    for index, character in enumerate(clean_text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(clean_text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"No valid JSON object found: {original_error}") from original_error

