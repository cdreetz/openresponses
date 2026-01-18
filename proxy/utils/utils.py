import time
import uuid
from typing import Any


def generate_id(prefix: str = "resp") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def timestamp() -> int:
    return int(time.time())


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("input_text") or ""
                parts.append(text)
        return "".join(parts)
    return ""
