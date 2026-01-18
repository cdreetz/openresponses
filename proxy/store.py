from typing import Any

_store: dict[str, dict[str, Any]] = {}


def save(response: dict[str, Any]) -> None:
    _store[response["id"]] = response


def get(response_id: str) -> dict[str, Any] | None:
    return _store.get(response_id)
