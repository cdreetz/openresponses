from typing import Any

_store: dict[tuple[str, str], dict[str, Any]] = {}


def _key(user_id: str, response_id: str) -> tuple[str, str]:
    return (user_id, response_id)


def save(user_id: str, response: dict[str, Any]) -> None:
    _store[_key(user_id, response["id"])] = response


def get(user_id: str, response_id: str) -> dict[str, Any] | None:
    return _store.get(_key(user_id, response_id))
