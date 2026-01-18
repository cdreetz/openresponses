from typing import Any

from proxy.utils.type_helpers import warn_unsupported
from proxy.utils.utils import generate_id, timestamp


def anthropic_to_chat(body: dict) -> dict[str, Any]:
    messages = []

    if body.get("system"):
        messages.append({"role": "system", "content": body["system"]})

    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                else:
                    warn_unsupported(f"content block type '{block.get('type')}'", "anthropic_to_chat")
            messages.append({"role": role, "content": "".join(text_parts)})

    result: dict[str, Any] = {"messages": messages}

    if body.get("model"):
        result["model"] = body["model"]
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        result["top_p"] = body["top_p"]
    if body.get("max_tokens") is not None:
        result["max_tokens"] = body["max_tokens"]

    return result


def chat_to_anthropic(chat_resp: dict, original_body: dict) -> dict[str, Any]:
    choice = chat_resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = chat_resp.get("usage", {})

    content = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})

    return {
        "id": generate_id("msg"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": chat_resp.get("model", original_body.get("model", "unknown")),
        "stop_reason": _map_finish_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _map_finish_reason(reason: str | None) -> str:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    return mapping.get(reason or "", "end_turn")
