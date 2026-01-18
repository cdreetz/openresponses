import time
import uuid


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                texts.append(part.get("text", ""))
        return "".join(texts)
    return ""


def _input_to_messages(inp) -> list[dict]:
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]

    messages = []
    for item in inp:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif item.get("type") == "message":
            messages.append({
                "role": item.get("role", "user"),
                "content": _extract_text(item.get("content", ""))
            })
    return messages


def _convert_tools(tools: list) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            }
        }
        for t in tools if t.get("type") == "function"
    ]


def request_to_chat(body: dict) -> dict:
    messages = []

    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})

    messages.extend(_input_to_messages(body.get("input", [])))

    result = {"messages": messages}

    if body.get("model"):
        result["model"] = body["model"]
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        result["top_p"] = body["top_p"]
    if body.get("max_output_tokens") is not None:
        result["max_tokens"] = body["max_output_tokens"]
    if body.get("tools"):
        result["tools"] = _convert_tools(body["tools"])
    if body.get("tool_choice"):
        result["tool_choice"] = body["tool_choice"]

    return result


def _build_output(message: dict) -> list[dict]:
    output = []

    if message.get("content"):
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": message["content"], "annotations": []}]
        })

    for tc in message.get("tool_calls", []):
        output.append({
            "type": "function_call",
            "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
            "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
            "status": "completed"
        })

    return output


def _map_usage(usage: dict) -> dict:
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def response_to_responses(chat_resp: dict, original: dict) -> dict:
    now = int(time.time())
    choice = chat_resp.get("choices", [{}])[0]

    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": now,
        "completed_at": now,
        "status": "completed",
        "model": chat_resp.get("model", original.get("model", "unknown")),
        "output": _build_output(choice.get("message", {})),
        "usage": _map_usage(chat_resp.get("usage", {})),
        "error": None,
        "incomplete_details": None,
        "instructions": original.get("instructions"),
        "previous_response_id": original.get("previous_response_id"),
        "metadata": original.get("metadata", {}),
        "temperature": original.get("temperature", 1.0),
        "top_p": original.get("top_p", 1.0),
        "max_output_tokens": original.get("max_output_tokens"),
        "tools": original.get("tools", []),
        "tool_choice": original.get("tool_choice", "auto"),
        "parallel_tool_calls": original.get("parallel_tool_calls", True),
        "truncation": original.get("truncation", "disabled"),
        "store": original.get("store", False),
    }
