from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.responses import (
    Response,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseFunctionToolCall,
)

from proxy.utils.type_helpers import (
    is_response_input_message,
    is_response_function_call,
    is_response_function_call_output,
    warn_unsupported,
    warn_conversion,
)
from proxy.utils.utils import generate_id, timestamp, extract_text_from_content


def responses_input_to_messages(
    input_items: str | list[ResponseInputItemParam],
    instructions: str | None = None,
) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []

    if instructions:
        messages.append(ChatCompletionSystemMessageParam(role="system", content=instructions))

    if isinstance(input_items, str):
        messages.append(ChatCompletionUserMessageParam(role="user", content=input_items))
        return messages

    for item in input_items:
        if isinstance(item, str):
            messages.append(ChatCompletionUserMessageParam(role="user", content=item))
            continue

        if not isinstance(item, dict):
            warn_unsupported(type(item).__name__, "responses_input_to_messages")
            continue

        item_type = item.get("type")

        if is_response_input_message(item):
            messages.extend(_convert_message_item(item))
        elif is_response_function_call(item):
            messages.append(_convert_function_call_to_assistant(item))
        elif is_response_function_call_output(item):
            messages.append(_convert_function_call_output(item))
        else:
            warn_unsupported(f"item type '{item_type}'", "responses_input_to_messages")

    return messages


def _convert_message_item(item: dict) -> list[ChatCompletionMessageParam]:
    role = item["role"]
    content = extract_text_from_content(item.get("content", ""))

    if role == "user":
        return [ChatCompletionUserMessageParam(role="user", content=content)]
    elif role == "assistant":
        return [ChatCompletionAssistantMessageParam(role="assistant", content=content)]
    elif role in ("system", "developer"):
        return [ChatCompletionSystemMessageParam(role="system", content=content)]
    else:
        warn_unsupported(f"role '{role}'", "_convert_message_item")
        return []


def _convert_function_call_to_assistant(item: dict) -> ChatCompletionAssistantMessageParam:
    call_id = item.get("call_id") or item.get("id") or generate_id("call")
    return ChatCompletionAssistantMessageParam(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCallParam(
                id=call_id,
                type="function",
                function=Function(name=item["name"], arguments=item["arguments"]),
            )
        ],
    )


def _convert_function_call_output(item: dict) -> ChatCompletionToolMessageParam:
    return ChatCompletionToolMessageParam(
        role="tool",
        tool_call_id=item["call_id"],
        content=item["output"],
    )


def responses_request_to_chat(body: dict) -> dict[str, Any]:
    messages = responses_input_to_messages(
        body.get("input", []),
        body.get("instructions"),
    )

    result: dict[str, Any] = {"messages": messages}

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


def _convert_tools(tools: list[dict]) -> list[dict]:
    converted = []
    for tool in tools:
        if tool.get("type") != "function":
            warn_unsupported(f"tool type '{tool.get('type')}'", "_convert_tools")
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        })
    return converted


def chat_response_to_responses(chat_resp: dict, original_body: dict) -> dict[str, Any]:
    now = timestamp()
    choice = chat_resp.get("choices", [{}])[0]
    message = choice.get("message", {})

    output = _build_output_items(message)
    usage = _build_usage(chat_resp.get("usage", {}))

    return {
        "id": generate_id("resp"),
        "object": "response",
        "created_at": now,
        "completed_at": now,
        "status": "completed",
        "model": chat_resp.get("model", original_body.get("model", "unknown")),
        "output": output,
        "usage": usage,
        "error": None,
        "incomplete_details": None,
        "instructions": original_body.get("instructions"),
        "previous_response_id": original_body.get("previous_response_id"),
        "metadata": original_body.get("metadata", {}),
        "temperature": original_body.get("temperature", 1.0),
        "top_p": original_body.get("top_p", 1.0),
        "max_output_tokens": original_body.get("max_output_tokens"),
        "tools": original_body.get("tools", []),
        "tool_choice": original_body.get("tool_choice", "auto"),
        "parallel_tool_calls": original_body.get("parallel_tool_calls", True),
        "truncation": original_body.get("truncation", "disabled"),
        "store": original_body.get("store", False),
    }


def _build_output_items(message: dict) -> list[dict]:
    output: list[dict] = []

    content = message.get("content")
    if content:
        output.append({
            "type": "message",
            "id": generate_id("msg"),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        })

    for tc in message.get("tool_calls", []):
        output.append({
            "type": "function_call",
            "id": generate_id("fc"),
            "call_id": tc.get("id", generate_id("call")),
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
            "status": "completed",
        })

    return output


def _build_usage(usage: dict) -> dict:
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
