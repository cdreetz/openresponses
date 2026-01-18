from typing import Any, Iterable, TypeAlias, TypeGuard
import warnings

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function as ToolCallFunction
from openai.types.responses import (
    ResponseInputItemParam,
    ResponseInputTextParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseFunctionToolCall,
)
from openai.types.responses.response_input_item_param import Message as ResponseInputMessage
from openai.types.responses.response_input_item_param import FunctionCallOutput


ResponseInputContent: TypeAlias = str | list[ResponseInputTextParam]
ChatCompletionContent: TypeAlias = str | Iterable[ChatCompletionContentPartTextParam] | None


def is_text_content_part(obj: Any) -> TypeGuard[ChatCompletionContentPartTextParam]:
    return (
        isinstance(obj, dict)
        and obj.get("type") == "text"
        and isinstance(obj.get("text"), str)
    )


def is_text_content_list(obj: Any) -> TypeGuard[list[ChatCompletionContentPartTextParam]]:
    if not isinstance(obj, list):
        return False
    return all(is_text_content_part(item) for item in obj)


def is_content_part(obj: Any) -> TypeGuard[ChatCompletionContentPartParam]:
    if not isinstance(obj, dict):
        return False
    t = obj.get("type")
    return (
        (t == "text" and "text" in obj)
        or (t == "image_url" and "image_url" in obj)
        or (t == "input_audio" and "input_audio" in obj)
    )


def is_content_part_list(obj: Any) -> TypeGuard[list[ChatCompletionContentPartParam]]:
    if not isinstance(obj, list):
        return False
    return all(is_content_part(item) for item in obj)


def is_response_input_message(obj: Any) -> TypeGuard[ResponseInputMessage]:
    return (
        isinstance(obj, dict)
        and obj.get("type") == "message"
        and obj.get("role") in ("user", "assistant", "system", "developer")
    )


def is_response_function_call(obj: Any) -> TypeGuard[dict]:
    return (
        isinstance(obj, dict)
        and obj.get("type") == "function_call"
        and "name" in obj
        and "arguments" in obj
    )


def is_response_function_call_output(obj: Any) -> TypeGuard[FunctionCallOutput]:
    return (
        isinstance(obj, dict)
        and obj.get("type") == "function_call_output"
        and "call_id" in obj
        and "output" in obj
    )


def is_chat_completion_message(obj: Any) -> TypeGuard[ChatCompletionMessageParam]:
    return isinstance(obj, dict) and obj.get("role") in ("user", "assistant", "system", "tool")


def warn_unsupported(item_type: str, context: str) -> None:
    warnings.warn(
        f"Unsupported {item_type} in {context}. Skipping.",
        UserWarning,
        stacklevel=3,
    )


def warn_conversion(source: str, target: str, detail: str) -> None:
    warnings.warn(
        f"Lossy conversion from {source} to {target}: {detail}",
        UserWarning,
        stacklevel=3,
    )
