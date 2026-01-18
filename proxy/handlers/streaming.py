"""
Streaming handler for Responses API -> Chat Completions conversion.

Converts Chat Completions SSE chunks to Responses API SSE events.
"""
import json
from typing import Any, AsyncIterator

from proxy.utils.utils import generate_id, timestamp


async def parse_sse_stream(response: Any) -> AsyncIterator[dict]:
    """Parse SSE stream from Chat Completions response."""
    buffer = ""
    async for chunk in response.content.iter_any():
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


def build_response_created_event(response_id: str, original_body: dict) -> dict:
    """Build the response.created event."""
    now = timestamp()
    return {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": now,
            "status": "in_progress",
            "model": original_body.get("model", "unknown"),
            "output": [],
            "usage": None,
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
            "frequency_penalty": original_body.get("frequency_penalty", 0.0),
            "presence_penalty": original_body.get("presence_penalty", 0.0),
        },
    }


def build_output_item_added_event(response_id: str, item_id: str) -> dict:
    """Build the response.output_item.added event for a message."""
    return {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": item_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        },
    }


def build_content_part_added_event(item_id: str, part_index: int) -> dict:
    """Build the response.content_part.added event."""
    return {
        "type": "response.content_part.added",
        "item_id": item_id,
        "output_index": 0,
        "content_index": part_index,
        "part": {
            "type": "output_text",
            "text": "",
            "annotations": [],
        },
    }


def build_text_delta_event(item_id: str, delta: str) -> dict:
    """Build the response.output_text.delta event."""
    return {
        "type": "response.output_text.delta",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }


def build_text_done_event(item_id: str, text: str) -> dict:
    """Build the response.output_text.done event."""
    return {
        "type": "response.output_text.done",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": text,
    }


def build_content_part_done_event(item_id: str, text: str) -> dict:
    """Build the response.content_part.done event."""
    return {
        "type": "response.content_part.done",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": text,
            "annotations": [],
        },
    }


def build_output_item_done_event(item_id: str, text: str) -> dict:
    """Build the response.output_item.done event."""
    return {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": item_id,
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                }
            ],
        },
    }


def build_response_completed_event(
    response_id: str,
    item_id: str,
    text: str,
    original_body: dict,
    usage: dict | None,
) -> dict:
    """Build the response.completed event."""
    now = timestamp()
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": now,
            "status": "completed",
            "model": original_body.get("model", "unknown"),
            "output": [
                {
                    "type": "message",
                    "id": item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
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
            "frequency_penalty": original_body.get("frequency_penalty", 0.0),
            "presence_penalty": original_body.get("presence_penalty", 0.0),
        },
    }


def build_function_call_item_added_event(
    output_index: int,
    call_id: str,
    item_id: str,
    name: str,
) -> dict:
    """Build the response.output_item.added event for a function call."""
    return {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "status": "in_progress",
        },
    }


def build_function_call_args_delta_event(
    item_id: str,
    output_index: int,
    delta: str,
) -> dict:
    """Build the response.function_call_arguments.delta event."""
    return {
        "type": "response.function_call_arguments.delta",
        "item_id": item_id,
        "output_index": output_index,
        "delta": delta,
    }


def build_function_call_args_done_event(
    item_id: str,
    output_index: int,
    arguments: str,
) -> dict:
    """Build the response.function_call_arguments.done event."""
    return {
        "type": "response.function_call_arguments.done",
        "item_id": item_id,
        "output_index": output_index,
        "arguments": arguments,
    }


def build_function_call_item_done_event(
    output_index: int,
    call_id: str,
    item_id: str,
    name: str,
    arguments: str,
) -> dict:
    """Build the response.output_item.done event for a function call."""
    return {
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        },
    }


def format_sse_event(event: dict) -> str:
    """Format an event as SSE."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
