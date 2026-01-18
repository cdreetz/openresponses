import os

import aiohttp
from aiohttp import web

from proxy.handlers.handle_responses_api import (
    responses_request_to_chat,
    chat_response_to_responses,
)
from proxy.handlers.handle_anthropic_api import (
    anthropic_to_chat,
    chat_to_anthropic,
)
from proxy.handlers.streaming import (
    parse_sse_stream,
    build_response_created_event,
    build_output_item_added_event,
    build_content_part_added_event,
    build_text_delta_event,
    build_text_done_event,
    build_content_part_done_event,
    build_output_item_done_event,
    build_response_completed_event,
    build_function_call_item_added_event,
    build_function_call_args_delta_event,
    build_function_call_args_done_event,
    build_function_call_item_done_event,
    format_sse_event,
)
from proxy.utils.utils import generate_id
from proxy import store

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")


def get_api_key(request: web.Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return UPSTREAM_API_KEY


async def forward_to_upstream(body: dict, api_key: str) -> tuple[dict, int]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession() as session:
        async with session.post(UPSTREAM_URL, json=body, headers=headers) as resp:
            return await resp.json(), resp.status


async def handle_responses(request: web.Request) -> web.Response:
    body = await request.json()
    api_key = get_api_key(request)

    if body.get("stream", False):
        return await handle_responses_streaming(request, body, api_key)

    chat_body = responses_request_to_chat(body)
    chat_resp, status = await forward_to_upstream(chat_body, api_key)

    if status != 200:
        return web.json_response(chat_resp, status=status)

    response = chat_response_to_responses(chat_resp, body)

    if body.get("store", False):
        response["input"] = body.get("input", [])
        store.save(response)

    return web.json_response(response)


async def handle_responses_streaming(
    request: web.Request,
    body: dict,
    api_key: str,
) -> web.StreamResponse:
    """Handle streaming Responses API request."""
    chat_body = responses_request_to_chat(body)
    chat_body["stream"] = True

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    response_id = generate_id("resp")
    message_id = generate_id("msg")
    accumulated_text = ""
    usage = None
    content_started = False

    tool_calls: dict[int, dict] = {}

    async with aiohttp.ClientSession() as session:
        async with session.post(UPSTREAM_URL, json=chat_body, headers=headers) as upstream_resp:
            if upstream_resp.status != 200:
                error_body = await upstream_resp.json()
                await response.write(
                    f"event: error\ndata: {error_body}\n\n".encode()
                )
                await response.write_eof()
                return response

            created_event = build_response_created_event(response_id, body)
            await response.write(format_sse_event(created_event).encode())

            async for chunk in parse_sse_stream(upstream_resp):
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                if delta.get("content"):
                    if not content_started:
                        item_event = build_output_item_added_event(response_id, message_id)
                        await response.write(format_sse_event(item_event).encode())

                        part_event = build_content_part_added_event(message_id, 0)
                        await response.write(format_sse_event(part_event).encode())
                        content_started = True

                    text_delta = delta["content"]
                    accumulated_text += text_delta
                    delta_event = build_text_delta_event(message_id, text_delta)
                    await response.write(format_sse_event(delta_event).encode())

                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": generate_id("fc"),
                                "call_id": tc.get("id", generate_id("call")),
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": "",
                            }
                            added_event = build_function_call_item_added_event(
                                idx,
                                tool_calls[idx]["call_id"],
                                tool_calls[idx]["id"],
                                tool_calls[idx]["name"],
                            )
                            await response.write(format_sse_event(added_event).encode())

                        if tc.get("function", {}).get("arguments"):
                            args_delta = tc["function"]["arguments"]
                            tool_calls[idx]["arguments"] += args_delta
                            args_event = build_function_call_args_delta_event(
                                tool_calls[idx]["id"],
                                idx,
                                args_delta,
                            )
                            await response.write(format_sse_event(args_event).encode())

                if chunk.get("usage"):
                    u = chunk["usage"]
                    usage = {
                        "input_tokens": u.get("prompt_tokens", 0),
                        "output_tokens": u.get("completion_tokens", 0),
                        "total_tokens": u.get("total_tokens", 0),
                    }

    if content_started:
        text_done = build_text_done_event(message_id, accumulated_text)
        await response.write(format_sse_event(text_done).encode())

        part_done = build_content_part_done_event(message_id, accumulated_text)
        await response.write(format_sse_event(part_done).encode())

        item_done = build_output_item_done_event(message_id, accumulated_text)
        await response.write(format_sse_event(item_done).encode())

    for idx, tc in sorted(tool_calls.items()):
        args_done = build_function_call_args_done_event(tc["id"], idx, tc["arguments"])
        await response.write(format_sse_event(args_done).encode())

        item_done = build_function_call_item_done_event(
            idx, tc["call_id"], tc["id"], tc["name"], tc["arguments"]
        )
        await response.write(format_sse_event(item_done).encode())

    completed_event = build_response_completed_event(
        response_id, message_id, accumulated_text, body, usage
    )
    await response.write(format_sse_event(completed_event).encode())

    if body.get("store", False):
        final_response = completed_event["response"]
        final_response["input"] = body.get("input", [])
        store.save(final_response)

    await response.write_eof()
    return response


async def handle_anthropic_messages(request: web.Request) -> web.Response:
    body = await request.json()
    chat_body = anthropic_to_chat(body)
    chat_resp, status = await forward_to_upstream(chat_body, get_api_key(request))

    if status != 200:
        return web.json_response(chat_resp, status=status)

    return web.json_response(chat_to_anthropic(chat_resp, body))


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/responses", handle_responses)
    app.router.add_post("/v1/messages", handle_anthropic_messages)
    app.router.add_get("/health", handle_health)
    return app
