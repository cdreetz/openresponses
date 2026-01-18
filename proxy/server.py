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
    chat_body = responses_request_to_chat(body)
    chat_resp, status = await forward_to_upstream(chat_body, get_api_key(request))

    if status != 200:
        return web.json_response(chat_resp, status=status)

    response = chat_response_to_responses(chat_resp, body)

    if body.get("store", False):
        response["input"] = body.get("input", [])
        store.save(response)

    return web.json_response(response)


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
