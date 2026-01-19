"""
Manual test: Multi-turn single tool call flow
Run: uv run proxy/tests/manual_tests/test_responses_multi_turn_tool.py
"""
import json
import os
import sys
import time
import subprocess
import signal
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

PROXY_PORT = 8771
PROXY_URL = f"http://localhost:{PROXY_PORT}/v1"
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
        },
        "required": ["location"],
    },
}


class TestFailure(Exception):
    pass


def assert_true(condition: bool, msg: str) -> None:
    if not condition:
        raise TestFailure(msg)


def assert_eq(actual: Any, expected: Any, msg: str) -> None:
    if actual != expected:
        raise TestFailure(f"{msg}: expected {expected!r}, got {actual!r}")


def get_repo_root() -> Path:
    path = Path(__file__).resolve()
    while path.parent != path:
        if (path / "proxy").is_dir() and (path / "proxy" / "cli.py").exists():
            return path
        path = path.parent
    raise TestFailure("Could not find repo root")


def wait_for_server(url: str, timeout: int = 10) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url.rstrip('/v1')}/health", timeout=1)
            return True
        except urllib.error.URLError:
            time.sleep(0.2)
    return False


def start_proxy() -> subprocess.Popen:
    repo_root = get_repo_root()
    env = os.environ.copy()
    env["UPSTREAM_URL"] = UPSTREAM_URL

    proc = subprocess.Popen(
        [sys.executable, "-m", "proxy.cli", "--port", str(PROXY_PORT)],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not wait_for_server(PROXY_URL):
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        raise TestFailure(
            f"Server failed to start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    return proc


def main() -> int:
    from openai import OpenAI

    print("=" * 60)
    print("TEST: Multi-turn Single Tool Call")
    print("=" * 60)

    print("\n[1/5] Starting proxy server...")
    proxy_proc = start_proxy()
    print("      Server ready")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))
        model = "gpt-4o-mini"

        print("\n[2/5] Making initial request with tools...")
        resp1 = proxy_client.responses.create(
            model=model,
            input="What's the weather in Paris?",
            tools=[WEATHER_TOOL],
        )
        resp1_dict = resp1.model_dump()
        print(f"      Response ID: {resp1_dict['id']}")

        # Find the function_call in output
        function_call = None
        for item in resp1_dict.get("output", []):
            if item.get("type") == "function_call":
                function_call = item
                break

        assert_true(function_call is not None, "Expected function_call in response")
        print(f"      Function call: {function_call['name']}({function_call['arguments']})")

        call_id = function_call.get("call_id") or function_call.get("id")
        assert_true(call_id is not None, "Function call must have call_id or id")

        print("\n[3/5] Sending follow-up with tool result...")
        # Simulate tool execution result
        tool_result = json.dumps({"temperature": "22°C", "condition": "Sunny"})

        # Send follow-up request with function_call and function_call_output
        resp2 = proxy_client.responses.create(
            model=model,
            input=[
                {"type": "message", "role": "user", "content": "What's the weather in Paris?"},
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": function_call["name"],
                    "arguments": function_call["arguments"],
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_result,
                },
            ],
            tools=[WEATHER_TOOL],
        )
        resp2_dict = resp2.model_dump()
        print(f"      Response ID: {resp2_dict['id']}")

        print("\n[4/5] Validating follow-up response...")
        assert_eq(resp2_dict["status"], "completed", "response status")

        # Should have a message output (not another tool call)
        message_output = None
        for item in resp2_dict.get("output", []):
            if item.get("type") == "message":
                message_output = item
                break

        assert_true(message_output is not None, "Expected message in follow-up response")
        content = message_output.get("content", [])
        assert_true(len(content) > 0, "Message should have content")

        text = content[0].get("text", "")
        print(f"      Response text: {text[:100]}...")

        print("\n[5/5] Validating response mentions weather...")
        text_lower = text.lower()
        assert_true(
            "paris" in text_lower or "22" in text_lower or "sunny" in text_lower,
            f"Response should mention Paris, temperature, or condition: {text}"
        )
        print("      Response correctly uses tool result")

        print("\n" + "=" * 60)
        print("PASSED: Multi-turn single tool call succeeded")
        print("=" * 60)
        return 0

    except TestFailure as e:
        print(f"\n{'=' * 60}")
        print(f"FAILED: {e}")
        print("=" * 60)
        return 1

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: Unexpected exception: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("\nStopping proxy server...")
        proxy_proc.send_signal(signal.SIGTERM)
        proxy_proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
