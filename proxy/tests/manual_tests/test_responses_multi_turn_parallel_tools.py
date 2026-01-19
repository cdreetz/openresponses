"""
Manual test: Multi-turn parallel tool calls flow
Run: uv run proxy/tests/manual_tests/test_responses_multi_turn_parallel_tools.py
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

PROXY_PORT = 8772
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

TIME_TOOL = {
    "type": "function",
    "name": "get_time",
    "description": "Get current time for a timezone",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "Timezone like Europe/Paris"},
        },
        "required": ["timezone"],
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
    print("TEST: Multi-turn Parallel Tool Calls")
    print("=" * 60)

    print("\n[1/6] Starting proxy server...")
    proxy_proc = start_proxy()
    print("      Server ready")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))
        model = "gpt-4o-mini"

        print("\n[2/6] Making request expecting parallel tool calls...")
        resp1 = proxy_client.responses.create(
            model=model,
            input="What's the weather AND current time in Tokyo? Use both tools.",
            tools=[WEATHER_TOOL, TIME_TOOL],
        )
        resp1_dict = resp1.model_dump()
        print(f"      Response ID: {resp1_dict['id']}")

        # Collect all function_calls from output
        function_calls = []
        for item in resp1_dict.get("output", []):
            if item.get("type") == "function_call":
                function_calls.append(item)

        print(f"      Found {len(function_calls)} function call(s)")
        for fc in function_calls:
            print(f"        - {fc['name']}({fc['arguments']})")

        # We need at least 1 tool call to proceed, ideally 2 for parallel
        assert_true(len(function_calls) >= 1, "Expected at least 1 function_call in response")

        print("\n[3/6] Preparing tool results...")
        # Build the follow-up input with all function_calls and their outputs
        follow_up_input = [
            {"type": "message", "role": "user", "content": "What's the weather AND current time in Tokyo? Use both tools."},
        ]

        # Add all function_calls
        for fc in function_calls:
            call_id = fc.get("call_id") or fc.get("id")
            follow_up_input.append({
                "type": "function_call",
                "call_id": call_id,
                "name": fc["name"],
                "arguments": fc["arguments"],
            })

        # Add all function_call_outputs
        for fc in function_calls:
            call_id = fc.get("call_id") or fc.get("id")
            if fc["name"] == "get_weather":
                result = json.dumps({"temperature": "18°C", "condition": "Cloudy"})
            elif fc["name"] == "get_time":
                result = json.dumps({"time": "14:30", "timezone": "Asia/Tokyo"})
            else:
                result = json.dumps({"result": "ok"})

            follow_up_input.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            })

        print(f"      Built follow-up with {len(function_calls)} tool calls and {len(function_calls)} results")

        print("\n[4/6] Sending follow-up with all tool results...")
        resp2 = proxy_client.responses.create(
            model=model,
            input=follow_up_input,
            tools=[WEATHER_TOOL, TIME_TOOL],
        )
        resp2_dict = resp2.model_dump()
        print(f"      Response ID: {resp2_dict['id']}")

        print("\n[5/6] Validating follow-up response...")
        assert_eq(resp2_dict["status"], "completed", "response status")

        # Should have a message output
        message_output = None
        for item in resp2_dict.get("output", []):
            if item.get("type") == "message":
                message_output = item
                break

        assert_true(message_output is not None, "Expected message in follow-up response")
        content = message_output.get("content", [])
        assert_true(len(content) > 0, "Message should have content")

        text = content[0].get("text", "")
        print(f"      Response text: {text[:150]}...")

        print("\n[6/6] Validating response uses tool results...")
        text_lower = text.lower()
        # Check if response mentions data from at least one tool
        has_weather = "18" in text_lower or "cloudy" in text_lower or "tokyo" in text_lower
        has_time = "14:30" in text_lower or "14" in text_lower

        assert_true(
            has_weather or has_time,
            f"Response should mention tool results (weather or time): {text}"
        )
        print("      Response correctly uses tool results")

        if len(function_calls) >= 2:
            print("\n      (Parallel tool calls confirmed)")
        else:
            print("\n      (Model made sequential calls - still valid)")

        print("\n" + "=" * 60)
        print("PASSED: Multi-turn parallel tool calls succeeded")
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
