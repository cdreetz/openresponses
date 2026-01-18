"""
Manual test: Responses API streaming with tools -> Chat Completions streaming conversion
Run: uv run proxy/tests/manual_tests/test_responses_streaming_tools.py
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

PROXY_PORT = 8769
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

REQUIRED_TOOL_EVENT_TYPES = [
    "response.created",
    "response.output_item.added",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.output_item.done",
    "response.completed",
]


class TestFailure(Exception):
    pass


def assert_eq(actual: Any, expected: Any, msg: str) -> None:
    if actual != expected:
        raise TestFailure(f"{msg}: expected {expected!r}, got {actual!r}")


def assert_true(condition: bool, msg: str) -> None:
    if not condition:
        raise TestFailure(msg)


def assert_field_exists(obj: dict, field: str, context: str) -> Any:
    if field not in obj:
        raise TestFailure(f"{context}: missing required field '{field}'")
    return obj[field]


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
    print("TEST: Responses API Streaming with Tools -> Chat Completions")
    print("=" * 60)

    print("\n[1/6] Starting proxy server...")
    proxy_proc = start_proxy()
    print("      Server ready")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))

        print("\n[2/6] Making streaming request with tools...")
        events_received: list[str] = []
        function_call_name: str | None = None
        accumulated_args = ""
        final_response = None

        with proxy_client.responses.create(
            model="gpt-4o-mini",
            input="What's the weather in Tokyo?",
            tools=[WEATHER_TOOL],
            stream=True,
        ) as stream:
            for event in stream:
                event_type = event.type
                events_received.append(event_type)

                if event_type == "response.output_item.added":
                    if hasattr(event, "item") and event.item.type == "function_call":
                        function_call_name = event.item.name
                        print(f"      Function call started: {function_call_name}")

                if event_type == "response.function_call_arguments.delta":
                    accumulated_args += event.delta
                    print(f"      Args delta: {event.delta!r}")

                if event_type == "response.completed":
                    final_response = event.response

        print(f"\n      Total events: {len(events_received)}")
        print(f"      Function: {function_call_name}")
        print(f"      Arguments: {accumulated_args}")

        print("\n[3/6] Validating required event types...")
        for required in REQUIRED_TOOL_EVENT_TYPES:
            assert_true(
                required in events_received,
                f"Missing required event type: {required}"
            )
            print(f"      Found: {required}")

        print("\n[4/6] Validating function call name...")
        assert_eq(function_call_name, "get_weather", "function call name")
        print(f"      Function name: {function_call_name}")

        print("\n[5/6] Validating function call arguments...")
        assert_true(len(accumulated_args) > 0, "No arguments were streamed")
        args = json.loads(accumulated_args)
        location = assert_field_exists(args, "location", "function_call.arguments")
        assert_true(
            "tokyo" in location.lower(),
            f"Expected 'tokyo' in location, got: {location}"
        )
        print(f"      Location: {location}")

        print("\n[6/6] Validating final response...")
        assert_true(final_response is not None, "No completed response received")
        assert_eq(final_response.status, "completed", "final response status")
        assert_true(len(final_response.output) > 0, "final response output is empty")

        fc_output = None
        for item in final_response.output:
            if item.type == "function_call":
                fc_output = item
                break

        assert_true(fc_output is not None, "No function_call in final output")
        assert_eq(fc_output.name, "get_weather", "final function_call name")
        print(f"      Final response validated")

        print("\n" + "=" * 60)
        print("PASSED: All streaming tools validations succeeded")
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
