"""
Manual test: Responses API with tools -> Chat Completions conversion
Run: uv run proxy/tests/manual_tests/test_responses_to_completions_tools.py
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

PROXY_PORT = 8766
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

REQUIRED_FUNCTION_CALL_FIELDS = {
    "type": str,
    "id": str,
    "name": str,
    "arguments": str,
    "status": str,
}


class TestFailure(Exception):
    pass


def assert_eq(actual: Any, expected: Any, msg: str) -> None:
    if actual != expected:
        raise TestFailure(f"{msg}: expected {expected!r}, got {actual!r}")


def assert_true(condition: bool, msg: str) -> None:
    if not condition:
        raise TestFailure(msg)


def assert_type(value: Any, expected_type: type | tuple, field: str) -> None:
    if not isinstance(value, expected_type):
        raise TestFailure(
            f"Field '{field}': expected type {expected_type}, got {type(value).__name__}"
        )


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
            urllib.request.urlopen(f"{url.removesuffix('/v1')}/health", timeout=1)
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


def validate_function_call(fc: dict, context: str) -> None:
    for field, expected_type in REQUIRED_FUNCTION_CALL_FIELDS.items():
        value = assert_field_exists(fc, field, context)
        assert_type(value, expected_type, f"{context}.{field}")

    assert_eq(fc["type"], "function_call", f"{context}.type")
    assert_eq(fc["status"], "completed", f"{context}.status")

    assert_true(len(fc["id"]) > 0, f"{context}.id is empty")
    assert_true(len(fc["name"]) > 0, f"{context}.name is empty")

    try:
        args = json.loads(fc["arguments"])
        assert_type(args, dict, f"{context}.arguments (parsed)")
    except json.JSONDecodeError as e:
        raise TestFailure(f"{context}.arguments is not valid JSON: {e}")


def main() -> int:
    from openai import OpenAI

    print("=" * 60)
    print("TEST: Responses API with Tools -> Chat Completions")
    print("=" * 60)

    print("\n[1/5] Starting proxy server...")
    proxy_proc = start_proxy()
    print("      Server ready")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))

        print("\n[2/5] Making request through proxy with tools...")
        proxy_response = proxy_client.responses.create(
            model="gpt-4o-mini",
            input="What's the weather in Tokyo?",
            tools=[WEATHER_TOOL],
        )
        proxy_dict = proxy_response.model_dump()
        print(f"      Response ID: {proxy_dict.get('id', 'MISSING')}")

        print("\n[3/5] Validating response has function_call output...")
        output = assert_field_exists(proxy_dict, "output", "response")
        assert_type(output, list, "response.output")
        assert_true(len(output) > 0, "response.output is empty")

        function_calls = [item for item in output if item.get("type") == "function_call"]
        assert_true(
            len(function_calls) > 0,
            f"No function_call in output. Got types: {[item.get('type') for item in output]}"
        )

        print(f"      Found {len(function_calls)} function call(s)")

        print("\n[4/5] Validating function call structure...")
        for i, fc in enumerate(function_calls):
            validate_function_call(fc, f"function_call[{i}]")
            print(f"      [{i}] name={fc['name']}, args={fc['arguments']}")

        print("\n[5/5] Validating function call content...")
        first_fc = function_calls[0]
        assert_eq(first_fc["name"], "get_weather", "function_call.name")

        args = json.loads(first_fc["arguments"])
        location = assert_field_exists(args, "location", "function_call.arguments")
        assert_type(location, str, "function_call.arguments.location")
        assert_true(len(location) > 0, "function_call.arguments.location is empty")
        assert_true(
            "tokyo" in location.lower(),
            f"Expected 'tokyo' in location, got: {location}"
        )
        print(f"      Location extracted: {location}")

        print("\n" + "=" * 60)
        print("PASSED: All validations succeeded")
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
