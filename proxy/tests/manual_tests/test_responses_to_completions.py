"""
Manual test: Basic Responses API -> Chat Completions conversion
Run: uv run proxy/tests/manual_tests/test_responses_to_completions.py
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

PROXY_PORT = 8765
PROXY_URL = f"http://localhost:{PROXY_PORT}/v1"
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")

REQUIRED_RESPONSE_FIELDS = {
    "id": str,
    "object": str,
    "created_at": (int, float),
    "status": str,
    "model": str,
    "output": list,
    "usage": (dict, type(None)),
    "error": type(None),
    "tools": list,
    "temperature": (int, float),
    "top_p": (int, float),
    "metadata": dict,
}

REQUIRED_OUTPUT_MESSAGE_FIELDS = {
    "type": str,
    "id": str,
    "role": str,
    "status": str,
    "content": list,
}

REQUIRED_OUTPUT_TEXT_FIELDS = {
    "type": str,
    "text": str,
    "annotations": list,
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


def validate_response_structure(resp: dict, context: str) -> None:
    for field, expected_type in REQUIRED_RESPONSE_FIELDS.items():
        value = assert_field_exists(resp, field, context)
        assert_type(value, expected_type, f"{context}.{field}")

    assert_eq(resp["object"], "response", f"{context}.object")
    assert_true(resp["status"] in ("completed", "in_progress", "failed", "incomplete"),
                f"{context}.status has invalid value: {resp['status']}")


def validate_output_message(msg: dict, context: str) -> None:
    for field, expected_type in REQUIRED_OUTPUT_MESSAGE_FIELDS.items():
        value = assert_field_exists(msg, field, context)
        assert_type(value, expected_type, f"{context}.{field}")

    assert_eq(msg["type"], "message", f"{context}.type")
    assert_eq(msg["role"], "assistant", f"{context}.role")
    assert_eq(msg["status"], "completed", f"{context}.status")

    content = msg["content"]
    assert_true(len(content) > 0, f"{context}.content is empty")

    for i, part in enumerate(content):
        validate_output_text(part, f"{context}.content[{i}]")


def validate_output_text(part: dict, context: str) -> None:
    for field, expected_type in REQUIRED_OUTPUT_TEXT_FIELDS.items():
        value = assert_field_exists(part, field, context)
        assert_type(value, expected_type, f"{context}.{field}")

    assert_eq(part["type"], "output_text", f"{context}.type")
    assert_true(len(part["text"]) > 0, f"{context}.text is empty")


def validate_usage(usage: dict, context: str) -> None:
    if usage is None:
        return

    for field in ("input_tokens", "output_tokens", "total_tokens"):
        value = assert_field_exists(usage, field, context)
        assert_type(value, int, f"{context}.{field}")
        assert_true(value >= 0, f"{context}.{field} is negative: {value}")


def main() -> int:
    from openai import OpenAI

    print("=" * 60)
    print("TEST: Basic Responses API -> Chat Completions")
    print("=" * 60)

    print("\n[1/4] Starting proxy server...")
    proxy_proc = start_proxy()
    print("      Server ready")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))

        print("\n[2/4] Making request through proxy...")
        proxy_response = proxy_client.responses.create(
            model="gpt-4o-mini",
            input="Say hello in exactly 3 words.",
        )
        proxy_dict = proxy_response.model_dump()
        print(f"      Response ID: {proxy_dict.get('id', 'MISSING')}")

        print("\n[3/4] Validating response structure...")

        validate_response_structure(proxy_dict, "response")

        output = proxy_dict["output"]
        assert_true(len(output) > 0, "response.output is empty")

        for i, item in enumerate(output):
            item_type = assert_field_exists(item, "type", f"output[{i}]")
            if item_type == "message":
                validate_output_message(item, f"output[{i}]")
            elif item_type == "function_call":
                assert_field_exists(item, "name", f"output[{i}]")
                assert_field_exists(item, "arguments", f"output[{i}]")
            else:
                raise TestFailure(f"output[{i}]: unexpected type '{item_type}'")

        validate_usage(proxy_dict.get("usage"), "usage")

        print("\n[4/4] Verifying response content...")
        first_output = output[0]
        assert_eq(first_output["type"], "message", "First output should be a message")
        text = first_output["content"][0]["text"]
        assert_true(len(text) > 0, "Response text is empty")
        print(f"      Text: {text[:100]}...")

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
