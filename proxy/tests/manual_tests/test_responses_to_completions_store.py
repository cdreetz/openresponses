"""
Manual test: Responses API with store -> Chat Completions conversion
Run: uv run proxy/tests/manual_tests/test_responses_to_completions_store.py
"""
import os
import sys
import time
import subprocess
import signal
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

PROXY_PORT = 8767
PROXY_URL = f"http://localhost:{PROXY_PORT}/v1"
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")

REQUIRED_RESPONSE_FIELDS = {
    "id": str,
    "object": str,
    "created_at": (int, float),
    "status": str,
    "model": str,
    "output": list,
    "usage": dict,
    "error": type(None),
    "incomplete_details": type(None),
    "metadata": dict,
    "temperature": (int, float),
    "top_p": (int, float),
    "tools": list,
    "tool_choice": str,
    "parallel_tool_calls": bool,
    "truncation": str,
    "store": bool,
    "frequency_penalty": (int, float),
    "presence_penalty": (int, float),
}

REQUIRED_USAGE_FIELDS = {
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
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


def validate_response_structure(resp: dict, context: str) -> None:
    for field, expected_type in REQUIRED_RESPONSE_FIELDS.items():
        value = assert_field_exists(resp, field, context)
        assert_type(value, expected_type, f"{context}.{field}")

    assert_eq(resp["object"], "response", f"{context}.object")
    assert_eq(resp["status"], "completed", f"{context}.status")
    assert_true(resp["id"].startswith("resp_"), f"{context}.id must start with 'resp_'")

    usage = resp["usage"]
    for field, expected_type in REQUIRED_USAGE_FIELDS.items():
        value = assert_field_exists(usage, field, f"{context}.usage")
        assert_type(value, expected_type, f"{context}.usage.{field}")


def validate_message_output(resp: dict, context: str) -> str:
    output = resp["output"]
    assert_true(len(output) > 0, f"{context}: output is empty")

    msg = output[0]
    assert_eq(msg.get("type"), "message", f"{context}: expected message output")
    assert_eq(msg.get("role"), "assistant", f"{context}: expected assistant role")
    assert_field_exists(msg, "id", f"{context}.output[0]")
    assert_field_exists(msg, "status", f"{context}.output[0]")

    content = assert_field_exists(msg, "content", f"{context}.output[0]")
    assert_type(content, list, f"{context}.output[0].content")
    assert_true(len(content) > 0, f"{context}.output[0].content is empty")

    text_item = content[0]
    assert_eq(text_item.get("type"), "output_text", f"{context}: expected output_text")
    text = assert_field_exists(text_item, "text", f"{context}.output[0].content[0]")
    assert_type(text, str, f"{context}.output[0].content[0].text")
    assert_true(len(text) > 0, f"{context}: response text is empty")

    return text


def main() -> int:
    from openai import OpenAI

    print("=" * 60)
    print("TEST: Responses API with Store -> Chat Completions")
    print("=" * 60)

    print("\n[1/8] Starting proxy server...")
    proxy_proc = start_proxy()
    print("      Server ready")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))
        model = "gpt-4o-mini"

        print("\n[2/8] Making initial request with store=True...")
        proxy_resp1 = proxy_client.responses.create(
            model=model,
            input="My name is Alice and I like Python.",
            store=True,
        )
        resp1_dict = proxy_resp1.model_dump()
        print(f"      Response ID: {resp1_dict['id']}")

        print("\n[3/8] Validating response 1 structure...")
        validate_response_structure(resp1_dict, "response1")
        assert_eq(resp1_dict["store"], True, "response1.store")
        text1 = validate_message_output(resp1_dict, "response1")
        print(f"      Output: {text1[:100]}...")

        print("\n[4/8] Making follow-up request with previous_response_id...")
        proxy_resp2 = proxy_client.responses.create(
            model=model,
            input="What is my name?",
            previous_response_id=proxy_resp1.id,
            store=True,
        )
        resp2_dict = proxy_resp2.model_dump()
        print(f"      Response ID: {resp2_dict['id']}")

        print("\n[5/8] Validating response 2 structure...")
        validate_response_structure(resp2_dict, "response2")
        assert_eq(resp2_dict["store"], True, "response2.store")
        assert_eq(
            resp2_dict["previous_response_id"],
            proxy_resp1.id,
            "response2.previous_response_id"
        )
        text2 = validate_message_output(resp2_dict, "response2")
        print(f"      Output: {text2[:100]}...")

        print("\n[6/8] Validating context retention (name remembered)...")
        assert_true(
            "alice" in text2.lower(),
            f"Context lost: expected 'alice' in response, got: {text2}"
        )
        print("      Name 'Alice' correctly remembered")

        print("\n[7/8] Making third request continuing the chain...")
        proxy_resp3 = proxy_client.responses.create(
            model=model,
            input="What programming language do I like?",
            previous_response_id=proxy_resp2.id,
            store=True,
        )
        resp3_dict = proxy_resp3.model_dump()
        print(f"      Response ID: {resp3_dict['id']}")

        validate_response_structure(resp3_dict, "response3")
        assert_eq(
            resp3_dict["previous_response_id"],
            proxy_resp2.id,
            "response3.previous_response_id"
        )
        text3 = validate_message_output(resp3_dict, "response3")
        print(f"      Output: {text3[:100]}...")

        print("\n[8/8] Validating chain context (language remembered)...")
        assert_true(
            "python" in text3.lower(),
            f"Context lost: expected 'python' in response, got: {text3}"
        )
        print("      Language 'Python' correctly remembered from chain")

        print("\n" + "=" * 60)
        print("PASSED: All store/chain validations succeeded")
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
