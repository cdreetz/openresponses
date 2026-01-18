"""
Manual test: Responses API streaming with store -> Chat Completions streaming conversion
Run: uv run proxy/tests/manual_tests/test_responses_streaming_store.py
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

PROXY_PORT = 8770
PROXY_URL = f"http://localhost:{PROXY_PORT}/v1"
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")


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


def stream_response(client: Any, **kwargs) -> tuple[str, str]:
    """Stream a response and return (response_id, accumulated_text)."""
    response_id = None
    accumulated_text = ""

    with client.responses.create(**kwargs, stream=True) as stream:
        for event in stream:
            if event.type == "response.created":
                response_id = event.response.id
            elif event.type == "response.output_text.delta":
                accumulated_text += event.delta

    assert_true(response_id is not None, "No response.created event received")
    return response_id, accumulated_text


def main() -> int:
    from openai import OpenAI

    print("=" * 60)
    print("TEST: Responses API Streaming with Store -> Chat Completions")
    print("=" * 60)

    print("\n[1/8] Starting proxy server...")
    proxy_proc = start_proxy()
    print("      Server ready")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))
        model = "gpt-4o-mini"

        print("\n[2/8] Making initial streaming request with store=True...")
        resp1_id, text1 = stream_response(
            proxy_client,
            model=model,
            input="My name is Bob and I enjoy hiking.",
            store=True,
        )
        print(f"      Response 1 ID: {resp1_id}")
        print(f"      Output: {text1[:100]}...")

        print("\n[3/8] Validating response 1 was stored...")
        assert_true(resp1_id.startswith("resp_"), "Response 1 ID format")
        print(f"      Response 1 stored with ID: {resp1_id}")

        print("\n[4/8] Making follow-up streaming request with previous_response_id...")
        resp2_id, text2 = stream_response(
            proxy_client,
            model=model,
            input="What is my name?",
            previous_response_id=resp1_id,
            store=True,
        )
        print(f"      Response 2 ID: {resp2_id}")
        print(f"      Output: {text2[:100]}...")

        print("\n[5/8] Validating context retention (name remembered)...")
        assert_true(
            "bob" in text2.lower(),
            f"Context lost: expected 'bob' in response, got: {text2}"
        )
        print("      Name 'Bob' correctly remembered")

        print("\n[6/8] Making third streaming request continuing the chain...")
        resp3_id, text3 = stream_response(
            proxy_client,
            model=model,
            input="What activity do I enjoy?",
            previous_response_id=resp2_id,
            store=True,
        )
        print(f"      Response 3 ID: {resp3_id}")
        print(f"      Output: {text3[:100]}...")

        print("\n[7/8] Validating chain context (activity remembered)...")
        assert_true(
            "hik" in text3.lower(),
            f"Context lost: expected 'hiking' in response, got: {text3}"
        )
        print("      Activity 'hiking' correctly remembered from chain")

        print("\n[8/8] Validating response IDs are unique...")
        ids = [resp1_id, resp2_id, resp3_id]
        assert_eq(len(ids), len(set(ids)), "All response IDs should be unique")
        print(f"      All 3 response IDs are unique")

        print("\n" + "=" * 60)
        print("PASSED: All streaming store validations succeeded")
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
