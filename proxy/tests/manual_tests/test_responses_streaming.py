"""
Manual test: Responses API streaming -> Chat Completions streaming conversion
Run: uv run proxy/tests/manual_tests/test_responses_streaming.py
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

PROXY_PORT = 8768
PROXY_URL = f"http://localhost:{PROXY_PORT}/v1"
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")

REQUIRED_EVENT_TYPES = [
    "response.created",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
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


def main() -> int:
    from openai import OpenAI

    print("=" * 60)
    print("TEST: Responses API Streaming -> Chat Completions")
    print("=" * 60)

    print("\n[1/5] Starting proxy server...")
    proxy_proc = start_proxy()
    print("      Server ready")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))

        print("\n[2/5] Making streaming request through proxy...")
        events_received: list[str] = []
        accumulated_text = ""

        with proxy_client.responses.create(
            model="gpt-4o-mini",
            input="Say 'Hello, streaming world!' and nothing else.",
            stream=True,
        ) as stream:
            for event in stream:
                event_type = event.type
                events_received.append(event_type)

                if event_type == "response.output_text.delta":
                    accumulated_text += event.delta
                    print(f"      Delta: {event.delta!r}")

        print(f"\n      Total events: {len(events_received)}")
        print(f"      Accumulated text: {accumulated_text!r}")

        print("\n[3/5] Validating required event sequence...")
        for required in REQUIRED_EVENT_TYPES:
            assert_true(
                required in events_received,
                f"Missing required event type: {required}"
            )
            print(f"      Found: {required}")

        print("\n[4/5] Validating event order...")
        created_idx = events_received.index("response.created")
        completed_idx = events_received.index("response.completed")
        assert_true(
            created_idx < completed_idx,
            "response.created must come before response.completed"
        )

        if "response.output_text.delta" in events_received:
            first_delta_idx = events_received.index("response.output_text.delta")
            text_done_idx = events_received.index("response.output_text.done")
            assert_true(
                first_delta_idx < text_done_idx,
                "delta events must come before done events"
            )
        print("      Event order is correct")

        print("\n[5/5] Validating accumulated text...")
        assert_true(len(accumulated_text) > 0, "No text was streamed")
        print(f"      Final text: {accumulated_text}")

        print("\n" + "=" * 60)
        print("PASSED: All streaming validations succeeded")
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
