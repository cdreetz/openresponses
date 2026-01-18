"""
Manual test: Responses API with store -> Chat Completions conversion
Run: uv run tests/manual_tests/test_responses_to_completions_store.py
"""
import os
import time
import subprocess
import signal
import sys
import urllib.request
import urllib.error
from pathlib import Path
from openai import OpenAI

PROXY_PORT = 8767
PROXY_URL = f"http://localhost:{PROXY_PORT}/v1"
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")


def get_repo_root() -> Path:
    path = Path(__file__).resolve()
    while path.parent != path:
        if (path / "proxy").is_dir() and (path / "proxy" / "cli.py").exists():
            return path
        path = path.parent
    raise RuntimeError("Could not find repo root")


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
        print("Server failed to start!")
        print(f"stdout: {stdout.decode()}")
        print(f"stderr: {stderr.decode()}")
        raise RuntimeError("Proxy server failed to start")

    return proc


def main():
    print("Starting proxy server...")
    proxy_proc = start_proxy()
    print("Server ready!")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))
        real_client = OpenAI()

        model = "gpt-4o-mini"

        print(f"\n{'='*60}")
        print("PROXY: Step 1 - Initial request with store=True")
        print(f"{'='*60}")

        proxy_resp1 = proxy_client.responses.create(
            model=model,
            input="My name is Alice and I like Python.",
            store=True,
        )
        print(f"Response 1 ID: {proxy_resp1.id}")
        print(f"Output: {proxy_resp1.output[0].content[0].text if proxy_resp1.output else 'None'}")

        print(f"\n{'='*60}")
        print("PROXY: Step 2 - Follow-up with previous_response_id")
        print(f"{'='*60}")

        proxy_resp2 = proxy_client.responses.create(
            model=model,
            input="What is my name?",
            previous_response_id=proxy_resp1.id,
            store=True,
        )
        print(f"Response 2 ID: {proxy_resp2.id}")
        print(f"Output: {proxy_resp2.output[0].content[0].text if proxy_resp2.output else 'None'}")

        print(f"\n{'='*60}")
        print("PROXY: Step 3 - Continue the chain")
        print(f"{'='*60}")

        proxy_resp3 = proxy_client.responses.create(
            model=model,
            input="What language do I like?",
            previous_response_id=proxy_resp2.id,
            store=True,
        )
        print(f"Response 3 ID: {proxy_resp3.id}")
        print(f"Output: {proxy_resp3.output[0].content[0].text if proxy_resp3.output else 'None'}")

        print(f"\n{'='*60}")
        print("REAL OpenAI: Same conversation flow")
        print(f"{'='*60}")

        real_resp1 = real_client.responses.create(
            model=model,
            input="My name is Alice and I like Python.",
            store=True,
        )
        print(f"Real Response 1 ID: {real_resp1.id}")

        real_resp2 = real_client.responses.create(
            model=model,
            input="What is my name?",
            previous_response_id=real_resp1.id,
            store=True,
        )
        print(f"Real Response 2 ID: {real_resp2.id}")
        print(f"Real Output: {real_resp2.output[0].content[0].text if real_resp2.output else 'None'}")

        print(f"\n{'='*60}")
        print("Verification")
        print(f"{'='*60}")

        proxy_mentions_alice = "alice" in proxy_resp2.output[0].content[0].text.lower()
        real_mentions_alice = "alice" in real_resp2.output[0].content[0].text.lower()

        print(f"Proxy remembered name: {proxy_mentions_alice}")
        print(f"Real remembered name: {real_mentions_alice}")

        proxy_mentions_python = "python" in proxy_resp3.output[0].content[0].text.lower()
        print(f"Proxy remembered language: {proxy_mentions_python}")

        if proxy_mentions_alice and proxy_mentions_python:
            print("\nStore/chain working correctly!")
        else:
            print("\nStore/chain may have issues.")

    finally:
        print("\nStopping proxy server...")
        proxy_proc.send_signal(signal.SIGTERM)
        proxy_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
