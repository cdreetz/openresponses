"""
Manual test: Basic Responses API -> Chat Completions conversion
Run from repo root: python -m tests.manual_tests.test_responses_to_completions
"""
import os
import time
import subprocess
import signal
import sys
import urllib.request
import urllib.error
from openai import OpenAI

PROXY_PORT = 8765
PROXY_URL = f"http://localhost:{PROXY_PORT}/v1"
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")


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
    env = os.environ.copy()
    env["UPSTREAM_URL"] = UPSTREAM_URL

    proc = subprocess.Popen(
        [sys.executable, "-m", "proxy.cli", "--port", str(PROXY_PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not wait_for_server(PROXY_URL):
        stdout, stderr = proc.communicate(timeout=2)
        print("Server failed to start!")
        print(f"stdout: {stdout.decode()}")
        print(f"stderr: {stderr.decode()}")
        raise RuntimeError("Proxy server failed to start")

    return proc


def compare_structure(proxy_resp: dict, real_resp: dict, path: str = "") -> list[str]:
    diffs = []

    proxy_keys = set(proxy_resp.keys()) if isinstance(proxy_resp, dict) else set()
    real_keys = set(real_resp.keys()) if isinstance(real_resp, dict) else set()

    missing = real_keys - proxy_keys
    extra = proxy_keys - real_keys

    if missing:
        diffs.append(f"{path}: missing keys {missing}")
    if extra:
        diffs.append(f"{path}: extra keys {extra}")

    for key in proxy_keys & real_keys:
        p_val = proxy_resp[key]
        r_val = real_resp[key]

        if type(p_val) != type(r_val) and not (p_val is None or r_val is None):
            diffs.append(f"{path}.{key}: type mismatch {type(p_val).__name__} vs {type(r_val).__name__}")
        elif isinstance(p_val, dict) and isinstance(r_val, dict):
            diffs.extend(compare_structure(p_val, r_val, f"{path}.{key}"))
        elif isinstance(p_val, list) and isinstance(r_val, list) and p_val and r_val:
            if isinstance(p_val[0], dict) and isinstance(r_val[0], dict):
                diffs.extend(compare_structure(p_val[0], r_val[0], f"{path}.{key}[0]"))

    return diffs


def main():
    print("Starting proxy server...")
    proxy_proc = start_proxy()
    print("Server ready!")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))
        real_client = OpenAI()

        test_input = "Say hello in exactly 3 words."
        model = "gpt-4o-mini"

        print(f"\n{'='*60}")
        print("Making request through PROXY...")
        print(f"{'='*60}")

        proxy_response = proxy_client.responses.create(
            model=model,
            input=test_input,
        )
        print(f"Proxy response ID: {proxy_response.id}")
        print(f"Proxy output: {proxy_response.output}")

        print(f"\n{'='*60}")
        print("Making request to REAL OpenAI...")
        print(f"{'='*60}")

        real_response = real_client.responses.create(
            model=model,
            input=test_input,
        )
        print(f"Real response ID: {real_response.id}")
        print(f"Real output: {real_response.output}")

        print(f"\n{'='*60}")
        print("Comparing response structures...")
        print(f"{'='*60}")

        proxy_dict = proxy_response.model_dump()
        real_dict = real_response.model_dump()

        diffs = compare_structure(proxy_dict, real_dict)

        if diffs:
            print("Structure differences found:")
            for diff in diffs:
                print(f"  - {diff}")
        else:
            print("Structures match!")

        print(f"\n{'='*60}")
        print("PROXY response keys:", list(proxy_dict.keys()))
        print(f"{'='*60}")
        print("REAL response keys:", list(real_dict.keys()))

    finally:
        print("\nStopping proxy server...")
        proxy_proc.send_signal(signal.SIGTERM)
        proxy_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
