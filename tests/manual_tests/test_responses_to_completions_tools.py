"""
Manual test: Responses API with tools -> Chat Completions conversion
Run from repo root: python -m tests.manual_tests.test_responses_to_completions_tools
"""
import os
import time
import subprocess
import signal
import sys
import urllib.request
import urllib.error
from openai import OpenAI

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


def main():
    print("Starting proxy server...")
    proxy_proc = start_proxy()
    print("Server ready!")

    try:
        proxy_client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))
        real_client = OpenAI()

        test_input = "What's the weather in Tokyo?"
        model = "gpt-4o-mini"

        print(f"\n{'='*60}")
        print("Making request through PROXY with tools...")
        print(f"{'='*60}")

        proxy_response = proxy_client.responses.create(
            model=model,
            input=test_input,
            tools=[WEATHER_TOOL],
        )
        print(f"Proxy response ID: {proxy_response.id}")
        print(f"Proxy output items: {len(proxy_response.output)}")
        for i, item in enumerate(proxy_response.output):
            print(f"  [{i}] type={item.type}")
            if item.type == "function_call":
                print(f"      name={item.name}, arguments={item.arguments}")

        print(f"\n{'='*60}")
        print("Making request to REAL OpenAI with tools...")
        print(f"{'='*60}")

        real_response = real_client.responses.create(
            model=model,
            input=test_input,
            tools=[WEATHER_TOOL],
        )
        print(f"Real response ID: {real_response.id}")
        print(f"Real output items: {len(real_response.output)}")
        for i, item in enumerate(real_response.output):
            print(f"  [{i}] type={item.type}")
            if item.type == "function_call":
                print(f"      name={item.name}, arguments={item.arguments}")

        print(f"\n{'='*60}")
        print("Comparing output item types...")
        print(f"{'='*60}")

        proxy_types = [item.type for item in proxy_response.output]
        real_types = [item.type for item in real_response.output]

        print(f"Proxy output types: {proxy_types}")
        print(f"Real output types: {real_types}")

        if proxy_types == real_types:
            print("Output item types match!")
        else:
            print("Output item types differ!")

    finally:
        print("\nStopping proxy server...")
        proxy_proc.send_signal(signal.SIGTERM)
        proxy_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
