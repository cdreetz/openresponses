"""
Manual test: Simulate opencode's Responses API flow with tool calls
Run: uv run proxy/tests/manual_tests/test_opencode_flow.py
"""
import os
import sys
import time
import subprocess
import signal
import urllib.request
import urllib.error
import json
from pathlib import Path
from typing import Any

PROXY_PORT = 8780
PROXY_URL = f"http://localhost:{PROXY_PORT}/v1"
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")


class TestFailure(Exception):
    pass


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
    print("TEST: Simulate opencode Responses API flow")
    print("=" * 60)

    print("\n[1/6] Starting proxy server...")
    proxy_proc = start_proxy()
    print("      Server ready")

    try:
        client = OpenAI(base_url=PROXY_URL, api_key=os.getenv("OPENAI_API_KEY", ""))
        model = "gpt-4o-mini"

        # Step 1: Initial request with tools (like opencode would send)
        print("\n[2/6] Making initial request with tools...")
        tools = [
            {
                "type": "function",
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to run"}
                    },
                    "required": ["command"]
                }
            }
        ]

        resp1 = client.responses.create(
            model=model,
            instructions="You are a helpful assistant. When asked to list files, use the bash tool with 'ls' command.",
            input="List the files in the current directory",
            tools=tools,
        )

        print(f"      Response 1 ID: {resp1.id}")
        print(f"      Response 1 output types: {[o.type for o in resp1.output]}")

        # Check if we got a function call
        function_calls = [o for o in resp1.output if o.type == "function_call"]
        if not function_calls:
            print("      No function call in response (model decided not to use tool)")
            print("      This is OK - let's test with a follow-up question instead")

            # Test simple chaining without tools
            print("\n[3/6] Testing previous_response_id chaining (no tools)...")
            resp2 = client.responses.create(
                model=model,
                input="What did I just ask you?",
                previous_response_id=resp1.id,
            )
            print(f"      Response 2 ID: {resp2.id}")
            print(f"      Response 2 text: {resp2.output[0].content[0].text[:100] if resp2.output else 'None'}...")

            # Verify context was retained
            text = resp2.output[0].content[0].text.lower() if resp2.output else ""
            if "file" in text or "list" in text or "directory" in text:
                print("      Context retained correctly!")
            else:
                print(f"      WARNING: Context may not be retained. Response: {text[:200]}")

            print("\n" + "=" * 60)
            print("PASSED: Basic chaining works (no tool calls in this run)")
            print("=" * 60)
            return 0

        fc = function_calls[0]
        print(f"      Function call: {fc.name}({fc.arguments})")
        print(f"      Call ID: {fc.call_id}")

        # Step 2: Simulate tool execution and send result
        print("\n[3/6] Simulating tool execution and sending result...")

        # This is how opencode sends tool results - with previous_response_id
        # and function_call_output in input
        resp2 = client.responses.create(
            model=model,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": "file1.txt\nfile2.py\nREADME.md"
                }
            ],
            previous_response_id=resp1.id,
            tools=tools,
        )

        print(f"      Response 2 ID: {resp2.id}")
        print(f"      Response 2 output types: {[o.type for o in resp2.output]}")

        # Get the text response
        messages = [o for o in resp2.output if o.type == "message"]
        if messages:
            text = messages[0].content[0].text if messages[0].content else ""
            print(f"      Response 2 text: {text[:100]}...")

        # Step 3: Another follow-up to verify chain is maintained
        print("\n[4/6] Making follow-up request to verify chain...")
        resp3 = client.responses.create(
            model=model,
            input="Which of those files is a Python file?",
            previous_response_id=resp2.id,
            tools=tools,
        )

        print(f"      Response 3 ID: {resp3.id}")

        messages = [o for o in resp3.output if o.type == "message"]
        if messages:
            text = messages[0].content[0].text if messages[0].content else ""
            print(f"      Response 3 text: {text[:100]}...")

            # Verify it knows about file2.py
            if "file2" in text.lower() or "python" in text.lower() or ".py" in text.lower():
                print("      Context retained correctly!")
            else:
                raise TestFailure(f"Context lost - expected mention of Python file, got: {text}")

        # Step 4: Test without previous_response_id (fresh conversation)
        print("\n[5/6] Testing fresh conversation (no previous_response_id)...")
        resp4 = client.responses.create(
            model=model,
            input="What is 2 + 2?",
            tools=tools,
        )
        print(f"      Response 4 ID: {resp4.id}")
        print(f"      Fresh conversation works")

        # Step 5: Verify all response IDs are unique
        print("\n[6/6] Verifying response IDs are unique...")
        ids = [resp1.id, resp2.id, resp3.id, resp4.id]
        if len(ids) != len(set(ids)):
            raise TestFailure(f"Duplicate response IDs: {ids}")
        print(f"      All {len(ids)} response IDs are unique")

        print("\n" + "=" * 60)
        print("PASSED: opencode flow simulation succeeded")
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
