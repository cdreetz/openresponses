"""
Manual test: Run actual opencode against the proxy
Run: uv run proxy/tests/manual_tests/test_with_opencode.py
"""
import os
import sys
import time
import subprocess
import signal
import urllib.request
import urllib.error
from pathlib import Path


PROXY_PORT = 8790
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
        stderr=subprocess.STDOUT,
    )

    if not wait_for_server(PROXY_URL):
        proc.terminate()
        stdout, _ = proc.communicate(timeout=5)
        raise RuntimeError(f"Server failed to start.\n{stdout.decode()}")

    return proc


def main() -> int:
    print("=" * 60)
    print("TEST: Run opencode against proxy")
    print("=" * 60)

    # Check opencode is installed
    opencode_path = subprocess.run(
        ["which", "opencode"], capture_output=True, text=True
    )
    if opencode_path.returncode != 0:
        print("ERROR: opencode not found. Install it first.")
        return 1
    print(f"Found opencode at: {opencode_path.stdout.strip()}")

    print("\n[1/3] Starting proxy server...")
    proxy_proc = start_proxy()
    print(f"      Proxy ready at {PROXY_URL}")

    try:
        print("\n[2/3] Running opencode with a simple task...")
        print("      (This will show what opencode actually sends)")
        print("-" * 60)

        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = PROXY_URL
        env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

        # Run opencode with a simple task that should trigger tool use
        result = subprocess.run(
            [
                "opencode",
                "--model", "openai/gpt-4o-mini",
                "run",
                "--format=json",
                "List the files in the current directory using ls, then tell me which ones are Python files."
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(get_repo_root()),
        )

        print("-" * 60)
        print("\n[3/3] Results:")
        print(f"      Exit code: {result.returncode}")

        if result.stdout:
            print("\n      STDOUT:")
            for line in result.stdout.split('\n')[:20]:
                print(f"        {line}")
            if len(result.stdout.split('\n')) > 20:
                print("        ... (truncated)")

        if result.stderr:
            print("\n      STDERR:")
            for line in result.stderr.split('\n')[:20]:
                print(f"        {line}")
            if len(result.stderr.split('\n')) > 20:
                print("        ... (truncated)")

        print("\n" + "=" * 60)
        print("Check the proxy server output above to see what opencode sent")
        print("=" * 60)
        return result.returncode

    except subprocess.TimeoutExpired:
        print("\nERROR: opencode timed out after 120 seconds")
        return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("\nStopping proxy server...")
        proxy_proc.send_signal(signal.SIGTERM)
        try:
            stdout, _ = proxy_proc.communicate(timeout=5)
            print("\n" + "=" * 60)
            print("PROXY SERVER OUTPUT:")
            print("=" * 60)
            print(stdout.decode())
        except:
            proxy_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
