import asyncio
import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from aiohttp import web
from datasets import Dataset
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
import verifiers as vf
from verifiers.types import State

from harbor.agents.factory import AgentFactory
from harbor.models.agent.name import AgentName

from src.harbor_tasks import HarborTask, load_tasks_from_dir
from src.harbor_sandbox_env import SandboxHarborEnvironment, get_bg_command

from proxy.handlers.handle_responses_api import (
    responses_request_to_chat,
    chat_response_to_responses,
)
from proxy.handlers.streaming import (
    build_response_created_event,
    build_output_item_added_event,
    build_content_part_added_event,
    build_text_done_event,
    build_content_part_done_event,
    build_output_item_done_event,
    build_response_completed_event,
    build_function_call_item_added_event,
    build_function_call_args_done_event,
    build_function_call_item_done_event,
    format_sse_event,
)
from proxy.utils.utils import generate_id
from proxy import store


class TerminalBenchEnv(vf.CliAgentEnv):
    def __init__(
        self,
        run_command=None,
        task_dir=None,
        task_paths=None,
        default_docker_image="ubuntu:24.04",
        interception_port=8765,
        agent_name: str = "claude-code",
        agent_kwargs: dict[str, Any] | None = None,
        max_turns=3,
        message_type="chat",
        **kwargs
    ):
        if task_dir:
            self.tasks = load_tasks_from_dir(Path(task_dir))
        elif task_paths:
            self.tasks = [HarborTask.from_path(Path(p)) for p in task_paths]
        else:
            raise ValueError("Need task_dir or task_paths")

        rows = [{
            "example_id": i,
            "prompt": [{"role": "user", "content": t.instruction}],
            "task_name": t.name,
            "task_path": str(t.task_dir),
        } for i, t in enumerate(self.tasks)]

        super().__init__(
            run_command=run_command,
            dataset=Dataset.from_list(rows),
            max_turns=max_turns,
            **kwargs,
        )
        self._task_cache = {t.name: t for t in self.tasks}
        self.sandbox_client = AsyncSandboxClient()
        self._default_image = default_docker_image
        self._tunnel = None
        self._tunnel_port = interception_port
        self._agent_name = agent_name
        self._agent_kwargs = agent_kwargs or {}
        self.run_command = run_command
        self._completed_rollouts: set[str] = set()

    async def post_sandbox_setup(self, state, sandbox_client):
        task = self._get_task(state)

        await self.sandbox_client.execute_command(
            state["sandbox_id"],
            "mkdir -p /app /logs/agent/sessions /logs/verifier /tests",
        )

        await self.sandbox_client.upload_file(
            state["sandbox_id"],
            "/app/instruction.md",
            task.task_dir / "instruction.md",
        )
        logs_dir = Path("/tmp") / "terminal-bench-agent" / state["rollout_id"]
        logs_dir.mkdir(parents=True, exist_ok=True)

        agent_model_name = state["model"]
        if "/" not in agent_model_name:
            agent_model_name = f"openai/{agent_model_name}"

        agent = AgentFactory.create_agent_from_name(
            AgentName(self._agent_name),
            logs_dir=logs_dir,
            model_name=agent_model_name,
            **self._agent_kwargs,
        )

        harbor_env = SandboxHarborEnvironment(
            sandbox_client=self.sandbox_client,
            sandbox_id=state["sandbox_id"],
        )

        await agent.setup(harbor_env)

        run_commands = agent.create_run_agent_commands(task.instruction)
        bg_command = await get_bg_command(sandbox_client, state["sandbox_id"], run_commands)
        state["bg_command"] = bg_command

        return state

    async def start_agent(
        self, state: State, sandbox_client: AsyncSandboxClient
    ) -> None:
        sandbox_id = state["sandbox_id"]
        run_command = state["bg_command"]

        print(f"=== STARTING AGENT ===")
        print(f"Sandbox ID: {sandbox_id}")
        print(f"Interception URL: {state.get('interception_base_url')}")
        print(f"Command: {run_command[:500]}...")

        wrapped_command = f"""
{run_command}
EXIT_CODE=$?
echo $EXIT_CODE > /tmp/vf_exit_code
touch /tmp/vf_complete
"""
        await sandbox_client.execute_command(
            sandbox_id,
            f"nohup bash -c {shlex.quote(wrapped_command)} "
            f"> /tmp/agent_stdout.log 2> /tmp/agent_stderr.log &"
        )

        state["completion_wait_task"] = asyncio.create_task(
            self.wait_for_completion(state),
        )

    async def ensure_interception_server(self):
        """Set up interception server with both chat completions and responses routes."""
        async with self.server_lock:
            if self.interception_server is not None:
                return

            app = web.Application()
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self.handle_intercepted_request,
            )
            app.router.add_post(
                "/rollout/{rollout_id}/v1/responses",
                self.handle_responses_request,
            )

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.interception_port)
            await site.start()

            self.interception_server = app
            self.server_runner = runner
            self.server_site = site

    async def handle_responses_request(self, request) -> web.Response:
        """Handle Responses API requests by converting to/from Chat Completions."""
        rollout_id = request.match_info["rollout_id"]

        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "Rollout not found"}, status=404)

        body = await request.json()

        # Skip title generation requests
        if body.get("model") == "gpt-5-nano":
            return self._fake_title_response()

        # Debug: see what opencode is sending
        print(f"=== RAW REQUEST ===")
        print(f"previous_response_id: {body.get('previous_response_id')}")
        print(f"stream: {body.get('stream')}")
        print(f"input type: {type(body.get('input'))}")
        if isinstance(body.get('input'), list):
            print(f"input items: {[item.get('type') if isinstance(item, dict) else type(item) for item in body.get('input', [])]}")
        else:
            print(f"input: {str(body.get('input', ''))[:100]}")

        # Convert Responses API request to Chat Completions format
        # This handles previous_response_id by looking up history in the store
        chat_body = responses_request_to_chat(body)
        messages = chat_body["messages"]
        tools = chat_body.get("tools")

        print(f"=== RESPONSES REQUEST (rollout={rollout_id}) ===")
        print(f"Messages: {len(messages)}, Tools: {len(tools) if tools else 0}")

        # Queue request for verifiers framework
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": messages,
            "model": body.get("model"),
            "tools": tools,
            "response_future": asyncio.Future(),
        }

        self.intercepts[request_id] = intercept
        await context["request_id_queue"].put(request_id)

        # Wait for model response
        try:
            response = await asyncio.wait_for(intercept["response_future"], timeout=300)
        except asyncio.CancelledError:
            return web.json_response({"error": "Cancelled"}, status=499)
        except asyncio.TimeoutError:
            return web.json_response({"error": "Timeout"}, status=504)

        response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)

        # Handle streaming vs non-streaming response
        if body.get("stream", False):
            return await self._stream_response(request, response_dict, body)

        # Non-streaming: return JSON
        responses_output = chat_response_to_responses(response_dict, body)
        responses_output["input"] = body.get("input", [])
        store.save(responses_output)
        print(f"=== SAVED TO STORE: {responses_output['id']} ===")
        return web.json_response(responses_output)

    async def _stream_response(
        self, request, chat_response: dict, original_body: dict
    ) -> web.StreamResponse:
        """Stream the response as SSE events."""
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        response_id = generate_id("resp")
        message_id = generate_id("msg")

        # Send response.created
        created_event = build_response_created_event(response_id, original_body)
        await response.write(format_sse_event(created_event).encode())

        # Get the message from chat response
        choice = chat_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        # Build output list for final event
        final_output = []

        # Handle text content
        if content:
            # Send output_item.added for message
            item_added = build_output_item_added_event(response_id, message_id, 0)
            await response.write(format_sse_event(item_added).encode())

            # Send content_part.added
            part_added = build_content_part_added_event(response_id, message_id, 0)
            await response.write(format_sse_event(part_added).encode())

            # Send text.done (all at once since we have complete response)
            text_done = build_text_done_event(response_id, message_id, 0, content)
            await response.write(format_sse_event(text_done).encode())

            # Send content_part.done
            part_done = build_content_part_done_event(response_id, message_id, 0, content)
            await response.write(format_sse_event(part_done).encode())

            # Send output_item.done
            item_done = build_output_item_done_event(response_id, message_id, 0, content)
            await response.write(format_sse_event(item_done).encode())

            final_output.append({
                "type": "message",
                "id": message_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            })

        # Handle tool calls
        for i, tc in enumerate(tool_calls):
            fc_id = generate_id("fc")
            call_id = tc.get("id", generate_id("call"))
            name = tc["function"]["name"]
            arguments = tc["function"]["arguments"]
            output_index = len(final_output)

            # Send function_call item added
            fc_added = build_function_call_item_added_event(
                response_id, fc_id, call_id, name, output_index
            )
            await response.write(format_sse_event(fc_added).encode())

            # Send arguments done
            args_done = build_function_call_args_done_event(
                response_id, fc_id, call_id, arguments, output_index
            )
            await response.write(format_sse_event(args_done).encode())

            # Send function_call item done
            fc_done = build_function_call_item_done_event(
                response_id, fc_id, call_id, name, arguments, output_index
            )
            await response.write(format_sse_event(fc_done).encode())

            final_output.append({
                "type": "function_call",
                "id": fc_id,
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed",
            })

        # Send response.completed
        usage = chat_response.get("usage", {})
        completed_event = build_response_completed_event(
            response_id, final_output, original_body, usage
        )
        await response.write(format_sse_event(completed_event).encode())

        # Save to store
        responses_output = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "output": final_output,
            "input": original_body.get("input", []),
        }
        store.save(responses_output)
        print(f"=== SAVED TO STORE (streaming): {response_id} ===")

        await response.write_eof()
        return response

    def _fake_title_response(self) -> web.Response:
        """Return a fake response for title generation requests."""
        return web.json_response({
            "id": f"resp_{uuid.uuid4().hex[:8]}",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "output": [{
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Task", "annotations": []}]
            }],
        })

    def _get_task(self, state) -> HarborTask:
        name = state.get("task_name")
        if name and name in self._task_cache:
            return self._task_cache[name]
        return HarborTask.from_path(Path(state["input"]["task_path"]))

    async def get_docker_image(self, state: State) -> str:
        task = self._get_task(state)
        return task.docker_image or self._default_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Build environment variables for the sandbox."""
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        return env_vars

    async def post_rollout(self, state: State):
        sandbox_id = state["sandbox_id"]

        print("=== AGENT STDOUT ===")
        r = await self.sandbox_client.execute_command(
            sandbox_id, "cat /tmp/agent_stdout.log 2>/dev/null || echo 'No stdout log'"
        )
        print(getattr(r, "stdout", "") or str(r))

        print("=== AGENT STDERR ===")
        r = await self.sandbox_client.execute_command(
            sandbox_id, "cat /tmp/agent_stderr.log 2>/dev/null || echo 'No stderr log'"
        )
        print(getattr(r, "stdout", "") or str(r))

        print("=== AGENT EXIT CODE ===")
        r = await self.sandbox_client.execute_command(
            sandbox_id, "cat /tmp/vf_exit_code 2>/dev/null || echo 'No exit code'"
        )
        print(getattr(r, "stdout", "") or str(r))

        task = self._get_task(state)
        local_tests_path = Path(state["input"]["task_path"]) / "tests"

        for f in local_tests_path.iterdir():
            if f.is_file():
                target_path = f"/tests/{f.name}"
                await self.sandbox_client.upload_file(
                    state["sandbox_id"],
                    target_path,
                    f,
                )

        await self.sandbox_client.execute_command(
            state["sandbox_id"],
            "chmod +x /tests/*.sh",
        )

        await self.sandbox_client.execute_command(
            state["sandbox_id"],
            "bash /tests/test.sh",
        )

        r = await self.sandbox_client.execute_command(
            state["sandbox_id"],
            "ls -1 /logs/verifier/*.txt /logs/verifier/*.json 2>/dev/null",
        )
        out = getattr(r, "stdout", "") or str(r)

        if ".txt" in out:
            reward = await parse_reward_text(self.sandbox_client, state["sandbox_id"])
        elif ".json" in out:
            reward = await parse_reward_json(self.sandbox_client, state["sandbox_id"])
        else:
            reward = 0.0

        state["reward"] = reward
        return state

    async def wait_for_completion(self, state: State) -> None:
        sandbox_id = state.get("sandbox_id")
        rollout_id = state.get("rollout_id")
        print(f"=== WAIT_FOR_COMPLETION STARTED: {rollout_id} ===")

        if not sandbox_id:
            print("=== NO SANDBOX_ID, exiting wait_for_completion ===")
            return

        try:
            sandbox_client = AsyncSandboxClient()
            max_wait = int(self.timeout_seconds)
            poll_interval = 5
            elapsed = 0

            while elapsed < max_wait:
                r = await sandbox_client.execute_command(
                    sandbox_id,
                    "test -f /tmp/vf_complete && echo 'done' || echo 'waiting'",
                    timeout=30,
                )
                stdout = getattr(r, "stdout", "") or ""
                print(f"=== POLL {rollout_id}: elapsed={elapsed}s, result={stdout.strip()} ===")

                if "done" in stdout:
                    print(f"=== AGENT FINISHED NORMALLY: {rollout_id} ===")
                    state["agent_completed"] = True
                    return

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            print(f"=== AGENT TIMED OUT: {rollout_id} after {max_wait}s ===")
            state["agent_completed"] = True

        except Exception as e:
            print(f"=== WAIT_FOR_COMPLETION EXCEPTION {rollout_id}: {type(e).__name__}: {e} ===")
            import traceback
            traceback.print_exc()
            state["agent_completed"] = True


async def parse_reward_text(
    sandbox_client: AsyncSandboxClient,
    sandbox_id: str
) -> float:
    r = await sandbox_client.execute_command(
        sandbox_id,
        "cat /logs/verifier/reward.txt",
    )
    exit_code = getattr(r, "exit_code", getattr(r, "return_code", None))
    if exit_code != 0:
        return 0.0

    stdout = getattr(r, "stdout", "") or ""
    if stdout.strip():
        return float(stdout.strip())
    return 0.0


async def parse_reward_json(
    sandbox_client: AsyncSandboxClient,
    sandbox_id: str
) -> float:
    r = await sandbox_client.execute_command(
        sandbox_id,
        "cat /logs/verifier/reward.json",
    )
    exit_code = getattr(r, "exit_code", getattr(r, "return_code", None))
    if exit_code != 0:
        return 0.0

    stdout = getattr(r, "stdout", "") or ""
    if stdout.strip():
        try:
            data = json.loads(stdout.strip())
            return float(data.get("reward", 0.0))
        except (json.JSONDecodeError, ValueError):
            return 0.0
    return 0.0


def load_environment(
    run_command="",
    task_dir="/Users/christian/.cache/harbor/tasks",
    task_paths=None,
    agent_name: str = "opencode",
    agent_kwargs: dict[str, Any] | None = None,
    timeout_seconds: int = 300,
    **kwargs
):
    return TerminalBenchEnv(
        run_command=run_command,
        task_dir=task_dir,
        task_paths=task_paths,
        agent_name=agent_name,
        agent_kwargs=agent_kwargs,
        timeout_seconds=timeout_seconds,
        **kwargs,
    )
