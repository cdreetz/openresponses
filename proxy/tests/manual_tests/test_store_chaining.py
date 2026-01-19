"""
Manual test: Test store and previous_response_id chaining without network
Run: uv run proxy/tests/manual_tests/test_store_chaining.py
"""
import sys
from typing import Any


class TestFailure(Exception):
    pass


def assert_eq(actual: Any, expected: Any, msg: str) -> None:
    if actual != expected:
        raise TestFailure(f"{msg}: expected {expected!r}, got {actual!r}")


def assert_true(condition: bool, msg: str) -> None:
    if not condition:
        raise TestFailure(msg)


def test_store_basic():
    """Test basic store save/get"""
    print("\n[1/5] Testing basic store operations...")
    from proxy import store

    # Clear any existing data
    store._store.clear()

    response = {
        "id": "resp_test1",
        "output": [{"type": "message", "content": [{"text": "Hello"}]}],
        "input": "Hi there",
    }
    store.save(response)

    retrieved = store.get("resp_test1")
    assert_true(retrieved is not None, "Response should be in store")
    assert_eq(retrieved["id"], "resp_test1", "ID should match")
    assert_eq(retrieved["input"], "Hi there", "Input should be saved")
    print("      Basic store operations work")


def test_store_chaining():
    """Test that previous_response_id lookups work"""
    print("\n[2/5] Testing store chaining...")
    from proxy import store
    from proxy.handlers.handle_responses_api import get_history_from_previous

    store._store.clear()

    # Save first response
    resp1 = {
        "id": "resp_chain1",
        "input": "What is 2+2?",
        "output": [{"type": "message", "role": "assistant", "content": [{"text": "4"}]}],
    }
    store.save(resp1)

    # Save second response that chains from first
    resp2 = {
        "id": "resp_chain2",
        "previous_response_id": "resp_chain1",
        "input": "And 3+3?",
        "output": [{"type": "message", "role": "assistant", "content": [{"text": "6"}]}],
    }
    store.save(resp2)

    # Get history from resp2
    history = get_history_from_previous("resp_chain2")
    print(f"      History messages: {len(history)}")
    for i, msg in enumerate(history):
        print(f"        [{i}] {msg.get('role')}: {str(msg.get('content', ''))[:50]}")

    # Should have: user(2+2), assistant(4), user(3+3), assistant(6)
    assert_true(len(history) >= 4, f"Should have at least 4 messages, got {len(history)}")
    print("      Store chaining works")


def test_tool_call_chaining():
    """Test that tool calls are properly chained"""
    print("\n[3/5] Testing tool call chaining...")
    from proxy import store
    from proxy.handlers.handle_responses_api import (
        get_history_from_previous,
        responses_input_to_messages,
    )

    store._store.clear()

    # First response: user asks something, assistant makes tool call
    resp1 = {
        "id": "resp_tool1",
        "input": "List files",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "bash",
                "arguments": '{"command": "ls"}',
            }
        ],
    }
    store.save(resp1)

    # Second response: tool result, then assistant responds
    resp2 = {
        "id": "resp_tool2",
        "previous_response_id": "resp_tool1",
        "input": [
            {"type": "function_call_output", "call_id": "call_123", "output": "file1.txt\nfile2.py"}
        ],
        "output": [
            {"type": "message", "role": "assistant", "content": [{"text": "I found 2 files"}]}
        ],
    }
    store.save(resp2)

    # Get history - should reconstruct the full conversation
    history = get_history_from_previous("resp_tool2")
    print(f"      History messages: {len(history)}")
    for i, msg in enumerate(history):
        role = msg.get('role')
        content = str(msg.get('content', msg.get('tool_calls', '')))[:50]
        print(f"        [{i}] {role}: {content}")

    # Should have: user, assistant(tool_calls), tool, assistant(response)
    roles = [m.get('role') for m in history]
    assert_true('tool' in roles, f"Should have tool message in history, got roles: {roles}")
    print("      Tool call chaining works")


def test_responses_request_to_chat_with_previous():
    """Test full conversion with previous_response_id"""
    print("\n[4/5] Testing responses_request_to_chat with previous_response_id...")
    from proxy import store
    from proxy.handlers.handle_responses_api import responses_request_to_chat

    store._store.clear()

    # Save a previous response
    prev_resp = {
        "id": "resp_prev",
        "input": "Hello",
        "output": [{"type": "message", "role": "assistant", "content": [{"text": "Hi!"}]}],
    }
    store.save(prev_resp)

    # New request with previous_response_id
    body = {
        "instructions": "You are helpful",
        "input": "How are you?",
        "previous_response_id": "resp_prev",
        "model": "gpt-4o-mini",
    }

    result = responses_request_to_chat(body)
    messages = result["messages"]

    print(f"      Messages after conversion: {len(messages)}")
    for i, msg in enumerate(messages):
        role = msg.get('role')
        content = str(msg.get('content', ''))[:50]
        print(f"        [{i}] {role}: {content}")

    # Should have: system, user(Hello), assistant(Hi!), user(How are you?)
    assert_true(len(messages) >= 4, f"Should have at least 4 messages, got {len(messages)}")
    assert_eq(messages[0]["role"], "system", "First should be system")
    assert_eq(messages[-1]["role"], "user", "Last should be user")
    assert_eq(messages[-1]["content"], "How are you?", "Last should be new input")
    print("      Conversion with previous_response_id works")


def test_missing_previous_response():
    """Test that missing previous_response_id is handled gracefully"""
    print("\n[5/5] Testing missing previous_response_id handling...")
    from proxy import store
    from proxy.handlers.handle_responses_api import responses_request_to_chat

    store._store.clear()

    # Request with non-existent previous_response_id
    body = {
        "instructions": "You are helpful",
        "input": "Hello",
        "previous_response_id": "resp_nonexistent",
        "model": "gpt-4o-mini",
    }

    result = responses_request_to_chat(body)
    messages = result["messages"]

    print(f"      Messages: {len(messages)}")
    # Should have just: system, user (no history since previous doesn't exist)
    assert_eq(len(messages), 2, "Should have 2 messages when previous not found")
    print("      Missing previous_response_id handled gracefully")


def main() -> int:
    print("=" * 60)
    print("TEST: Store and previous_response_id chaining")
    print("=" * 60)

    try:
        test_store_basic()
        test_store_chaining()
        test_tool_call_chaining()
        test_responses_request_to_chat_with_previous()
        test_missing_previous_response()

        print("\n" + "=" * 60)
        print("PASSED: All store chaining tests succeeded")
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


if __name__ == "__main__":
    sys.exit(main())
