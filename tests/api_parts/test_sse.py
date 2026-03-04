# tests/api_parts/test_sse.py

import json

from llm_local.api_parts.sse import sse


def test_sse_with_event_includes_event_line_and_double_newline():
    msg = sse("delta", {"content": "hello"})

    assert msg.startswith("event: delta\n")
    assert "data: " in msg
    assert msg.endswith("\n\n")

    # Extract the data JSON and verify it parses
    data_line = [line for line in msg.splitlines() if line.startswith("data: ")][0]
    payload = json.loads(data_line[len("data: ") :])
    assert payload == {"content": "hello"}


def test_sse_without_event_omits_event_line_and_ends_with_double_newline():
    msg = sse(None, {"ok": True})

    assert msg.startswith("data: ")
    assert "event:" not in msg
    assert msg.endswith("\n\n")

    data_line = msg.splitlines()[0]
    payload = json.loads(data_line[len("data: ") :])
    assert payload == {"ok": True}


def test_sse_uses_utf8_json_without_ascii_escaping():
    # ensure_ascii=False should keep unicode characters readable
    msg = sse("meta", {"text": "café"})

    data_line = [line for line in msg.splitlines() if line.startswith("data: ")][0]
    json_text = data_line[len("data: ") :]

    assert "café" in json_text
    assert json.loads(json_text) == {"text": "café"}