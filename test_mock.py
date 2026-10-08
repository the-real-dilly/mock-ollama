import json
import threading
import urllib.error
import urllib.request

import pytest

import mock_ollama


@pytest.fixture(scope="module")
def base():
    server = mock_ollama.make_server(port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def post(base, body, path="/v1/chat/completions"):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def history(n_tool_msgs):
    msgs = [{"role": "user", "content": "How many albums?"}]
    for i in range(n_tool_msgs):
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    return msgs


def test_root_and_models(base):
    assert urllib.request.urlopen(base + "/").read() == b"Ollama is running"
    models = json.load(urllib.request.urlopen(base + "/v1/models"))
    assert models["data"][0]["id"] == "mock"


def test_unknown_route_404(base):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/nope")
    assert e.value.code == 404


def test_tool_call_sequence(base):
    expected = ["sql_db_list_tables", "sql_db_schema", "sql_db_query"]
    for n, name in enumerate(expected):
        r = json.load(post(base, {"model": "gpt-5.5", "messages": history(n)}))
        choice = r["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == name
        assert isinstance(json.loads(call["function"]["arguments"]), dict)
    final = json.load(post(base, {"messages": history(3)}))["choices"][0]
    assert final["finish_reason"] == "stop"
    assert "347" in final["message"]["content"]


def test_last_step_repeats(base):
    r = json.load(post(base, {"messages": history(10)}))
    assert r["choices"][0]["finish_reason"] == "stop"


def sse(resp):
    events = [l[6:] for l in resp.read().decode().split("\n\n") if l.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def test_stream_tool_call(base):
    chunks = sse(post(base, {"messages": history(0), "stream": True}))
    calls = [tc for c in chunks for ch in c["choices"] for tc in ch["delta"].get("tool_calls", [])]
    assert calls[0]["function"]["name"] == "sql_db_list_tables"
    assert json.loads("".join(tc["function"]["arguments"] for tc in calls)) == {"tool_input": ""}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_text_and_usage(base):
    chunks = sse(post(base, {"messages": history(3), "stream": True,
                             "stream_options": {"include_usage": True}}))
    text = "".join(ch["delta"].get("content") or "" for c in chunks for ch in c["choices"])
    assert text == "There are 347 albums in the database."
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert "usage" in chunks[-1]


def test_bad_json_400(base):
    req = urllib.request.Request(base + "/v1/chat/completions", b"{nope", {})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
