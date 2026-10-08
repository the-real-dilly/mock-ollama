"""Mock OpenAI-compatible endpoint (as served by Ollama at /v1) for testing agents.

Stateless and deterministic: the step played is the number of `role: "tool"`
messages already in the request, indexed into script.json. Past the end, the
last step repeats.

    python mock_ollama.py [--host 127.0.0.1] [--port 11434] [--script script.json]
"""
import argparse
import json
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_SCRIPT = Path(__file__).with_name("script.json")


def load_script(path):
    steps = json.loads(Path(path).read_text(encoding="utf-8"))
    if not steps:
        raise ValueError("script must contain at least one step")
    return steps


def pick_step(script, messages):
    done = sum(1 for m in messages if m.get("role") == "tool")
    return script[min(done, len(script) - 1)]


def build_tool_calls(step):
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments", {}))},
        }
        for tc in step.get("tool_calls", [])
    ]


def completion(model, step):
    tool_calls = build_tool_calls(step)
    message = {"role": "assistant", "content": step.get("content")}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def stream_chunks(model, step, include_usage):
    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }

    def chunk(delta, finish=None):
        return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    tool_calls = build_tool_calls(step)
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            yield chunk({"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["function"]["name"], "arguments": ""},
            }]})
            yield chunk({"tool_calls": [{
                "index": i, "function": {"arguments": tc["function"]["arguments"]},
            }]})
        yield chunk({}, "tool_calls")
    else:
        for piece in re.findall(r"\S+\s*", step.get("content") or ""):
            yield chunk({"content": piece})
        yield chunk({}, "stop")
    if include_usage:
        yield {**base, "choices": [],
               "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


class Handler(BaseHTTPRequestHandler):
    script = None  # set by make_server

    def log_message(self, fmt, *args):
        sys.stderr.write("[mock] " + fmt % args + "\n")

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            data = b"Ollama is running"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "mock", "object": "model", "created": 0, "owned_by": "mock"}]})
        else:
            self._json(404, {"error": {"message": f"not found: {path}", "type": "not_found"}})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": f"not found: {path}", "type": "not_found"}})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})

        model = body.get("model", "mock")
        step = pick_step(self.script, body.get("messages", []))

        if not body.get("stream"):
            return self._json(200, completion(model, step))

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for c in stream_chunks(model, step, include_usage):
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def make_server(host="127.0.0.1", port=11434, script_path=DEFAULT_SCRIPT):
    handler = type("ScriptedHandler", (Handler,), {"script": load_script(script_path)})
    return ThreadingHTTPServer((host, port), handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--script", default=str(DEFAULT_SCRIPT))
    args = ap.parse_args()
    server = make_server(args.host, args.port, args.script)
    print(f"Mock server on http://{args.host}:{args.port}/v1 (script: {args.script})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
