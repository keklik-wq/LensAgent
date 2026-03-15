from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockLlmHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        request = json.loads(raw_body.decode("utf-8"))
        messages = request.get("messages", [])
        user_content = ""
        for message in messages:
            if message.get("role") == "user":
                user_content = str(message.get("content", ""))
        payload = json.loads(user_content) if user_content else {}
        params = _choose_params(payload)
        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "params": params,
                                "rationale": "Mock LLM picked a deterministic local config.",
                            }
                        ),
                    },
                }
            ],
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format
        del args


def _choose_params(payload: dict[str, object]) -> dict[str, object]:
    history = payload.get("history", [])
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict):
            last_params = last.get("params")
            if isinstance(last_params, dict):
                return {
                    "spark.sql.shuffle.partitions": max(
                        4, int(last_params.get("spark.sql.shuffle.partitions", 8))
                    ),
                    "executor.cores": max(1, int(last_params.get("executor.cores", 1))),
                    "executor.instances": max(1, int(last_params.get("executor.instances", 1))),
                    "executor.memory_gb": max(1, int(last_params.get("executor.memory_gb", 1))),
                }
    base_params = payload.get("base_params", {})
    if isinstance(base_params, dict):
        return {
            "spark.sql.shuffle.partitions": int(base_params.get("spark.sql.shuffle.partitions", 8)),
            "executor.cores": int(base_params.get("executor.cores", 1)),
            "executor.instances": int(base_params.get("executor.instances", 1)),
            "executor.memory_gb": int(base_params.get("executor.memory_gb", 1)),
        }
    return {
        "spark.sql.shuffle.partitions": 8,
        "executor.cores": 1,
        "executor.instances": 1,
        "executor.memory_gb": 1,
    }


def main() -> None:
    server = HTTPServer(("0.0.0.0", 8000), MockLlmHandler)
    print("mock llm listening on :8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
