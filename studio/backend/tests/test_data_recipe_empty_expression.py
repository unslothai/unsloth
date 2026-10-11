# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture(name = "provider")
def fixture_provider():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            # Row 1 leaves the optional question blank, as a model with nothing to say does.
            if "row 1" in body:
                answer = {"question": "", "answer": "A."}
            else:
                answer = {"question": "Q?", "answer": "B."}
            content = "```json\n" + json.dumps(answer) + "\n```"
            reply = json.dumps(
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def _preview(
    tmp_path: Path,
    recipe: dict,
    num_records: int = 2,
):
    from core.data_recipe.service import build_config_builder, create_data_designer
    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    return designer.preview(build_config_builder(recipe), num_records = num_records).dataset


def test_flattening_an_empty_structured_field_keeps_the_run(tmp_path: Path, provider) -> None:
    # unslothai/unsloth#4675: {{ response.question }} on an empty optional field failed the whole preview.
    pytest.importorskip("data_designer")
    seed = tmp_path / "seed.csv"
    seed.write_text("n\n1\n2\n", encoding = "utf-8")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [],
        "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    }
    recipe = {
        "model_providers": [{"name": "p", "endpoint": provider, "provider_type": "openai"}],
        "model_configs": [
            {
                "alias": "m",
                "model": "x",
                "provider": "p",
                "inference_parameters": {"max_parallel_requests": 1},
            }
        ],
        "seed_config": {
            "source": {"seed_type": "local", "path": str(seed)},
            "sampling_strategy": "ordered",
        },
        "columns": [
            {
                "column_type": "llm-structured",
                "name": "response",
                "model_alias": "m",
                "prompt": "Write a QA pair for row {{ n }}",
                "output_format": schema,
            },
            {"column_type": "expression", "name": "question", "expr": "{{ response.question }}"},
            {
                "column_type": "expression",
                "name": "guarded",
                "expr": "{% if response.question %}{{ response.question }}{% endif %}",
            },
            {"column_type": "expression", "name": "answer", "expr": "{{ response.answer }}"},
        ],
    }

    dataset = _preview(tmp_path, recipe)

    assert sorted(dataset["question"]) == ["", "Q?"]
    assert sorted(dataset["guarded"]) == ["", "Q?"]
    assert sorted(dataset["answer"]) == ["A.", "B."]


@pytest.mark.parametrize(
    "expr",
    ["{{ response.question }}", "{{ response.missing }}", "{% if response.question %}x{% endif %}"],
)
def test_text_expression_may_render_empty(tmp_path: Path, expr: str) -> None:
    pytest.importorskip("data_designer")
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        json.dumps({"response": {"question": ""}})
        + "\n"
        + json.dumps({"response": {"question": "Q?"}})
        + "\n",
        encoding = "utf-8",
    )
    recipe = {
        "seed_config": {
            "source": {"seed_type": "local", "path": str(seed)},
            "sampling_strategy": "ordered",
        },
        "columns": [{"column_type": "expression", "name": "out", "expr": expr}],
    }

    dataset = _preview(tmp_path, recipe)

    assert "" in list(dataset["out"])


def test_number_expression_still_rejects_empty(tmp_path: Path) -> None:
    pytest.importorskip("data_designer")
    seed = tmp_path / "seed.jsonl"
    seed.write_text(json.dumps({"v": ""}) + "\n", encoding = "utf-8")
    recipe = {
        "seed_config": {
            "source": {"seed_type": "local", "path": str(seed)},
            "sampling_strategy": "ordered",
        },
        "columns": [
            {"column_type": "expression", "name": "out", "expr": "{{ v }}", "dtype": "int"}
        ],
    }

    with pytest.raises(Exception, match = "template is invalid"):
        _preview(tmp_path, recipe, num_records = 1)
