# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture(name = "provider")
def fixture_provider():
    prompts: list[str] = []
    systems: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            for message in body["messages"]:
                if message["role"] == "user":
                    prompts.extend(part["text"] for part in message["content"])
                elif message["role"] == "system":
                    content = message["content"]
                    systems.append(content if isinstance(content, str) else content[0]["text"])
            reply = json.dumps(
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
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
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", prompts, systems
    server.shutdown()


def test_blank_seed_cells_reach_the_model_as_empty_text(tmp_path: Path, provider) -> None:
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    endpoint, prompts, _ = provider
    seed = tmp_path / "seed.csv"
    seed.write_text("question,year,context\nWho?,,\nWhen?,2019,ctx\n", encoding = "utf-8")
    recipe = {
        "model_providers": [{"name": "p", "endpoint": endpoint, "provider_type": "openai"}],
        "model_configs": [
            {"alias": "m", "model": "x", "provider": "p", "inference_parameters": {}}
        ],
        "seed_config": {
            "source": {"seed_type": "local", "path": str(seed)},
            "sampling_strategy": "ordered",
        },
        "columns": [
            {
                "column_type": "llm-text",
                "name": "answer",
                "model_alias": "m",
                "prompt": "{{ question }} ({{ year }}) {{ context }}",
            }
        ],
    }

    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    designer.preview(build_config_builder(recipe), num_records = 2)

    assert "Who? () " in prompts
    assert any(p.startswith("When? (2019") and p.endswith(") ctx") for p in prompts)
    assert not [p for p in prompts if "None" in p or "nan" in p]


def test_blank_system_prompt_cell_keeps_the_row(tmp_path: Path, provider) -> None:
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    endpoint, prompts, systems = provider
    seed = tmp_path / "seed.csv"
    seed.write_text("question,context\nWho?,\nWhen?,ctx\n", encoding = "utf-8")
    recipe = {
        "model_providers": [{"name": "p", "endpoint": endpoint, "provider_type": "openai"}],
        "model_configs": [
            {"alias": "m", "model": "x", "provider": "p", "inference_parameters": {}}
        ],
        "seed_config": {
            "source": {"seed_type": "local", "path": str(seed)},
            "sampling_strategy": "ordered",
        },
        "columns": [
            {
                "column_type": "llm-text",
                "name": "answer",
                "model_alias": "m",
                "system_prompt": "{{ context }}",
                "prompt": "{{ question }}",
            }
        ],
    }

    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    result = designer.preview(build_config_builder(recipe), num_records = 2)

    assert sorted(result.dataset["question"]) == ["When?", "Who?"]
    assert "Who?" in prompts and "When?" in prompts
    assert "ctx" in systems
    assert "None" not in systems


@pytest.mark.parametrize(
    "template", ["{{ a | string }}", "{{ a | lower }}", "{{ a | title }}", "{{ a | trim }}"]
)
def test_stringifying_filters_render_blank_cells_as_empty_text(template: str) -> None:
    pytest.importorskip("data_designer")
    from core.data_recipe.service import _apply_data_designer_prompt_blank_patch
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
    )
    from data_designer.engine.models.recipes.response_recipes import TextResponseRecipe

    _apply_data_designer_prompt_blank_patch()

    def render(value):
        return RecordBasedPromptRenderer(TextResponseRecipe()).render(
            prompt_template = f"<{template}>", record = {"a": value}, prompt_type = PromptType.USER_PROMPT
        )

    assert render(None) == "<>"
    assert render(float("nan")) == "<>"
    assert render("Ab c") not in ("<>", "<None>")


def test_reused_renderer_does_not_rewrap_filters_per_record() -> None:
    pytest.importorskip("data_designer")
    from core.data_recipe.service import _apply_data_designer_prompt_blank_patch
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
    )
    from data_designer.engine.models.recipes.response_recipes import TextResponseRecipe

    _apply_data_designer_prompt_blank_patch()
    renderer = RecordBasedPromptRenderer(TextResponseRecipe())
    for _ in range(1500):
        rendered = renderer.render(
            prompt_template = "{{ a | lower }}",
            record = {"a": None},
            prompt_type = PromptType.USER_PROMPT,
        )
    assert rendered == ""
