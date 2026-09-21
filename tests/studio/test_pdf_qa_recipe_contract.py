# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contracts and opt-in runtime coverage for the PDF grounded QA recipe."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECIPE_PATH = (
    REPO / "studio/frontend/src/features/data-recipes/learning-recipes/pdf-grounded-qa.json"
)
TRAINING_START_PATH = REPO / "studio/frontend/src/features/training/lib/start-fresh-training-run.ts"
SEED_BUILDER_PATH = (
    REPO / "studio/frontend/src/features/recipe-studio/utils/payload/builders-seed.ts"
)
RECIPE_IMPORTER_PATH = REPO / "studio/frontend/src/features/recipe-studio/utils/import/importer.ts"
SEED_PARSER_PATH = (
    REPO / "studio/frontend/src/features/recipe-studio/utils/import/parsers/seed-config-parser.ts"
)
FORMAT_DETECTION_PATH = REPO / "studio/backend/utils/datasets/format_detection.py"


def _load_payload() -> dict:
    return json.loads(RECIPE_PATH.read_text(encoding = "utf-8"))


def _render_expression(template: str, row: dict) -> str:
    def replace(match: re.Match[str]) -> str:
        value = row
        for part in match.group(1).strip().split("."):
            value = value[part]
        return str(value)

    return re.sub(r"\{\{\s*([^}]+?)\s*\}\}", replace, template)


def test_pdf_qa_recipe_projects_and_cleans_training_columns():
    recipe = _load_payload()["recipe"]
    columns = {column["name"]: column for column in recipe["columns"]}

    assert list(columns) == ["llm_structured_1", "instruction", "output", "input"]
    assert columns["llm_structured_1"]["drop"] is True
    assert columns["instruction"]["expr"] == "{{ llm_structured_1.question }}"
    assert columns["output"]["expr"] == "{{ llm_structured_1.answer }}"
    assert "llm_structured_1.evidence_quote" in columns["input"]["expr"]
    assert "chunk_text" in columns["input"]["expr"]
    assert recipe["processors"] == [
        {
            "processor_type": "drop_columns",
            "name": "drop_seed_columns",
            "column_names": ["chunk_text", "source_file"],
        }
    ]


def test_pdf_qa_recipe_sample_row_is_qlora_ready():
    recipe = _load_payload()["recipe"]
    row = {
        "chunk_text": "Paris is the capital of France.",
        "source_file": "facts.pdf",
        "llm_structured_1": {
            "question": "What is the capital of France?",
            "answer": "Paris.",
            "evidence_quote": "Paris is the capital of France.",
        },
    }

    for column in recipe["columns"]:
        if column["column_type"] == "expression":
            row[column["name"]] = _render_expression(column["expr"], row)
    for column in recipe["columns"]:
        if column.get("drop"):
            row.pop(column["name"], None)
    for processor in recipe["processors"]:
        for name in processor["column_names"]:
            row.pop(name, None)

    assert row == {
        "instruction": "What is the capital of France?",
        "output": "Paris.",
        "input": (
            "Evidence quote: Paris is the capital of France.\n\n"
            "Source context: Paris is the capital of France."
        ),
    }


def test_pdf_qa_canvas_edges_cover_expression_dependencies():
    payload = _load_payload()
    recipe = payload["recipe"]
    node_ids = {node["id"] for node in payload["ui"]["nodes"]}
    edges = {(edge["from"], edge["to"]) for edge in payload["ui"]["edges"]}

    assert all(source in node_ids and target in node_ids for source, target in edges)
    assert ("seed", "llm_structured_1") in edges
    assert ("llm_structured_1", "instruction") in edges
    assert ("llm_structured_1", "output") in edges
    assert ("llm_structured_1", "input") in edges
    assert ("seed", "input") in edges

    column_names = {column["name"] for column in recipe["columns"]}
    assert {"instruction", "output"} <= column_names


def test_pdf_qa_fields_match_studio_alpaca_mapping():
    source = TRAINING_START_PATH.read_text(encoding = "utf-8")
    manual_mapping = source.split("function hasManualMapping", 1)[1]
    manual_mapping = manual_mapping.split("\n}\n", 1)[0]
    assert 'alpaca: { user: "instruction", system: "input", assistant: "output" }' in source
    assert 'if (config.datasetFormat === "alpaca") {' in manual_mapping
    assert 'return roles.has("instruction") && roles.has("output");' in manual_mapping


def test_pdf_qa_fields_are_detected_as_alpaca():
    spec = importlib.util.spec_from_file_location("_pdf_qa_format_detection", FORMAT_DETECTION_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    detected = module.detect_dataset_format(
        [{"instruction": "What is the capital?", "input": "source", "output": "Paris."}]
    )
    assert detected["format"] == "alpaca"
    assert detected["needs_standardization"] is False


def test_unstructured_seed_drop_toggle_round_trip_contract():
    builder = SEED_BUILDER_PATH.read_text(encoding = "utf-8")
    importer = RECIPE_IMPORTER_PATH.read_text(encoding = "utf-8")
    parser = SEED_PARSER_PATH.read_text(encoding = "utf-8")

    assert 'if (seedSourceType === "unstructured")' in builder
    assert "if (!config.drop)" in builder
    assert "selectedDropColumns.length > 0" in builder
    assert ': ["chunk_text", "source_file"];' in builder
    assert "payloadSeedSourceIsUnstructured && payloadSeedDropColumns.length > 0" in importer
    assert "payloadSeedSourceIsUnstructured" in importer
    assert '? ["chunk_text", "source_file"]' in importer
    assert "drop?: boolean;" in parser
    assert "...(options?.drop !== undefined ? { drop: options.drop } : {})" in parser


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, format: str, *args) -> None:
        return

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.requests.append(json.loads(raw or b"{}"))
        structured = {
            "question": "What is the capital of France?",
            "answer": "Paris.",
            "evidence_quote": "Paris is the capital of France.",
        }
        body = json.dumps(
            {
                "id": "chatcmpl-pdf-qa-test",
                "object": "chat.completion",
                "created": 0,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": f"```json\n{json.dumps(structured)}\n```",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_pdf_qa_recipe_runs_with_pinned_data_designer(tmp_path, monkeypatch):
    if os.environ.get("UNSLOTH_PDF_QA_MANAGED_INTEGRATION") != "1":
        pytest.skip("set UNSLOTH_PDF_QA_MANAGED_INTEGRATION=1 to run this integration")

    backend = REPO / "studio/backend"
    sys.path.insert(0, str(backend))
    pytest.importorskip("data_designer")
    pytest.importorskip("data_designer_unstructured_seed")
    from core.data_recipe import service

    source_path = tmp_path / "facts.txt"
    source_path.write_text("Paris is the capital of France.", encoding = "utf-8")
    monkeypatch.setattr(service, "recipe_datasets_root", lambda: tmp_path / "artifacts")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        recipe = copy.deepcopy(_load_payload()["recipe"])
        recipe["seed_config"]["source"] = {
            "seed_type": "unstructured",
            "paths": [str(source_path)],
            "chunk_size": 1200,
            "chunk_overlap": 200,
        }
        recipe["model_providers"][0].update(
            {
                "endpoint": f"http://127.0.0.1:{server.server_port}/v1",
                "api_key": "test-only",
            }
        )
        recipe["model_configs"][0].update({"model": "mock-model", "skip_health_check": True})
        dataset, _, _ = service.preview_recipe(recipe, 1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout = 5)

    assert dataset == [
        {
            "instruction": "What is the capital of France?",
            "output": "Paris.",
            "input": (
                "Evidence quote: Paris is the capital of France.\n\n"
                "Source context: Paris is the capital of France."
            ),
        }
    ]
    assert _MockOpenAIHandler.requests
