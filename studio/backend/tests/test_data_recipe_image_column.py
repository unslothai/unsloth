# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import io
import json
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from auth import storage
from core.data_recipe.jsonable import image_cell_to_preview, render_image_columns
from routes.data_recipe import jobs as route

IMAGE_MODEL = "stabilityai/sdxl-turbo"


def _png(color = (200, 30, 30), size = (8, 8)) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format = "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture(name = "provider")
def fixture_provider():
    requests: list[tuple[str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path.endswith("/images/generations"):
                reply = {"created": 0, "data": [{"b64_json": _png()}]}
            else:
                reply = {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "a picture"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests
    server.shutdown()


def _recipe(
    endpoint,
    *,
    model = IMAGE_MODEL,
    local = True,
):
    model_config = {
        "alias": "painter",
        "model": model,
        "provider": "p",
        "skip_health_check": True,
        "inference_parameters": {
            "generation_type": "image",
            "extra_body": {
                "size": "512x512",
                "steps": 1,
                "guidance": 0,
                "response_format": "b64_json",
            },
        },
    }
    if local:
        model_config["unsloth_diffusion_route"] = True
    return {
        "model_providers": [{"name": "p", "endpoint": endpoint, "provider_type": "openai"}],
        "model_configs": [model_config],
        "columns": [
            {
                "column_type": "sampler",
                "name": "subject",
                "sampler_type": "category",
                "params": {"values": ["mug", "lamp"]},
            },
            {
                "column_type": "image",
                "name": "photo",
                "model_alias": "painter",
                "prompt": "A product photo of a {{ subject }}",
            },
        ],
    }


def _paths(requests):
    return [path for path, _ in requests]


def test_a_local_image_model_is_generated_through_images_generations(tmp_path, provider):
    pytest.importorskip("data_designer")
    from core.data_recipe.jsonable import image_column_names
    from core.data_recipe.service import build_config_builder, create_data_designer

    endpoint, requests = provider
    recipe = _recipe(endpoint)
    designer = create_data_designer(recipe, artifact_path = str(tmp_path))
    results = designer.preview(build_config_builder(recipe), num_records = 2)

    assert _paths(requests) == ["/v1/images/generations"] * 2
    for _, body in requests:
        assert body["model"] == IMAGE_MODEL
        assert body["prompt"] in {"A product photo of a mug", "A product photo of a lamp"}
        assert body["response_format"] == "b64_json"
        assert (body["size"], body["steps"], body["guidance"]) == ("512x512", 1, 0)
    rows = render_image_columns(
        results.dataset.to_dict(orient = "records"), image_column_names(recipe)
    )
    assert [row["photo"]["type"] for row in rows] == ["image", "image"]
    assert rows[0]["photo"]["mime"] == "image/jpeg"


def test_a_cloud_image_model_keeps_data_designers_own_route(tmp_path, provider):
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    endpoint, requests = provider
    # Same name as a local model, but nothing marked it as one.
    recipe = _recipe(endpoint, local = False)
    designer = create_data_designer(recipe, artifact_path = str(tmp_path))
    with pytest.raises(Exception):
        designer.preview(build_config_builder(recipe), num_records = 1)
    assert set(_paths(requests)) == {"/v1/chat/completions"}

    requests.clear()
    recipe = _recipe(endpoint, model = "gpt-image-1", local = False)
    designer = create_data_designer(recipe, artifact_path = str(tmp_path))
    designer.preview(build_config_builder(recipe), num_records = 1)
    assert _paths(requests) == ["/v1/images/generations"]


def test_a_remote_preset_naming_the_local_model_keeps_its_own_route(tmp_path, provider):
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    endpoint, requests = provider
    recipe = _recipe(endpoint + "/local")
    recipe["model_providers"].append(
        {"name": "remote", "endpoint": endpoint + "/remote", "provider_type": "openai"}
    )
    remote = json.loads(json.dumps(recipe["model_configs"][0]))
    remote.pop("unsloth_diffusion_route")
    remote.update(alias = "remote_painter", provider = "remote")
    recipe["model_configs"].append(remote)
    recipe["columns"].append(
        {
            "column_type": "image",
            "name": "remote_photo",
            "model_alias": "remote_painter",
            "prompt": "a cat",
        }
    )
    designer = create_data_designer(recipe, artifact_path = str(tmp_path))
    with pytest.raises(Exception):
        designer.preview(build_config_builder(recipe), num_records = 1)

    assert "/v1/local/images/generations" in _paths(requests)
    assert "/v1/remote/images/generations" not in _paths(requests)
    assert "/v1/remote/chat/completions" in _paths(requests)


def test_a_full_run_writes_images_that_dataset_pages_show(tmp_path, provider, monkeypatch):
    pytest.importorskip("data_designer")
    from core.data_recipe import export
    from core.data_recipe.jobs.manager import JobManager
    from core.data_recipe.service import build_config_builder, create_data_designer

    endpoint, _ = provider
    recipe = _recipe(endpoint)
    designer = create_data_designer(recipe, artifact_path = str(tmp_path))
    results = designer.create(build_config_builder(recipe), num_records = 3, dataset_name = "run")
    dataset_dir = Path(results.artifact_storage.base_dataset_path)

    images = sorted((dataset_dir / "images").rglob("*.png"))
    assert len(images) == 3
    page = JobManager._load_dataset_page(
        parquet_dir = dataset_dir / "parquet-files", limit = 10, offset = 0
    )
    raw = [row["photo"] for row in page["dataset"]]
    assert all(cell[0].startswith("images/") for cell in raw)
    render_image_columns(page["dataset"], ["photo"], dataset_dir = dataset_dir)
    assert [row["photo"]["type"] for row in page["dataset"]] == ["image"] * 3

    monkeypatch.setattr(export, "_resolve_recipe_artifact_path", lambda path: Path(path))
    for export_format in ("jsonl", "parquet"):
        zip_path, _, filename = export.build_dataset_download(
            artifact_path = str(dataset_dir), export_format = export_format, filename_stem = "run"
        )
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
        zip_path.unlink()
        assert filename.endswith(".zip")
        assert sorted(n for n in names if n.startswith("images/")) == [
            str(p.relative_to(dataset_dir)) for p in images
        ]


def test_image_paths_outside_the_dataset_images_folder_are_not_read(tmp_path):
    dataset_dir = tmp_path / "run"
    (dataset_dir / "images" / "photo").mkdir(parents = True)
    png = base64.b64decode(_png())
    (dataset_dir / "images" / "photo" / "ok.png").write_bytes(png)
    (dataset_dir / "secret.png").write_bytes(png)
    (tmp_path / "outside.png").write_bytes(png)
    (dataset_dir / "images" / "photo" / "link.png").symlink_to(tmp_path / "outside.png")

    assert (
        image_cell_to_preview(["images/photo/ok.png"], dataset_dir = dataset_dir)["type"] == "image"
    )
    for cell in (
        ["secret.png"],
        ["images/../secret.png"],
        ["images/photo/../../../outside.png"],
        [str(tmp_path / "outside.png")],
        ["images/photo/link.png"],
    ):
        assert image_cell_to_preview(cell, dataset_dir = dataset_dir) == cell


def test_several_images_in_one_cell_show_the_first_with_a_count():
    cell = image_cell_to_preview([_png((0, 0, 255), (900, 600)), _png()])

    assert cell["count"] == 2
    assert max(cell["width"], cell["height"]) == 384


def test_only_declared_image_columns_are_read_as_images():
    rows = [{"photo": [_png()], "note": _png(), "broken": ["not base64!"]}]

    render_image_columns(rows, ["photo", "broken"])

    assert rows[0]["photo"]["type"] == "image"
    assert rows[0]["note"] == _png()
    assert rows[0]["broken"] == ["not base64!"]


def test_an_image_only_recipe_without_a_provider_gets_the_provider_error(tmp_path):
    pytest.importorskip("data_designer")
    from core.data_recipe.service import create_data_designer

    recipe = _recipe("http://127.0.0.1:1/v1")
    recipe["model_providers"] = []
    with pytest.raises(ValueError, match = "Add a Provider connection block"):
        create_data_designer(recipe, artifact_path = str(tmp_path))


def _local_recipe(*, with_llm = False):
    columns = [
        {"column_type": "image", "name": "photo", "model_alias": "painter", "prompt": "a cat"}
    ]
    if with_llm:
        columns.append(
            {"column_type": "llm-text", "name": "caption", "model_alias": "writer", "prompt": "hi"}
        )
    return {
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [
            {
                "alias": "painter",
                "model": "local",
                "provider": "local",
                "inference_parameters": {"generation_type": "image", "extra_body": {"steps": 1}},
            },
            {"alias": "writer", "model": "local", "provider": "local"},
        ],
        "columns": columns,
    }


@pytest.fixture(name = "inject")
def fixture_inject(monkeypatch):
    def run(
        recipe,
        *,
        image_model = IMAGE_MODEL,
        chat_loaded = True,
    ):
        llama = SimpleNamespace(is_loaded = False, model_identifier = "", hf_variant = "")
        backend = SimpleNamespace(active_model_name = "unsloth/Qwen3-0.6B" if chat_loaded else "")
        monkeypatch.setitem(
            sys.modules, "routes.inference", SimpleNamespace(get_llama_cpp_backend = lambda: llama)
        )
        monkeypatch.setitem(
            sys.modules, "core.inference", SimpleNamespace(get_inference_backend = lambda: backend)
        )
        monkeypatch.setattr(route, "_loaded_local_image_model", lambda: image_model)
        monkeypatch.setattr(
            route, "_resolve_local_v1_endpoint", lambda request: "http://127.0.0.1:8888/v1"
        )
        monkeypatch.setattr(
            storage, "create_api_key", lambda **kwargs: ("sk-unsloth-test", {"id": 7})
        )
        return route._inject_local_providers(recipe, SimpleNamespace())

    return run


def test_an_image_only_local_recipe_gets_a_key_and_the_loaded_image_model(inject):
    recipe = _local_recipe()

    assert inject(recipe, chat_loaded = False) == 7
    assert recipe["model_providers"][0]["api_key"] == "sk-unsloth-test"
    painter = recipe["model_configs"][0]
    assert painter["model"] == IMAGE_MODEL
    assert painter["unsloth_diffusion_route"] is True
    assert painter["skip_health_check"] is True
    assert painter["inference_parameters"] == {
        "generation_type": "image",
        "timeout": 600,
        "extra_body": {"steps": 1, "response_format": "b64_json"},
    }
    writer = recipe["model_configs"][1]
    assert writer["inference_parameters"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_a_local_image_column_with_no_image_model_loaded_is_refused(inject):
    with pytest.raises(ValueError, match = "No image model loaded"):
        inject(_local_recipe(), image_model = None)


def test_a_chat_and_image_recipe_explains_keeping_both_models_loaded(inject):
    with pytest.raises(ValueError, match = "Keep multiple models loaded to When they fit"):
        inject(_local_recipe(with_llm = True), chat_loaded = False)

    recipe = _local_recipe(with_llm = True)
    assert inject(recipe) == 7


def test_the_loaded_image_model_is_read_from_the_active_engine(monkeypatch):
    status = {"loaded": True, "repo_id": IMAGE_MODEL}
    engine = SimpleNamespace(status = lambda: status)
    monkeypatch.setitem(
        sys.modules,
        "core.inference.diffusion_engine_router",
        SimpleNamespace(get_active_diffusion_engine = lambda: engine),
    )

    assert route._loaded_local_image_model() == IMAGE_MODEL
    status["loaded"] = False
    assert route._loaded_local_image_model() is None


def test_image_columns_count_toward_column_progress(monkeypatch):
    from core.data_recipe.jobs import manager as manager_module

    started = {}

    class _Proc:
        pid = 1

        def start(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(manager_module._CTX, "Process", lambda **kwargs: _Proc())
    monkeypatch.setattr(
        manager_module, "account_thread", lambda **kwargs: SimpleNamespace(start = lambda: None)
    )
    manager = manager_module.JobManager()
    manager.start(recipe = _local_recipe(with_llm = True), run = {})
    started["total"] = manager._job.progress_columns_total
    started["image_columns"] = manager._job.image_columns

    assert started == {"total": 2, "image_columns": ["photo"]}
