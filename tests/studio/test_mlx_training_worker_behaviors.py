# SPDX-License-Identifier: AGPL-3.0-only

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = REPO_ROOT / "studio" / "backend" / "core" / "training" / "worker.py"


def _find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_run_mlx_training_passes_token_to_from_pretrained():
    tree = ast.parse(WORKER.read_text(encoding = "utf-8"))
    fn = _find_func(tree, "_run_mlx_training")
    assert fn is not None
    found = False
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "FastMLXModel"
        ):
            kwarg_names = {kw.arg for kw in node.keywords if kw.arg}
            assert (
                "token" in kwarg_names
            ), f"FastMLXModel.from_pretrained must forward token=hf_token; got {kwarg_names!r}"
            found = True
    assert found, "FastMLXModel.from_pretrained call not found in _run_mlx_training"


def test_wandb_init_strips_secret_keys():
    src = WORKER.read_text(encoding = "utf-8")
    assert "_wandb_sensitive" in src, "expected a sensitive-key set near wandb.init"
    assert '"hf_token"' in src and '"wandb_token"' in src
    assert (
        "config = dict(config)" not in src
    ), "wandb.init received raw config dict; secrets would leak"


def test_local_dataset_loader_uses_load_dataset_path():
    src = WORKER.read_text(encoding = "utf-8")
    assert "_resolve_mlx_local_dataset_files" in src
    assert "_mlx_local_dataset_loader_for_files" in src
    assert "data_files = all_files" in src or "data_files=all_files" in src


def test_send_aliases_status_message_to_message():
    src = WORKER.read_text(encoding = "utf-8")
    assert 'kwargs["message"] = sm' in src or 'kwargs["message"]=sm' in src


def test_slice_uses_inclusive_end_and_handles_zero():
    src = WORKER.read_text(encoding = "utf-8")
    assert "min(end + 1, len(ds))" in src or "min(end+1, len(ds))" in src
    assert "slice_start if slice_start is not None else 0" in src
    assert "slice_end if slice_end is not None else len(ds) - 1" in src


def test_poll_stop_returns_on_broken_pipe():
    tree = ast.parse(WORKER.read_text(encoding = "utf-8"))
    fn = _find_func(tree, "_start_worker_stop_poller")
    assert fn is not None
    handlers = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.ExceptHandler) or not isinstance(node.type, ast.Tuple):
            continue
        exception_names = {item.id for item in node.type.elts if isinstance(item, ast.Name)}
        if {"EOFError", "OSError"}.issubset(exception_names):
            handlers.append(node)
    assert handlers
    assert any(handler.body and isinstance(handler.body[0], ast.Return) for handler in handlers)


def test_unsloth_zoo_mlx_imports_have_friendly_error():
    src = WORKER.read_text(encoding = "utf-8")
    assert "from unsloth_zoo.mlx.loader import FastMLXModel" in src
    assert "from unsloth_zoo.mlx.trainer import" in src
    assert "raise ImportError" in src
    assert "install.sh" in src
