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
    tree = ast.parse(WORKER.read_text())
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
    src = WORKER.read_text()
    assert "_wandb_sensitive" in src, "expected a sensitive-key set near wandb.init"
    assert '"hf_token"' in src and '"wandb_token"' in src
    assert (
        "config = dict(config)" not in src
    ), "wandb.init received raw config dict; secrets would leak"


def test_local_dataset_loader_uses_load_dataset_path():
    src = WORKER.read_text()
    assert "_resolve_local_files" in src
    assert "_loader_for_files" in src
    assert "data_files = all_files" in src or "data_files=all_files" in src


def test_send_aliases_status_message_to_message():
    src = WORKER.read_text()
    assert 'kwargs["message"] = sm' in src or 'kwargs["message"]=sm' in src


def test_slice_uses_inclusive_end_and_handles_zero():
    src = WORKER.read_text()
    assert "min(end + 1, len(ds))" in src or "min(end+1, len(ds))" in src
    assert "slice_start if slice_start is not None else 0" in src
    assert "slice_end if slice_end is not None else len(ds) - 1" in src


def test_poll_stop_returns_on_broken_pipe():
    src = WORKER.read_text()
    assert "except (EOFError, OSError)" in src
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "except (EOFError, OSError)" in line:
            for j in range(i + 1, min(i + 6, len(lines))):
                stripped = lines[j].strip()
                if not stripped or stripped.startswith("#"):
                    continue
                assert stripped.startswith(
                    "return"
                ), f"expected return after EOFError/OSError, got {stripped!r}"
                break
            break
    else:
        raise AssertionError("EOFError/OSError handler not found in worker.py")


def test_unsloth_zoo_mlx_imports_have_friendly_error():
    src = WORKER.read_text()
    assert "from unsloth_zoo.mlx_loader import FastMLXModel" in src
    assert "from unsloth_zoo.mlx_trainer import" in src
    assert "raise ImportError" in src
    assert "install.sh" in src
