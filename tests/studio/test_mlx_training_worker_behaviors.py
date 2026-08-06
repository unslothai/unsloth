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


def test_mlx_dora_decided_before_load_and_merged_into_peft_kwargs():
    """The DoRA decision must be wired to happen before the base model is
    fetched, and to feed the wrap call.

    Dropping the merge would accept a DoRA request and silently train plain
    LoRA; deciding after the load would make an unsupported unsloth-zoo cost
    a multi-gigabyte download first.

    This checks WIRING, not runtime behavior: the decision precedes the load,
    takes the run config, probes the same callable the wrap invokes, and its
    result is merged into the mapping that wrap expands without being rebound
    in between. It cannot see through code that keeps the shape and defeats
    the effect -- burying the merge under a false condition, unsetting the key
    afterwards -- because the test and the code it reads live together; the
    real-model behavior is covered by the MLX backend's own tests.

    This asserts one specific shape — a `peft_kwargs.update(...)` call and a
    `**peft_kwargs` wrap call — rather than trying to recognize every
    equivalent formulation. Rewriting that seam another way (a merge
    operator, explicit unpacking) is expected to fail here and to be
    re-expressed, not worked around.
    """
    tree = ast.parse(WORKER.read_text(encoding = "utf-8"))
    fn = _find_func(tree, "_run_mlx_training")
    assert fn is not None

    decided_at = None
    decision_target = None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_mlx_dora_peft_kwargs"
        ):
            decided_at = node.lineno
            decision_target = node.targets[0].id
            # The run's own config and the same callable the wrap invokes --
            # probing anything else answers for something else.
            passed = [ast.unparse(arg) for arg in node.value.args]
            assert passed == [
                "config",
                "FastMLXModel.get_peft_model",
            ], f"unexpected arguments to _mlx_dora_peft_kwargs: {passed}"
    assert decided_at is not None, "_mlx_dora_peft_kwargs is never called"

    loaded_at = min(
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    )
    assert decided_at < loaded_at, (
        "DoRA support must be decided before the base model loads; "
        f"decided at line {decided_at}, loaded at line {loaded_at}"
    )

    merge = f"peft_kwargs.update({decision_target})"
    merged_at = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and ast.unparse(node) == merge
    ]
    assert merged_at, f"expected {merge}, or a DoRA request silently trains plain LoRA"
    wraps = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "FastMLXModel.get_peft_model"
    ]
    assert wraps, "FastMLXModel.get_peft_model is never called"
    for wrap in wraps:
        # A real ** expansion, not a string that merely looks like one.
        assert any(
            kw.arg is None and ast.unparse(kw.value) == "peft_kwargs" for kw in wrap.keywords
        ), (
            "get_peft_model must expand **peft_kwargs, or the merged DoRA "
            f"kwargs never reach it: {ast.unparse(wrap)}"
        )
    first_wrap = min(wrap.lineno for wrap in wraps)
    assert (
        min(merged_at) < first_wrap
    ), "peft_kwargs must be updated before get_peft_model is called"
    # ...and not replaced in between, which would drop the merge again.
    # A Store/Del context covers the rebinding forms a refactor realistically
    # produces. This is a regression check over one function's line range, not
    # a proof: string-bound names (import aliases, `case` captures) and
    # anything sharing the merge's line are outside it.
    rebound = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Name)
        and node.id == "peft_kwargs"
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and min(merged_at) < node.lineno < first_wrap
    ]
    assert not rebound, f"peft_kwargs rebound at {rebound}, discarding the merge"


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
    src = WORKER.read_text(encoding = "utf-8")
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
    src = WORKER.read_text(encoding = "utf-8")
    assert "from unsloth_zoo.mlx.loader import FastMLXModel" in src
    assert "from unsloth_zoo.mlx.trainer import" in src
    assert "raise ImportError" in src
    assert "install.sh" in src
