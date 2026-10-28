"""apply_unsloth_gradient_checkpointing with several models in one process.

Smart offloading is one process-wide patch. A second model loaded with the short sequence
fallback (max_seq_length < 512) used to unpatch it, so a model loaded earlier with
use_gradient_checkpointing = "unsloth" silently stopped offloading (offloads per step went from
28 to 0 for Qwen3-0.6B) and could run out of memory. The fallback is only a speed heuristic, so
it now keeps "unsloth" when an earlier model turned offloading on; an explicit True / False still turns it off, and
says so. The function is read from source and run against a stub torch, so no GPU is needed.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
NAMES = ("_smart_gradient_checkpointing_patched", "apply_unsloth_gradient_checkpointing")


def _load():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    found = {n.name for n in nodes}
    if "apply_unsloth_gradient_checkpointing" not in found:
        pytest.fail("apply_unsloth_gradient_checkpointing not found")

    class Plain: pass
    class UnslothCheckpointFunction: pass
    ckpt = SimpleNamespace(CheckpointFunction = Plain)
    torch = SimpleNamespace(utils = SimpleNamespace(checkpoint = ckpt))
    log = []

    def patch(dtype = None):
        ckpt.CheckpointFunction = UnslothCheckpointFunction
        log.append("patch")

    def unpatch():
        ckpt.CheckpointFunction = Plain
        log.append("unpatch")

    ns = {"torch": torch, "patch_unsloth_smart_gradient_checkpointing": patch,
          "unpatch_unsloth_smart_gradient_checkpointing": unpatch}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE), "exec"), ns)
    patched = lambda: ckpt.CheckpointFunction is UnslothCheckpointFunction  # noqa: E731
    return ns["apply_unsloth_gradient_checkpointing"], patched


def test_short_sequence_model_keeps_an_earlier_models_offload():
    apply, patched = _load()
    assert apply("unsloth", 2048, None) == "unsloth" and patched()
    # Stays "unsloth": a True here makes prepare_model_for_training unpatch it anyway.
    assert apply("unsloth", 256, None) == "unsloth"
    assert patched(), "a <512 model turned smart offloading off for the model loaded before it"


def test_short_sequence_alone_still_uses_plain_checkpointing():
    apply, patched = _load()
    assert apply("unsloth", 256, None) is True
    assert not patched()
    assert apply("unsloth", 4096, None) == "unsloth" and patched()


@pytest.mark.parametrize("value", [True, False])
def test_explicit_true_or_false_turns_it_off_and_says_so(value, capsys):
    apply, patched = _load()
    apply("unsloth", 2048, None)
    capsys.readouterr()
    assert apply(value, 2048, None) is value
    assert not patched()
    assert "turns off smart offloading" in capsys.readouterr().out


def test_explicit_true_without_offload_is_quiet(capsys):
    apply, patched = _load()
    assert apply(True, 2048, None) is True and not patched()
    assert capsys.readouterr().out == ""
