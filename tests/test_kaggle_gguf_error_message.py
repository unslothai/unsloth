"""The Kaggle 20GB message must only fire on failures that are about disk.

It used to fire for every GGUF failure, so an unconvertible architecture, a
missing tokenizer and a bad quant method all told the user to free up space.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

save = pytest.importorskip("unsloth.save")
_looks_like_disk = save._gguf_failure_looks_like_disk


# ---- failures that ARE about disk -----------------------------------------

@pytest.mark.parametrize("msg", [
    "OSError: [Errno 28] No space left on device",
    "Not enough free space to write 262144 bytes",
    "Disk quota exceeded",
    "write failed: no space left",
])
def test_disk_wordings_are_recognised(msg):
    assert _looks_like_disk(RuntimeError(msg)) is True


def test_errno_attribute_is_enough_on_its_own():
    exc = OSError("something opaque")
    exc.errno = 28
    assert _looks_like_disk(exc) is True


def test_the_check_is_case_insensitive():
    assert _looks_like_disk(RuntimeError("NO SPACE LEFT ON DEVICE")) is True


# ---- failures that are NOT about disk -------------------------------------

def test_an_unconvertible_architecture_is_not_a_disk_problem():
    """The bert_classification case."""
    exc = NotImplementedError(
        "Model ModernBertForSequenceClassification is not supported")
    assert _looks_like_disk(exc, os.getcwd()) is False


def test_a_missing_tokenizer_is_not_a_disk_problem():
    exc = ValueError("Unsloth: Saving to GGUF must have a tokenizer.")
    assert _looks_like_disk(exc, os.getcwd()) is False


def test_a_bad_quant_method_is_not_a_disk_problem():
    exc = RuntimeError("Unknown quantization method: q9_k_xxl")
    assert _looks_like_disk(exc, os.getcwd()) is False


# ---- the guard must never be what raises ---------------------------------

def test_a_nonexistent_directory_does_not_raise():
    assert _looks_like_disk(RuntimeError("boom"),
                            "/definitely/not/a/real/path") in (True, False)


def test_none_directory_does_not_raise():
    assert _looks_like_disk(RuntimeError("boom"), None) in (True, False)


def test_an_exception_with_no_message_does_not_raise():
    assert _looks_like_disk(RuntimeError()) in (True, False)


# ---- the call site --------------------------------------------------------

def test_the_kaggle_branch_is_gated_on_the_check():
    """Source-level, because reaching the branch needs a real conversion."""
    import ast
    src = Path(save.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    gated = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "IS_KAGGLE_ENVIRONMENT" in names and \
                "_gguf_failure_looks_like_disk" in names:
            gated = True
    assert gated, (
        "the Kaggle 20GB message is no longer gated on the disk check")


def test_the_real_error_survives_either_way():
    """Both branches must carry the original error text."""
    src = Path(save.__file__).read_text(encoding="utf-8")
    i = src.index("GGUF conversion failed in Kaggle environment")
    window = src[i - 200:i + 900]
    assert window.count("from e") >= 2, (
        "the original exception must be chained so the traceback survives")
    assert "GGUF conversion failed: {e}" in window


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- a converter killed by the OOM-killer ---------------------------------

def test_sigkill_is_recognised_from_the_message():
    """subprocess renders it as text, which is all Unsloth re-raises."""
    from unsloth.save import _gguf_child_was_oom_killed
    exc = RuntimeError("Command '[...]' died with <Signals.SIGKILL: 9>.")
    assert _gguf_child_was_oom_killed(exc)


@pytest.mark.parametrize("code", [-9, 137])
def test_sigkill_is_recognised_from_the_returncode(code):
    """CalledProcessError uses -9; a shell wrapper reports 137."""
    from unsloth.save import _gguf_child_was_oom_killed

    class _Called(Exception):
        returncode = code

    assert _gguf_child_was_oom_killed(_Called())


def test_an_ordinary_converter_failure_is_not_called_an_oom():
    """A converter that fails on its own must keep its own message."""
    from unsloth.save import _gguf_child_was_oom_killed
    exc = RuntimeError("NotImplementedError: Unknown tensor name audio_tower.x")
    assert not _gguf_child_was_oom_killed(exc)


def test_a_disk_failure_is_not_called_an_oom():
    from unsloth.save import _gguf_child_was_oom_killed
    assert not _gguf_child_was_oom_killed(OSError("No space left on device"))


def test_the_oom_branch_runs_before_the_kaggle_disk_branch():
    """A SIGKILL on Kaggle with a full-ish disk would otherwise be reported as
    a disk problem, which is the wrong advice."""
    import inspect
    from unsloth import save as _s
    src = inspect.getsource(_s.unsloth_save_pretrained_gguf)
    assert src.index("_gguf_child_was_oom_killed(e)") < src.index(
        "IS_KAGGLE_ENVIRONMENT and _gguf_failure_looks_like_disk")


def test_the_message_says_host_ram_not_gpu_or_disk():
    """The whole point: SIGKILL names no resource, and the user's first guess
    is usually VRAM."""
    import inspect
    from unsloth import save as _s
    src = inspect.getsource(_s.unsloth_save_pretrained_gguf)
    i = src.index("_gguf_child_was_oom_killed(e)")
    body = src[i:i + 900]
    assert "host RAM" in body
    assert "rather than GPU memory or disk" in body


def test_it_chains_the_original():
    import inspect
    from unsloth import save as _s
    src = inspect.getsource(_s.unsloth_save_pretrained_gguf)
    i = src.index("_gguf_child_was_oom_killed(e)")
    assert "from e" in src[i:i + 900]


# ---- the inner conversion/quantize branches are gated too ------------------

def test_no_kaggle_disk_message_is_left_ungated():
    """The outer gate cannot undo a disk explanation already baked into the
    inner RuntimeError's message, so save_to_gguf's own missing-output and
    quantize handlers have to make the same check. unslothai/unsloth#835."""
    import ast
    src = Path(save.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    ungated = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "IS_KAGGLE_ENVIRONMENT" not in names:
            continue
        if "20GB" not in (ast.get_source_segment(src, node) or ""):
            continue
        if "_gguf_failure_looks_like_disk" not in names:
            ungated.append(node.lineno)
    assert not ungated, f"20GB disk message still ungated at lines {ungated}"


def test_a_broken_quantizer_is_not_a_disk_problem():
    """The failure the inner quantize handler used to blame on disk."""
    exc = RuntimeError("llama-quantize: unknown quantization type")
    assert _looks_like_disk(exc, os.getcwd()) is False


def test_the_inner_quantize_branch_chains_the_original():
    src = Path(save.__file__).read_text(encoding="utf-8")
    i = src.index("Unsloth: Quantization failed for {output_location}")
    assert "from e" in src[i:i + 900]
