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
