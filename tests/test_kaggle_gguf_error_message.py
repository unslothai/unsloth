# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Kaggle 20GB message must only fire on failures that are about disk.

It used to fire for every GGUF failure, so an unconvertible architecture, a
missing tokenizer and a bad quant method all told the user to free up space.
"""

import os
import shutil
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

save = pytest.importorskip("unsloth.save")
_looks_like_disk = save._gguf_failure_looks_like_disk

_Usage = namedtuple("_Usage", ("total", "used", "free"))


@pytest.fixture
def plenty_of_free_space(monkeypatch):
    """State the premise the "not a disk problem" tests are written under.

    `_gguf_failure_looks_like_disk` has a second, independent signal: a
    filesystem with less than `_DISK_HEADROOM_BYTES` (2GiB) free is a disk
    failure whatever the exception says. It probes `save_directory` and then
    `os.getcwd()`, so on a host whose working directory is that full, every
    "this is NOT a disk problem" test inverts and fails for a reason that has
    nothing to do with the message it is asserting about.

    Report ample space so the message is the only signal left, which is what
    those tests are about ("a broken quantizer with 19GB free is not a disk
    problem", save.py). The real threshold is left alone, so the comparison
    still runs and a nonsensical headroom would still be caught.
    """
    real_disk_usage = shutil.disk_usage
    ample = _Usage(total = 100 * 1024**3, used = 1 * 1024**3, free = 99 * 1024**3)

    def plenty(path):
        # Keep the real failure modes;
        real_disk_usage(path)
        return ample

    monkeypatch.setattr(shutil, "disk_usage", plenty)
    return ample




@pytest.mark.parametrize(
    "msg",
    [
        "OSError: [Errno 28] No space left on device",
        "Not enough free space to write 262144 bytes",
        "Disk quota exceeded",
        "write failed: no space left",
    ],
)
def test_disk_wordings_are_recognised(msg):
    assert _looks_like_disk(RuntimeError(msg)) is True


def test_errno_attribute_is_enough_on_its_own():
    exc = OSError("something opaque")
    exc.errno = 28
    assert _looks_like_disk(exc) is True


def test_the_check_is_case_insensitive():
    assert _looks_like_disk(RuntimeError("NO SPACE LEFT ON DEVICE")) is True




@pytest.mark.usefixtures("plenty_of_free_space")
def test_an_unconvertible_architecture_is_not_a_disk_problem():
    """The bert_classification case."""
    exc = NotImplementedError("Model ModernBertForSequenceClassification is not supported")
    assert _looks_like_disk(exc, os.getcwd()) is False


@pytest.mark.usefixtures("plenty_of_free_space")
def test_a_missing_tokenizer_is_not_a_disk_problem():
    exc = ValueError("Unsloth: Saving to GGUF must have a tokenizer.")
    assert _looks_like_disk(exc, os.getcwd()) is False


@pytest.mark.usefixtures("plenty_of_free_space")
def test_a_bad_quant_method_is_not_a_disk_problem():
    exc = RuntimeError("Unknown quantization method: q9_k_xxl")
    assert _looks_like_disk(exc, os.getcwd()) is False




# ---- the guard must never be what raises ---------------------------------
def test_a_nonexistent_directory_does_not_raise():
    assert _looks_like_disk(RuntimeError("boom"), "/definitely/not/a/real/path") in (True, False)


def test_none_directory_does_not_raise():
    assert _looks_like_disk(RuntimeError("boom"), None) in (True, False)


def test_an_exception_with_no_message_does_not_raise():
    assert _looks_like_disk(RuntimeError()) in (True, False)




def test_the_kaggle_branch_is_gated_on_the_check():
    """Source-level, because reaching the branch needs a real conversion."""
    import ast

    src = Path(save.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(src)
    gated = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "IS_KAGGLE_ENVIRONMENT" in names and "_gguf_failure_looks_like_disk" in names:
            gated = True
    assert gated, "the Kaggle 20GB message is no longer gated on the disk check"


def test_the_real_error_survives_either_way():
    """Both branches must carry the original error text."""
    src = Path(save.__file__).read_text(encoding = "utf-8")
    i = src.index("GGUF conversion failed in Kaggle environment")
    window = src[i - 200 : i + 900]
    assert (
        window.count("from e") >= 2
    ), "the original exception must be chained so the traceback survives"
    # `{e}` is empty when the exception has no args, so the type-leading form counts too.
    assert (
        "GGUF conversion failed: {e}" in window
        or "GGUF conversion failed: {_describe_exception(e)}" in window
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))




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


def test_a_shell_wrapped_137_is_recognised():
    """llama-quantize runs under `shell = True`, so /bin/sh reports a SIGKILLed
    child as exit status 137 and never names the signal. unsloth_zoo then
    re-raises a plain RuntimeError, dropping `returncode`, so the wording is
    the only thing left."""
    from unsloth.save import _gguf_child_was_oom_killed

    exc = RuntimeError(
        "Failed to quantize model.BF16.gguf to q4_k_m: Command "
        "'/root/llama.cpp/llama-quantize model.BF16.gguf model.Q4_K_M.gguf q4_k_m 8' "
        "returned non-zero exit status 137."
    )
    assert _gguf_child_was_oom_killed(exc)


def test_a_chained_cause_is_inspected():
    """`raise RuntimeError(...) from e` keeps the CalledProcessError, and its
    returncode is a stronger signal than any wording."""
    import subprocess

    from unsloth.save import _gguf_child_was_oom_killed

    cause = subprocess.CalledProcessError(137, "llama-quantize ...")
    outer = RuntimeError("Unsloth: Quantization failed for model.Q4_K_M.gguf")
    outer.__cause__ = cause
    assert _gguf_child_was_oom_killed(outer)


def test_an_implicit_context_is_inspected():
    """Layers that re-raise without `from` still leave __context__ behind."""
    import subprocess

    from unsloth.save import _gguf_child_was_oom_killed

    try:
        try:
            raise subprocess.CalledProcessError(-9, "convert_hf_to_gguf.py ...")
        except subprocess.CalledProcessError:
            raise RuntimeError("Unsloth: GGUF conversion failed")
    except RuntimeError as outer:
        assert _gguf_child_was_oom_killed(outer)


def test_the_quantize_wrapper_chains_its_cause():
    """The build-llama.cpp branch must chain too, else the 137 is unreachable
    from the outer handler."""
    src = Path(save.__file__).read_text(encoding = "utf-8")
    i = src.index("You might have to compile llama.cpp yourself")
    assert "from e" in src[i : i + 900]


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
        "IS_KAGGLE_ENVIRONMENT and _gguf_failure_looks_like_disk"
    )


def test_the_message_says_host_ram_not_gpu_or_disk():
    """The whole point: SIGKILL names no resource, and the user's first guess
    is usually VRAM."""
    import inspect
    from unsloth import save as _s

    src = inspect.getsource(_s.unsloth_save_pretrained_gguf)
    i = src.index("_gguf_child_was_oom_killed(e)")
    body = src[i : i + 900]
    assert "host RAM" in body
    assert "rather than GPU memory or disk" in body


def test_it_chains_the_original():
    import inspect
    from unsloth import save as _s

    src = inspect.getsource(_s.unsloth_save_pretrained_gguf)
    i = src.index("_gguf_child_was_oom_killed(e)")
    assert "from e" in src[i : i + 900]




def test_no_kaggle_disk_message_is_left_ungated():
    """The outer gate cannot undo a disk explanation already baked into the
    inner RuntimeError's message, so save_to_gguf's own missing-output and
    quantize handlers have to make the same check. unslothai/unsloth#835."""
    import ast

    src = Path(save.__file__).read_text(encoding = "utf-8")
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


@pytest.mark.usefixtures("plenty_of_free_space")
def test_a_broken_quantizer_is_not_a_disk_problem():
    """The failure the inner quantize handler used to blame on disk."""
    exc = RuntimeError("llama-quantize: unknown quantization type")
    assert _looks_like_disk(exc, os.getcwd()) is False


def test_the_inner_quantize_branch_chains_the_original():
    src = Path(save.__file__).read_text(encoding = "utf-8")
    i = src.index("Unsloth: Quantization failed for {output_location}")
    assert "from e" in src[i : i + 900]
