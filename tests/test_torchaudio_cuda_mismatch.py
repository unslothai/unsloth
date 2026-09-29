# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A text model should not die on an audio library it never asked for.

`torchaudio._extension.utils._check_cuda_version` compares the CUDA version
torchaudio was BUILT against with torch's, and raises on any difference:

    RuntimeError: Detected that PyTorch and TorchAudio were compiled with
    different CUDA versions.

That runs at extension init, so it takes the whole import with it. Measured on
a Kaggle 2xT4 session running `Kaggle-Muse_Glimmer_(30B)-GRPO` -- a text model
-- which died at cell 4 having never reached anything audio-shaped.

The repair is the one `disable_torchcodec_if_broken` already makes for the same
structural reason: the package resolves, `find_spec` says so, and the failure is
at native init, so every downstream `except ImportError` handler is bypassed.
Seating the absence sentinel gives them their chance back.

What it must NOT do is patch out `_check_cuda_version`. That check is correct --
torchaudio's CUDA ops really are unusable against a different runtime -- and
silencing it in place would leave those ops reachable and wrong. The last test
here is the one that pins that distinction.
"""

from __future__ import annotations

import sys

import pytest


# The repair flips availability state on the REAL transformers and datasets modules, not on copies.
# Restoring only sys.modules would leave `is_torchaudio_available` bound to `lambda: False` for every later test in the
# process, so the fixture snapshots these too.
_PATCH_SITES = (
    ("transformers.utils.import_utils", "_torchaudio_available"),
    ("transformers.utils.import_utils", "is_torchaudio_available"),
    ("transformers.utils.import_utils", "is_speech_available"),
    ("datasets.config", "TORCHAUDIO_AVAILABLE"),
)

_MISSING = object()


@pytest.fixture
def fresh(monkeypatch):
    """Import the repair without importing unsloth's whole init."""
    import importlib

    module = importlib.import_module("unsloth.import_fixes")
    saved = {k: v for k, v in sys.modules.items() if k.startswith("torchaudio")}

    flags = []
    for mod_name, attr in _PATCH_SITES:
        try:
            owner = importlib.import_module(mod_name)
        except ImportError:
            continue
        flags.append((owner, attr, getattr(owner, attr, _MISSING)))

    yield module

    for key in [k for k in sys.modules if k.startswith("torchaudio")]:
        sys.modules.pop(key, None)
    sys.modules.update(saved)
    for owner, attr, value in flags:
        if value is _MISSING:
            if hasattr(owner, attr):
                delattr(owner, attr)
        else:
            setattr(owner, attr, value)


def _stage(monkeypatch, fresh, error):
    """Present a torchaudio that resolves and then fails at init."""
    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "torchaudio" else None,
    )

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def fake_import(name, *args, **kwargs):
        if name == "torchaudio" or name.startswith("torchaudio."):
            if error is None:
                module = type(sys)("torchaudio")
                sys.modules["torchaudio"] = module
                return module
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "torchaudio", None)
    sys.modules.pop("torchaudio")
    monkeypatch.setattr("builtins.__import__", fake_import)


MISMATCH = RuntimeError(
    "Detected that PyTorch and TorchAudio were compiled with different CUDA "
    "versions. PyTorch has CUDA version 12.8 whereas TorchAudio has CUDA "
    "version 12.6."
)


def test_a_mismatched_torchaudio_is_made_absent(monkeypatch, fresh):
    _stage(monkeypatch, fresh, MISMATCH)
    with pytest.warns(UserWarning, match = "torchaudio cannot initialise"):
        fresh.disable_torchaudio_if_cuda_mismatched()
    assert sys.modules.get("torchaudio", "missing") is None


def test_the_speech_backend_goes_down_with_torchaudio(monkeypatch, fresh):
    """`speech` is torchaudio wearing a different name, so it has to follow.

    On transformers 5 `is_speech_available` is separately `@lru_cache`d, so a
    `speech` answer computed before the repair survives it. Callers gated on
    `requires_backends(..., "speech")` are then waved into a torchaudio that is
    now a None sentinel, which is the crash this whole file exists to prevent.
    """
    from functools import lru_cache

    tf_iu = pytest.importorskip("transformers.utils.import_utils")

    # Stand up the 5.x shape explicitly rather than asking whichever
    # transformers happens to be installed: on 4.x both readers share one
    # module global, so the 4.x version of this test cannot fail.
    monkeypatch.delattr(tf_iu, "_torchaudio_available", raising = False)
    monkeypatch.setattr(tf_iu, "is_torchaudio_available", lru_cache(lambda: True))
    monkeypatch.setattr(
        tf_iu, "is_speech_available", lru_cache(lambda: tf_iu.is_torchaudio_available())
    )

    _stage(monkeypatch, fresh, MISMATCH)
    assert tf_iu.is_speech_available() is True  # warmed, as a live process would be
    with pytest.warns(UserWarning, match = "torchaudio cannot initialise"):
        fresh.disable_torchaudio_if_cuda_mismatched()
    assert tf_iu.is_torchaudio_available() is False
    assert tf_iu.is_speech_available() is False


def test_a_healthy_torchaudio_is_left_alone(monkeypatch, fresh):
    """The repair must not cost anything on the machines that are fine."""
    _stage(monkeypatch, fresh, None)
    fresh.disable_torchaudio_if_cuda_mismatched()
    assert sys.modules.get("torchaudio") is not None


def test_an_absent_torchaudio_is_not_invented(monkeypatch, fresh):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **k: None)
    fresh.disable_torchaudio_if_cuda_mismatched()
    assert "torchaudio" not in sys.modules or sys.modules["torchaudio"] is not None


def test_an_unrelated_failure_is_re_raised(monkeypatch, fresh):
    """Swallowing it would hide a real error behind a message about CUDA
    versions, which is the failure mode this whole file exists to avoid."""
    _stage(monkeypatch, fresh, RuntimeError("something else entirely"))
    with pytest.raises(RuntimeError, match = "something else entirely"):
        fresh.disable_torchaudio_if_cuda_mismatched()


def test_warning_filters_promoted_to_errors_do_not_abort_the_repair(monkeypatch, fresh):
    """PYTHONWARNINGS=error and `pytest -W error` are both real. The repair is
    more important than its own announcement."""
    import warnings

    _stage(monkeypatch, fresh, MISMATCH)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fresh.disable_torchaudio_if_cuda_mismatched()
    assert sys.modules.get("torchaudio", "missing") is None


def test_the_check_itself_is_never_patched_out():
    """The distinction the docstring turns on, asserted rather than trusted.

    Monkeypatching `_check_cuda_version` to return would leave torchaudio's
    CUDA ops importable and broken. Making the package absent is the honest
    repair; a future edit that reaches for the shortcut fails here.
    """
    import ast
    import inspect
    import textwrap

    from unsloth import import_fixes

    func = import_fixes.disable_torchaudio_if_cuda_mismatched
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    # The docstring names it, deliberately, to say why it is NOT touched, so strip it by AST rather than by string
    # surgery. Only the body is a claim.
    body = tree.body[0].body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    code = "\n".join(ast.unparse(node) for node in body)
    assert "_check_cuda_version" not in code


def test_it_runs_before_the_torchcodec_repair_because_it_has_to():
    """This assertion used to run the other way round, and was wrong.

    Both repairs seat sentinels, and the audio decoder path touches both, so
    ordering them torchcodec-first looked natural. But torchcodec is only
    reached lazily, while torchaudio is imported eagerly by
    transformers.audio_utils as soon as unsloth_zoo is imported -- which
    happens ~95 lines BEFORE the late fix block where torchcodec is repaired.
    Ordering by tidiness rather than by when each package actually gets
    imported is what let Kaggle-Muse_Glimmer_(30B)-GRPO keep dying at cell 4
    with the guard present and shipped.
    """
    from pathlib import Path

    init = (Path(import_fixes_dir()) / "_gpu_init.py").read_text()
    assert init.index("disable_torchaudio_if_cuda_mismatched()") < init.index(
        "disable_torchcodec_if_broken()"
    )


def import_fixes_dir():
    import unsloth
    from pathlib import Path
    return Path(unsloth.__file__).parent


def test_the_guard_runs_before_anything_can_import_torchaudio():
    """Defined is not the same as run in time.

    The guard shipped invoked at line 250 of _gpu_init, and `import
    unsloth_zoo` sits at line 155. unsloth_zoo's temporary_patches reach
    transformers.processing_utils -> transformers.audio_utils -> torchaudio,
    so a torchaudio that raises at extension init took the whole unsloth
    import down 95 lines before the repair would have run. Measured:
    Kaggle-Muse_Glimmer_(30B)-GRPO still died at cell 4 with the fix present.

    Ordering is the property that matters, so assert on it directly.
    """
    from pathlib import Path

    src = (
        (Path(__file__).resolve().parents[1] / "unsloth" / "_gpu_init.py")
        .read_text(encoding = "utf-8")
        .splitlines()
    )

    def line_of(predicate):
        for i, line in enumerate(src):
            if predicate(line):
                return i
        return None

    call = line_of(lambda l: l.strip() == "disable_torchaudio_if_cuda_mismatched()")
    zoo = line_of(lambda l: l.strip() == "import unsloth_zoo")
    assert call is not None, "the guard is never called"
    assert zoo is not None, "could not find the unsloth_zoo import"
    assert call < zoo, (
        f"disable_torchaudio_if_cuda_mismatched() runs at line {call + 1}, "
        f"after `import unsloth_zoo` at line {zoo + 1}; torchaudio is already "
        f"imported by then and the guard cannot help"
    )
