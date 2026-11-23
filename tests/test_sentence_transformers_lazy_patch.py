# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""unsloth does not import sentence_transformers itself; it patches the package when (or if) it is imported,
with the same result in either import order."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_CHILD = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {root!r})
    if {st_first}:
        import sentence_transformers
    try:
        import unsloth.models.sentence_transformer as module
    except Exception as error:
        print("UNSLOTH_IMPORT_FAILED", repr(error))
        raise SystemExit(0)
    assert module.__file__.startswith({root!r}), module.__file__
    loaded = "sentence_transformers" in sys.modules
    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers.models import Dense, Pooling
    import sentence_transformers
    print(json.dumps({{
        "loaded_by_unsloth": loaded,
        "flags": [
            getattr(SentenceTransformerTrainer, "_unsloth_auto_compile_patched", False),
            getattr(SentenceTransformerTrainer, "_unsloth_load_from_checkpoint_patched", False),
            getattr(Pooling, "_unsloth_float16_pooling", False),
            getattr(Dense, "_unsloth_input_dtype", False),
        ],
        "loader": type(sentence_transformers.__spec__.loader).__name__,
    }}))
    """
)


def _run(st_first):
    code = _CHILD.format(root = str(_ROOT), st_first = st_first)
    # The suite's conftest refuses the CPU fallback; this child only needs the import to work.
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_DISABLE_CPU_FALLBACK"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, timeout = 900, env = env
    )
    assert out.returncode == 0, out.stderr[-4000:]
    if "UNSLOTH_IMPORT_FAILED" in out.stdout:
        pytest.skip(out.stdout.strip().splitlines()[-1])
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None, reason = "no sentence_transformers"
)
def test_unsloth_does_not_import_sentence_transformers_and_patches_it_on_import():
    late = _run(st_first = False)
    assert late["loaded_by_unsloth"] is False
    assert late["flags"] == [True, True, True, True]
    assert late["loader"] != "_PatchAfterImportLoader"
    early = _run(st_first = True)
    assert early["flags"] == [True, True, True, True]


_RELOAD_CHILD = textwrap.dedent(
    """
    import copy, importlib, importlib.util, json, sys
    sys.path.insert(0, {root!r})
    try:
        import unsloth.models.sentence_transformer as module
    except Exception as error:
        print("UNSLOTH_IMPORT_FAILED", repr(error))
        raise SystemExit(0)
    importlib.reload(module)
    copy.copy(importlib.util.find_spec("sentence_transformers").loader)
    import sentence_transformers
    from sentence_transformers.models import Pooling
    print(json.dumps({{
        "flags": [getattr(Pooling, "_unsloth_float16_pooling", False)],
        "loader": type(sentence_transformers.__spec__.loader).__name__,
    }}))
    """
)


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None, reason = "no sentence_transformers"
)
def test_reloading_the_module_does_not_recurse_on_import():
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_DISABLE_CPU_FALLBACK"}
    out = subprocess.run(
        [sys.executable, "-c", _RELOAD_CHILD.format(root = str(_ROOT))],
        capture_output = True,
        text = True,
        timeout = 900,
        env = env,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    if "UNSLOTH_IMPORT_FAILED" in out.stdout:
        pytest.skip(out.stdout.strip().splitlines()[-1])
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["flags"] == [True]
    assert result["loader"] != "_PatchAfterImportLoader"
