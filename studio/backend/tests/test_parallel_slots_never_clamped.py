# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launched --parallel must equal the requested slot count.

#7717 clamped it to 1 whenever MTP resolved, which cost batched API callers up
to 4x throughput behind a single log line. These drive the real load path and
read the slot count back off the launched argv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Reuses the only harness in the suite that yields a real launched argv.
from test_llama_cpp_placement import _backend, _launch, _write_gguf  # noqa: E402

_MEMORY = [(0, 40 * 1024**3, 48 * 1024**3)]

# Everything the MTP branch of _build_speculative_flags probes for.
_CAPS = {
    "found": True,
    "supports_kv_unified": True,
    "supports_mtp": True,
    "mtp_token": "draft-mtp",
    "spec_draft_n_max_flag": "--spec-draft-n-max",
}


@pytest.fixture
def mtp_backend(tmp_path, monkeypatch):
    backend, _ = _backend(tmp_path, vulkan = False, memory = _MEMORY)
    monkeypatch.setattr(
        type(backend), "probe_server_capabilities", classmethod(lambda cls, binary = None: _CAPS)
    )
    # The name is what _is_mtp_model_name reads, so this GGUF resolves to MTP.
    return backend, _write_gguf(tmp_path / "Qwen3.5-9B-MTP.gguf")


def _slots(cmd: list[str]) -> int:
    return int(cmd[cmd.index("--parallel") + 1])


@pytest.mark.parametrize("requested", [1, 2, 4, 8])
@pytest.mark.parametrize("speculative", ["auto", "mtp", "mtp+ngram", "ngram", "off"])
def test_the_launch_serves_the_slots_that_were_asked_for(mtp_backend, requested, speculative):
    backend, gguf = mtp_backend
    cmd = _launch(backend, gguf, n_parallel = requested, speculative_type = speculative)["cmd"]
    assert _slots(cmd) == requested


@pytest.mark.parametrize("requested", [2, 4, 8])
def test_extras_owned_mtp_keeps_the_slots_too(mtp_backend, requested):
    """The other route into MTP: --spec-type in the user's own extra args."""
    backend, gguf = mtp_backend
    cmd = _launch(
        backend,
        gguf,
        n_parallel = requested,
        extra_args = ["--spec-type", "draft-mtp"],
    )["cmd"]
    assert _slots(cmd) == requested


@pytest.mark.parametrize("requested", [2, 4, 8])
def test_an_inherited_spec_env_does_not_take_the_slots(mtp_backend, monkeypatch, requested):
    """LLAMA_ARG_SPEC_TYPE cannot be cleared by a later flag, so it used to clamp."""
    backend, gguf = mtp_backend
    monkeypatch.setenv("LLAMA_ARG_SPEC_TYPE", "draft-mtp")
    cmd = _launch(backend, gguf, n_parallel = requested)["cmd"]
    assert _slots(cmd) == requested


@pytest.mark.parametrize("requested", [2, 8])
def test_the_batch_floor_follows_the_slot_count(mtp_backend, requested):
    """llama-server aborts below the slot count, so -b rises with an MTP load too."""
    backend, gguf = mtp_backend
    cmd = _launch(
        backend, gguf, n_parallel = requested, n_batch = 1, speculative_type = "mtp"
    )["cmd"]
    assert _slots(cmd) == requested
    assert int(cmd[cmd.index("--batch-size") + 1]) >= max(2, requested)


def test_a_build_without_kv_unified_still_falls_back_to_one_slot(tmp_path, monkeypatch):
    """The one downgrade that survives: more slots would split the context window."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = _MEMORY)
    monkeypatch.setattr(
        type(backend),
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {**_CAPS, "supports_kv_unified": False}),
    )
    cmd = _launch(backend, gguf, n_parallel = 4, speculative_type = "mtp")["cmd"]
    assert _slots(cmd) == 1
