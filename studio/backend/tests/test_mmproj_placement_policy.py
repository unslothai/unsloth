# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vision-projector placement policy: GPU, CPU, or not loaded at all.

Model + speculative decoding + projector + 4096 context in VRAM if they fit;
drop speculative decoding first; pin the projector to the CPU only as a last
resort; load no projector at all when vision is switched off.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

MIB = 1024 * 1024
GIB = 1024**3
_REAL_POPEN = subprocess.Popen

# Model + projector fit at 4096 but not at the native length: the placement loop
# shrinks the context long before it would spill a layer, so pricing residency at
# the native length is the bug this policy exists to avoid.
NATIVE_CTX = 262144
KV_PER_TOKEN = 64 * 1024  # 4096 ctx -> 256 MiB, NATIVE_CTX -> 16 GiB


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _backend(
    tmp_path: Path,
    *,
    memory,
    model_bytes: int = 6 * GIB,
    mmproj_bytes: int = 1 * GIB,
    drafter_bytes: int = 0,
):
    """A GGUF vision load with the fit inputs pinned to known numbers."""
    backend = LlamaCppBackend()
    gguf = _write_gguf(tmp_path / "model.gguf")
    mmproj = _write_gguf(tmp_path / "mmproj-F16.gguf")
    drafter = _write_gguf(tmp_path / "mtp.gguf")

    def read_metadata(_path):
        backend._context_length = NATIVE_CTX
        backend._n_layers = 32
        backend._n_heads = 32
        backend._n_kv_heads = 8
        backend._embedding_length = 4096
        backend._vocab_size = 32000

    backend._read_gguf_metadata = read_metadata
    backend._get_gpu_memory = lambda _binary = None, **_kw: list(memory)
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [
        (index, free) for index, free, _total in memory
    ]
    backend._estimate_kv_cache_bytes = lambda ctx, *_a, **_kw: max(0, ctx) * KV_PER_TOKEN
    backend._compute_buffer_ctx_bytes = lambda *_a, **_kw: 0
    backend._estimate_compute_buffer_bytes = lambda **_kw: 256 * MIB
    backend._get_gguf_size_bytes = lambda path: (
        drafter_bytes if Path(path).name == "mtp.gguf" else model_bytes
    )
    backend._mmproj_vram_bytes = lambda _path: mmproj_bytes
    backend._resolve_launch_mmproj_path = lambda **_kw: str(mmproj)
    # Only the speculative-decoding test asks for a drafter; everywhere else the
    # resolution has to come back empty or MTP engages behind the scenes.
    backend._resolve_launch_mtp_path = lambda **_kw: str(drafter) if drafter_bytes else None
    backend._apu_ram_shortfall_message = lambda *_a, **_kw: None
    backend._amd_apu_wants_unified_memory = lambda *_a, **_kw: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_no_mmproj_offload": True,
        "mtp_token": "draft-mtp",
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    return backend, gguf


def _launch(backend, gguf, **load_kwargs):
    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env") or dict(os.environ)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                is_vision = True,
                # Auto context, the mode the policy is about: 0 resolves to the
                # model's native length and lets the placement loop shrink it. A
                # request pinned at 4096 would hide the whole floor question.
                n_ctx = 0,
                **load_kwargs,
            )
        )
    return captured


def test_projector_stays_on_gpu_when_it_fits_at_the_floor(tmp_path):
    """Model + projector fit at 4096 but not at the native 262144 context.

    The placement loop shrinks the context, it does not spill layers, so both
    arms are fully GPU-resident and pinning would buy nothing while costing
    ~8.8x on every image encode.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" not in cmd
    # The premise the pin would have been traded against: every layer is already
    # resident, so what the native length cost was context, not residency.
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert int(cmd[cmd.index("-c") + 1]) < NATIVE_CTX


def test_projector_pinned_to_cpu_when_it_does_not_fit(tmp_path):
    """The 6 GiB model fits at 4096 on this card, the projector on top does not.

    Vision is preserved rather than dropped: --mmproj still goes out, only the
    offload is disabled. Sized so the _MMPROJ_VRAM_SAFETY surcharge decides it:
    the budget is 8692 - 3% of 16384 = 8200 MiB, the footprint at 4096 is 6144
    model + 1024 projector + 256 compute + 320 CUDA context + 256 KV = 8000 MiB,
    and only the 409 MiB a projector costs beyond its file size puts it over.
    """
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf)["cmd"]

    assert "--mmproj" in cmd
    assert "--no-mmproj-offload" in cmd
    # And the trade was paid for: the model alone is fully resident, which is the
    # only thing the slower image encode is bought with.
    assert cmd[cmd.index("--fit") + 1] == "off"


def test_user_owns_the_placement_when_they_name_either_spelling(tmp_path):
    """llama.cpp is last-wins on the placement pair, so an explicit
    --mmproj-offload must not be raced by the automatic pin."""
    backend, gguf = _backend(tmp_path, memory = [(0, 8_692, 16_384)])

    cmd = _launch(backend, gguf, extra_args = ["--mmproj-offload"])["cmd"]

    assert cmd.count("--no-mmproj-offload") == 0


def test_vision_switched_off_loads_no_projector_anywhere(tmp_path, monkeypatch):
    """Not on the GPU, not on the CPU, and not through an inherited env var:
    common/arg.cpp reads LLAMA_ARG_MMPROJ straight into params.mmproj.path."""
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", "/ambient/mmproj.gguf")
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
    backend, gguf = _backend(tmp_path, memory = [(0, 7_600, 8_192)])

    result = _launch(backend, gguf, disable_vision = True)

    assert "--mmproj" not in result["cmd"]
    assert "--no-mmproj-offload" not in result["cmd"]
    assert "LLAMA_ARG_MMPROJ" not in result["env"]
    assert "LLAMA_ARG_MMPROJ_URL" not in result["env"]
    assert backend.is_vision is False


@pytest.mark.parametrize(
    "memory,label",
    [
        ([], "apple_unified"),
        ([(0, 7_600, 0)], "amd_apu_or_igpu"),
    ],
)
def test_shared_memory_pools_are_not_charged_as_discrete(tmp_path, memory, label):
    """Apple Silicon enumerates no GPU and an APU / iGPU reports free SYSTEM RAM
    as its free VRAM with a total of 0. Moving the encoder inside one pool frees
    nothing, so the same shortfall that pins a discrete card must not pin here."""
    backend, gguf = _backend(tmp_path, memory = memory)

    cmd = _launch(backend, gguf)["cmd"]

    assert "--no-mmproj-offload" not in cmd, label


def test_speculative_decoding_is_dropped_before_the_projector_is_pinned(tmp_path):
    """The drafter's reserve is not what the projector is asked to make room for.

    Model + projector fit at the floor; the drafter on top of them does not. The
    projector stays on the GPU and speculative decoding gives way instead.
    """
    backend, gguf = _backend(
        tmp_path,
        memory = [(0, 9_400, 24_000)],
        drafter_bytes = 2 * GIB,
    )

    cmd = _launch(
        backend,
        gguf,
        mtp_draft_path = str(tmp_path / "mtp.gguf"),
        speculative_type = "auto",
    )["cmd"]

    # The premise: Auto gave the drafter up, so its reserve is not part of the
    # footprint the projector is measured against.
    assert "--model-draft" not in cmd
    assert "--no-mmproj-offload" not in cmd
