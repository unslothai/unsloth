# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF that misses VRAM spills into host RAM under `--fit on`, unpriced. When
that spill is larger than available RAM the mmap'd weights thrash rather than
fail, the desktop stops responding and the OS kills the app, so the load must be
refused before llama-server is spawned."""

from __future__ import annotations

from core.inference.llama_cpp import LlamaCppBackend

_GB = 1024**3
_MIB_PER_GB = 1024
# Module-level (not a class attr) so it stays a plain function, not a bound method.
_shortfall = LlamaCppBackend._host_offload_shortfall_message


class TestHostOffloadShortfall:
    def test_field_case_refuses(self):
        # 13.3 GB GGUF + 1.1 GB mmproj + 1.8 GB KV on a 6 GB RTX 4050 laptop holding
        # 4.8 GB free, against ~10 GB of RAM: about 11 GB has to run from host memory.
        offload = int(16.2 * _GB) - int(4.8 * _GB)
        msg = _shortfall(offload, 10 * _MIB_PER_GB)
        assert msg is not None
        # need rounds up and usable rounds down, so the pair never reads as a tie
        assert "12 GB" in msg and "10 GB" in msg and "8 GB usable" in msg
        assert "quantized GGUF" in msg
        # the guard prices weights only, so context length cannot change its verdict
        assert "context" not in msg

    def test_same_spill_on_a_large_ram_host_allows(self):
        # Deliberate CPU offload is a supported mode; only a shortfall refuses.
        offload = int(16.2 * _GB) - int(4.8 * _GB)
        assert _shortfall(offload, 64 * _MIB_PER_GB) is None

    def test_vram_resident_load_never_refuses(self):
        # More VRAM than the load needs, so the subtraction goes negative.
        assert _shortfall(-4 * _GB, 1 * _MIB_PER_GB) is None
        assert _shortfall(0, 1 * _MIB_PER_GB) is None

    def test_unknown_available_never_refuses(self):
        assert _shortfall(40 * _GB, None) is None

    def test_boundary_at_headroom(self):
        # 20 GB spill, headroom 2 GB. avail 23 GB -> fits; 21 GB -> refuse.
        assert _shortfall(20 * _GB, 23 * _MIB_PER_GB) is None
        assert _shortfall(20 * _GB, 21 * _MIB_PER_GB) is not None

    def test_a_refusal_never_prints_a_need_at_or_under_the_usable_figure(self):
        """A spill inside available RAM but inside the headroom too is still refused, so
        the message must not read as 7 GB not fitting in 8 GB."""
        msg = _shortfall(7 * _GB, 8 * _MIB_PER_GB)
        assert msg is not None
        assert "About 7 GB" in msg and "6 GB usable" in msg
