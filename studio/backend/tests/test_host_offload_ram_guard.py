# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF that misses VRAM spills into host RAM under `--fit on`, unpriced. When that
spill is larger than available RAM the weights page in from disk as the model runs, so
generation is slow.

This used to REFUSE the load. It no longer does: the spill is mmap'd, so an oversized
model pages rather than failing, and running a quant larger than fast memory off an SSD
is deliberate and supported, which this check cannot tell apart from a mistake. Same
arithmetic, different consequence -- it warns, and the load proceeds."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import core.inference.llama_cpp as llama_cpp_module
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

    def test_the_warning_says_the_load_goes_ahead(self):
        """Nothing here blocks a load any more, so the message must not read as a
        refusal or send the user hunting for an env var. It states the cost and says
        the load continues."""
        msg = _shortfall(20 * _GB, 21 * _MIB_PER_GB)
        assert msg is not None
        assert "Loading anyway" in msg
        assert "UNSLOTH_ALLOW_HOST_OFFLOAD" not in msg

    def test_a_refusal_never_prints_a_need_at_or_under_the_usable_figure(self):
        """A spill inside available RAM but inside the headroom too is still refused, so
        the message must not read as 7 GB not fitting in 8 GB."""
        msg = _shortfall(7 * _GB, 8 * _MIB_PER_GB)
        assert msg is not None
        assert "About 7 GB" in msg and "6 GB usable" in msg


def test_available_ram_is_capped_by_cgroup_v2_remainder(tmp_path, monkeypatch):
    """A container sees host-wide MemAvailable through psutil, but can only charge
    memory.max - memory.current before the kernel enforces its own OOM boundary."""
    root = tmp_path / "cgroup"
    leaf = root / "studio.slice"
    leaf.mkdir(parents = True)
    (leaf / "memory.max").write_text(str(16 * _GB), encoding = "utf-8")
    (leaf / "memory.current").write_text(str(4 * _GB), encoding = "utf-8")
    proc_cgroup = tmp_path / "self.cgroup"
    proc_cgroup.write_text("0::/studio.slice\n", encoding = "utf-8")

    monkeypatch.setattr(llama_cpp_module, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(llama_cpp_module, "_PROC_SELF_CGROUP", str(proc_cgroup))
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory = lambda: SimpleNamespace(available = 64 * _GB)),
    )

    assert LlamaCppBackend._available_system_memory_mib() == 12 * _MIB_PER_GB

    backend = object.__new__(LlamaCppBackend)
    backend._get_gguf_size_bytes = lambda _path: 20 * _GB
    msg = backend._launch_host_shortfall_message(
        ["llama-server", "-m", str(tmp_path / "model.gguf")],
        [(0, 4 * _MIB_PER_GB)],
    )
    assert msg is not None
    assert "16 GB" in msg and "10 GB usable" in msg


def test_cgroup_v2_reclaims_inactive_file_cache_for_ram_admission(tmp_path, monkeypatch):
    """Cached GGUF pages are reclaimable, not another permanent host-RAM charge."""
    root = tmp_path / "cgroup"
    leaf = root / "studio.slice"
    leaf.mkdir(parents = True)
    (leaf / "memory.max").write_text(str(16 * _GB), encoding = "utf-8")
    (leaf / "memory.current").write_text(str(12 * _GB), encoding = "utf-8")
    (leaf / "memory.stat").write_text(f"inactive_file {8 * _GB}\n", encoding = "utf-8")
    proc_cgroup = tmp_path / "self.cgroup"
    proc_cgroup.write_text("0::/studio.slice\n", encoding = "utf-8")

    monkeypatch.setattr(llama_cpp_module, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(llama_cpp_module, "_PROC_SELF_CGROUP", str(proc_cgroup))
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory = lambda: SimpleNamespace(available = 64 * _GB)),
    )

    assert LlamaCppBackend._available_system_memory_mib() == 12 * _MIB_PER_GB

    backend = object.__new__(LlamaCppBackend)
    backend._get_gguf_size_bytes = lambda _path: 12 * _GB
    assert (
        backend._launch_host_shortfall_message(
            ["llama-server", "-m", str(tmp_path / "model.gguf")],
            [(0, 4 * _MIB_PER_GB)],
        )
        is None
    )


def test_cgroup_v1_reclaims_hierarchical_inactive_file_cache(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    leaf = root / "memory" / "studio.slice"
    leaf.mkdir(parents = True)
    (leaf / "memory.limit_in_bytes").write_text(str(16 * _GB), encoding = "utf-8")
    (leaf / "memory.usage_in_bytes").write_text(str(12 * _GB), encoding = "utf-8")
    (leaf / "memory.stat").write_text(
        f"inactive_file {2 * _GB}\ntotal_inactive_file {8 * _GB}\n",
        encoding = "utf-8",
    )
    proc_cgroup = tmp_path / "self.cgroup"
    proc_cgroup.write_text("5:memory:/studio.slice\n", encoding = "utf-8")

    monkeypatch.setattr(llama_cpp_module, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(llama_cpp_module, "_PROC_SELF_CGROUP", str(proc_cgroup))

    assert LlamaCppBackend._cgroup_available_memory_mib() == 12 * _MIB_PER_GB
