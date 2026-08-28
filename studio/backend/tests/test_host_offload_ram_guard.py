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


# ------------------------------------------------- prompt cache in the footprint

def _cache_bytes(cache_ram, caps = None):
    from core.inference.llama_cpp import LlamaCppBackend

    return LlamaCppBackend._effective_prompt_cache_bytes(cache_ram, caps)


def test_the_prompt_cache_defaults_to_llama_cpps_own_8192_mib():
    """Unset means llama-server's default applies, and that default is 8 GiB of
    host RAM (common/common.h:632) that no footprint term used to charge."""
    assert _cache_bytes(None) == 8192 * 1024 * 1024


def test_an_explicit_zero_disables_the_cache_and_costs_nothing():
    assert _cache_bytes(0) == 0


def test_a_typed_ceiling_is_charged_at_what_was_typed():
    assert _cache_bytes(512) == 512 * 1024 * 1024


def test_no_limit_is_charged_as_the_default_not_as_infinity():
    """-1 is llama.cpp's "no limit". Charging infinity would answer "never fits"
    for every load and take --load-mode none away from hosts that are fine; the
    default is the size it is actually likely to reach."""
    assert _cache_bytes(-1) == 8192 * 1024 * 1024


def test_a_build_without_the_flag_has_no_prompt_cache_to_charge():
    """--cache-ram predates nothing here: a server that does not accept it has no
    prompt cache, so charging one would refuse fits that are real."""
    assert _cache_bytes(None, {"supports_cache_ram": False}) == 0
    assert _cache_bytes(None, {"supports_cache_ram": True}) == 8192 * 1024 * 1024


def test_the_load_mode_rule_matches_a_measured_ram_boundary_crossing():
    """Pin the (VRAM + RAM) switch point, and record that the measurement
    DISAGREES with it at the margin.

    gemma-4-31B Q4 on a G4 forced to 12 GiB free VRAM, host spill 10757 MiB
    (10.50 GiB), RAM swept. Same cell, same placement, only RAM differs
    (generation t/s):

        RAM 15.76 GiB   --fit on 22.62      --fit on --load-mode none 23.23
        RAM 10.44 GiB   --fit on TIMED OUT  --fit on --load-mode none 23.18
                                 (>2400 s)

    At 10.44 GiB the 10.50 GiB spill does not fit, so `A` fails by about 60 MiB
    and `_fits_without_paging` returns False, i.e. mmap. But mmap is the arm that
    DIED there, while --load-mode none ran at full speed. mmap's benefit is
    demand-paging a footprint the machine cannot hold; its cost is thrashing
    precisely when the machine cannot hold it, so at the margin it is the worse
    of the two options the rule chooses between.

    This test pins the current switch point so a change to it is deliberate. It
    is NOT a validation of that switch point: on this evidence the boundary is
    in the wrong place, and the honest fix is to move it down (or price the
    thrash) rather than to trust `A` alone. One cell, one host, so it is not
    enough to move the rule on -- but it is enough to stop calling the rule
    validated, which an earlier version of this docstring wrongly did after
    reading the first arm's timeout without waiting for the second.
    """
    mib = 1024 ** 2
    gib = 1024 ** 3
    backend = object.__new__(LlamaCppBackend)

    need = int(18.4 * gib) + 5520 * mib
    gpus = [(0, 12 * 1024)]

    def mode_at(ram_gib):
        fits = backend._fits_without_paging(
            need, gpus, avail_mib = int(ram_gib * 1024), headroom_mib = 2048,
        )
        return "none" if fits is True else ("mmap" if fits is False else None)

    assert mode_at(24) == "none"
    assert mode_at(15.76) == "none"
    # The switch. Measured: mmap timed out here and none ran at 23.18 t/s.
    assert mode_at(10.44) == "mmap"
