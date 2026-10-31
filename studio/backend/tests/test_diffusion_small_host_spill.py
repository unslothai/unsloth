# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small-host int8 spill: the denoiser's int8 weights live in a memory-mapped scratch file instead of host RAM.
Sized to a Colab standard T4 (12.7 GB host RAM) loading FLUX.2-klein-9B; CPU only."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import core.inference.diffusion_small_host as sh

KLEIN9 = {
    "transformer": sh.StoredComponent("transformer", 17317, "bfloat16", 0, "diffusers"),
    "text_encoder": sh.StoredComponent(
        "text_encoder", 15623, "bfloat16", 0, "transformers", embed_mib = 1187
    ),
    "vae": sh.StoredComponent("vae", 161, "bfloat16", 0, "diffusers"),
}
COLAB_TOTAL = 12977
COLAB_AVAILABLE = 11161
ROOMY_DISK = 200 * 1024

_LINUX = sys.platform.startswith("linux")


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        sh.SMALL_HOST_ENV,
        sh.HOST_RAM_CHECK_ENV,
        sh.SPILL_ENV,
        sh.SPILL_DIR_ENV,
    ):
        monkeypatch.delenv(name, raising = False)


def _decide(
    comps = KLEIN9,
    total = COLAB_TOTAL,
    available = COLAB_AVAILABLE,
    **kw,
):
    return sh.decide_small_host(
        comps,
        torch.float16,
        device = "cuda",
        host_total_mib = total,
        host_available_mib = available,
        **kw,
    )


def test_klein9_on_colab_refuses_without_spill():
    d = _decide()
    assert d.engaged and not d.int8_spill
    assert d.refuse and "11.5 GB" in d.refuse and "10.9 GB" in d.refuse


def test_klein9_on_colab_spills_and_loads():
    d = _decide(spill_free_mib = ROOMY_DISK)
    assert d.engaged and d.int8_spill and d.refuse is None
    assert d.storage_dtypes == {"transformer": "bfloat16", "text_encoder": "bfloat16"}
    # int8 DiT leaves the host estimate; the encoder's converted embedding joins it
    assert d.spill_route_host_mib == 1187 + sh._SPILL_TRANSIENT_MIB
    assert d.route_host_mib == 17317 // 2


def test_spill_kill_switch_keeps_the_refusal(monkeypatch):
    monkeypatch.setenv(sh.SPILL_ENV, "0")
    d = _decide(spill_free_mib = ROOMY_DISK)
    assert not d.int8_spill and d.refuse == _decide().refuse


def test_spill_needs_disk_room():
    d = _decide(spill_free_mib = 17317 // 2 + sh._SPILL_DISK_MARGIN_MIB - 1)
    assert not d.int8_spill and d.refuse == _decide().refuse


def test_spill_route_still_refuses_a_host_too_small_for_it():
    d = _decide(available = 3000, spill_free_mib = ROOMY_DISK)
    assert d.int8_spill and d.refuse and "Not enough system RAM" in d.refuse


@pytest.mark.parametrize("available", [11800, 14000, 20000, 30000, 40000])
def test_hosts_that_load_today_keep_their_exact_decision(available):
    base = _decide(available = available)
    head = _decide(available = available, spill_free_mib = ROOMY_DISK)
    assert base.refuse is None and not head.int8_spill
    assert (base.engaged, base.reason, base.storage_dtypes, base.refuse, base.route_host_mib) == (
        head.engaged,
        head.reason,
        head.storage_dtypes,
        head.refuse,
        head.route_host_mib,
    )


def test_spill_forced_on_a_host_that_fits(monkeypatch):
    monkeypatch.setenv(sh.SPILL_ENV, "1")
    d = _decide(available = 30000, spill_free_mib = ROOMY_DISK)
    assert d.engaged and d.int8_spill and d.refuse is None


def test_no_denoiser_no_spill():
    comps = {"text_encoder": KLEIN9["text_encoder"]}
    d = _decide(comps, available = 2000, spill_free_mib = ROOMY_DISK)
    assert not d.int8_spill


def test_check_disabled_still_spills(monkeypatch):
    # the in-RAM route was OOM-killed with the check off: spill whenever it does not fit
    monkeypatch.setenv(sh.HOST_RAM_CHECK_ENV, "0")
    d = _decide(spill_free_mib = ROOMY_DISK)
    assert d.int8_spill and d.refuse is None


def _write_st(path: Path, tensors: dict) -> None:
    header, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = 2
        for s in shape:
            n *= s
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * off)


def test_stored_components_reads_the_encoder_embedding(tmp_path):
    te = tmp_path / "text_encoder"
    te.mkdir()
    _write_st(
        te / "model.safetensors",
        {
            "model.embed_tokens.weight": ("BF16", (1024, 512)),
            "model.layers.0.mlp.up_proj.weight": ("BF16", (512, 512)),
        },
    )
    dit = tmp_path / "transformer"
    dit.mkdir()
    _write_st(
        dit / "diffusion_pytorch_model.safetensors", {"x_embedder.weight": ("BF16", (1024, 1024))}
    )
    comps = sh.stored_components(tmp_path)
    assert comps["text_encoder"].embed_mib == 1
    assert comps["transformer"].embed_mib == 0
    assert sh._safetensors_bytes_by_dtype(te / "model.safetensors") == {
        "bfloat16": 1024 * 512 * 2 + 512 * 512 * 2
    }


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(2048, 2048)
        self.b = torch.nn.Linear(2048, 4096, bias = False)
        self.small = torch.nn.Linear(64, 64)
        self.norm = torch.nn.LayerNorm(2048)

    def forward(self, x):
        return self.b(self.norm(self.a(x)))


def _tiny(seed = 0):
    torch.manual_seed(seed)
    return _Tiny().to(torch.bfloat16)


@pytest.mark.skipif(not _LINUX, reason = "spill is Linux only")
def test_spilled_int8_is_bit_identical_and_file_backed(tmp_path):
    ram, spilled = _tiny(), _tiny()
    s_ram = sh.quantize_int8_weight_(ram, compute_dtype = torch.float16, work_device = "cpu")
    s_sp = sh.quantize_int8_weight_(
        spilled, compute_dtype = torch.float16, work_device = "cpu", spill_dir = tmp_path
    )
    assert s_ram["linears"] == s_sp["linears"] == 2
    assert s_sp["spilled_bytes"] == s_sp["int8_bytes"] and s_ram.get("spilled_bytes", 0) == 0
    for name in ("a", "b"):
        r, s = getattr(ram, name), getattr(spilled, name)
        assert torch.equal(r.qweight, s.qweight) and torch.equal(r.scale, s.scale)
        assert s.qweight.untyped_storage().filename is not None
        assert r.qweight.untyped_storage().filename is None
    assert not list(tmp_path.iterdir())
    x = torch.randn(3, 2048, dtype = torch.float16)
    with torch.no_grad():
        assert torch.equal(ram(x), spilled(x))


@pytest.mark.skipif(not _LINUX, reason = "spill is Linux only")
def test_spill_capacity_reads_free_disk_and_refuses_ram_filesystems(tmp_path, monkeypatch):
    free = sh.spill_capacity_mib(tmp_path)
    assert free is not None and free > 0
    if Path("/dev/shm").is_dir() and sh._fs_type(Path("/dev/shm")) == "tmpfs":
        assert sh.spill_capacity_mib(Path("/dev/shm")) is None
    monkeypatch.setenv(sh.SPILL_ENV, "0")
    assert sh.spill_capacity_mib(tmp_path) is None


def test_spill_capacity_is_none_off_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(sh.sys, "platform", "win32")
    assert sh.spill_capacity_mib(tmp_path) is None


def test_spill_dir_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(sh.SPILL_DIR_ENV, str(tmp_path / "x"))
    assert sh.spill_dir() == tmp_path / "x"


@pytest.mark.skipif(not _LINUX, reason = "spill is Linux only")
def test_stale_spill_files_of_dead_processes_are_swept(tmp_path):
    dead = tmp_path / f"{sh._SPILL_PREFIX}999999999-abc.bin"
    dead.write_bytes(b"x")
    mine = tmp_path / f"{sh._SPILL_PREFIX}{os.getpid()}-abc.bin"
    mine.write_bytes(b"x")
    sh._sweep_stale_spills(tmp_path)
    assert not dead.exists() and mine.exists()


_RSS_CHILD = textwrap.dedent(
    """
    import sys, torch
    sys.path.insert(0, {backend!r})
    import core.inference.diffusion_small_host as sh

    def anon():
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) // 1024
    import safetensors.torch as st
    torch.manual_seed(0)
    def build():
        return torch.nn.Sequential(*[torch.nn.Linear(4096, 4096, bias=False) for _ in range(16)])
    src = {tmp!r} + "/src.safetensors"
    st.save_file({{k: v.to(torch.bfloat16) for k, v in build().state_dict().items()}}, src)
    with torch.device("meta"):
        m = build()
    # memory-mapped bf16 source, as the route loads the stored shards
    m.load_state_dict(st.load_file(src), assign=True)
    before = anon()
    kw = dict(spill_dir={spill!r}) if {spill!r} else dict()
    sh.quantize_int8_weight_(m, compute_dtype=torch.float16, work_device="cpu", **kw)
    print("DELTA", anon() - before)
    """
)


@pytest.mark.skipif(not _LINUX, reason = "RssAnon is Linux only")
@pytest.mark.parametrize("spill", [False, True])
def test_int8_host_residency_bound(tmp_path, spill):
    # 16 x 4096^2 int8 = 256 MiB: the in-RAM route keeps all of it anonymous, the spill keeps it file-backed
    backend = str(Path(sh.__file__).resolve().parents[2])
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    code = _RSS_CHILD.format(
        backend = backend, tmp = str(tmp_path), spill = str(spill_dir) if spill else ""
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout = 300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    delta = int(out.stdout.split("DELTA")[-1].split()[0])
    if spill:
        assert delta < 64, delta
    else:
        assert delta > 200, delta


@pytest.mark.skipif(not _LINUX, reason = "spill is Linux only")
def test_spill_capacity_creates_nothing(tmp_path):
    target = tmp_path / "a" / "b"
    assert sh.spill_capacity_mib(target) is not None
    assert not (tmp_path / "a").exists()
