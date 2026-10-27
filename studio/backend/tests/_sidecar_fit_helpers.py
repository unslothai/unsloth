# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared launch harness for the DSpark/DFlash sidecar reserve tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as placement  # noqa: E402
import test_metal_explicit_context_guard as metal_guard  # noqa: E402
from core.inference import llama_cpp as lc  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB
CAPS = {
    "supports_dspark": True,
    "supports_dflash": True,
    "supports_fit_target": True,
    "supports_fit_ctx": True,
    "supports_kv_unified": True,
    "spec_draft_n_max_flag": "--spec-draft-n-max",
}
# llama.cpp's default 1024 MiB margin plus the 8 GiB sidecar.
FLOOR = 1024 + 8 * 1024
ONE = ((0, 24_000, 24_000),)
TWO = ((0, 24_000, 24_000), (1, 24_000, 24_000))
MANUAL_AUTO = {"gpu_memory_mode": "manual", "gpu_layers": -1}
DRAFT_KV = 768 * MIB
NATIVE_MARKER = b"measuring the extra model next to an allocation-free target context"
UNSET = object()


def last(cmd, name):
    """Last-wins value of ``name``, in either the spaced or the ``=`` spelling."""
    value = None
    for i, tok in enumerate(cmd):
        if tok == name and i + 1 < len(cmd):
            value = cmd[i + 1]
        elif tok.startswith(name + "="):
            value = tok.split("=", 1)[1]
    return value


def margins(cmd):
    raw = last(cmd, "--fit-target") or "1024"
    return [int(float(e)) for e in str(raw).split(",")]


def margin(cmd):
    return max(margins(cmd))


def fit_targets(cmd):
    return [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "--fit-target"]


def sidecar(tmp_path, kind = "dspark"):
    path = tmp_path / f"{kind}-model-Q8_0.gguf"
    path.write_bytes(b"draft")
    return str(path)


def cuda_backend(
    tmp_path,
    memory = ONE,
    vulkan = False,
    draft_key = "dspark",
):
    backend, gguf = placement._backend(tmp_path, vulkan = vulkan, memory = list(memory))
    backend._select_gpus = lambda *a, **k: (None, True)
    backend.probe_server_capabilities = lambda _binary = None: dict(CAPS)
    backend._get_gguf_size_bytes = lambda path: 8 * GIB if draft_key in str(path) else 1024
    # A discrete-GPU host on every OS, not an Apple Silicon runner's unified budget.
    backend._apple_metal_memory_budget_bytes = lambda: 0
    return backend, gguf


def cards(
    tmp_path,
    memory = TWO,
    vulkan = False,
):
    backend, gguf = cuda_backend(tmp_path, memory = memory, vulkan = vulkan)
    backend._arch_gate_survivors = lambda binary = None: []
    return backend, gguf


def dspark(backend, gguf, tmp_path, **kw):
    return placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = sidecar(tmp_path), **kw
    )["cmd"]


def per_token_estimate(backend, per_token):
    backend._estimate_mtp_overhead_bytes = lambda n_ctx, **kw: (
        8 * GIB + max(0, int(n_ctx)) * per_token
    )


def slot_priced(backend):
    backend._estimate_mtp_overhead_bytes = lambda n_ctx, n_parallel = 1, **kw: (
        8 * GIB + max(1, int(n_parallel or 1)) * GIB
    )
    return backend


def three_block_drafter(backend):
    read = backend._draft_backend_for

    def three_blocks(path):
        db = read(path)
        db._n_layers = 3
        return db

    backend._draft_backend_for = three_blocks


def elf(path, payload = b""):
    path.write_bytes(b"\x7fELF" + payload + b"\x00")
    return path


def info_messages(monkeypatch):
    seen = []
    real_info = lc.logger.info

    def info(msg, *args, **kwargs):
        try:
            seen.append(str(msg) % args if args else str(msg))
        except TypeError:
            seen.append(str(msg))
        return real_info(msg, *args, **kwargs)

    monkeypatch.setattr(lc.logger, "info", info)
    return seen


class _Spawns(list):
    loaded = None


def spawns(
    backend,
    gguf,
    *,
    healthy,
    crash = lambda i, cmd: None,
    stdout = lambda i, cmd: (),
    late_exit = False,
    **intent,
):
    """Every llama-server argv the load spawns. ``crash(i, cmd)`` is spawn i's exit code
    (None = running), ``stdout`` its output, ``healthy`` its health answer. ``late_exit``
    makes the first poll see the child running: it exits between the startup retry's check
    and the outer --flash-attn off check."""
    cmds = _Spawns()

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return placement._REAL_POPEN(cmd, **kwargs)
        cmds.append(list(cmd))
        rc = crash(len(cmds) - 1, list(cmd))
        polls = [0]

        def poll(self):
            polls[0] += 1
            return None if late_exit and polls[0] == 1 else rc

        return type(
            "Process",
            (),
            {
                "pid": 2**31 - 1,
                "stdout": (),
                "returncode": rc,
                "poll": poll,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: rc or 0,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None, **_kw):
        i = len(cmds) - 1
        backend._stdout_lines = list(stdout(i, cmds[-1]))
        return healthy(i, cmds[-1])

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        try:
            cmds.loaded = backend.load_model(
                GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", **intent)
            )
        except Exception:
            pass
    return cmds


class _ReserveReport:
    """Whether the user is told the sidecar is reserved: planning's verdict, then the
    spawn's report off the argv it launched (the last outcome a retry logged)."""

    def __init__(self):
        self.planned = None
        self.spawned: list = []

    def __getitem__(self, key):
        assert key == "fit_sized"
        if self.planned is not True:
            return self.planned
        return self.spawned[-1] if self.spawned else True


def spy_fit_sized(backend):
    seen = _ReserveReport()
    build = backend._build_speculative_flags
    report = backend._log_sidecar_fit_reserve

    def spy(**kw):
        seen.planned = kw.get("dspark_fit_sized")
        return build(**kw)

    def log(run_cmd, env, kind, path, reserved_mib, logged):
        before = len(logged)
        report(run_cmd, env, kind, path, reserved_mib, logged)
        if len(logged) > before:
            seen.spawned.append(logged[-1][1])

    backend._build_speculative_flags = spy
    backend._log_sidecar_fit_reserve = log
    return seen


def host_kv_delta(
    tmp_path,
    monkeypatch,
    *,
    memory = ONE,
    vulkan = False,
    listed = UNSET,
    apu = UNSET,
    extras = (),
    **kw,
):
    """MiB the -nkvo launch's largest margin is below the KV-on-card launch's.
    ``listed`` is what --list-devices reports (None: no answer); ``apu`` the card whose
    memory is the host's (None: none)."""
    if listed is not UNSET:
        monkeypatch.setattr(
            LlamaCppBackend,
            "_enumerated_gpu_devices",
            classmethod(
                lambda cls, binary = None, env = None: None if listed is None else list(listed)
            ),
        )
    out = []
    for name, more in (("on", []), ("off", ["-nkvo"])):
        d = tmp_path / name
        d.mkdir()
        backend, gguf = cards(d, memory = memory, vulkan = vulkan)
        backend._mtp_draft_kv_bytes = lambda *a, **k: DRAFT_KV
        if apu is not UNSET:
            backend._amd_apu_wants_unified_memory = lambda gpu_indices = None: (
                apu is not None and (gpu_indices is None or apu in gpu_indices)
            )
        backend._integrated_cuda_unified_memory = lambda gpu_indices = None: False
        out.append(margin(dspark(backend, gguf, d, extra_args = [*extras, *more], **kw)))
    return out[0] - out[1]


def metal_cmd(
    tmp_path,
    monkeypatch,
    mode,
    *,
    budget_gib,
    wired_gib = None,
    sub = None,
    n_ctx = 0,
    **kw,
):
    """A real-fit Metal load with a DSpark sidecar; every GGUF here reads 4 GiB, the
    sidecar too. Undoes ``monkeypatch`` so a test can load twice."""
    sub = tmp_path / (sub or mode)
    sub.mkdir()
    path = sidecar(sub)
    real_init = GgufLoadIntent.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("dspark_draft_path", path)
        kwargs.setdefault("speculative_type", mode)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(GgufLoadIntent, "__init__", init)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: dict(CAPS)),
    )
    backend = LlamaCppBackend()
    wired_gib = budget_gib if wired_gib is None else wired_gib
    cmd = metal_guard._launch(
        sub,
        monkeypatch,
        n_ctx = n_ctx,
        backend = backend,
        real_fit = True,
        budget_bytes = int(budget_gib * GIB),
        wired_bytes = int(wired_gib * GIB),
        weights_bytes = 4 * GIB,
        **kw,
    )["cmd"]
    monkeypatch.undo()
    return cmd, backend, path
