# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where llama.cpp put a model's weights, and load progress that sees weights headed for VRAM."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _extra_weight_file_bytes,
    _gguf_set_bytes,
    parse_weight_placement,
)

# Gemma 4 26B A4B Q8_0 on a 24 GB RTX 3090 under --fit on (trimmed from a real llama-server log).
_MOE_SPILL_LOG = """\
0.00.997.437 I common_params_fit_impl: filling dense-only layers back-to-front:
0.01.282.617 I common_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 3090): 31 layers,   5550 MiB used,  18347 MiB free
0.02.623.039 I common_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 3090): 31 layers ( 8 overflowing),  22815 MiB used,   1082 MiB free
0.02.623.041 I common_fit_params: successfully fit params to free device memory
0.03.897.106 I load_tensors: offloading output layer to GPU
0.03.897.109 I load_tensors: offloading 29 repeating layers to GPU
0.03.897.109 I load_tensors: offloaded 31/31 layers to GPU
0.03.897.113 I load_tensors:        CUDA0 model buffer size = 20709.15 MiB
0.03.897.113 I load_tensors:    CUDA_Host model buffer size =  5639.40 MiB
0.23.263.171 I llama_kv_cache:      CUDA0 KV buffer size =   160.00 MiB
""".splitlines()


def test_moe_spill_is_read_from_fit_not_from_the_layer_count():
    assert parse_weight_placement(_MOE_SPILL_LOG) == {
        "gpu_mib": 20709.2,
        "host_mib": 5639.4,
        "overflow_layers": 7,
        "all_experts_in_ram": False,
        "expert_cache_mib": None,
    }


def test_dense_model_fully_on_gpu():
    lines = [
        "0.00.365.476 I load_tensors: offloaded 41/41 layers to GPU",
        "0.00.365.484 I load_tensors:        CUDA0 model buffer size =  8161.75 MiB",
        "0.00.365.484 I load_tensors:    CUDA_Host model buffer size =   417.30 MiB",
    ]
    assert parse_weight_placement(lines) == {
        "gpu_mib": 8161.8,
        "host_mib": 417.3,
        "overflow_layers": None,
        "all_experts_in_ram": False,
        "expert_cache_mib": None,
    }


def test_dry_run_and_drafter_blocks_do_not_count():
    lines = [
        # --fit's no-alloc pass logs the same block with nothing allocated.
        "I load_tensors: offloaded 31/31 layers to GPU",
        "I load_tensors:        CUDA0 model buffer size =     0.00 MiB",
        "I load_tensors:    CUDA_Host model buffer size =     0.00 MiB",
        "I load_tensors: offloaded 31/31 layers to GPU",
        "I load_tensors:   CPU_Mapped model buffer size = 13986.81 MiB",
        "I load_tensors:        CUDA0 model buffer size = 13258.54 MiB",
        # A drafter logs its own, smaller block after the main model.
        "I load_tensors: offloaded 3/3 layers to GPU",
        "I load_tensors:        CUDA0 model buffer size =   600.00 MiB",
    ]
    # A drafter can also be DEEPER than its target (Gemma 4 E2B, 35 layers, for 26B A4B, 30): still the main model.
    deeper = lines + [
        "I load_tensors: offloaded 36/36 layers to GPU",
        "I load_tensors:        CUDA0 model buffer size =  2000.00 MiB",
    ]
    assert parse_weight_placement(deeper)["gpu_mib"] == 13258.5
    assert parse_weight_placement(lines) == {
        "gpu_mib": 13258.5,
        "host_mib": 13986.8,
        "overflow_layers": None,
        "all_experts_in_ram": False,
        "expert_cache_mib": None,
    }


def test_multi_gpu_extra_layer_overflows_to_the_next_gpu_not_ram():
    lines = [
        "I common_params_fit_impl: filling dense-only layers back-to-front:",
        "I common_params_fit_impl: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:",
        # Device 0 took one extra partial layer; it overflows into CUDA1, not into RAM.
        "I common_params_fit_impl: set ngl_per_device[0].(n_layer, n_part, overflow_type)=(20, 1, GATE), id_dense_start=1",
        "I common_params_fit_impl:   - CUDA0 (A): 20 layers ( 1 overflowing),  23000 MiB used,   900 MiB free",
        "I common_params_fit_impl:   - CUDA1 (B): 11 layers ( 4 overflowing),  15000 MiB used,   800 MiB free",
        "I load_tensors: offloaded 31/31 layers to GPU",
        "I load_tensors:        CUDA0 model buffer size = 20000.00 MiB",
        "I load_tensors:        CUDA1 model buffer size = 12000.00 MiB",
        "I load_tensors:          CPU model buffer size =  4000.00 MiB",
    ]
    out = parse_weight_placement(lines)
    # CUDA1's 4 partial slots end at the output layer, which spills nothing.
    assert (out["gpu_mib"], out["host_mib"], out["overflow_layers"]) == (32000.0, 4000.0, 3)


def test_moe_fit_that_stops_after_the_dense_only_fill_has_every_gpu_layer_spilled():
    """The dense-only lines carry no count, but every GPU layer there holds dense weights only."""
    lines = [
        "I common_params_fit_impl: filling dense-only layers back-to-front:",
        "I common_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 5060 Ti): 60 layers,  15000 MiB used,    900 MiB free",
        "I load_tensors: offloaded 60/95 layers to GPU",
        "I load_tensors:        CUDA0 model buffer size = 14000.00 MiB",
        "I load_tensors:    CUDA_Host model buffer size = 120000.00 MiB",
    ]
    # One device, so it is also the last: its output layer carries no experts.
    # 60 slots, 59 partial; the last partial slot is the output layer and the first layer stays full.
    assert parse_weight_placement(lines)["overflow_layers"] == 58
    dense = [line.replace("dense-only layers", "dense layers") for line in lines]
    assert parse_weight_placement(dense)["overflow_layers"] == 0, "a dense model's layers are full"


def test_moe_cache_auto_puts_every_expert_in_ram():
    lines = [
        "I common_params_fit_impl: moe cache auto: all routed experts in host RAM, cache 6144 MiB (margin 1024 MiB)",
        "I common_fit_params: successfully fit params to free device memory",
        "I load_tensors: offloaded 31/31 layers to GPU",
        "I load_tensors:        CUDA0 model buffer size =  3900.00 MiB",
        "I load_tensors:    CUDA_Host model buffer size = 22400.00 MiB",
    ]
    out = parse_weight_placement(lines)
    assert out["all_experts_in_ram"] is True
    assert out["expert_cache_mib"] == 6144.0
    assert (
        out["overflow_layers"] is None
    ), "no per-device line on this path; the backend supplies the MoE layer count"


def test_cpu_repack_buffers_count_as_ram_and_remote_buffers_parse():
    lines = [
        "I load_tensors: offloaded 0/49 layers to GPU",
        "I load_tensors:          AMX model buffer size =  9000.00 MiB",
        "I load_tensors:   CPU_Mapped model buffer size =   500.00 MiB",
        "I load_tensors: RPC0[192.168.1.2:50052] model buffer size =  1000.00 MiB",
    ]
    out = parse_weight_placement(lines)
    assert (out["gpu_mib"], out["host_mib"]) == (1000.0, 9500.0)


def test_no_buffer_lines_means_nothing_to_report():
    assert parse_weight_placement(["I srv load_model: loading model"]) is None


def test_a_cpu_only_build_reports_its_ram_placement():
    # llama.cpp prints "offloaded N/M layers" only when the build supports GPU offload.
    lines = [
        "load_tensors:   CPU_Mapped model buffer size = 21000.0 MiB",
        "load_tensors:          CPU model buffer size =     0.6 MiB",
        "llama_context: n_ctx = 4096",
        # A drafter's buffers after other lines are its own block, not the main model's.
        "load_tensors:   CPU_Mapped model buffer size =   500.0 MiB",
    ]
    placement = parse_weight_placement(lines)
    assert placement is not None
    assert (placement["gpu_mib"], placement["host_mib"]) == (0.0, 21000.6)
    assert placement["overflow_layers"] is None


def test_projector_and_drafter_on_the_command_line_count_toward_the_load(tmp_path):
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"x" * 1000)
    drafter = tmp_path / "draft.gguf"
    drafter.write_bytes(b"x" * 300)
    argv = [
        "llama-server",
        "-m",
        "model.gguf",
        "--mmproj",
        str(projector),
        "-md",
        str(drafter),
        "--port",
        "1",
    ]
    assert _extra_weight_file_bytes(argv) == 1300
    assert _extra_weight_file_bytes(None) == 0
    assert (
        _extra_weight_file_bytes(["llama-server", "--mmproj", str(tmp_path / "missing.gguf")]) == 0
    )


def test_a_split_drafter_counts_every_shard(tmp_path):
    # llama-server opens a split drafter's siblings from the named first shard.
    for index in (1, 2, 3):
        (tmp_path / f"Model-Q4_K_M-MTP-{index:05d}-of-00003.gguf").write_bytes(b"x" * 100)
    # The main model's set in the same folder, whose prefix the drafter's starts with, is not the drafter's.
    for index in (1, 2):
        (tmp_path / f"Model-Q4_K_M-{index:05d}-of-00002.gguf").write_bytes(b"x" * 7)
    # Nor is a stale set under the same prefix with another shard count.
    for index in (1, 2, 3, 4):
        (tmp_path / f"Model-Q4_K_M-MTP-{index:05d}-of-00004.gguf").write_bytes(b"x" * 1000)
    argv = ["llama-server", "-md", str(tmp_path / "Model-Q4_K_M-MTP-00001-of-00003.gguf")]
    assert _extra_weight_file_bytes(argv) == 300
    assert _gguf_set_bytes(tmp_path / "Model-Q4_K_M-00001-of-00002.gguf") == 14


def test_companions_picked_by_the_launch_env_count_too(tmp_path):
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"x" * 1000)
    drafter = tmp_path / "draft.gguf"
    drafter.write_bytes(b"x" * 300)
    argv = ["llama-server", "-m", "model.gguf"]
    env = {"LLAMA_ARG_MMPROJ": str(projector), "LLAMA_ARG_SPEC_DRAFT_MODEL": str(drafter)}
    assert _extra_weight_file_bytes(argv, env) == 1300
    # Pre-b8955 drafter spelling.
    assert _extra_weight_file_bytes(argv, {"LLAMA_ARG_MODEL_DRAFT": str(drafter)}) == 300
    # argv overrides the env, as in arg.cpp; a Hugging Face repo is not a local file.
    other = tmp_path / "mmproj-BF16.gguf"
    other.write_bytes(b"x" * 50)
    assert _extra_weight_file_bytes(argv + ["--mmproj", str(other)], env) == 350
    assert _extra_weight_file_bytes(argv, {"LLAMA_ARG_SPEC_DRAFT_HF_REPO": "org/draft"}) == 0


def test_adapters_on_the_command_line_count_toward_the_load(tmp_path, monkeypatch):
    lora = tmp_path / "lora.gguf"
    lora.write_bytes(b"x" * 400)
    (tmp_path / "vector.gguf").write_bytes(b"x" * 20)
    # Relative: upstream splits FNAME:SCALE on every ':', so a drive-lettered path is its own error.
    monkeypatch.chdir(tmp_path)
    argv = ["llama-server", "-m", "model.gguf", "--lora", str(lora)]
    assert _extra_weight_file_bytes(argv + ["--control-vector-scaled", "vector.gguf:0.5"]) == 420


def test_load_progress_sizes_companions_from_the_env_its_process_got(tmp_path, monkeypatch):
    inst = _fake_load(tmp_path, 1000)
    projector = tmp_path / "mmproj-F16.gguf"
    projector.write_bytes(b"x" * 1000)
    monkeypatch.setattr(LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: None))
    monkeypatch.setattr(LlamaCppBackend, "_read_load_bytes", staticmethod(lambda pid: 500))
    inst._process_env = (inst._process, {"LLAMA_ARG_MMPROJ": str(projector)})
    assert inst.load_progress()["bytes_total"] == 2000
    # An env recorded for an earlier process does not describe this one.
    inst._process_env = (object(), {"LLAMA_ARG_MMPROJ": str(projector)})
    assert inst.load_progress()["bytes_total"] == 1000


@pytest.mark.skipif(not Path("/proc").exists(), reason = "live /proc test is Linux-only")
def test_load_progress_follows_bytes_read_when_rss_stays_flat(tmp_path):
    """A server that streams its weights to VRAM reads every byte but keeps little resident:
    progress must follow the reads (it froze at the RSS share before)."""
    weights = tmp_path / "model.gguf"
    weights.write_bytes(os.urandom(64 * 1024 * 1024))
    script = tmp_path / "reader.py"
    script.write_text(
        "import sys, time\n"
        "with open(sys.argv[1], 'rb') as f:\n"
        "    while f.read(1 << 20):\n"
        "        pass\n"
        "print('read', flush=True)\n"
        "time.sleep(10)\n"
    )
    proc = subprocess.Popen([sys.executable, str(script), str(weights)], stdout = subprocess.PIPE)
    try:
        assert proc.stdout.readline().strip() == b"read"
        inst = LlamaCppBackend.__new__(LlamaCppBackend)
        inst._process = proc
        inst._gguf_path = str(weights)
        inst._healthy = False
        rss = LlamaCppBackend._read_rss_bytes(proc.pid)
        assert rss is not None and rss < 64 * 1024 * 1024, "the reader never holds the file"
        out = inst.load_progress()
        assert out is not None
        assert out["bytes_total"] == 64 * 1024 * 1024
        assert out["fraction"] == 1.0, out
    finally:
        proc.kill()
        proc.wait(timeout = 5)


def _fake_load(tmp_path, size):
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"\0" * size)
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._process = type("P", (), {"pid": 4242, "args": []})()
    inst._gguf_path = str(weights)
    inst._healthy = False
    return inst


def test_windows_load_progress_follows_bytes_read_alone(tmp_path, monkeypatch):
    """A Windows working set takes in the buffers llama.cpp allocates before reading a byte (DevLab:
    9.1 of 9.0 GB a second into the load, reads still at 0.01 GB), so only the read counter counts."""
    inst = _fake_load(tmp_path, 1000)
    monkeypatch.setattr(LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: None))
    monkeypatch.setattr(LlamaCppBackend, "_read_load_bytes", staticmethod(lambda pid: 250))
    assert inst.load_progress()["fraction"] == 0.25


def test_linux_rss_term_leaves_out_the_pinned_weight_buffer():
    """With --load-mode none, CUDA pins the host buffer for the weights kept in RAM before a byte is
    read (3090: VmRSS 6.39 GB, 5.99 GB of it RssShmem, 0.16 GB read), which put the bar at 23% and held
    it there until the reads passed it. Only the file-backed share is what a mapped load pages in."""
    status = "VmRSS:\t6390000 kB\nRssAnon:\t250000 kB\nRssFile:\t150000 kB\nRssShmem:\t5990000 kB\n"
    with patch("builtins.open", side_effect = lambda *a, **kw: io.StringIO(status)):
        assert LlamaCppBackend._read_rss_bytes(4242) == 150000 * 1024
    # A kernel without the split (before 4.5) still gives VmRSS.
    with patch("builtins.open", side_effect = lambda *a, **kw: io.StringIO("VmRSS:\t2048 kB\n")):
        assert LlamaCppBackend._read_rss_bytes(4242) == 2048 * 1024


def test_no_load_signal_reports_no_figure(tmp_path, monkeypatch):
    inst = _fake_load(tmp_path, 1000)
    monkeypatch.setattr(LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: None))
    monkeypatch.setattr(LlamaCppBackend, "_read_load_bytes", staticmethod(lambda pid: None))
    assert inst.load_progress() is None


def test_each_os_reads_its_own_load_counter(monkeypatch):
    """Windows: psutil read bytes. macOS: the larger of disk bytes read (a cold memory-mapped Metal
    load) and physical footprint (a copy into Metal or CPU buffers); RSS stays at ~20% of a Metal load."""
    import core.inference.llama_cpp as llama_cpp

    calls = []

    class _Process:
        def __init__(self, pid):
            calls.append(pid)

        def io_counters(self):
            return type("C", (), {"read_bytes": 123})()

    monkeypatch.setitem(sys.modules, "psutil", type("M", (), {"Process": _Process}))
    monkeypatch.setattr(llama_cpp.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(LlamaCppBackend, "_darwin_rusage", staticmethod(lambda pid: (700, 300)))
    monkeypatch.setattr(llama_cpp.sys, "platform", "win32")
    assert LlamaCppBackend._read_load_bytes(7) == 123
    monkeypatch.setattr(llama_cpp.sys, "platform", "darwin")
    assert LlamaCppBackend._read_load_bytes(7) == 700
    monkeypatch.setattr(LlamaCppBackend, "_darwin_rusage", staticmethod(lambda pid: None))
    assert LlamaCppBackend._read_load_bytes(7) is None
    monkeypatch.setattr(llama_cpp.sys, "platform", "freebsd14")
    assert LlamaCppBackend._read_load_bytes(7) is None
    assert calls == [7]


def test_load_progress_follows_a_real_process_on_this_os(tmp_path):
    """No mocks: a child reads a weight file into memory, and load_progress must see all of it.
    Linux rchar and Windows read_bytes count the read; on macOS a page-cached file reads nothing from
    disk, so the physical footprint of the held bytes carries it. Before, macOS and Windows got None."""
    if not (os.path.isdir("/proc") or sys.platform in ("darwin", "win32")):
        pytest.skip("no load counter on this platform")
    size = 64 * 1024 * 1024
    weights = tmp_path / "model.gguf"
    weights.write_bytes(os.urandom(size))
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "data = open(sys.argv[1], 'rb').read()\n"
            "print(len(data), flush = True)\n"
            "time.sleep(60)\n",
            str(weights),
        ],
        stdout = subprocess.PIPE,
        text = True,
    )
    try:
        assert int(child.stdout.readline()) == size
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = type("P", (), {"pid": child.pid, "args": None})()
        backend._gguf_path = str(weights)
        backend._healthy = False
        progress = backend.load_progress()
        assert progress is not None
        assert progress["phase"] == "mmap"
        assert progress["bytes_total"] == size
        assert progress["fraction"] >= 0.99
    finally:
        child.kill()
        child.wait(10)


def test_an_mmap_span_over_gpu_tensors_is_capped_at_the_file_minus_the_gpu_side():
    """With experts spilled under mmap, CPU_Mapped logs the span from the first to the last host tensor,
    which covers the GPU tensors in between: 25 GiB "in RAM" for a 25.3 GiB file with 21 GiB on the GPU."""
    lines = [
        "load_tensors: offloaded 49/49 layers to GPU",
        "load_tensors:        CUDA0 model buffer size = 21504.00 MiB",
        "load_tensors:   CPU_Mapped model buffer size = 25600.00 MiB",
    ]
    model_bytes = 26419 * 1024 * 1024
    out = parse_weight_placement(lines, model_bytes)
    assert out["gpu_mib"] == 21504.0
    assert out["host_mib"] == 26419 - 21504.0
    # Without the file size, and for a buffer that is not a mapping, the logged size stands.
    assert parse_weight_placement(lines)["host_mib"] == 25600.0
    host = [line.replace("CPU_Mapped", "CUDA_Host") for line in lines]
    assert parse_weight_placement(host, model_bytes)["host_mib"] == 25600.0
