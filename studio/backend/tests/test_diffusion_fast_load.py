# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image-load cold-start I/O: the page-cache prefetch and the pinned-ring upload (diffusion_fast_load)."""

import json
import threading

import pytest

from core.inference import diffusion_fast_load as fl


@pytest.fixture(autouse = True)
def _solid_state_disk(monkeypatch):
    # CI disks can report rotational=1; the rotational skip has its own test below.
    monkeypatch.setattr(fl, "_on_rotational_disk", lambda path: False)


def _join_prefetch(handle, timeout = 30.0):
    assert handle is not None
    assert handle.join(timeout), "prefetch threads never finished"
    assert not [t for t in threading.enumerate() if t.name.startswith(fl._PREFETCH_THREAD_PREFIX)]


def test_kill_switches(monkeypatch):
    for env, probe in (
        (fl.PREFETCH_ENV, fl.prefetch_enabled),
        (fl.FAST_UPLOAD_ENV, fl.fast_upload_enabled),
    ):
        monkeypatch.delenv(env, raising = False)
        assert probe()
        for off in ("0", "off", "false", "no", " OFF "):
            monkeypatch.setenv(env, off)
            assert not probe()


def test_slices_cover_the_file_contiguously_on_mib_boundaries():
    for size, parts in ((1, 8), ((5 << 20) + 3, 8), (64 << 20, 8), ((100 << 20) + 1, 3)):
        got = fl._slices(size, parts)
        assert got[0][0] == 0 and got[-1][1] == size
        assert all(a[1] == b[0] for a, b in zip(got, got[1:]))
        assert all(start % (1 << 20) == 0 for start, _ in got)
        assert len(got) <= parts


def test_prefetch_reads_every_uncached_byte_in_order(tmp_path, monkeypatch):
    files = []
    for name, size in (("dit.pt", (3 << 20) + 17), ("te.safetensors", 5 << 20)):
        path = tmp_path / name
        path.write_bytes(b"z" * size)
        files.append(str(path))
    read: list = []
    real_open = open

    class _Spy:
        def __init__(self, path, *args, **kwargs):
            self._handle = real_open(path, *args, **kwargs)
            self._path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

        def seek(self, at):
            self._at = at
            return self._handle.seek(at)

        def readinto(self, view):
            got = self._handle.readinto(view)
            read.append((self._path, self._at, got))
            self._at += got
            return got

    monkeypatch.setattr(fl, "_uncached_bytes", lambda path: __import__("os").path.getsize(path))
    monkeypatch.setattr(fl, "_available_host_mib", lambda: 1 << 20)
    monkeypatch.setattr("builtins.open", _Spy)
    handle = fl.start_prefetch(
        files + [files[0], str(tmp_path / "missing")], min_bytes = 0, threads = 4
    )
    _join_prefetch(handle)
    assert handle.files == files
    for path in files:
        size = __import__("os").path.getsize(path)
        covered = sorted((at, at + got) for p, at, got in read if p == path)
        assert covered[0][0] == 0 and covered[-1][1] == size
        assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))


def test_prefetch_declines_cached_small_or_ram_tight_loads(tmp_path, monkeypatch):
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"y" * (2 << 20))
    monkeypatch.setattr(fl, "_available_host_mib", lambda: 1 << 20)
    monkeypatch.setattr(fl, "_uncached_bytes", lambda p: 0)
    assert fl.start_prefetch([str(path)], min_bytes = 0) is None
    monkeypatch.setattr(fl, "_uncached_bytes", lambda p: 2 << 20)
    assert fl.start_prefetch([str(path)]) is None
    monkeypatch.setattr(fl, "_available_host_mib", lambda: 3)  # 2 MiB needed > half of 3 MiB
    assert fl.start_prefetch([str(path)], min_bytes = 0) is None
    monkeypatch.setattr(fl, "_available_host_mib", lambda: None)
    assert fl.start_prefetch([str(path)], min_bytes = 0) is None
    monkeypatch.setattr(fl, "_available_host_mib", lambda: 1 << 20)
    monkeypatch.setenv(fl.PREFETCH_ENV, "0")
    assert fl.start_prefetch([str(path)], min_bytes = 0) is None
    monkeypatch.delenv(fl.PREFETCH_ENV)
    _join_prefetch(fl.start_prefetch([str(path)], min_bytes = 0))
    monkeypatch.setattr(
        fl, "_on_rotational_disk", lambda p: True
    )  # parallel slices would seek-thrash a disk
    assert fl.start_prefetch([str(path)], min_bytes = 0) is None


def _fake_snapshot(tmp_path):
    snap = tmp_path / "snap"
    index = {
        "_class_name": "FluxPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "image_encoder": [None, None],
    }
    snap.mkdir()
    (snap / "model_index.json").write_text(json.dumps(index))
    layout = {
        "text_encoder": ["model.safetensors", "model.fp16.safetensors"],
        "text_encoder_2": [
            "model-00002-of-00002.safetensors",
            "model-00001-of-00002.safetensors",
            "index.json",
        ],
        "transformer": [
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
        ],
        "vae": ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"],
        "tokenizer": ["vocab.json"],
    }
    for folder, names in layout.items():
        (snap / folder).mkdir()
        for name in names:
            (snap / folder / name).write_bytes(b"w")
    return snap


def test_component_files_are_the_weights_from_pretrained_reads_text_encoders_first(tmp_path):
    snap = _fake_snapshot(tmp_path)
    rel = lambda paths: [p[len(str(snap)) + 1 :] for p in paths]  # noqa: E731
    assert rel(fl.pipeline_component_files(str(snap), skip_denoiser = False)) == [
        "text_encoder/model.safetensors",
        "text_encoder_2/model-00001-of-00002.safetensors",
        "text_encoder_2/model-00002-of-00002.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model.safetensors",
    ]
    assert rel(
        fl.pipeline_component_files(
            str(snap), skip_denoiser = True, skip_components = {"text_encoder_2"}
        )
    ) == [
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    assert fl.pipeline_component_files(None, skip_denoiser = False) == []
    assert fl.pipeline_component_files(str(tmp_path), skip_denoiser = False) == []


def test_load_prefetch_puts_the_seeded_checkpoint_first_and_skips_the_dense_denoiser(
    tmp_path, monkeypatch
):
    snap = _fake_snapshot(tmp_path)
    ckpt = tmp_path / "FLUX.1-schnell-INT8.pt"
    ckpt.write_bytes(b"q")
    seen = {}

    class _Source:
        kind = "repo"

    import core.inference.diffusion_denoiser_prequant as dp
    import core.inference.diffusion_prequant as pq

    monkeypatch.setattr(
        dp,
        "denoiser_prequant_source",
        lambda fam, scheme, **kw: _Source() if scheme == "int8" else None,
    )
    monkeypatch.setattr(pq, "cached_checkpoint_path", lambda source, cache_dir = None: str(ckpt))
    monkeypatch.setattr(
        fl, "start_prefetch", lambda paths, logger = None: seen.setdefault("paths", list(paths))
    )

    fl.start_load_prefetch(object(), str(snap), prequant_scheme = "int8")
    assert seen.pop("paths")[0] == str(ckpt)
    fl.start_load_prefetch(object(), str(snap), prequant_scheme = "int8")
    assert not any("transformer/" in p for p in seen.pop("paths"))
    fl.start_load_prefetch(object(), str(snap), prequant_scheme = None)
    assert seen["paths"][-1].endswith("transformer/diffusion_pytorch_model.safetensors")
    monkeypatch.setattr(pq, "cached_checkpoint_path", lambda source, cache_dir = None: None)
    seen.clear()
    fl.start_load_prefetch(
        object(),
        str(snap),
        prequant_scheme = "int8",
        text_encoders_replaced = {"text_encoder", "text_encoder_2"},
    )
    assert [p.split("/snap/")[-1] for p in seen["paths"]] == [
        "vae/diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model.safetensors",
    ]


def test_only_hosted_precast_encoders_leave_the_prefetch(monkeypatch):
    import types

    import torch
    import core.inference.diffusion_precision as prec
    from core.inference.diffusion_families import detect_family

    monkeypatch.setattr(prec, "te_quant_supported", lambda target, mode: True)
    fam = detect_family("black-forest-labs/FLUX.1-schnell")
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", dtype = torch.bfloat16)
    base = "black-forest-labs/FLUX.1-schnell"
    for mode in (None, "off", "int8"):
        assert fl.te_precast_components(fam, base, mode, target) == frozenset()
    replaced = fl.te_precast_components(fam, base, "fp8", target)
    assert replaced and replaced <= {"text_encoder_2"}
    assert fl.te_precast_components(object(), base, "fp8", target) == frozenset()


def test_load_prefetch_never_raises(monkeypatch):
    import core.inference.diffusion_denoiser_prequant as dp

    def _boom(*a, **k):
        raise RuntimeError("registry down")

    monkeypatch.setattr(dp, "denoiser_prequant_source", _boom)
    assert fl.start_load_prefetch(object(), "no/such-repo", prequant_scheme = "int8") is None
    fl.stop_prefetch(None)


def test_fast_upload_is_inert_off_cuda():
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(4, 4)
    with fl.fast_upload([module], "cpu") as staged:
        module.to("cpu")
    assert staged == 0


torch_cuda = pytest.mark.skipif(
    not (pytest.importorskip("torch").cuda.is_available()), reason = "needs a CUDA device"
)


def _model(torch):
    torch.manual_seed(0)

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(512, 1024).to(torch.bfloat16)
            self.b = torch.nn.Linear(1024, 512, bias = False).to(torch.bfloat16)
            self.c = torch.nn.Linear(512, 1024)
            self.c.weight = (
                self.a.weight
            )  # tied (a bf16 weight on an fp32 Linear is fine for .to())
            self.register_buffer("table", torch.randn(700, 1024))
            self.register_buffer("ids", torch.arange(300000, dtype = torch.int64))
            self.small = torch.nn.Parameter(torch.randn(3, 5), requires_grad = False)
            self.view_param = torch.nn.Parameter(
                torch.randn(1024, 600)[:, :512].clone().t(), requires_grad = False
            )

    return _M()


@torch_cuda
def test_fast_upload_matches_a_plain_to_byte_for_byte(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(fl, "_UPLOAD_MIN_TOTAL_BYTES", 0)
    monkeypatch.setattr(fl, "_UPLOAD_CHUNK_BYTES", 1 << 20)  # many chunks per tensor, ring wraps
    ref = _model(torch).to("cuda")
    got = _model(torch)
    names_before = {n: type(p) for n, p in got.named_parameters()}
    with fl.fast_upload([got, None, "tokenizer"], "cuda") as staged:
        out = got.to("cuda")
    assert out is got and staged > 0
    assert got.c.weight is got.a.weight
    ref_state, got_state = ref.state_dict(keep_vars = True), got.state_dict(keep_vars = True)
    assert ref_state.keys() == got_state.keys()
    for name, want in ref_state.items():
        have = got_state[name]
        assert (
            have.device.type == "cuda" and have.dtype == want.dtype and have.shape == want.shape
        ), name
        assert have.stride() == want.stride(), name
        assert have.requires_grad == want.requires_grad, name
        assert torch.equal(have, want), name
    assert {n: type(p) for n, p in got.named_parameters()} == names_before


@torch_cuda
def test_fast_upload_leaves_other_to_calls_alone(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(fl, "_UPLOAD_MIN_TOTAL_BYTES", 0)
    module = _model(torch)
    weight = module.a.weight
    with fl.fast_upload([module], "cuda") as staged:
        assert staged > 0
        assert weight.to("cuda", dtype = torch.float32).dtype == torch.float32
        assert weight.to("cuda", copy = True).data_ptr() != weight.to("cuda").data_ptr()
        assert weight.to("cpu").device.type == "cpu"
        module.to("cuda")
    assert module.a.weight.device.type == "cuda"


@torch_cuda
def test_fast_upload_kill_switch_and_failure_fall_back_to_the_stock_copy(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(fl, "_UPLOAD_MIN_TOTAL_BYTES", 0)
    monkeypatch.setenv(fl.FAST_UPLOAD_ENV, "0")
    module = _model(torch)
    with fl.fast_upload([module], "cuda") as staged:
        module.to("cuda")
    assert staged == 0 and module.a.weight.device.type == "cuda"
    monkeypatch.delenv(fl.FAST_UPLOAD_ENV)

    def _boom(*a, **k):
        raise RuntimeError("pinned memory exhausted")

    monkeypatch.setattr(fl, "_ring_copy", _boom)
    module = _model(torch)
    with fl.fast_upload([module], "cuda") as staged:
        module.to("cuda")
    assert staged == 0 and torch.equal(module.table, _model(torch).table.cuda())


@torch_cuda
def test_fast_upload_passes_tensor_subclasses_through(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(fl, "_UPLOAD_MIN_TOTAL_BYTES", 0)

    class _Tagged(torch.Tensor):
        pass

    module = torch.nn.Linear(1024, 1024)
    module.weight = torch.nn.Parameter(
        module.weight.detach().as_subclass(_Tagged), requires_grad = False
    )
    assert fl._upload_candidates([module], "cuda") == [] or all(
        type(t) in (torch.Tensor, torch.nn.Parameter)
        for t in fl._upload_candidates([module], "cuda")
    )
    with fl.fast_upload([module], "cuda"):
        module.to("cuda")
    assert module.weight.device.type == "cuda"


def test_rotational_lookup_never_raises(tmp_path):
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"z")
    assert fl._on_rotational_disk(str(path)) in (True, False)
    assert fl._on_rotational_disk(str(tmp_path / "missing")) is False
