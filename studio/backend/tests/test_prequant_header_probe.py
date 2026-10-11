# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A hosted pre-quant checkpoint is judged from its header BEFORE its weights are downloaded, and its
packed-weight fingerprint verifies whichever torchao generation built or loads it.

The Hub side is a real HTTP server on localhost serving real ``torch.save`` / safetensors files with
Range support, so the probe's range requests, zip directory walk and inert unpickle all run for real;
only the two huggingface_hub URL / HEAD helpers are pointed at it.
"""

from __future__ import annotations

import http.server
import json
import os
import threading
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_prequant as pq  # noqa: E402
from core.inference.diffusion_prequant import PREQUANT_FORMAT, PrequantSource  # noqa: E402

_STALE_INT8_EXCLUDES = [
    "norm",
    "_mod",
    "modulation",
    "timestep_embed",
    "guidance_embed",
    "time_text_embed",
    "pooled",
    "time_embed",
]


@pytest.fixture(autouse = True)
def _fresh_probe(monkeypatch):
    # Tolerant of a build without the probe, so the behavioural tests below run (and fail) there too.
    monkeypatch.delenv(
        getattr(pq, "HEADER_PROBE_ENV", "UNSLOTH_PREQUANT_HEADER_PROBE"), raising = False
    )
    monkeypatch.setattr(pq, "_HEADER_PROBE_MEMO", {}, raising = False)
    monkeypatch.setattr(pq, "_HEADER_PROBE_SAID", set(), raising = False)
    monkeypatch.setattr(pq, "hub_offline", lambda: False)
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme, name = None: True)


def _qwen_int8_meta(excludes):
    return {
        "scheme": "int8",
        "family": "qwen-image",
        "base_model_id": "Qwen/Qwen-Image-2512",
        "min_features": 512,
        "exclude_name_tokens": list(excludes),
        "torch_version": torch.__version__,  # a TorchVersion, as the builder records it
    }


def _write_pt(
    path,
    meta,
    *,
    mb = 8,
):
    """A real torch.save checkpoint whose weight bytes dwarf its metadata."""
    state_dict = {
        f"blocks.{i}.weight": torch.zeros(256, 1024, dtype = torch.int8) for i in range(mb * 4)
    }
    torch.save({"format": PREQUANT_FORMAT, "metadata": meta, "state_dict": state_dict}, path)
    return path


def _write_safetensors(
    path,
    meta,
    *,
    mb = 8,
):
    from safetensors.torch import save_file

    from core.inference.prequant_safetensors import UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY

    tensors = {
        f"blocks.{i}.weight": torch.zeros(256, 1024, dtype = torch.int8) for i in range(mb * 4)
    }
    save_file(
        tensors,
        str(path),
        metadata = {
            UNSLOTH_FORMAT_KEY: PREQUANT_FORMAT,
            UNSLOTH_METADATA_KEY: json.dumps(meta, default = str),
        },
    )
    return path


class _RangeServer:
    """Serves files from a directory with HTTP Range support and counts every byte sent."""

    def __init__(self, root):
        self.root = root
        self.sent = 0
        self.full_gets = 0
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = os.path.join(owner.root, self.path.lstrip("/").split("?")[0])
                if not os.path.isfile(path):
                    self.send_response(404)
                    self.end_headers()
                    return
                size = os.path.getsize(path)
                spec = self.headers.get("Range")
                if not spec:
                    owner.full_gets += 1
                    start, end, code = 0, size - 1, 200
                else:
                    a, _, b = spec.replace("bytes=", "").partition("-")
                    if a == "":
                        start, end = max(0, size - int(b)), size - 1
                    else:
                        start, end = int(a), min(size - 1, int(b) if b else size - 1)
                    code = 206
                with open(path, "rb") as handle:
                    handle.seek(start)
                    body = handle.read(end - start + 1)
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                if code == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                self.wfile.write(body)
                owner.sent += len(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target = self.httpd.serve_forever, daemon = True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hub: ``<repo>/<file>`` under tmp_path served over real HTTP; downloads are recorded."""
    import huggingface_hub

    root = tmp_path / "hub"
    root.mkdir()
    server = _RangeServer(str(root))

    def hf_hub_url(
        repo_id,
        filename,
        revision = None,
        **_,
    ):
        return f"{server.url}/{repo_id}/{filename}"

    def get_hf_file_metadata(
        url,
        token = None,
        timeout = None,
        **_,
    ):
        path = os.path.join(str(root), url[len(server.url) + 1 :])
        if not os.path.isfile(path):
            err = Exception("404")
            err.response = types.SimpleNamespace(status_code = 404)
            raise err
        return types.SimpleNamespace(size = os.path.getsize(path), commit_hash = None)

    downloads: list = []

    def hf_hub_download(
        repo_id = None,
        filename = None,
        **kwargs,
    ):
        downloads.append(filename)
        raise RuntimeError("weight download attempted")

    monkeypatch.setattr(huggingface_hub, "hf_hub_url", hf_hub_url)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", get_hf_file_metadata)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(pq, "cached_checkpoint_path", lambda *a, **k: None)
    yield types.SimpleNamespace(root = root, server = server, downloads = downloads)
    server.close()


def _repo_file(hub, repo, name):
    folder = hub.root / repo
    folder.mkdir(parents = True, exist_ok = True)
    return folder / name


def _source(repo, *names):
    return PrequantSource(
        kind = "repo", location = repo, filename = names[0], fallback_filenames = tuple(names[1:])
    )


def _load(source, logger = None):
    return pq.load_prequantized_transformer(
        object,
        "Qwen/Qwen-Image-2512",
        source,
        device = "cpu",
        dtype = None,
        scheme = "int8",
        min_features = 512,
        logger = logger,
    )


class _Lines:
    def __init__(self):
        self.lines: list = []

    def _say(self, msg, *args):
        self.lines.append(msg % args if args else msg)

    debug = info = warning = error = _say


@pytest.mark.parametrize("container", ["pt", "safetensors"])
def test_a_stale_hosted_checkpoint_is_refused_before_any_weight_download(hub, container):
    repo = "unsloth/Qwen-Image-2512-FP8"
    name = f"Qwen-Image-2512-INT8.{container}"
    path = _repo_file(hub, repo, name)
    writer = _write_pt if container == "pt" else _write_safetensors
    writer(path, _qwen_int8_meta(_STALE_INT8_EXCLUDES), mb = 16)
    logger = _Lines()
    # A missing ComfyUI twin ahead of it (404) is walked past, exactly as the resolver does.
    source = _source(repo, "Qwen-Image-2512-INT8-ComfyUI.safetensors", name)

    assert _load(source, logger) is None
    assert hub.downloads == []  # the multi-GB hf_hub_download never started
    assert hub.server.full_gets == 0
    assert hub.server.sent < path.stat().st_size // 4, (hub.server.sent, path.stat().st_size)
    said = [line for line in logger.lines if "skipping hosted int8 checkpoint" in line]
    assert len(said) == 1 and "txt_in" in said[0] and name in said[0]
    assert "txt_in" in (pq.last_prequant_failure() or "")


def test_a_current_hosted_checkpoint_still_goes_to_the_download(hub):
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme

    repo = "unsloth/Qwen-Image-FP8"
    path = _repo_file(hub, repo, "Qwen-Image-INT8.pt")
    _write_pt(path, _qwen_int8_meta(exclude_tokens_for_scheme("int8", "qwen-image")))
    source = _source(repo, "Qwen-Image-INT8.pt")
    _load(source)
    assert hub.downloads == ["Qwen-Image-INT8.pt"]


def test_planning_sees_a_stale_hosted_checkpoint_as_no_source(hub):
    """``usable_prequant_source`` is what the download plan and auto policy ask, so a stale file is
    never staged: the dense path is budgeted up front instead of after a wasted download."""
    from core.inference.diffusion_families import detect_family

    fam = detect_family("Qwen/Qwen-Image-2512")
    src = pq.resolve_prequant_source(fam, "int8", base_repo = "Qwen/Qwen-Image-2512")
    assert src is not None and src.kind == "repo"
    stale = next(n for n in pq.candidate_filenames_of(src) if n.endswith("INT8.safetensors"))
    _write_safetensors(_repo_file(hub, src.location, stale), _qwen_int8_meta(_STALE_INT8_EXCLUDES))
    assert pq.usable_prequant_source(fam, "int8", base_repo = "Qwen/Qwen-Image-2512") is None
    assert hub.downloads == []


def test_an_unreadable_or_foreign_header_is_never_a_refusal(hub):
    repo = "org/odd"
    _repo_file(hub, repo, "odd-INT8.pt").write_bytes(b"not a zip at all" * 1000)
    from safetensors.torch import save_file

    # A ComfyUI-style file: a valid safetensors without Unsloth's header keys.
    save_file({"a.weight": torch.zeros(4, 4)}, str(_repo_file(hub, repo, "odd-INT8.safetensors")))
    for name in ("odd-INT8.pt", "odd-INT8.safetensors", "absent-INT8.pt"):
        assert pq.hosted_prequant_refusal(_source(repo, name), "int8") is None


def test_offline_with_nothing_cached_asks_nothing(hub, monkeypatch):
    monkeypatch.setattr(pq, "hub_offline", lambda: True)
    repo = "unsloth/Qwen-Image-2512-FP8"
    _write_pt(_repo_file(hub, repo, "x-INT8.pt"), _qwen_int8_meta(_STALE_INT8_EXCLUDES))
    assert pq.hosted_prequant_refusal(_source(repo, "x-INT8.pt"), "int8") is None
    assert (
        pq.hosted_prequant_refusal(_source(repo, "x-INT8.pt"), "int8", local_files_only = True)
        is None
    )
    assert hub.server.sent == 0


def test_a_cached_checkpoint_is_judged_locally_without_the_network(hub, monkeypatch, tmp_path):
    cached = _write_pt(tmp_path / "cached-INT8.pt", _qwen_int8_meta(_STALE_INT8_EXCLUDES))
    monkeypatch.setattr(pq, "cached_checkpoint_path", lambda *a, **k: str(cached))
    monkeypatch.setattr(pq, "hub_offline", lambda: True)
    reason = pq.hosted_prequant_refusal(_source("unsloth/x", "cached-INT8.pt"), "int8")
    assert reason is not None and "txt_in" in reason
    assert hub.server.sent == 0


def test_the_probe_reads_pickle_metadata_without_constructing_anything(tmp_path):
    from core.inference.prequant_header_probe import read_local_checkpoint_header

    class Boom:
        def __reduce__(self):
            return (os.system, ("echo pwned > " + str(tmp_path / "pwned"),))

    meta = _qwen_int8_meta(["norm"])
    torch.save(
        {"format": PREQUANT_FORMAT, "metadata": meta, "state_dict": {"a.weight": Boom()}},
        tmp_path / "evil.pt",
    )
    header = read_local_checkpoint_header(str(tmp_path / "evil.pt"))
    assert header["format"] == PREQUANT_FORMAT
    assert header["metadata"]["exclude_name_tokens"] == ["norm"]
    assert header["metadata"]["torch_version"] == str(torch.__version__)
    assert not (tmp_path / "pwned").exists()


def test_the_kill_switch_turns_the_probe_off(hub, monkeypatch):
    repo = "unsloth/Qwen-Image-2512-FP8"
    _write_pt(_repo_file(hub, repo, "y-INT8.pt"), _qwen_int8_meta(_STALE_INT8_EXCLUDES))
    monkeypatch.setenv(pq.HEADER_PROBE_ENV, "0")
    assert pq.hosted_prequant_refusal(_source(repo, "y-INT8.pt"), "int8") is None
    assert hub.server.sent == 0


# ── fingerprint across torchao generations ───────────────────────────────────────
# Stand-ins named like torchao's classes (the fingerprint keys on the class NAME) holding REAL tensors,
# laid out exactly as each generation stores one symmetric per-row int8 weight (probed on 0.17 / 0.18):
#   0.17: LinearActivationQuantizedTensor -> AffineQuantizedTensor -> PlainAQTTensorImpl(int_data, bf16 scale [n])
#   0.18: Int8Tensor(qdata, float32 scale [n, 1], int8 zero_point [n, 1] of zeros)
class PlainAQTTensorImpl:
    def __init__(self, q, scale):
        self.int_data, self.scale, self.zero_point = q, scale, None


class AffineQuantizedTensor:
    def __init__(self, impl):
        self.tensor_impl = impl


class LinearActivationQuantizedTensor:
    def __init__(self, aqt):
        self.original_weight_tensor = aqt


class Int8Tensor:
    def __init__(
        self,
        q,
        scale,
        zero_point = None,
    ):
        self.qdata, self.scale, self.zero_point = q, scale, zero_point
        self.act_quant_scale = self.act_quant_zero_point = self.act_pre_scale = None


def _int8_values(
    seed,
    n = 8,
    k = 32,
):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8)
    scale = (torch.rand(n, generator = g) / 100 + 1e-4).to(torch.bfloat16)
    return q, scale


def _as_017(q, scale):
    return LinearActivationQuantizedTensor(AffineQuantizedTensor(PlainAQTTensorImpl(q, scale)))


def _as_018(
    q,
    scale,
    *,
    zero_point = True,
):
    n = q.shape[0]
    zp = torch.zeros(n, 1, dtype = torch.int8) if zero_point else None
    return Int8Tensor(q, scale.float().reshape(n, 1), zp)


def _state(layout, values):
    return {f"blocks.{i}.weight": layout(*v) for i, v in enumerate(values)}


_GENERATIONS = {
    "0.17": _as_017,
    "0.18": _as_018,
    "0.18-legacy-decode": lambda q, s: Int8Tensor(q, s.reshape(-1, 1)),  # prequant_legacy_int8
    "0.16": lambda q, s: _as_018(q, s, zero_point = False),
}


@pytest.mark.parametrize("built", sorted(_GENERATIONS))
@pytest.mark.parametrize("loaded", sorted(_GENERATIONS))
def test_a_legacy_v1_fingerprint_verifies_on_another_torchao_generation(built, loaded):
    values = [_int8_values(i) for i in range(4)]
    recorded = pq.packed_weight_fingerprint(_state(_GENERATIONS[built], values))
    assert recorded["count"] == 4
    assert pq._verify_packed_fingerprint(
        _state(_GENERATIONS[loaded], values), {"fingerprint": recorded}
    )


@pytest.mark.parametrize("built", sorted(_GENERATIONS))
@pytest.mark.parametrize("loaded", sorted(_GENERATIONS))
def test_the_canonical_fingerprint_is_the_same_on_every_generation(built, loaded):
    values = [_int8_values(i) for i in range(4)]
    recorded = pq.canonical_weight_fingerprint(_state(_GENERATIONS[built], values))
    assert "modules" not in recorded  # an older loader skips the check instead of refusing
    assert recorded["count"] == 4 and recorded["algo"] == pq.CANONICAL_FINGERPRINT_ALGO
    assert recorded == pq.canonical_weight_fingerprint(_state(_GENERATIONS[loaded], values))
    assert pq._verify_packed_fingerprint(
        _state(_GENERATIONS[loaded], values), {"fingerprint": recorded}
    )


@pytest.mark.parametrize("algo", ["v1", "canonical"])
@pytest.mark.parametrize("what", ["q", "scale"])
def test_a_changed_value_is_still_refused_across_generations(algo, what):
    values = [_int8_values(i) for i in range(4)]
    fingerprint = pq.packed_weight_fingerprint if algo == "v1" else pq.canonical_weight_fingerprint
    recorded = fingerprint(_state(_as_017, values))
    q, scale = values[2]
    if what == "q":
        q = q.clone()
        q[0, 0] ^= 1
    else:
        scale = scale.clone()
        scale[0] = scale[0] * 2
    values[2] = (q, scale)
    logger = _Lines()
    assert not pq._verify_packed_fingerprint(
        _state(_as_018, values), {"fingerprint": recorded}, logger = logger
    )
    assert "blocks.2.weight" in "\n".join(logger.lines)
    assert "blocks.1.weight" not in "\n".join(logger.lines)


def test_fp8_fingerprints_are_unchanged_by_the_canonical_form():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch has no float8")

    class Float8Tensor:
        def __init__(self, q, scale):
            self.qdata, self.scale = q, scale

    g = torch.Generator().manual_seed(0)
    q = torch.randn(8, 32, generator = g).to(torch.float8_e4m3fn)
    scale = torch.rand(8, 1, generator = g)
    sd = {"a.weight": Float8Tensor(q, scale)}
    assert pq._verify_packed_fingerprint(sd, {"fingerprint": pq.packed_weight_fingerprint(sd)})
    assert pq._verify_packed_fingerprint(sd, {"fingerprint": pq.canonical_weight_fingerprint(sd)})


def test_real_torchao_int8_weight_matches_the_other_generations_layout():
    """The weight the INSTALLED torchao quantizes, re-laid out as the other generation stores it."""
    pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    from core.inference.diffusion_transformer_quant import _make_quant_config

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias = False)).to(torch.bfloat16)
    quantize_(model, _make_quant_config("int8"))
    real = model.state_dict()
    parts = pq._canonical_parts(real["0.weight"])
    q, scale = parts["q"], parts["scale"]
    native = type(real["0.weight"]).__name__
    other = (
        _as_018(q, scale.reshape(-1).to(torch.bfloat16))
        if native == "LinearActivationQuantizedTensor"
        else _as_017(q, scale.reshape(-1).to(torch.bfloat16))
    )
    if not torch.equal(scale.reshape(-1).to(torch.bfloat16).float(), scale.reshape(-1).float()):
        pytest.skip("this torchao's scale is not bf16-representable; no 0.17 twin exists")
    assert pq._verify_packed_fingerprint(
        {"0.weight": other}, {"fingerprint": pq.packed_weight_fingerprint(real)}
    )
    assert pq._verify_packed_fingerprint(
        {"0.weight": other}, {"fingerprint": pq.canonical_weight_fingerprint(real)}
    )
