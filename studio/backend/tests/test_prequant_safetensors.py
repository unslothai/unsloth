# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The safetensors pre-quant container: its own round-trip, and how the loaders dispatch to it.

Two layers, deliberately separated. The dispatch tests are hermetic -- they never touch torchao or
safetensors -- because what they assert is routing: which reader a name selects, which gate a name
answers, which source survives planning. The round-trip tests are the opposite and are skipped
without the real libraries, because a fake that "round-trips" proves nothing about a format whose
entire risk is in the metadata contract.
"""

from __future__ import annotations

import json
import sys

import pytest

import core.inference.diffusion_prequant as pq
import core.inference.prequant_safetensors as ps
from core.inference.diffusion_families import DiffusionFamily


def _family(**overrides) -> DiffusionFamily:
    return DiffusionFamily(
        **{
            "name": "test-fam",
            "base_repo": "org/base",
            "pipeline_class": "TestPipeline",
            "transformer_class": "TestTransformer",
            **overrides,
        }
    )


# --------------------------------------------------------------------------------------- dispatch


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Model-FP8.safetensors", True),
        ("Model-FP8.SAFETENSORS", True),
        ("/abs/path/to/Model-FP8.safetensors", True),
        ("Model-FP8.pt", False),
        ("transformer_fp8.pt", False),
        # A name that merely CONTAINS the word is not the container. Dispatch is on the extension
        # because that is what was asked of the Hub, not on anything resembling a sniff.
        ("safetensors-notes.pt", False),
        (None, False),
        ("", False),
    ],
)
def test_is_safetensors_checkpoint(name, expected):
    assert ps.is_safetensors_checkpoint(name) is expected


def test_load_dispatches_on_extension(monkeypatch):
    """The two readers are picked by name, and neither is consulted for the other's file."""
    calls = []
    monkeypatch.setattr(
        ps, "load_prequant_safetensors", lambda path, **kw: calls.append(("safetensors", path))
    )
    monkeypatch.setattr(
        pq, "_torch_load_prequant", lambda path, **kw: calls.append(("pickle", path))
    )
    pq._load_prequant_checkpoint("/x/Model-FP8.safetensors")
    pq._load_prequant_checkpoint("/x/Model-FP8.pt", map_location = "cpu")
    assert calls == [("safetensors", "/x/Model-FP8.safetensors"), ("pickle", "/x/Model-FP8.pt")]


def test_safetensors_needs_no_pickle_allowlist(monkeypatch):
    """The whole point of the container: an install that cannot open a pickle can still load one.

    ``restricted_prequant_load_supported`` is the gate that makes ``video.py`` refuse a pre-quant
    checkpoint outright. A safetensors artifact names no constructors, so answering the pickle's
    question for it would strand exactly the installs this format exists to unblock.
    """
    monkeypatch.setattr(pq, "_register_prequant_safe_globals", lambda: False)
    monkeypatch.setattr(ps, "safetensors_prequant_supported", lambda: True)

    assert pq.restricted_prequant_load_supported("fp8") is False
    assert pq.restricted_prequant_load_supported("fp8", "Model-FP8.pt") is False
    assert pq.restricted_prequant_load_supported("fp8", "Model-FP8.safetensors") is True


def test_safetensors_gate_still_needs_torchao_helpers(monkeypatch):
    """The other direction: no flatten/unflatten means no safetensors load, so say so up front
    rather than letting planning drop the dense shards for a file nothing can read."""
    monkeypatch.setattr(pq, "_register_prequant_safe_globals", lambda: True)
    monkeypatch.setattr(ps, "safetensors_prequant_supported", lambda: False)
    assert pq.restricted_prequant_load_supported("fp8", "Model-FP8.safetensors") is False


def test_usable_source_survives_a_pickle_only_refusal(monkeypatch):
    """Regression for the ordering: the gate is asked about the RESOLVED names, not the scheme.

    Asking first and resolving second (what this did before safetensors existed) returns None for a
    repo whose primary artifact is safetensors, on an install that merely cannot open pickles. The
    pre-quant is then invisible to memory planning and the load silently runs dense.
    """
    fam = _family(
        prequant_repos = (("fp8", "org/model-fp8"),),
        prequant_filenames = (("fp8", "Model-FP8.safetensors"),),
    )
    monkeypatch.setattr(pq, "_register_prequant_safe_globals", lambda: False)
    monkeypatch.setattr(ps, "safetensors_prequant_supported", lambda: True)

    src = pq.usable_prequant_source(fam, "fp8")
    assert src is not None
    assert src.filename == "Model-FP8.safetensors"


def test_usable_source_still_refused_when_nothing_is_readable(monkeypatch):
    """A family with no DECLARED safetensors name, on a pickle-less install, keeps answering None.

    Regression for a hazard the safetensors-first chain introduced. Every family now derives a
    ``<Model>-<SCHEME>.safetensors`` candidate, so a naive "is any candidate readable" gate answers
    yes on a pickle-less install even for a repo that hosts only ``.pt``. Planning would then budget
    a 6 GB artifact, the download would 404 to the pickle, the pickle would be refused, and the
    load would fall back to dense under a plan that never budgeted it: the evict-then-OOM this
    function exists to prevent. A derived name is a guess and does not count as evidence.
    """
    fam = _family(prequant_repos = (("fp8", "org/model-fp8"),))
    monkeypatch.setattr(pq, "_register_prequant_safe_globals", lambda: False)
    monkeypatch.setattr(ps, "safetensors_prequant_supported", lambda: True)
    monkeypatch.setattr(pq, "cached_checkpoint_path", lambda source, **kw: None)
    assert pq.usable_prequant_source(fam, "fp8") is None


def test_a_cached_safetensors_artifact_is_evidence_enough(monkeypatch):
    """The other side of that guard: once the file is actually THERE, the pickle-less install may
    plan for it. A cached candidate is evidence, exactly as a family-declared name is."""
    fam = _family(prequant_repos = (("fp8", "org/model-fp8"),))
    monkeypatch.setattr(pq, "_register_prequant_safe_globals", lambda: False)
    monkeypatch.setattr(ps, "safetensors_prequant_supported", lambda: True)
    monkeypatch.setattr(
        pq, "cached_checkpoint_path", lambda source, **kw: "/cache/model-FP8.safetensors"
    )
    src = pq.usable_prequant_source(fam, "fp8")
    assert src is not None and src.filename == "model-FP8.safetensors"


def test_the_derived_chain_puts_safetensors_first_and_keeps_the_pickles(monkeypatch):
    """The preference itself, and the promise that nothing is dropped behind it: an existing
    .pt-only repo still resolves both of the names it resolves today."""
    fam = _family(prequant_repos = (("fp8", "unsloth/Model-FP8"),))
    src = pq.resolve_prequant_source(fam, "fp8")
    assert src.candidate_filenames == (
        "Model-FP8.safetensors",
        "Model-FP8.pt",
        "transformer_fp8.pt",
    )


def test_local_scheme_probe_reads_the_header_only(monkeypatch, tmp_path):
    """The probe must not parse a multi-GB file to learn one string. For safetensors it reads the
    header, and it must NOT reach the pickle reader on the way."""
    path = tmp_path / "local-FP8.safetensors"
    path.write_bytes(b"not really safetensors")

    def _boom(*a, **kw):
        raise AssertionError("the pickle reader was used for a safetensors artifact")

    monkeypatch.setattr(pq, "_torch_load_prequant", _boom)
    monkeypatch.setattr(
        ps,
        "read_prequant_header",
        lambda p: {"format": pq.PREQUANT_FORMAT, "metadata": {"scheme": "int8"}},
    )
    assert pq.local_prequant_scheme(str(path)) == "int8"


def test_local_scheme_probe_rejects_a_foreign_format(monkeypatch, tmp_path):
    path = tmp_path / "other.safetensors"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        ps,
        "read_prequant_header",
        lambda p: {"format": "someone_elses_v1", "metadata": {"scheme": "int8"}},
    )
    assert pq.local_prequant_scheme(str(path)) is None


def test_undotted_keys_are_refused_before_a_file_exists():
    """torchao's unflatten splits every key on '.', so a root-level tensor writes fine and can
    never be read back. Refuse at build time, naming the keys."""
    assert ps.unsupported_state_dict_keys({"a.weight": 1, "pos_embed": 2}) == ["pos_embed"]
    assert ps.unsupported_state_dict_keys({"a.weight": 1}) == []


# ------------------------------------------------------------------------------------ round-trip


def _real_libs():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("torchao")
    if not ps.safetensors_prequant_supported():
        pytest.skip("torchao is too old for flatten/unflatten_tensor_state_dict")


def test_plain_tensor_round_trip(tmp_path):
    """Plain tensors (the text-encoder artifact's whole state dict) survive with their metadata."""
    _real_libs()
    import torch

    sd = {"enc.layer.weight": torch.ones(4, 4), "enc.layer.bias": torch.zeros(4)}
    meta = {"scheme": "fp8", "base_model_id": "org/base", "min_features": 512}
    path = str(tmp_path / "te.safetensors")
    ps.save_prequant_safetensors(path, fmt = pq.PREQUANT_FORMAT, state_dict = sd, metadata = meta)

    header = ps.read_prequant_header(path)
    assert header == {"format": pq.PREQUANT_FORMAT, "metadata": meta}

    ckpt = ps.load_prequant_safetensors(path)
    assert ckpt["format"] == pq.PREQUANT_FORMAT
    assert ckpt["metadata"] == meta
    assert set(ckpt["state_dict"]) == set(sd)
    for key, value in sd.items():
        assert torch.equal(ckpt["state_dict"][key], value)


def test_header_is_readable_as_a_torchao_checkpoint(tmp_path):
    """Interop, and the trap that makes it fail silently.

    torchao hands back metadata ALREADY JSON-encoded. Encoding it again round-trips perfectly
    against our own reader and still produces a header ``is_metadata_torchao`` rejects, so every
    transformers / diffusers loader refuses the file and nothing here would notice.
    """
    _real_libs()
    import torch
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_utils import is_metadata_torchao

    path = str(tmp_path / "x.safetensors")
    ps.save_prequant_safetensors(
        path,
        fmt = pq.PREQUANT_FORMAT,
        state_dict = {"enc.layer.weight": torch.ones(2, 2)},
        metadata = {"scheme": "fp8"},
    )
    with safe_open(path, framework = "pt") as handle:
        raw = handle.metadata() or {}

    assert is_metadata_torchao(raw)
    # Our two keys ride alongside torchao's and are ignored by it.
    assert raw[ps.UNSLOTH_FORMAT_KEY] == pq.PREQUANT_FORMAT
    assert json.loads(raw[ps.UNSLOTH_METADATA_KEY]) == {"scheme": "fp8"}
    assert "tensor_names" in raw


def test_truncated_checkpoint_is_named_not_silently_partial(tmp_path):
    """A file whose header does not account for every tensor is refused with the tensor named,
    rather than reaching load_state_dict as a bare missing-key error."""
    _real_libs()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    path = str(tmp_path / "x.safetensors")
    ps.save_prequant_safetensors(
        path,
        fmt = pq.PREQUANT_FORMAT,
        state_dict = {"enc.layer.weight": torch.ones(2, 2)},
        metadata = {"scheme": "fp8"},
    )
    with safe_open(path, framework = "pt") as handle:
        raw = dict(handle.metadata() or {})
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}

    # An extra tensor the header never describes: the shape a truncated or edited artifact takes.
    tensors["enc.layer.stray"] = torch.zeros(2)
    stray = str(tmp_path / "stray.safetensors")
    save_file(tensors, stray, metadata = raw)

    with pytest.raises(ValueError, match = "does not account for"):
        ps.load_prequant_safetensors(stray)


@pytest.mark.parametrize("scheme", ["fp8", "int8"])
def test_quantized_round_trip_is_exact(tmp_path, scheme):
    """The case the format exists for: torchao weight subclasses, reconstructed bit for bit and
    accepted by ``load_state_dict(strict=True)`` on a module quantized the same way."""
    _real_libs()
    import torch

    if not torch.cuda.is_available():
        pytest.skip("fp8 / int8 quantization needs CUDA")
    from torchao.quantization import quantize_

    from core.inference.diffusion_transformer_quant import _make_quant_config, make_filter_fn

    def build():
        module = (
            torch.nn.Sequential(
                torch.nn.Linear(1024, 1024, bias = False),
                torch.nn.Linear(1024, 1024, bias = True),
            )
            .cuda()
            .bfloat16()
        )
        quantize_(module, _make_quant_config(scheme), filter_fn = make_filter_fn(512))
        return module

    source = build()
    state = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in source.state_dict().items()
    }
    path = str(tmp_path / f"{scheme}.safetensors")
    try:
        ps.save_prequant_safetensors(
            path, fmt = pq.PREQUANT_FORMAT, state_dict = state, metadata = {"scheme": scheme}
        )
    except ValueError as exc:
        # A torchao too old to QUANTISE into a flattenable subclass (int8 before 0.18) must say so
        # in its own terms rather than leak torchao's bare "Unsupported tensor type". Assert the
        # message, then skip: the round-trip below is not a question this install can answer.
        assert "torchao >= 0.18" in str(exc), exc
        pytest.skip(f"this torchao cannot write {scheme} to safetensors: {exc}")

    loaded = ps.load_prequant_safetensors(path)["state_dict"]
    assert set(loaded) == set(state)
    for key, value in state.items():
        got, want = loaded[key], value
        if hasattr(want, "dequantize"):
            assert type(got) is type(want)
            got, want = got.dequantize(), want.dequantize()
        assert torch.equal(got, want)

    build().load_state_dict(loaded, strict = True, assign = True)


def test_the_windows_rocm_torchao_stub_is_not_safetensors_support(monkeypatch):
    """The stub answers every ``torchao.*`` import with a fabricated callable, so importing the
    flatten/unflatten pair proves nothing on Windows ROCm: both names bind, both return None, and
    an install that reported support here would have planning drop the dense shards for a
    checkpoint the loader can never rebuild. The pickle probe already asks the same question."""
    import core._torchao_stub as stub
    from core.inference import prequant_safetensors as ps
    from core.inference.diffusion_prequant import restricted_prequant_load_supported

    for name in [k for k in list(sys.modules) if k == "torchao" or k.startswith("torchao.")]:
        monkeypatch.delitem(sys.modules, name, raising = False)
    monkeypatch.setattr(stub, "_is_windows_rocm", lambda: True)
    stub.install_torchao_windows_rocm_stub()
    assert stub.is_stubbed("torchao"), "the stub did not install, so this proves nothing"
    # The negative control: the import really does succeed against the stub.
    from torchao.prototype.safetensors.safetensors_support import (  # noqa: F401
        unflatten_tensor_state_dict,
    )

    assert ps._torchao_helpers() is None
    assert ps.safetensors_prequant_supported() is False
    # And the capability the planners ask is False for a safetensors name too, not only the pickle.
    assert restricted_prequant_load_supported("fp8", "Model-FP8.safetensors") is False
    assert restricted_prequant_load_supported("fp8", "Model-FP8.pt") is False


def test_a_scheme_this_torchao_cannot_flatten_is_reported_before_the_build(monkeypatch):
    """The helpers importing is not the same question as this scheme producing something they take.

    Through torchao 0.17 int8 quantises to LinearActivationQuantizedTensor, which flatten refuses,
    so the builder's container preflight passed and the failure arrived only after the download and
    the hours of GPU quantization.
    """
    import torchao.quantization as tq
    from core.inference import prequant_safetensors as ps

    quantized: list = []
    monkeypatch.setattr(tq, "quantize_", lambda mod, cfg, **kw: quantized.append(cfg))

    def _refuses(state_dict):
        raise ValueError("Unsupported tensor type: <class 'LinearActivationQuantizedTensor'>")

    monkeypatch.setattr(ps, "_torchao_helpers", lambda: (_refuses, object()))
    assert ps.scheme_is_flattenable("cfg") is False
    assert quantized == ["cfg"], "the probe must actually quantize, not guess from a version"

    # Flattens cleanly: supported.
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: (lambda sd: ({}, {}), object()))
    assert ps.scheme_is_flattenable("cfg") is True

    # An unrelated flatten failure is NOT evidence the scheme is unsupported, and a build must not
    # be refused on it.
    def _other(state_dict):
        raise ValueError("something else entirely")

    monkeypatch.setattr(ps, "_torchao_helpers", lambda: (_other, object()))
    assert ps.scheme_is_flattenable("cfg") is None

    # A config this torchao will not even apply is the same "no evidence" answer.
    def _boom(mod, cfg, **kw):
        raise RuntimeError("config needs CUDA")

    monkeypatch.setattr(tq, "quantize_", _boom)
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: (_refuses, object()))
    assert ps.scheme_is_flattenable("cfg") is None

    # No helpers at all is a flat no.
    monkeypatch.setattr(ps, "_torchao_helpers", lambda: None)
    assert ps.scheme_is_flattenable("cfg") is False


def test_the_real_installed_torchao_answers_the_int8_question(monkeypatch):
    """Not a mock: the probe against the torchao actually installed, whichever way it answers."""
    from core.inference.diffusion_transformer_quant import _make_quant_config
    from core.inference.prequant_safetensors import scheme_is_flattenable

    import torchao

    answer = scheme_is_flattenable(_make_quant_config("int8"))
    version = tuple(int(p) for p in torchao.__version__.split("+")[0].split(".")[:2])
    if version < (0, 18):
        assert answer is False, f"torchao {torchao.__version__} should refuse int8 flatten"
    else:
        assert answer is not False, f"torchao {torchao.__version__} should flatten int8"
