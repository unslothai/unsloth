# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DSpark drafter schema gate, the one-folder-up mmproj lookup, and the drafterless Auto kind."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.models import model_config as mc
from utils.models.drafters import dspark_schema as ds
from utils.models.gguf_metadata import read_gguf_header_summary

PADDED, UNPADDED = 201024, 200058


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value) -> bytes:
    if isinstance(value, bool):
        return _s(key) + struct.pack("<IB", 7, int(value))
    if isinstance(value, int):
        return _s(key) + struct.pack("<II", 4, value)
    if isinstance(value, str):
        return _s(key) + struct.pack("<I", 8) + _s(value)
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return _s(key) + struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_s(v) for v in value)
        return (
            _s(key)
            + struct.pack("<IIQ", 9, 4, len(value))
            + b"".join(struct.pack("<I", v) for v in value)
        )
    raise TypeError(value)


def _write_gguf(path: Path, kvs: dict, tensors: dict) -> Path:
    body = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs))
    body += b"".join(_kv(k, v) for k, v in kvs.items())
    for name, shape in tensors.items():
        body += _s(name) + struct.pack("<I", len(shape))
        body += b"".join(struct.pack("<Q", d) for d in shape) + struct.pack("<IQ", 0, 0)
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(body + b"\0" * 64)
    return path


def _target(
    path: Path,
    n_embd: int = 2304,
    name: str = "Inkling-2-Mini",
) -> Path:
    return _write_gguf(
        path,
        {
            "general.architecture": "inkling2_loop",
            "general.type": "model",
            "general.name": name,
            "general.basename": name,
            "inkling2_loop.embedding_length": n_embd,
            "inkling2_loop.vocab_size_unpadded": UNPADDED,
            # A long string array the summary must skip, as it skips the real vocabulary.
            "tokenizer.ggml.tokens": [f"t{i}" for i in range(300)],
        },
        {"token_embd.weight": (n_embd, PADDED), "output.weight": (n_embd, PADDED)},
    )


def _fused_drafter(path: Path, rows: int = PADDED) -> Path:
    """The 0a9105de4 layout the published dspark-Inkling-2-Mini-Q8_0.gguf uses."""
    return _write_gguf(
        path,
        {
            "general.architecture": "dflash",
            "dflash.vocab_size": PADDED,
            "dflash.vocab_size_unpadded": UNPADDED,
            "dflash.dspark.markov_rank": 256,
            "dflash.attention.sliding_window": 1024,
        },
        {
            "dspark_fuse.weight": (4, 4),
            "joint_proj.weight": (5120, 768),
            "markov_w1.weight": (256, rows),
        },
    )


def _markov2_drafter(path: Path) -> Path:
    """The ggml-org#32218 e4b1cb531 layout of the Mainline_Support drafter."""
    return _write_gguf(
        path,
        {
            "general.architecture": "dflash",
            "dflash.attention.sliding_window": 1025,
            "dflash.has_markov_context": True,
        },
        {
            "joint_proj.weight": (5120, 768),
            "markov_ctx_mix.weight": (4,),
            "markov_w1.weight": (256, UNPADDED),
            "markov_w2.weight": (256, UNPADDED),
            "markov_w2.bias": (UNPADDED,),
        },
    )


def _mmproj(
    path: Path,
    name = "Inkling-2-Mini",
    projection_dim = 2304,
    **extra,
) -> Path:
    kvs = {"general.architecture": "clip", "general.type": "mmproj"}
    if name is not None:
        kvs["general.name"] = name
    kvs["clip.vision.projection_dim"] = projection_dim
    kvs["clip.vision.projector_type"] = "inkling2v"
    kvs.update(extra)
    return _write_gguf(path, kvs, {"mm.2.weight": (4096, projection_dim)})


@pytest.fixture(autouse = True)
def _bundled_schemas(monkeypatch):
    # The published prebuilts (b12588, b12631) load the fused layout only.
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "fused")


# ---- DSpark drafter schema -------------------------------------------------


def test_header_summary_reads_shapes_and_skips_the_vocabulary(tmp_path):
    kvs, tensors = read_gguf_header_summary(str(_target(tmp_path / "m.gguf")))
    assert tensors["token_embd.weight"] == (2304, PADDED)
    assert kvs["inkling2_loop.vocab_size_unpadded"] == UNPADDED
    assert "tokenizer.ggml.tokens" not in kvs


def test_schema_markers(tmp_path):
    assert ds.dspark_drafter_schema(str(_fused_drafter(tmp_path / "a.gguf"))) == "fused"
    assert ds.dspark_drafter_schema(str(_markov2_drafter(tmp_path / "b.gguf"))) == "markov2"
    # No Inkling 2 head tensor: a generic DSpark drafter, whatever its markov_w1 rows.
    bare = _write_gguf(
        tmp_path / "c.gguf",
        {"dflash.vocab_size": PADDED},
        {"markov_w1.weight": (256, UNPADDED), "markov_w2.weight": (256, UNPADDED)},
    )
    assert ds.dspark_drafter_schema(str(bare)) == "generic"
    assert ds.dspark_drafter_incompatibility(str(bare)) is None
    assert ds.dspark_drafter_schema(str(tmp_path / "missing.gguf")) is None
    assert ds.dspark_drafter_incompatibility(str(tmp_path / "missing.gguf")) is None  # fails open


def test_a_fused_only_build_rejects_the_markov2_layout(tmp_path):
    target = str(_target(tmp_path / "m.gguf"))
    assert (
        ds.dspark_drafter_incompatibility(str(_fused_drafter(tmp_path / "a.gguf")), target) is None
    )
    reason = ds.dspark_drafter_incompatibility(str(_markov2_drafter(tmp_path / "b.gguf")), target)
    assert reason is not None and "markov_ctx_mix" in reason and "loads fused" in reason


def test_a_markov2_build_rejects_the_fused_layout(tmp_path, monkeypatch):
    target = str(_target(tmp_path / "m.gguf"))
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "markov2")
    assert (
        ds.dspark_drafter_incompatibility(str(_markov2_drafter(tmp_path / "b.gguf")), target)
        is None
    )
    assert ds.dspark_drafter_incompatibility(str(_fused_drafter(tmp_path / "a.gguf")), target)


def test_env_override_for_a_custom_llama_server(tmp_path, monkeypatch):
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "fused,markov2")
    assert ds.dspark_drafter_incompatibility(str(_markov2_drafter(tmp_path / "b.gguf"))) is None
    assert ds.dspark_drafter_incompatibility(str(_fused_drafter(tmp_path / "a.gguf"))) is None
    monkeypatch.setenv(ds.DSPARK_SCHEMAS_ENV, "bogus")
    monkeypatch.setattr(ds, "_default_llama_server_binary", lambda: None)
    assert ds.supported_dspark_schemas() is None


def test_markov_w1_rows_must_match_the_target_vocab(tmp_path):
    target = str(_target(tmp_path / "m.gguf"))
    short = _fused_drafter(tmp_path / "a.gguf", rows = 151936)
    reason = ds.dspark_drafter_incompatibility(str(short), target)
    assert reason is not None and "151936" in reason


def _layout(tmp_path: Path, *drafters):
    weight = _target(tmp_path / "Inkling-2-Mini-UD-Q4_K_XL.gguf")
    for kind, name in drafters:
        (_fused_drafter if kind == "fused" else _markov2_drafter)(tmp_path / name)
    return str(weight)


def test_detect_skips_a_markov2_drafter_on_a_fused_build(tmp_path, caplog):
    weight = _layout(tmp_path, ("markov2", "dspark-Inkling-2-Mini-Q8_0.gguf"))
    assert mc.detect_dspark_file(weight) is None


def test_detect_prefers_the_loadable_drafter_over_a_better_ranked_one(tmp_path):
    # Q8_0 outranks BF16 by precision, but only the BF16 copy is in the layout this build loads.
    weight = _layout(
        tmp_path,
        ("markov2", "dspark-Inkling-2-Mini-Q8_0.gguf"),
        ("fused", "dspark-Inkling-2-Mini-BF16.gguf"),
    )
    assert Path(mc.detect_dspark_file(weight)).name == "dspark-Inkling-2-Mini-BF16.gguf"


def test_detect_keeps_the_published_drafter(tmp_path):
    weight = _layout(tmp_path, ("fused", "dspark-Inkling-2-Mini-Q8_0.gguf"))
    assert Path(mc.detect_dspark_file(weight)).name == "dspark-Inkling-2-Mini-Q8_0.gguf"


def test_hub_sidecar_in_the_wrong_schema_counts_as_absent(tmp_path):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    weight = _layout(tmp_path, ("markov2", "dspark-Inkling-2-Mini-Q8_0.gguf"))
    drafter = str(tmp_path / "dspark-Inkling-2-Mini-Q8_0.gguf")
    assert backend._dspark_sidecar_unloadable(drafter, weight, frozenset({"fused"})) is True
    assert backend._dspark_sidecar_absent is True
    backend._dspark_sidecar_absent = False
    _fused_drafter(Path(drafter))
    assert backend._dspark_sidecar_unloadable(drafter, weight, frozenset({"fused"})) is False
    assert backend._dspark_sidecar_absent is False


# ---- mmproj one folder up ---------------------------------------------------


def _quant_subfolder(
    tmp_path: Path,
    sub = "Mainline_Support",
    name = "Inkling-2-Mini",
) -> str:
    return str(_target(tmp_path / "repo" / sub / "Inkling-2-Mini-UD-Q4_K_XL.gguf", name = name))


def test_parent_mmproj_of_the_same_model_is_found(tmp_path):
    weight = _quant_subfolder(tmp_path)
    proj = _mmproj(tmp_path / "repo" / "mmproj-F16.gguf")
    assert mc.detect_mmproj_file(weight) == str(proj.resolve())


def test_a_projector_beside_the_weight_still_wins(tmp_path):
    weight = _quant_subfolder(tmp_path)
    _mmproj(tmp_path / "repo" / "mmproj-F16.gguf")
    near = _mmproj(tmp_path / "repo" / "Mainline_Support" / "mmproj-BF16.gguf")
    assert mc.detect_mmproj_file(weight) == str(near.resolve())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Qwen3-VL-8B"},  # another model's projector in a shared models folder
        {"name": None},  # no identity at all: a wildcard name is not enough one folder up
        {"projection_dim": 4096},  # same name, but llama-server would refuse the width
        # A matching general.basename (identity score) must not override a known width mismatch.
        {"projection_dim": 4096, **{"general.basename": "Inkling-2-Mini"}},
    ],
)
def test_unrelated_parent_mmproj_is_ignored(tmp_path, kwargs):
    weight = _quant_subfolder(tmp_path)
    _mmproj(tmp_path / "repo" / "mmproj-F16.gguf", **kwargs)
    assert mc.detect_mmproj_file(weight) is None


def test_a_fitting_projector_of_another_model_one_folder_up_is_ignored(tmp_path, monkeypatch):
    """Type and width that happen to fit are compatibility, not identity: in a shared parent folder the projector
    must also name this model."""
    weight = _quant_subfolder(tmp_path)
    _mmproj(tmp_path / "repo" / "mmproj-F16.gguf", name = "Other-Model")
    monkeypatch.setattr(mc, "mmproj_functional_match", lambda *a, **k: (True, "type and width"))
    assert mc.detect_mmproj_file(weight) is None


def test_the_matching_width_projector_wins_over_a_same_name_wrong_width_one(tmp_path):
    weight = _quant_subfolder(tmp_path)
    _mmproj(
        tmp_path / "repo" / "mmproj-F16.gguf",
        projection_dim = 4096,
        **{"general.basename": "Inkling-2-Mini"},
    )
    good = _mmproj(tmp_path / "repo" / "mmproj-BF16.gguf")
    assert mc.detect_mmproj_file(weight) == str(good.resolve())


def test_parent_lookup_is_one_level_only(tmp_path):
    weight = str(_target(tmp_path / "repo" / "a" / "b" / "Inkling-2-Mini-UD-Q4_K_XL.gguf"))
    _mmproj(tmp_path / "repo" / "mmproj-F16.gguf")
    assert mc.detect_mmproj_file(weight) is None


def test_parent_lookup_respects_accept(tmp_path):
    weight = _quant_subfolder(tmp_path)
    _mmproj(tmp_path / "repo" / "mmproj-F16.gguf")
    assert mc.detect_mmproj_file(weight, accept = lambda p: "repo/mmproj" not in p) is None


# ---- spec_drafter_kind under Auto with no drafter --------------------------


def _spec_backend(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    caps = {
        "found": True,
        "mtp_token": "draft-mtp",
        "supports_mtp": True,
        "supports_dspark": True,
        "supports_dflash": True,
        "mtp_probe_inconclusive": False,
        "ngram_mod_flavor": "new",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, binary = None: caps)
    )
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = None
    return backend


def _flags(backend, **kwargs):
    base = dict(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "/models/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    base.update(kwargs)
    return backend._build_speculative_flags(**base)


def test_auto_without_any_drafter_reports_no_kind(monkeypatch):
    backend = _spec_backend(monkeypatch)
    flags = _flags(backend)
    assert flags == ["--spec-default"]
    assert backend._spec_drafter_kind is None
    assert backend.spec_drafter_kind is None


def test_auto_with_an_mtp_drafter_still_reports_mtp(monkeypatch):
    backend = _spec_backend(monkeypatch)
    _flags(backend, model_identifier = "unsloth/gemma-4-31b-it-GGUF", mtp_draft_path = "/m/mtp-x.gguf")
    assert backend._spec_drafter_kind == "mtp"


def test_auto_with_a_dspark_sidecar_reports_dspark(monkeypatch):
    backend = _spec_backend(monkeypatch)
    flags = _flags(backend, dspark_draft_path = "/m/dspark-Inkling-2-Mini-Q8_0.gguf")
    assert "draft-dspark" in flags
    assert backend._spec_drafter_kind == "dspark"


def test_forced_mtp_on_a_plain_model_keeps_the_mtp_kind(monkeypatch):
    # The fallback notice names the drafter the user asked for.
    backend = _spec_backend(monkeypatch)
    _flags(backend, speculative_type = "mtp")
    assert backend._spec_drafter_kind == "mtp"
