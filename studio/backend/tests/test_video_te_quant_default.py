# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video families default to a hosted quantized text encoder, the scheme ComfyUI's templates load."""

from __future__ import annotations

import json
import re
import types

import pytest
import torch

import core.inference.diffusion_te_prequant as tpq
from core.inference.video import video_te_quant_request
from core.inference.video_families import _FAMILIES, detect_video_family

CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
WAN_REPO = "unsloth/Wan2.2-TI2V-5B-FP8"
WAN_BASES = (
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
)

# family -> (default scheme, hosted repo, preferred filename)
EXPECTED_DEFAULTS = {
    "wan2.2-ti2v-5b": ("fp8", WAN_REPO, "Wan2.2-TI2V-5B-text_encoder-FP8.safetensors"),
    "wan2.2-t2v-a14b": ("fp8", WAN_REPO, "Wan2.2-TI2V-5B-text_encoder-FP8.safetensors"),
    "wan2.1-t2v-1.3b": ("fp8", WAN_REPO, "Wan2.2-TI2V-5B-text_encoder-FP8.safetensors"),
    "wan2.1-t2v-14b": ("fp8", WAN_REPO, "Wan2.2-TI2V-5B-text_encoder-FP8.safetensors"),
    "ltx-2": ("fp8", "unsloth/LTX-2-FP8", "LTX-2-text_encoder-FP8.safetensors"),
    "hunyuanvideo-1.5": (
        "fp8",
        "unsloth/HunyuanVideo-1.5-FP8",
        "HunyuanVideo-1.5-text_encoder-FP8.safetensors",
    ),
    "hunyuanvideo-1.5-720p": (
        "fp8",
        "unsloth/HunyuanVideo-1.5-FP8",
        "HunyuanVideo-1.5-text_encoder-FP8.safetensors",
    ),
}
HV15_BASES = (
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
)


def test_every_generic_path_video_family_has_a_default():
    """Every family that casts its own encoder in place defaults to a hosted one; only H3 and LTX-2.5 resolve theirs,
    and Cosmos3 has no text encoder."""
    assert {f.name for f in _FAMILIES if f.te_quant_auto} == set(EXPECTED_DEFAULTS)
    assert {f.name for f in _FAMILIES if not f.te_quant_auto} == {
        "minimax-h3",
        "ltx-2.5",
        "cosmos3-nano",
        "cosmos3-edge",
    }


def test_the_two_hunyuanvideo_repacks_share_one_encoder_and_not_qwen_images():
    assert tpq.te_base_equivalent(*HV15_BASES)
    # Qwen-Image's hosted file is the full Qwen2.5-VL (vision tower, lm_head), a different key layout.
    assert not tpq.te_base_equivalent(HV15_BASES[0], "Qwen/Qwen-Image")


def _family(name):
    return detect_video_family("", override = name)


def test_every_video_family_default_scheme_is_one_we_host():
    """The video twin of the image sweep: a te_quant_auto with no hosted file would cost a dense download plus an
    in-place cast on every default load."""
    offenders = []
    for fam in _FAMILIES:
        scheme = getattr(fam, "te_quant_auto", None)
        if scheme is None:
            continue
        if not any(
            tpq.resolve_te_prequant_source(fam, c, scheme) for c in tpq.TE_PREQUANT_COMPONENTS
        ):
            offenders.append(f"{fam.name}: te_quant_auto={scheme!r} is not hosted")
    assert not offenders, offenders


@pytest.mark.parametrize("name", sorted(EXPECTED_DEFAULTS))
def test_an_unset_request_resolves_to_the_hosted_encoder(name):
    scheme, repo, filename = EXPECTED_DEFAULTS[name]
    fam = _family(name)
    for raw in (None, "", "auto", "AUTO"):
        assert video_te_quant_request(fam, raw) == (scheme, True), raw
    sources = tpq.te_prequant_sources_for_base(
        fam, fam.base_repo, te_quant_mode = video_te_quant_request(fam, None)[0], target = CUDA_BF16
    )
    source = sources["text_encoder"]
    assert (source.kind, source.location, source.filename) == ("repo", repo, filename)


@pytest.mark.parametrize("name", sorted(EXPECTED_DEFAULTS))
@pytest.mark.parametrize("raw", ["none", "off", "Off"])
def test_none_and_off_pin_the_dense_encoder(name, raw):
    fam = _family(name)
    mode, auto = video_te_quant_request(fam, raw)
    assert not auto
    assert (
        tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = mode, target = CUDA_BF16)
        == {}
    )


def test_an_explicit_scheme_pins_itself_and_is_not_marked_automatic():
    fam = _family("wan2.2-ti2v-5b")
    assert video_te_quant_request(fam, "int8") == ("int8", False)
    assert video_te_quant_request(fam, "fp8") == ("fp8", False)


@pytest.mark.parametrize("name", ["minimax-h3", "ltx-2.5"])
def test_families_with_their_own_tristate_get_the_raw_request(name):
    """H3 and LTX-2.5 read the raw request themselves (unset vs none), so nothing is folded for them."""
    fam = _family(name)
    assert fam.te_quant_auto is None
    for raw in (None, "none", "int8"):
        assert video_te_quant_request(fam, raw) == (raw, False)


@pytest.mark.parametrize("name", ["cosmos3-nano", "cosmos3-edge"])
def test_a_cosmos3_family_has_no_text_encoder_to_default(name):
    # No text encoder, so nothing hosted to default to: every request comes back as given.
    fam = _family(name)
    for raw in (None, "", "auto", "none", "fp8"):
        for speed_mode in ("default", "off"):
            got = video_te_quant_request(fam, raw, base = fam.base_repo, speed_mode = speed_mode)
            assert got == (raw, False), (raw, speed_mode)
    for mode in (None, "fp8", "int8"):
        assert (
            tpq.te_prequant_sources_for_base(
                fam, fam.base_repo, te_quant_mode = mode, target = CUDA_BF16
            )
            == {}
        )


def test_unsupported_hardware_gets_no_hosted_encoder_for_the_default():
    fam = _family("wan2.2-ti2v-5b")
    mps = types.SimpleNamespace(device = "mps", dtype = torch.bfloat16)
    mode, _ = video_te_quant_request(fam, None)
    assert (
        tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = mode, target = mps) == {}
    )


@pytest.mark.parametrize("a", WAN_BASES)
@pytest.mark.parametrize("b", WAN_BASES)
def test_every_wan_base_takes_the_one_umt5_file(a, b):
    assert tpq.te_base_equivalent(a, b)


def test_wan_is_not_equivalent_to_other_encoders():
    assert not tpq.te_base_equivalent(WAN_BASES[0], "Lightricks/LTX-2")
    assert not tpq.te_base_equivalent(WAN_BASES[2], "Qwen/Qwen-Image")


def test_wan_int8_resolves_the_convrot_file_then_the_fp8_one():
    for name in ("wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "wan2.1-t2v-1.3b", "wan2.1-t2v-14b"):
        source = tpq.resolve_te_prequant_source(_family(name), "text_encoder", "int8")
        assert source.location == WAN_REPO
        assert tpq.te_candidate_filenames(source) == (
            "Wan2.2-TI2V-5B-text_encoder-INT8-ConvRot.safetensors",
            "Wan2.2-TI2V-5B-text_encoder-FP8.safetensors",
            "Wan2.2-TI2V-5B-text_encoder-FP8.pt",
        ), name


def test_wan_int8_needs_no_lm_head_trim(monkeypatch):
    """UMT5EncoderModel has no lm_head, so the keep-lm_head switch cannot push it onto the fp8 file."""
    from core.inference.diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV

    monkeypatch.setenv(KEEP_LM_HEAD_ENV, "1")
    source = tpq.resolve_te_prequant_source(_family("wan2.2-ti2v-5b"), "text_encoder", "int8")
    assert source.filename == "Wan2.2-TI2V-5B-text_encoder-INT8-ConvRot.safetensors"


# ---- the int8 ConvRot runtime on a UMT5 encoder ---------------------------------------------------------------------

GROUP = 256
_UMT5_LINEAR = re.compile(
    r"^encoder\.block\.\d+\.layer\.(0\.SelfAttention\.[qkvo]|1\.DenseReluDense\.(wi_0|wi_1|wo))\.weight$"
)


def _umt5_config():
    from transformers import UMT5Config
    return UMT5Config(
        vocab_size = 64,
        d_model = 256,
        d_kv = 64,
        d_ff = 512,
        num_layers = 2,
        num_heads = 4,
        feed_forward_proj = "gated-gelu",
        relative_attention_num_buckets = 8,
        dropout_rate = 0.0,
    )


def _umt5(seed = 0):
    from transformers import UMT5EncoderModel

    torch.manual_seed(seed)
    model = UMT5EncoderModel(_umt5_config()).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std = 0.05)
    return model


def _int8_state(encoder, *, keep_wo: bool):
    blob = torch.tensor(
        list(
            json.dumps(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}
            ).encode()
        ),
        dtype = torch.uint8,
    )
    state = {}
    for k, v in encoder.state_dict().items():
        if _UMT5_LINEAR.match(k) and not (keep_wo and ".wo." in k):
            q, s = tpq.quantize_int8_convrot_weight(v, GROUP)
            state[k], state[k[:-7] + ".weight_scale"], state[k[:-7] + ".comfy_quant"] = (
                q,
                s,
                blob.clone(),
            )
        else:
            state[k] = v.detach().clone().float()
    return state


def _dequantized(encoder, state):
    from core.inference.diffusion_convrot import build_convrot_hadamard

    h = build_convrot_hadamard(GROUP, dtype = torch.float32)
    sd = encoder.state_dict()
    for k in list(sd):
        if k + "_scale" in state:
            q, s = state[k].float(), state[k + "_scale"]
            o, i = q.shape
            sd[k] = ((q * s).reshape(o, i // GROUP, GROUP) @ h).reshape(o, i)
    encoder.load_state_dict(sd)
    return encoder


def _hidden(encoder, ids):
    with torch.no_grad():
        return encoder(input_ids = ids, attention_mask = torch.ones_like(ids)).last_hidden_state


@pytest.mark.parametrize("keep_wo", [False, True])
def test_a_umt5_int8_convrot_file_builds_and_matches_its_dequantized_weights(keep_wo):
    from transformers import UMT5EncoderModel

    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class

    reference = _umt5(seed = 3)
    state = _int8_state(reference, keep_wo = keep_wo)
    encoder = tpq._build_int8_convrot_encoder(
        UMT5EncoderModel, _umt5_config(), dict(state), dtype = torch.float32, trim = False
    )
    cls = _int8_convrot_linear_class()
    swapped = sum(isinstance(m, cls) for m in encoder.encoder.block.modules())
    assert swapped == 2 * (6 if keep_wo else 7)
    ids = torch.randint(0, 64, (1, 11))
    expected = _hidden(_dequantized(_umt5(seed = 3), state), ids)
    torch.testing.assert_close(_hidden(encoder, ids), expected, rtol = 1e-4, atol = 1e-4)


def test_a_dense_kept_umt5_wo_stays_float32_under_a_bf16_load():
    """T5's ``wo`` is in _keep_in_fp32_modules (it overflows in low precision); a dense one keeps float32 storage."""
    from transformers import UMT5EncoderModel

    state = _int8_state(_umt5(seed = 1), keep_wo = True)
    encoder = tpq._build_int8_convrot_encoder(
        UMT5EncoderModel, _umt5_config(), dict(state), dtype = torch.bfloat16, trim = False
    )
    ff = encoder.encoder.block[0].layer[1].DenseReluDense
    assert ff.wo.weight.dtype == torch.float32
    assert encoder.shared.weight.dtype == torch.bfloat16


def test_a_umt5_file_storing_the_tied_embedding_once_loads_tied():
    """The hosted UMT5 files hold shared.weight only, like the dense shards; the alias is filled from the model's tie."""
    from transformers import UMT5EncoderModel

    state = _int8_state(_umt5(seed = 2), keep_wo = False)
    state.pop("encoder.embed_tokens.weight")
    encoder = tpq._build_int8_convrot_encoder(
        UMT5EncoderModel, _umt5_config(), dict(state), dtype = torch.float32, trim = False
    )
    assert encoder.encoder.embed_tokens.weight.data_ptr() == encoder.shared.weight.data_ptr()
    ids = torch.randint(0, 64, (1, 7))
    expected = _hidden(_dequantized(_umt5(seed = 2), state), ids)
    torch.testing.assert_close(_hidden(encoder, ids), expected, rtol = 1e-4, atol = 1e-4)


def test_fill_tied_aliases_leaves_a_head_alone():
    model = types.SimpleNamespace(
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight", "b.weight": "a.weight"},
        named_parameters = lambda remove_duplicate = False: iter(()),
    )
    state = {"model.embed_tokens.weight": torch.zeros(1), "a.weight": torch.ones(1)}
    tpq._fill_tied_aliases(model, state)
    assert "lm_head.weight" not in state
    assert state["b.weight"] is state["a.weight"]


def test_a_umt5_file_missing_a_projection_is_refused():
    from transformers import UMT5EncoderModel

    state = _int8_state(_umt5(), keep_wo = False)
    name = "encoder.block.0.layer.0.SelfAttention.q"
    q, s = state.pop(name + ".weight"), state.pop(name + ".weight_scale")
    state.pop(name + ".comfy_quant")
    state[name + ".weight"] = (q.float() * s).float()
    with pytest.raises(ValueError, match = "not quantized"):
        tpq._build_int8_convrot_encoder(
            UMT5EncoderModel, _umt5_config(), state, dtype = torch.float32, trim = False
        )


def _converter():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts" / "convert_prequant_to_safetensors.py"
    spec = importlib.util.spec_from_file_location("convert_prequant_to_safetensors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_converter_can_store_a_tied_embedding_once(tmp_path, monkeypatch):
    """--drop-tied-copies writes UMT5's shared/embed_tokens pair once, the round trip still verifies, and the fp8
    loader fills the alias back from the model's own tie."""
    pytest.importorskip("diffusers")  # _cast_fp8 installs diffusers' layerwise hooks
    import core.inference.prequant_safetensors as ps
    from core.inference.diffusion_precision import _cast_fp8
    from core.inference.diffusion_prequant import ALLOW_LOCAL_PREQUANT_PATH_ENV

    conv = _converter()
    encoder = _umt5(seed = 4)
    target = types.SimpleNamespace(dtype = torch.float32)
    _cast_fp8(encoder, target)
    state = encoder.state_dict()
    assert conv.tied_copies(state) == {"encoder.embed_tokens.weight": "shared.weight"}
    src = tmp_path / "Wan-text_encoder-FP8.pt"
    torch.save(
        {
            "format": tpq.TE_PREQUANT_FORMAT,
            "state_dict": state,
            "metadata": {
                "scheme": "fp8",
                "component": "text_encoder",
                "base_model_id": WAN_BASES[0],
                "te_class": "UMT5EncoderModel",
            },
        },
        src,
    )
    rec = conv.convert(str(src), str(tmp_path), repo = None, revision = None, sha = None, drop_tied = True)
    assert rec["verified_bit_identical"] and rec["tied_copies_dropped"]
    assert (
        "encoder.embed_tokens.weight"
        not in ps.load_plain_prequant_safetensors(rec["dst"])["state_dict"]
    )
    plain = conv.convert(str(src), str(tmp_path / "plain"), repo = None, revision = None, sha = None)
    assert plain["dst_bytes"] > rec["dst_bytes"]

    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: _umt5_config()
    )
    monkeypatch.setenv(ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    loaded = tpq.load_prequant_text_encoder(
        WAN_BASES[2],
        "text_encoder",
        tpq.TePrequantSource(kind = "path", location = rec["dst"]),
        dtype = torch.float32,
        scheme = "fp8",
        local_files_only = True,
    )
    assert loaded is not None
    assert loaded.encoder.embed_tokens.weight.data_ptr() == loaded.shared.weight.data_ptr()
    ids = torch.randint(0, 64, (1, 9))
    torch.testing.assert_close(_hidden(loaded, ids), _hidden(encoder, ids), rtol = 0, atol = 0)


@pytest.mark.parametrize("name", ["hunyuanvideo-1.5", "hunyuanvideo-1.5-720p"])
def test_hunyuanvideo_default_fp8_leaves_the_byt5_encoder_dense(name, monkeypatch):
    import inspect

    import core.inference.diffusion_precision as dp
    import core.inference.video as video

    fam = next(f for f in _FAMILIES if f.name == name)
    assert fam.te_quant_auto_encoders == ("text_encoder",)
    cast: list = []
    monkeypatch.setattr(dp, "te_quant_supported", lambda *a, **k: True)
    monkeypatch.setattr(dp, "_cast_fp8", lambda enc, tgt: cast.append(enc))
    qwen, byt5 = object(), object()
    pipe = types.SimpleNamespace(text_encoder = qwen, text_encoder_2 = byt5)
    outcome = dp.quantize_text_encoders(
        pipe, CUDA_BF16, mode = "fp8", family = name, encoders = fam.te_quant_auto_encoders
    )
    assert outcome.mode == "fp8" and not outcome.partial
    assert cast == [qwen]
    # An explicit fp8 (no encoders limit) still covers both towers.
    cast.clear()
    dp.quantize_text_encoders(pipe, CUDA_BF16, mode = "fp8", family = name)
    assert cast == [qwen, byt5]
    # The loader passes the limit only for the automatic pick.
    src = inspect.getsource(video)
    call = src[src.index("text_encoder_outcome = quantize_text_encoders(") :][:600]
    assert 'getattr(fam, "te_quant_auto_encoders", None)' in call
    assert "if text_encoder_quant_auto" in call


MPS_BF16 = types.SimpleNamespace(device = "mps", dtype = torch.bfloat16)


def test_an_explicit_wan_int8_on_mps_reaches_the_hosted_convrot_file():
    # MPS runs the hosted int8 ConvRot encoder without fp8, so the gate must not rewrite int8 to fp8 and refuse it.
    import core.inference.video as v
    v._assert_video_precision_for_target(
        _family("wan2.2-ti2v-5b"), MPS_BF16, model_kind = "pipeline", text_encoder_quant = "int8"
    )
    # A family with no hosted int8 file still meets the generic gate.
    with pytest.raises(RuntimeError, match = "text_encoder_quant"):
        v._assert_video_precision_for_target(
            _family("ltx-2"), MPS_BF16, model_kind = "pipeline", text_encoder_quant = "int8"
        )


def test_the_builder_quantizes_a_non_gated_t5_feed_forward():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts" / "build_te_int8_convrot_checkpoint.py"
    spec = importlib.util.spec_from_file_location("build_te_int8_convrot_checkpoint", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    pattern = builder._linear_pattern("T5EncoderModel")
    for proj in ("wi", "wi_0", "wi_1", "wo"):
        assert pattern.match(f"encoder.block.3.layer.1.DenseReluDense.{proj}.weight"), proj
    assert not pattern.match("encoder.block.3.layer.1.DenseReluDense.wix.weight")


def test_the_transformers_builder_honours_keep_dense(tmp_path):
    import importlib.util
    from pathlib import Path

    import torch
    import transformers
    from safetensors import safe_open

    path = Path(__file__).resolve().parents[3] / "scripts" / "build_te_int8_convrot_checkpoint.py"
    spec = importlib.util.spec_from_file_location("build_te_int8_convrot_checkpoint", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    config = transformers.T5Config(
        vocab_size = 32,
        d_model = 64,
        d_kv = 16,
        d_ff = 128,
        num_layers = 1,
        num_heads = 4,
        feed_forward_proj = "gated-gelu",
    )
    src = tmp_path / "te"
    transformers.T5EncoderModel(config).save_pretrained(src)
    out = tmp_path / "te-int8.safetensors"
    builder.main(
        [
            "--base",
            "someone/t5",
            "--family",
            "flux.1",
            "--src",
            str(src),
            "--out",
            str(out),
            "--load",
            "transformers",
            "--te-class",
            "T5EncoderModel",
            "--keep-dense",
            "wo",
            "--group",
            "64",
            "--device",
            "cpu",
        ]
    )
    with safe_open(str(out), framework = "pt") as handle:
        keys = set(handle.keys())
        wo = handle.get_tensor("encoder.block.0.layer.1.DenseReluDense.wo.weight")
    assert "encoder.block.0.layer.1.DenseReluDense.wi_0.weight_scale" in keys
    assert "encoder.block.0.layer.1.DenseReluDense.wo.weight_scale" not in keys
    # T5 keeps ``wo`` in float32, and a dense one is stored that way.
    assert wo.dtype == torch.float32


def test_a_wan_load_claims_the_hosted_encoder_repo_while_it_fetches():
    # The hosted file lives in a third repo; without the claim the delete-cached guard admits deleting it mid-load.
    import inspect

    import core.inference.video as v

    src = inspect.getsource(v.VideoBackend)
    claim = src[src.index("te_repos = tuple(") :][:400]
    assert "te_sources" in claim and 'kind", None) == "repo"' in claim
    assert "self._loading.asset_repos = tuple(" in src[src.index("te_repos = tuple(") :][:700]


@pytest.mark.parametrize(
    "base, hosted",
    [
        ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", True),
        ("/hf/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/abc123", True),
        ("/packs/custom/Wan2.2-TI2V-5B-Diffusers", False),
        ("someone/Wan2.2-TI2V-5B-Diffusers", False),
    ],
)
def test_the_automatic_encoder_needs_the_exact_official_base(base, hosted):
    # A custom pack that only shares the folder name keeps its own encoder weights.
    mode, auto = video_te_quant_request(_family("wan2.2-ti2v-5b"), None, base = base)
    assert (mode, auto) == (("fp8", True) if hosted else (None, False))
    # An explicit scheme is never second-guessed.
    assert video_te_quant_request(_family("wan2.2-ti2v-5b"), "fp8", base = base) == ("fp8", False)


@pytest.mark.parametrize("name", ["wan2.2-ti2v-5b", "ltx-2", "ltx-2.5"])
def test_speed_off_keeps_the_released_encoder_unless_a_scheme_is_named(name):
    fam = _family(name)
    assert video_te_quant_request(fam, None, speed_mode = "off") == ("none", False)
    assert video_te_quant_request(fam, "auto", speed_mode = "off") == ("none", False)
    assert video_te_quant_request(fam, "int8", speed_mode = "off") == ("int8", False)
    if name != "ltx-2.5":
        assert video_te_quant_request(fam, None, speed_mode = "default")[1] is True


def test_wan_int8_budget_covers_the_measured_file():
    # 6,735,404,944 bytes int8 against 11,361,851,208 dense (Wan2.2-TI2V-5B UMT5).
    for name in ("wan2.2-ti2v-5b", "wan2.1-t2v-14b"):
        assert 11_361_851_208 * tpq.te_int8_convrot_budget_scale(name) >= 6_735_404_944
    assert tpq.te_int8_convrot_budget_scale("qwen-image-2.1") == tpq.TE_INT8_CONVROT_BUDGET_SCALE


def test_ltx25_budgets_the_dense_peak_unless_the_hosted_encoder_applies():
    fam = types.SimpleNamespace(name = "ltx-2.5", bf16_components_gb = (40.0, 23.9))
    hosted = tpq.te_prequant_budget_scale(
        fam, te_quant_mode = None, target = None, base = "Lightricks/LTX-2.5-Diffusers"
    )
    assert hosted < 1.0
    # A custom pack quantizes the dense encoder at load: its peak is the dense one.
    assert (
        tpq.te_prequant_budget_scale(
            fam, te_quant_mode = None, target = None, base = "someone/ltx25-pack"
        )
        == 1.0
    )
    assert (
        tpq.te_prequant_budget_scale(
            fam, te_quant_mode = "none", target = None, base = "Lightricks/LTX-2.5-Diffusers"
        )
        == 1.0
    )


def test_drop_tied_copies_keeps_a_tied_lm_head(tmp_path):
    import core.inference.prequant_safetensors as ps

    conv = _converter()
    embed = torch.randn(8, 4)
    shared = torch.randn(8, 4)
    state = {
        "shared.weight": shared,
        "encoder.embed_tokens.weight": shared,
        "model.embed_tokens.weight": embed,
        "lm_head.weight": embed,
    }
    src = tmp_path / "te-FP8.pt"
    torch.save(
        {
            "format": tpq.TE_PREQUANT_FORMAT,
            "state_dict": state,
            "metadata": {
                "scheme": "fp8",
                "component": "text_encoder",
                "base_model_id": WAN_BASES[0],
            },
        },
        src,
    )
    rec = conv.convert(str(src), str(tmp_path), repo = None, revision = None, sha = None, drop_tied = True)
    # The loader fills the embedding alias from the model's tie, never a head one.
    assert rec["tied_copies_dropped"] == {"encoder.embed_tokens.weight": "shared.weight"}
    assert "lm_head.weight" in ps.load_plain_prequant_safetensors(rec["dst"])["state_dict"]
