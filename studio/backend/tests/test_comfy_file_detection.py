# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI file detection: separator-insensitive family names and header-based DiT recognition.

The fixture holds the real tensor names/shapes (trimmed to the first block of each stack) of 216
ComfyUI-ecosystem files (Comfy-Org repackages, Kijai fp8, Lightricks, BFL, city96 / QuantStack /
unsloth GGUFs) plus the diffusers layout of every supported transformer class. Expectations are
reviewed per file; the name columns also record what main detected before this change."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest

from core.inference import diffusion_content as dc
from core.inference import diffusion_families as df
from core.inference import video_families as vf
from core.inference.family_name_match import normalize_family_name, token_in_name, token_length

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "comfy_checkpoint_headers.json").read_text()
)
_AUDIT = {k: v for k, v in _FIXTURE.items() if not k.startswith("diffusers/")}
_DIFFUSERS = {k: v for k, v in _FIXTURE.items() if k.startswith("diffusers/")}


# --------------------------------------------------------------------------------------------
# Name matching
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Qwen_Image_2.1_BF16", "qwen-image-2-1-bf16"),
        ("flux-2--klein  4b", "flux-2-klein-4b"),
        ("Wan2_2-TI2V-5B", "wan2-2-ti2v-5b"),
        ("a/b_c\\d.e", "a/b-c\\d-e"),
    ],
)
def test_normalize_family_name(text, expected):
    assert normalize_family_name(text) == expected


def test_token_in_name_is_separator_insensitive_but_segment_bounded():
    assert token_in_name("qwen-image", "qwen_image_bf16.safetensors")
    assert token_in_name("flux.2-klein", "flux-2-klein-4b.safetensors")
    assert token_in_name("wan2.2-ti2v-5b", "Wan2_2-TI2V-5B_fp8_e4m3fn_scaled_KJ.safetensors")
    assert token_in_name("lumina-2", "ComfyUI/models/checkpoints/lumina_2.safetensors")
    assert not token_in_name("kontext", "kontextual.safetensors")
    assert not token_in_name("ltx", "voltxa.safetensors")
    # a token never spans a path separator
    assert not token_in_name("qwen-image", "qwen/image.safetensors")


@pytest.mark.parametrize(
    "name, image, video",
    [
        ("wan2.2_ti2v_5B_fp16.safetensors", None, "wan2.2-ti2v-5b"),
        ("Wan2_2-TI2V-5B_fp8_e5m2_scaled_KJ.safetensors", None, "wan2.2-ti2v-5b"),
        ("wan2.2_t2v_low_noise_14B_fp16.safetensors", None, "wan2.2-t2v-a14b"),
        ("wan2.2_i2v_low_noise_14B_fp16.safetensors", None, None),
        ("flux-2-klein-4b.safetensors", "flux.2-klein", None),
        ("flux-2-klein-base-4b.safetensors", "flux.2-klein", None),
        ("Flux-2-Klein-9B-KV-Q4_K_M.gguf", "flux.2-klein", None),
        ("flux2_dev_fp8mixed.safetensors", "flux.2-dev", None),
        ("lumina_2.safetensors", "lumina-2", None),
        ("lumina_2_model_bf16.safetensors", "lumina-2", None),
        ("qwen_image_2.1_bf16.safetensors", "qwen-image-2.1", None),
        ("qwen_image_2512_bf16.safetensors", "qwen-image", None),
        ("qwen_image_edit_2511_bf16.safetensors", "qwen-image-edit", None),
        ("qwen_image_layered_bf16.safetensors", "qwen-image-layered", None),
        ("hunyuanvideo1.5_720p_t2v_fp16.safetensors", None, "hunyuanvideo-1.5-720p"),
    ],
)
def test_comfy_spellings_detect(name, image, video):
    fam = df.detect_family(name)
    vfam = vf.detect_video_family(name)
    assert (fam.name if fam else None) == image
    assert (vfam.name if vfam else None) == video


def test_override_accepts_any_separator_spelling():
    assert df.detect_family("x", override = "flux_2_klein").name == "flux.2-klein"
    assert df.detect_family("x", override = "Qwen Image 2.1").name == "qwen-image-2.1"
    assert vf.detect_video_family("x", override = "wan2_2_ti2v_5b").name == "wan2.2-ti2v-5b"
    assert df.detect_family("x", override = "not-a-family") is None


def _old_token_in_needle(token: str, needle: str) -> bool:
    """The matcher main shipped before separator normalisation, kept to prove nothing regressed."""
    return re.search(r"(?:^|[-_./\\])" + re.escape(token) + r"(?:$|[-_./\\])", needle) is not None


def _old_best(families, needle):
    best = None
    for fam in families:
        for token in (fam.name, *fam.aliases):
            if _old_token_in_needle(token, needle) and (best is None or len(token) > best[1]):
                best = (fam, len(token))
    return best[0] if best else None


def _new_best(families, needle):
    best = None
    for fam in families:
        for token in (fam.name, *fam.aliases):
            if token_in_name(token, needle) and (best is None or token_length(token) > best[1]):
                best = (fam, token_length(token))
    return best[0] if best else None


def _corpus() -> set[str]:
    names = set()
    for fams in (df._FAMILIES, vf._FAMILIES):
        for fam in fams:
            for token in (fam.name, *fam.aliases):
                names |= {token, f"org/{token}-gguf", f"{token}-Q4_K_M.gguf", f"org/{token.upper()}"}
            if getattr(fam, "base_repo", None):
                names.add(fam.base_repo)
    for rel in _AUDIT:
        base = rel.split("/")[-1]
        names |= {base, rel, f"ComfyUI/models/diffusion_models/{base}"}
    return names


# Names main matched to one family and this change deliberately matches to another: ComfyUI's
# ``qwen_image_2.1_*`` files were read as plain Qwen-Image because ``qwen-image-2.1`` needed hyphens.
_INTENDED_RENAMES = {("qwen-image", "qwen-image-2.1")}


def test_every_previously_detected_name_keeps_its_family():
    """Regression over the whole alias table and the audit's file names: separator normalisation
    only ADDS matches (None -> family), never moves a name to another family, except the
    documented Qwen-Image-2.1 correction."""
    changed = []
    for needle in sorted(_corpus()):
        low = needle.lower()
        old_img = _old_best(df._FAMILIES, low)
        new_img = df._best_family_match(low)
        old_vid = _old_best(vf._FAMILIES, low)
        new_vid = _new_best(vf._FAMILIES, needle)
        if old_img is not None and new_img is not old_img:
            pair = (old_img.name, new_img.name if new_img else None)
            if pair not in _INTENDED_RENAMES:
                changed.append((needle, pair))
        if old_vid is not None and new_vid is not old_vid:
            changed.append((needle, (old_vid.name, new_vid.name if new_vid else None)))
    assert changed == []


def test_audit_name_table_before_vs_after():
    """The recorded before/after name verdicts for the 216 audit files: no regression, the listed
    gaps closed."""
    regressions, gained, renamed = [], 0, 0
    for rel, row in _AUDIT.items():
        base = rel.split("/")[-1]
        fam = df.detect_family(base)
        vfam = vf.detect_video_family(base)
        assert (fam.name if fam else None) == row["name_image"], rel
        assert (vfam.name if vfam else None) == row["name_video"], rel
        for before, after in ((row["name_image_before"], row["name_image"]), (row["name_video_before"], row["name_video"])):
            if before and after != before:
                if (before, after) in _INTENDED_RENAMES:
                    renamed += 1
                else:
                    regressions.append((rel, before, after))
            elif not before and after:
                gained += 1
    assert regressions == []
    assert gained >= 20 and renamed >= 2


def test_generation_defaults_read_comfy_names():
    assert df.default_generation_params("z_image_turbo_bf16.safetensors") == (8, 0.0)
    assert df.default_generation_params("hidream_i1_dev_fp8.safetensors") == (28, 0.0)
    assert df.default_generation_params("qwen_image_2512_bf16.safetensors") == (50, 4.0)
    assert df.default_generation_params("flux-2-klein-base-4b.safetensors") == (20, 5.0)
    # unchanged spellings keep their row
    assert df.default_generation_params("Tongyi-MAI/Z-Image-Turbo") == (8, 0.0)
    assert vf.video_generation_variant("Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors") == "a14b"


# --------------------------------------------------------------------------------------------
# Header classification
# --------------------------------------------------------------------------------------------


def _audit_role_from_path(rel: str) -> str | None:
    if "/text_encoders/" in rel:
        return dc.ROLE_TEXT_ENCODER
    if "/vae/" in rel:
        return dc.ROLE_VAE
    if "/loras/" in rel:
        return dc.ROLE_LORA
    return None


@pytest.mark.parametrize("rel", sorted(_FIXTURE))
def test_header_classification(rel):
    row = _FIXTURE[rel]
    info = dc.classify_tensors(row["shapes"], row["meta"])
    assert (info.role, info.family) == (row["expect_role"], row["expect_family"])
    folder_role = _audit_role_from_path(rel)
    if folder_role:
        assert info.role == folder_role
    if info.family:
        assert info.page == dc.family_page(info.family)
        known = {f.name for f in df._FAMILIES} | {f.name for f in vf._FAMILIES}
        assert info.family in known


def test_fixture_covers_every_single_file_family_in_both_layouts():
    comfy = {r["expect_family"] for r in _AUDIT.values() if r["expect_role"] == dc.ROLE_DIT}
    diffusers = {r["expect_family"] for r in _DIFFUSERS.values()}
    for fam in (
        "flux.1", "flux.2-klein", "flux.2-dev", "qwen-image", "qwen-image-2.1", "z-image", "krea-2",
        "lumina-2", "hunyuanimage-2.1", "hidream-i1", "wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "ltx-2",
        "hunyuanvideo-1.5", "minimax-h3",
    ):
        assert fam in comfy, fam
    assert len(diffusers) >= 14


def test_unsupported_architectures_are_dits_without_a_family():
    wan_i2v = [r for k, r in _AUDIT.items() if re.search(r"(?<!t)i2v", k.lower()) and "wan" in k.lower()]
    assert wan_i2v and all(r["expect_role"] == dc.ROLE_DIT and r["expect_family"] is None for r in wan_i2v)


# --------------------------------------------------------------------------------------------
# File-level behaviour: header-only reads, refusal, family resolution for a pick
# --------------------------------------------------------------------------------------------


def _write_safetensors(path: Path, shapes: dict, meta: dict | None = None) -> Path:
    """Header-only safetensors (data section absent): the classifier never reads past the header."""
    header = {k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}
    if meta:
        header["__metadata__"] = meta
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _write_gguf(path: Path, shapes: dict, arch: str) -> Path:
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    out = struct.pack("<IIQQ", 0x46554747, 3, len(shapes), 1)
    out += s("general.architecture") + struct.pack("<I", 8) + s(arch)
    for name, shape in shapes.items():
        dims = list(reversed(shape))
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", 0, 0)
    path.write_bytes(out)
    return path


def _row(suffix: str) -> dict:
    return next(v for k, v in _AUDIT.items() if k.endswith(suffix))


def test_renamed_comfy_dit_resolves_by_content(tmp_path):
    _write_safetensors(tmp_path / "my_model.safetensors", _row("z_image_int8_convrot.safetensors")["shapes"])
    assert df.detect_family("my_model.safetensors") is None
    fam = df.detect_family_for_pick(str(tmp_path), "my_model.safetensors")
    assert fam is not None and fam.name == "z-image"
    dc.assert_local_pick_is_dit(str(tmp_path), "my_model.safetensors", "image")


def test_misnamed_dit_follows_its_header(tmp_path):
    _write_safetensors(tmp_path / "flux1-dev-mine.safetensors", _row("flux-2-klein-4b.safetensors")["shapes"])
    assert df.detect_family("flux1-dev-mine.safetensors").name == "flux.1"
    assert df.detect_family_for_pick(str(tmp_path), "flux1-dev-mine.safetensors").name == "flux.2-klein"


def test_name_breaks_ties_between_same_architecture_variants(tmp_path):
    shapes = _row("qwen_image_edit_2509_bf16.safetensors")["shapes"]
    _write_safetensors(tmp_path / "qwen_image_edit_2509_bf16.safetensors", shapes)
    _write_safetensors(tmp_path / "qwen_image_2512_bf16.safetensors", shapes)
    _write_safetensors(tmp_path / "anything.safetensors", shapes)
    pick = lambda f: df.detect_family_for_pick(str(tmp_path), f).name  # noqa: E731
    assert pick("qwen_image_edit_2509_bf16.safetensors") == "qwen-image-edit"
    assert pick("qwen_image_2512_bf16.safetensors") == "qwen-image"
    assert pick("anything.safetensors") == "qwen-image"
    kontext = _row("flux1-dev-kontext_fp8_scaled.safetensors")["shapes"]
    _write_safetensors(tmp_path / "flux1-kontext-dev.safetensors", kontext)
    assert pick("flux1-kontext-dev.safetensors") == "flux.1-kontext"


@pytest.mark.parametrize(
    "suffix, fragment",
    [
        ("Qwen-Image_ComfyUI/split_files/vae/qwen_image_vae.safetensors", "is a VAE"),
        ("clip_l_hidream.safetensors", "is a CLIP text encoder"),
        ("qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors", "text encoder"),
        ("krea2_darkbrush.safetensors", "is a LoRA adapter"),
        ("flux_shakker_labs_union_pro-fp8_e4m3fn.safetensors", "is a ControlNet"),
    ],
)
def test_non_dit_named_for_a_family_is_refused(tmp_path, suffix, fragment):
    name = suffix.split("/")[-1]
    _write_safetensors(tmp_path / name, _row(suffix)["shapes"])
    with pytest.raises(ValueError, match = re.escape(fragment)) as err:
        dc.assert_local_pick_is_dit(str(tmp_path), name, "image")
    assert "not a diffusion model" in str(err.value) and name in str(err.value)
    assert df.detect_family_for_pick(str(tmp_path), name) is None


def test_gguf_text_encoder_and_dit(tmp_path):
    _write_gguf(tmp_path / "renamed.gguf", _row("qwen-image-Q4_K_M.gguf")["shapes"], "qwen_image")
    _write_gguf(tmp_path / "qwen_image_te.gguf", {"token_embd.weight": [151936, 3584]}, "qwen2vl")
    assert df.detect_family_for_pick(str(tmp_path), "renamed.gguf").name == "qwen-image"
    with pytest.raises(ValueError, match = "text encoder"):
        dc.assert_local_pick_is_dit(str(tmp_path), "qwen_image_te.gguf", "image")


def test_wrong_page_is_refused_with_the_right_page(tmp_path):
    _write_safetensors(tmp_path / "flux_like.safetensors", _row("wan2.2_ti2v_5B_fp16.safetensors")["shapes"])
    with pytest.raises(ValueError, match = "Video page"):
        dc.assert_local_pick_is_dit(str(tmp_path), "flux_like.safetensors", "image")
    dc.assert_local_pick_is_dit(str(tmp_path), "flux_like.safetensors", "video")
    assert df.detect_family_for_pick(str(tmp_path), "flux_like.safetensors") is None


def test_name_veto_is_not_revived_by_content(tmp_path):
    _write_safetensors(tmp_path / "flux1-dev-inpaint.safetensors", _row("flux1-dev.safetensors")["shapes"])
    assert df.detect_family_for_pick(str(tmp_path), "flux1-dev-inpaint.safetensors") is None


def test_unreadable_or_unknown_header_keeps_the_name_verdict(tmp_path):
    (tmp_path / "flux1-dev-broken.safetensors").write_bytes(b"\x00" * 4)
    _write_safetensors(tmp_path / "flux1-dev-odd.safetensors", {"foo.weight": [4, 4]})
    assert dc.inspect_checkpoint(str(tmp_path / "flux1-dev-broken.safetensors")).role == dc.ROLE_UNKNOWN
    assert dc.offer_as_dit(str(tmp_path / "flux1-dev-odd.safetensors"))
    for name in ("flux1-dev-broken.safetensors", "flux1-dev-odd.safetensors"):
        assert df.detect_family_for_pick(str(tmp_path), name).name == "flux.1"
        dc.assert_local_pick_is_dit(str(tmp_path), name, "image")


def test_remote_and_override_picks_are_untouched(tmp_path):
    assert df.detect_family_for_pick("Comfy-Org/Qwen-Image_ComfyUI", "qwen_image_vae.safetensors").name == "qwen-image"
    _write_safetensors(tmp_path / "x.safetensors", _row("flux-2-klein-4b.safetensors")["shapes"])
    assert df.detect_family_for_pick(str(tmp_path), "x.safetensors", "flux.1").name == "flux.1"
    dc.assert_local_pick_is_dit("Comfy-Org/Qwen-Image_ComfyUI", "qwen_image_vae.safetensors", "image")


def test_header_read_is_bounded_and_cached(tmp_path, monkeypatch):
    big = tmp_path / "huge.safetensors"
    big.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    assert dc.inspect_checkpoint(str(big)).role == dc.ROLE_UNKNOWN
    path = _write_safetensors(tmp_path / "z.safetensors", _row("z_image_bf16.safetensors")["shapes"])
    first = dc.inspect_checkpoint(str(path))
    monkeypatch.setattr(dc, "_read_safetensors_header", lambda p: pytest.fail("re-read a cached header"))
    assert dc.inspect_checkpoint(str(path)) is first


def test_local_pick_file_rejects_escaping_names(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"")
    assert dc.local_pick_file(str(tmp_path), "../a.safetensors") is None
    assert dc.local_pick_file(str(tmp_path), "a.safetensors") == str(tmp_path / "a.safetensors")
    assert dc.local_pick_file(str(tmp_path / "a.safetensors"), None) == str(tmp_path / "a.safetensors")
    assert dc.local_pick_file("org/repo", "a.safetensors") is None


def test_classifier_is_torch_free():
    import subprocess
    import sys

    code = (
        "import sys; import core.inference.diffusion_content, core.inference.family_name_match; "
        "assert 'torch' not in sys.modules and 'diffusers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check = True, cwd = Path(__file__).resolve().parents[1])


def test_qwen_image_21_header_wins_over_a_plain_qwen_image_name(tmp_path):
    """A Qwen-Image-2.1 DiT read as plain Qwen-Image is rebuilt at the wrong width (4096 vs 3072)."""
    _write_safetensors(tmp_path / "qwen_image_mine.safetensors", _row("qwen_image_2.1_bf16.safetensors")["shapes"])
    assert df.detect_family("qwen_image_mine.safetensors").name == "qwen-image"
    assert df.detect_family_for_pick(str(tmp_path), "qwen_image_mine.safetensors").name == "qwen-image-2.1"


def test_flux1_dev_krea_schnell_defaults(tmp_path):
    base = "black-forest-labs/FLUX.1-schnell"  # the flux.1 family's companion base
    assert df.default_generation_params("flux1-krea-dev_fp8_scaled.safetensors", base) == (20, 3.5)
    assert df.default_generation_params("flux1-dev-fp8.safetensors", base) == (20, 3.5)
    assert df.default_generation_params("flux1-schnell-fp8.safetensors", base) == (4, 0.0)
    assert df.default_generation_params("flux1-dev-kontext_fp8_scaled.safetensors", base) == (20, 2.5)
    assert df.default_generation_params("krea2_turbo_bf16.safetensors") == (8, 0.0)
    # renamed files: the keys separate schnell (no guidance_in) from dev / Krea-dev
    _write_safetensors(tmp_path / "a.safetensors", _row("flux1-krea-dev_fp8_scaled.safetensors")["shapes"])
    _write_safetensors(tmp_path / "b.safetensors", _row("Comfy-Org/flux1-schnell/flux1-schnell.safetensors")["shapes"])
    hint_a = dc.content_variant_hint(str(tmp_path), "a.safetensors")
    hint_b = dc.content_variant_hint(str(tmp_path), "b.safetensors")
    assert (hint_a, hint_b) == ("flux.1-dev", "flux.1-schnell")
    assert df.default_generation_params("a.safetensors", hint_a, str(tmp_path), base) == (20, 3.5)
    assert df.default_generation_params("b.safetensors", hint_b, str(tmp_path), base) == (4, 0.0)
    fam = df.detect_family("flux.1")
    assert df.comfy_flow_shift_for(fam, "a.safetensors", hint_a, base) == df.comfy_flow_shift_for(fam, "flux1-dev")
    assert df.comfy_flow_shift_for(fam, "b.safetensors", hint_b, base) is None
