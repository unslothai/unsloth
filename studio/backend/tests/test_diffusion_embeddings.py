# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Textual-inversion embeddings: discovery, A1111 trigger matching, and apply / restore on real tiny
SD 1.x and SDXL pipelines (bit-identical once restored, nothing touched when no trigger is named)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

from core.inference import diffusion_embeddings as de  # noqa: E402


@pytest.fixture
def emb_dir(tmp_path, monkeypatch):
    d = tmp_path / "embeddings"
    d.mkdir()
    monkeypatch.setattr(de, "embeddings_dir", lambda: d)
    return d


def _entry(
    stem,
    path = "/nonexistent",
    kind = None,
):
    return de.EmbeddingEntry(id = stem, path = path, kind = kind, vectors = 1, size_bytes = 0)


def test_classify_formats():
    assert de.classify({"clip_l": (8, 768), "clip_g": (8, 1280)}) == (de.KIND_SDXL, 8)
    assert de.classify({"emb_params": (4, 768)}) == (de.KIND_SD1, 4)
    assert de.classify({"<cat-toy>": (768,)}) == (de.KIND_SD1, 1)
    assert de.classify({"emb_params": (2, 1024)}) == (de.KIND_SD2, 2)
    assert de.classify({"clip_l": (8, 768), "clip_g": (4, 1280)}) == (None, 0)
    assert de.classify({"a": (1, 768), "b": (1, 768)}) == (None, 0)


def test_listing_reads_headers_and_filters_by_kind(emb_dir):
    save_file(
        {"clip_l": torch.zeros(2, 768), "clip_g": torch.zeros(2, 1280)},
        str(emb_dir / "EasyNegXL.safetensors"),
    )
    save_file({"emb_params": torch.zeros(3, 768)}, str(emb_dir / "EasyNegative.safetensors"))
    (emb_dir / "notes.txt").write_text("x")
    (emb_dir / "broken.safetensors").write_bytes(b"not a safetensors file")
    (emb_dir / "pickled.pt").write_bytes(b"x")
    rows = {e.id: (e.kind, e.vectors) for e in de.list_embeddings()}
    assert rows == {"EasyNegXL": (de.KIND_SDXL, 2), "EasyNegative": (de.KIND_SD1, 3)}
    assert [e.id for e in de.list_embeddings(de.KIND_SDXL)] == ["EasyNegXL"]


def test_triggers_follow_a1111_and_comfyui_syntax():
    entries = [_entry("EasyNegative"), _entry("bad-hands-5"), _entry("bad"), _entry("cat")]

    def hits(text):
        return sorted(e.id for e in de.find_triggers([text], entries))

    assert hits("blurry, easynegative, lowres") == ["EasyNegative"]
    assert hits("(embedding:EasyNegative:1.2)") == ["EasyNegative"]
    assert hits("bad-hands-5") == ["bad-hands-5"]  # longest stem wins, "bad" is not also triggered
    assert hits("bad, bad-hands-5") == ["bad", "bad-hands-5"]
    assert hits("concatenate a category") == []  # whole words only
    assert hits("") == []


def test_rewrite_replaces_only_triggers():
    applied = de.AppliedEmbeddings(
        pipe = None,
        tokenizers = [],
        encoders = [],
        tokens = {"easynegative": "<unsloth-ti-0-easynegative>"},
        used = ["EasyNegative"],
        skipped = [],
        pattern = de._trigger_pattern(["EasyNegative"]),
        marker_owners = [],
    )
    src = {
        "prompt": ["a cat", "EasyNegative cat"],
        "negative_prompt": "embedding:easynegative, blurry",
        "x": 1,
    }
    out = applied.rewrite(src)
    assert out["prompt"] == ["a cat", "<unsloth-ti-0-easynegative> cat"]
    assert out["negative_prompt"] == "<unsloth-ti-0-easynegative>, blurry"
    assert out["x"] == 1
    assert src["prompt"] == ["a cat", "EasyNegative cat"]  # the caller's kwargs are kept


def test_no_trigger_touches_nothing(emb_dir):
    save_file({"emb_params": torch.zeros(1, 768)}, str(emb_dir / "EasyNegative.safetensors"))

    class Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"pipe.{name} read without a trigger")

    assert (
        de.apply(Exploding(), de.KIND_SD1, {"prompt": ["a cat"], "negative_prompt": "blurry"})
        is None
    )
    assert de.apply(Exploding(), None, {"prompt": ["EasyNegative"]}) is None


def _tiny(repo, cls_name):
    diffusers = pytest.importorskip("diffusers")
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(repo)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny pipeline not reachable: {exc}")
    return getattr(diffusers, cls_name).from_pretrained(path, torch_dtype = torch.float32)


def _encode(pipe, prompt, negative):
    with torch.no_grad():
        out = pipe.encode_prompt(
            prompt,
            device = "cpu",
            num_images_per_prompt = 1,
            do_classifier_free_guidance = True,
            negative_prompt = negative,
        )
    return [t for t in out if isinstance(t, torch.Tensor)]


def _check_pipeline(pipe, kind, emb_dir, tensors_for):
    widths = [pipe.text_encoder.get_input_embeddings().embedding_dim]
    if kind == de.KIND_SDXL:
        widths.append(pipe.text_encoder_2.get_input_embeddings().embedding_dim)
    save_file(tensors_for(widths, 3), str(emb_dir / "EasyNeg.safetensors"))
    # A file for another text encoder (wrong width): skipped like A1111, never applied.
    save_file({"emb_params": torch.ones(2, widths[0] + 8)}, str(emb_dir / "OtherModel.safetensors"))
    originals = {attr: getattr(pipe, attr, None) for attr in ("tokenizer", "tokenizer_2")}
    embeds = {
        attr: getattr(pipe, attr).get_input_embeddings()
        for attr in ("text_encoder", "text_encoder_2")
        if getattr(pipe, attr, None) is not None
    }
    prompt, negative = "a photo of a cat", "blurry, EasyNeg, othermodel"
    base = _encode(pipe, prompt, negative)

    call = {"prompt": [prompt], "negative_prompt": negative}
    applied = de.apply(pipe, kind, call, marker_owners = ())
    assert applied is not None and applied.used == ["EasyNeg"] and applied.skipped == ["OtherModel"]
    assert pipe.__dict__["_unsloth_embeddings"][0][0] == "EasyNeg"
    rewritten = applied.rewrite(call)
    token = applied.tokens["easyneg"]
    assert token in rewritten["negative_prompt"] and "othermodel" in rewritten["negative_prompt"]
    # Each tokenizer the pipeline encodes with carries the 3-vector expansion, past the base vocabulary.
    for tok_attr, enc_attr in (("tokenizer", "text_encoder"), ("tokenizer_2", "text_encoder_2")):
        tok = getattr(pipe, tok_attr, None)
        if tok is None or enc_attr not in embeds:
            continue
        ids = tok(pipe.maybe_convert_prompt(rewritten["negative_prompt"], tok)).input_ids
        base_n = embeds[enc_attr].num_embeddings
        assert sum(1 for i in ids if i >= base_n) == 3
    with_ti = _encode(pipe, rewritten["prompt"][0], rewritten["negative_prompt"])
    # The positive prompt names no embedding: its encoding is unchanged; the negative's changes.
    assert torch.equal(with_ti[0], base[0])
    assert not torch.equal(with_ti[1], base[1])

    applied.restore()
    for attr, original in originals.items():
        assert getattr(pipe, attr, None) is original
    for attr, module in embeds.items():
        assert getattr(pipe, attr).get_input_embeddings() is module
    assert "_unsloth_embeddings" not in pipe.__dict__
    after = _encode(pipe, prompt, negative)
    assert all(torch.equal(a, b) for a, b in zip(after, base))


@pytest.mark.allow_network
def test_sd1_pipeline_applies_and_restores(emb_dir):
    pipe = _tiny("hf-internal-testing/tinier-stable-diffusion-pipe", "StableDiffusionPipeline")
    _check_pipeline(pipe, de.KIND_SD1, emb_dir, lambda w, n: {"emb_params": torch.randn(n, w[0])})


@pytest.mark.allow_network
def test_sdxl_pipeline_applies_both_encoders_and_restores(emb_dir):
    pipe = _tiny("hf-internal-testing/tiny-stable-diffusion-xl-pipe", "StableDiffusionXLPipeline")
    _check_pipeline(
        pipe,
        de.KIND_SDXL,
        emb_dir,
        lambda w, n: {"clip_l": torch.randn(n, w[0]), "clip_g": torch.randn(n, w[1])},
    )


def test_prompt_cache_keys_on_applied_embeddings():
    from core.inference import diffusion_prompt_cache as pc

    class Pipe:
        def __init__(self):
            self.calls = 0

        def encode_prompt(
            self,
            prompt,
            device = None,
        ):
            self.calls += 1
            return torch.full((1,), float(self.calls))

    pipe = Pipe()
    cache = pc.PromptCache(budget = 1 << 20)
    assert pc._wrap_encode_prompt(pipe, cache, {}, pipe, None)
    first = pipe.encode_prompt("x")
    assert torch.equal(pipe.encode_prompt("x"), first) and pipe.calls == 1
    pipe.__dict__["_unsloth_embeddings"] = (("EasyNeg", "abc"),)
    assert not torch.equal(pipe.encode_prompt("x"), first) and pipe.calls == 2
    del pipe.__dict__["_unsloth_embeddings"]
    assert torch.equal(pipe.encode_prompt("x"), first) and pipe.calls == 2


def test_families_gate_textual_inversion():
    from core.inference.diffusion_families import _FAMILIES
    assert {f.name for f in _FAMILIES if f.textual_inversion} == {"sdxl"}
    assert next(f for f in _FAMILIES if f.name == "sdxl").textual_inversion == de.KIND_SDXL
