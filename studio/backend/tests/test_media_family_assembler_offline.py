# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The FAMILY-SPECIFIC assemblers keep the no-download promise too.

``test_diffusion_offline_load.py`` and ``test_video_offline_load.py`` pin the SHARED path: the
staging phase, the byte estimate, the prefetch, the guarded ``pipe_kwargs`` every ordinary family
is built from. Four assemblers do not go through that dict, and each of them was still reaching
the Hub on a load nobody asked for:

- the MiniMax-H3 hosted conditioner (~27 GB), fetched by ``load_h3_quantized_text_encoder``;
- the image checkpoint, REOPENED by ``_resolve_gguf_path`` under the generation lock;
- Krea 2, assembled per-component because the repo ships transformers-5.x configs;
- LTX 2.3, assembled per-component because its vocoder class differs from the base pin.

Krea and LTX are reachable with no race at all: both are handed a REPO ID rather than a staged
snapshot (``_base_local_dir`` is None for 2.3 by design), so an assembler that resolves it without
the flag downloads whatever the caller's cache root does not hold. Every test below therefore
records what each component load was actually asked for, and the mirror tests keep the
user-initiated path fetching exactly as it did before.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from core.inference import diffusion as diffusion_mod


def _call_keyword_sets(module_path: str, function: str, callee: str) -> list[set[str]]:
    """The keywords EVERY call to *callee* inside *function* spells out, one set per call site.

    Read from the source rather than driven: these branches need a real Krea single-file build, a
    real 2.3 checkpoint header or a Modular Diffusers H3 pipeline to reach, which no unit test can
    stage -- but the keyword either is written there or is not. ``callee`` matches a bare name
    (``load_krea2_pipeline``) or the last attribute segment (``self._resolve_gguf_path``,
    ``LTX2Pipeline.load_config``), so a call site that moves onto or off a receiver still counts.

    Anchored on the package, not on the process CWD: CI runs pytest from the repo root with the
    backend merely on PYTHONPATH, where a relative open raises FileNotFoundError.
    """
    backend_root = pathlib.Path(diffusion_mod.__file__).resolve().parents[2]
    tree = ast.parse((backend_root / module_path).read_text(encoding = "utf-8"))
    found: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != function:
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
            if name == callee:
                found.append({kw.arg for kw in call.keywords if kw.arg})
    if not found:
        raise AssertionError(f"no call to {callee}() inside {function}()")
    return found


# ── [A] the MiniMax-H3 hosted conditioner ────────────────────────────────────


def _h3_te_module():
    from core.inference import video_minimax_h3_te as te_mod
    return te_mod


def _drive_h3_conditioner(monkeypatch, *, local_files_only):
    """Call the conditioner loader far enough to record its two Hub reads.

    The loader is best-effort by contract and swallows everything into a None return, so the
    RECORD is the result: the artifact fetch and the config read are the only two calls that can
    leave the process, and a stub that raises after recording stops the 62 GB meta-init below.
    """
    # The bare CI runners ship neither, and this driver reaches the real library rather than
    # a stub, so the honest answer there is a skip.
    transformers = pytest.importorskip("transformers")

    import utils.hf_xet_fallback as xet

    seen: dict = {}
    # accelerate is imported at the top of the loader body, ahead of both Hub reads, and it is not
    # a hard dependency of this backend; without the stub the whole function degrades to its
    # best-effort None return before it asks for anything and the test would pass vacuously.
    monkeypatch.setitem(
        sys.modules, "accelerate", SimpleNamespace(init_empty_weights = lambda **_k: None)
    )

    def _download(
        repo_id,
        filename,
        token = None,
        **kwargs,
    ):
        seen["download"] = (repo_id, filename, kwargs)
        return "/nowhere/artifact.safetensors"

    def _config(*args, **kwargs):
        seen["config"] = kwargs
        raise RuntimeError("stop after the two Hub reads")

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _config, raising = False)
    assert (
        _h3_te_module().load_h3_quantized_text_encoder(
            "MiniMaxAI/MiniMax-H3",
            "int8",
            dtype = None,
            cache_dir = "/live/root",
            local_files_only = local_files_only,
        )
        is None
    )
    return seen


def test_the_h3_conditioner_is_opened_from_the_cache_on_a_load_nobody_asked_for(monkeypatch):
    """The artifact is ~27 GB and the staging fetch already accepted it, so the loader may only
    look it up -- through the SAME both-roots rule the stager used, or a moved cache folder makes
    it re-pull what the load was cleared on."""
    seen = _drive_h3_conditioner(monkeypatch, local_files_only = True)
    repo, filename, kwargs = seen["download"]
    te_mod = _h3_te_module()
    assert (repo, filename) == (te_mod.H3_TE_QUANT_REPO, te_mod.H3_TE_QUANT_FILES["int8"])
    assert kwargs["local_files_only"] is True
    assert kwargs["reuse_other_cache_root"] is True
    # The component config is a hub read too: _base_local_dir is None on an offline load, because
    # the scoped base predownload stands down, so `local_base or base` resolves the repo id.
    assert seen["config"]["local_files_only"] is True


def test_a_user_initiated_h3_load_still_fetches_the_conditioner(monkeypatch):
    """The pre-PR behaviour, unchanged: a load the user asked for pulls the artifact."""
    seen = _drive_h3_conditioner(monkeypatch, local_files_only = False)
    assert seen["download"][2]["local_files_only"] is False
    assert seen["config"]["local_files_only"] is False


def test_the_h3_modular_build_hands_the_flag_to_the_conditioner_loader():
    # The flag protected load_components() on one side and load_prequantized_transformer() on the
    # other; the conditioner load between them was the remaining multi-GB fetch on that path.
    for keywords in _call_keyword_sets(
        "core/inference/video.py",
        "_load_h3_modular_pipeline",
        "load_h3_quantized_text_encoder",
    ):
        assert "local_files_only" in keywords


# ── [B] reopening the image checkpoint under the generation lock ─────────────


def _drive_resolve_gguf(monkeypatch, *, cached_here, local_files_only):
    """``_resolve_gguf_path`` with the cache answering ``cached_here`` for the live root."""
    import huggingface_hub

    seen: list[dict] = []

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, name, cache_dir = None: (
            "/live/checkpoint.gguf" if cache_dir is not None and cached_here else "/other/ck.gguf"
        ),
        raising = False,
    )

    def _download(repo_id, filename, **kwargs):
        seen.append(kwargs)
        return "/resolved/checkpoint.gguf"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download, raising = False)
    diffusion_mod.DiffusionBackend()._resolve_gguf_path(
        "unsloth/FLUX.1-dev-GGUF",
        "flux1-dev-Q4_K_M.gguf",
        None,
        local_files_only = local_files_only,
    )
    return seen


@pytest.mark.parametrize("cached_here", [True, False])
def test_reopening_the_image_checkpoint_is_a_cache_lookup_offline(monkeypatch, cached_here):
    """Both resolutions, the live root and the other-root reuse. Neither is a no-op even on a hit:
    ``hf_hub_download`` re-resolves the revision against the Hub, so a checkpoint republished since
    the cache was filled is a multi-GB pull taken AFTER the resident pipeline was evicted, inside
    the generation lock, with progress already reading 100%."""
    for kwargs in _drive_resolve_gguf(monkeypatch, cached_here = cached_here, local_files_only = True):
        assert kwargs["local_files_only"] is True


@pytest.mark.parametrize("cached_here", [True, False])
def test_a_user_initiated_image_load_still_revalidates_the_checkpoint(monkeypatch, cached_here):
    """Unchanged for the UI load: it still goes to the Hub, which is how a republished GGUF is
    picked up."""
    for kwargs in _drive_resolve_gguf(monkeypatch, cached_here = cached_here, local_files_only = False):
        assert kwargs["local_files_only"] is False


def test_the_image_assembly_hands_the_flag_to_the_checkpoint_resolver():
    for keywords in _call_keyword_sets(
        "core/inference/diffusion.py", "load_pipeline", "_resolve_gguf_path"
    ):
        assert "local_files_only" in keywords


# ── [C] the Krea 2 per-component assembler ───────────────────────────────────


def _drive_krea(
    monkeypatch,
    tmp_path,
    *,
    local_files_only,
    with_transformer = True,
):
    """Assemble a Krea pipeline against fakes, recording what each component was asked for."""
    from core.inference.diffusion_krea2 import load_krea2_pipeline

    import huggingface_hub

    index = tmp_path / "model_index.json"
    index.write_text(json.dumps({"patch_size": 2}), encoding = "utf-8")
    seen: dict = {}
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo_id, filename, **kwargs: seen.update(model_index = kwargs) or str(index),
        raising = False,
    )

    class _Component:
        def __init__(self, tag):
            self.tag = tag

        def from_pretrained(self, repo_id, **kwargs):
            seen[self.tag] = kwargs
            return SimpleNamespace(tag = self.tag)

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            FlowMatchEulerDiscreteScheduler = _Component("scheduler"),
            AutoencoderKLQwenImage = _Component("vae"),
            Krea2Transformer2DModel = _Component("transformer"),
            Krea2Pipeline = lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    for name in ("tokenizer", "text_encoder"):
        monkeypatch.setattr(
            f"core.inference.diffusion_krea2.load_krea2_{name}",
            (
                lambda *_a, tag = name, **kwargs: (
                    seen.__setitem__(tag, kwargs),
                    SimpleNamespace(tag = tag),
                )[1]
            ),
        )
    load_krea2_pipeline(
        # A hub id, not the local dir the other Krea tests use: the branch that reaches this passes
        # ``fetch_base`` (or ``base_local_dir or base``, which is the id whenever nothing staged),
        # and a local dir would resolve every component off disk and prove nothing.
        "krea/Krea-2-Turbo",
        "bf16",
        with_transformer = with_transformer,
        local_files_only = local_files_only,
    )
    return seen


def test_the_krea_assembler_opens_every_component_from_the_cache_offline(monkeypatch, tmp_path):
    """The 26 GB transformer, the 8.88 GB Qwen3-VL encoder, the VAE, the tokenizer and the
    scheduler: this branch never sees the guarded pipe_kwargs, so each one has to carry the flag
    itself or a load that promised nothing pulls it."""
    seen = _drive_krea(monkeypatch, tmp_path, local_files_only = True)
    assert set(seen) == {
        "scheduler",
        "vae",
        "transformer",
        "tokenizer",
        "text_encoder",
        "model_index",
    }
    for tag, kwargs in seen.items():
        assert kwargs.get("local_files_only") is True, tag


def test_a_user_initiated_krea_load_still_fetches_every_component(monkeypatch, tmp_path):
    seen = _drive_krea(monkeypatch, tmp_path, local_files_only = False)
    for tag, kwargs in seen.items():
        assert kwargs.get("local_files_only") is False, tag


def test_the_krea_model_index_read_is_a_cache_lookup_offline(monkeypatch):
    """A few KB, but still a fetch: the assembly reads the init config (``is_distilled`` carries
    Turbo's mu shift) straight off the hub id when the repo is not a local directory."""
    import huggingface_hub

    from core.inference.diffusion_krea2 import _load_model_index

    seen: dict = {}

    def _download(repo_id, filename, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop before the read")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download, raising = False)
    with pytest.raises(RuntimeError):
        _load_model_index("krea/Krea-2-Turbo", None, local_files_only = True)
    assert seen["local_files_only"] is True


def test_every_krea_call_site_hands_over_the_flag():
    # Three: the full-pipeline branch, the transformer-only single-file branch, and _assemble_pipe.
    sites = _call_keyword_sets(
        "core/inference/diffusion.py", "load_pipeline", "load_krea2_pipeline"
    ) + _call_keyword_sets("core/inference/diffusion.py", "_assemble_pipe", "load_krea2_pipeline")
    assert len(sites) == 3
    for keywords in sites:
        assert "local_files_only" in keywords


# ── [D] the LTX 2.3 per-component assembler ──────────────────────────────────


def test_the_ltx23_extras_fetch_is_a_cache_lookup_offline(monkeypatch):
    """The text projections, the video VAE and the audio VAE/vocoder: the switch's locality gate
    clears these three by name, so a miss here is a promise it cannot keep."""
    # monkeypatched by dotted path below, which imports the module to patch it.
    pytest.importorskip("safetensors")
    import utils.hf_xet_fallback as xet
    from core.inference import video_ltx2

    seen: dict = {}
    monkeypatch.setattr(
        xet,
        "hf_hub_download_with_xet_fallback",
        lambda repo_id, filename, token = None, **kwargs: seen.update(kwargs) or "/nowhere.st",
    )
    monkeypatch.setattr("safetensors.torch.load_file", lambda _path: {}, raising = False)
    video_ltx2._load_extras_file("vae/x.safetensors", None, True)
    assert seen["local_files_only"] is True
    assert seen["reuse_other_cache_root"] is True

    seen.clear()
    video_ltx2._load_extras_file("vae/x.safetensors", None)
    assert seen["local_files_only"] is False


def _drive_ltx23(monkeypatch, *, local_files_only):
    """Assemble a 2.3 pipeline against fakes, recording what the base repo reads were asked for."""
    diffusers = pytest.importorskip("diffusers")

    from core.inference import video_ltx2

    seen: dict = {}

    class _Sub:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            seen[kwargs["subfolder"]] = kwargs
            return SimpleNamespace(tag = kwargs["subfolder"])

    class _FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def load_config(cls, repo_id, **kwargs):
            seen["load_config"] = kwargs
            return {
                name: ["diffusers", "_LtxOfflineSub"]
                for name in ("scheduler", "tokenizer", "text_encoder")
            }

    monkeypatch.setattr(diffusers, "LTX2Pipeline", _FakePipeline, raising = False)
    monkeypatch.setattr(diffusers, "_LtxOfflineSub", _Sub, raising = False)
    monkeypatch.setattr(
        "diffusers.loaders.single_file_utils.load_single_file_checkpoint",
        lambda _path: {},
        raising = False,
    )
    monkeypatch.setattr(video_ltx2, "checkpoint_variant", lambda _p: "dev")
    for name in ("transformer", "connectors", "vae", "audio_vae_and_vocoder"):
        monkeypatch.setattr(
            video_ltx2,
            f"load_ltx23_{name}",
            (
                lambda *_a, tag = name, **kwargs: (
                    seen.__setitem__(tag, kwargs),
                    (None, None) if tag == "audio_vae_and_vocoder" else None,
                )[1]
            ),
        )
    video_ltx2.load_ltx23_pipeline(
        "/models/ltx-2.3-dev-Q4_K_M.gguf",
        # A repo id, which is what this branch always gets: the 2.3 snapshot lacks the base VAEs,
        # so _run_load sets _base_local_dir to None for it deliberately.
        base_repo = "Lightricks/LTX-Video-2",
        torch_dtype = "bf16",
        is_gguf = True,
        local_files_only = local_files_only,
    )
    return seen


def test_the_ltx23_assembler_opens_every_component_from_the_cache_offline(monkeypatch):
    """The base model_index, the scheduler, the tokenizer, the dense Gemma3 encoder (~50 GB) and
    every companion loader: none of them sees the guarded pipe_kwargs, and there is no staged
    snapshot to fall back on, so each resolves the hub id itself."""
    seen = _drive_ltx23(monkeypatch, local_files_only = True)
    assert seen["load_config"]["local_files_only"] is True
    for name in ("scheduler", "tokenizer", "text_encoder"):
        assert seen[name]["local_files_only"] is True, name
    for name in ("transformer", "connectors", "vae", "audio_vae_and_vocoder"):
        assert seen[name]["local_files_only"] is True, name


def test_a_user_initiated_ltx23_load_still_fetches_every_component(monkeypatch):
    seen = _drive_ltx23(monkeypatch, local_files_only = False)
    assert seen["load_config"]["local_files_only"] is False
    for name in ("scheduler", "tokenizer", "text_encoder", "transformer", "connectors"):
        assert seen[name]["local_files_only"] is False, name


def test_the_video_assembly_hands_the_flag_to_the_ltx23_assembler():
    for keywords in _call_keyword_sets(
        "core/inference/video.py", "load_pipeline", "load_ltx23_pipeline"
    ):
        assert "local_files_only" in keywords


# ── the live cache root ──────────────────────────────────────────────────────────
# Unsloth's HF cache folder is a SETTING (PUT /settings/hugging-face-cache), and changing it only
# rewrites the DB: os.environ and huggingface_hub's import-time constant keep the startup value.
# So after a change the live root and the import-time root differ, and an unset cache_dir resolves
# through the stale one. That mismatch predates this PR and used to be survivable, because a miss
# in the stale root just downloaded again. It is not survivable with local_files_only: the switch's
# locality gate reads the LIVE root (media_locality passes cache_dir = hub_cache_dir()), so it
# clears a model that is fully present, the resident pipeline is evicted, and the assembler then
# raises LocalEntryNotFoundError against the other root. Pinning is what keeps the gate's verdict
# and the load looking in the same place.

LIVE_ROOT = "/live-hub"


@pytest.fixture
def live_cache_root(monkeypatch):
    """Point the live root somewhere unmistakable, so a stale-root read cannot pass by accident."""
    import utils.hf_cache_settings as cache_settings

    monkeypatch.setattr(cache_settings, "active_hf_hub_cache", lambda: LIVE_ROOT)
    return LIVE_ROOT


def test_the_krea_assembler_pins_every_component_to_the_live_cache(
    monkeypatch, tmp_path, live_cache_root
):
    seen = _drive_krea(monkeypatch, tmp_path, local_files_only = True)
    # The direct loader calls. "tokenizer" and "text_encoder" are absent by design: those two tags
    # record the kwargs handed to load_krea2_tokenizer / load_krea2_text_encoder, which are Unsloth
    # helpers rather than hub calls, so they take no cache_dir and pin internally instead. The test
    # below drives them for real.
    for tag in ("scheduler", "vae", "transformer", "model_index"):
        assert seen[tag].get("cache_dir") == live_cache_root, tag


def test_the_krea_tokenizer_and_encoder_helpers_pin_internally(monkeypatch, live_cache_root):
    """Both build their own kwargs dict, so the pin has to be inside each one."""
    transformers = pytest.importorskip("transformers")

    from core.inference import diffusion_krea2

    seen: dict = {}

    class _Component:
        def __init__(self, tag):
            self.tag = tag

        def from_pretrained(self, repo_id, **kwargs):
            seen[self.tag] = kwargs
            return SimpleNamespace(tag = self.tag, text_config = SimpleNamespace())

    monkeypatch.setattr(transformers, "AutoTokenizer", _Component("tokenizer"), raising = False)
    monkeypatch.setattr(transformers, "AutoConfig", _Component("config"), raising = False)
    monkeypatch.setattr(transformers, "Qwen3VLModel", _Component("text_encoder"), raising = False)

    diffusion_krea2.load_krea2_tokenizer("krea/Krea-2-Turbo", local_files_only = True)
    diffusion_krea2.load_krea2_text_encoder("krea/Krea-2-Turbo", "bf16", local_files_only = True)

    assert set(seen) == {"tokenizer", "config", "text_encoder"}
    for tag, kwargs in seen.items():
        assert kwargs.get("cache_dir") == live_cache_root, tag
        assert kwargs.get("local_files_only") is True, tag


def test_the_ltx23_assembler_pins_the_base_reads_to_the_live_cache(monkeypatch, live_cache_root):
    seen = _drive_ltx23(monkeypatch, local_files_only = True)
    # The base-repo reads only: the companion loaders take a checkpoint path, not a hub id.
    assert seen["load_config"]["cache_dir"] == live_cache_root
    for name in ("scheduler", "tokenizer", "text_encoder"):
        assert seen[name]["cache_dir"] == live_cache_root, name


def test_the_hidream_external_encoder_is_pinned_to_the_live_cache(monkeypatch, live_cache_root):
    """TE4 lives in its own standalone repo, so it never rides the pipeline's pipe_kwargs (which
    do carry cache_dir); it is the one 16 GB component that resolves its hub id unaided."""
    transformers = pytest.importorskip("transformers")

    from core.inference import diffusion_hidream

    seen: dict = {}

    class _Component:
        def __init__(self, tag):
            self.tag = tag

        def from_pretrained(self, repo_id, **kwargs):
            seen[self.tag] = kwargs
            return SimpleNamespace(tag = self.tag)

    monkeypatch.setattr(transformers, "AutoTokenizer", _Component("tokenizer_4"), raising = False)
    monkeypatch.setattr(
        transformers, "LlamaForCausalLM", _Component("text_encoder_4"), raising = False
    )
    diffusion_hidream.hidream_te4_kwargs(
        dtype = "bf16",
        hf_token = None,
        local_files_only = True,
    )
    assert set(seen) == {"tokenizer_4", "text_encoder_4"}
    for tag, kwargs in seen.items():
        assert kwargs.get("cache_dir") == live_cache_root, tag
        assert kwargs.get("local_files_only") is True, tag


# ── [E] the transformer-only single-file build ───────────────────────────────
# from_single_file(config = <repo id>, subfolder = "transformer") is not a local read: diffusers
# forwards local_files_only into the load_config() that resolves that id (single_file_model.py in
# 0.39 pops the kwarg and passes it on), and an unset flag is None, which is falsy, which permits
# the network. The pipeline assembly after it was already guarded, so this one call was the last
# unguarded Hub read on the GGUF/safetensors path -- and it runs AFTER eviction.
#
# The flag alone would not have been enough. transformer/config.json was deliberately excluded from
# the staged base file set on both paths (the shards come from the checkpoint), so the locality gate
# cleared picks that had never cached it and local_files_only would have turned a silent ~1 KB fetch
# into a hard failure on essentially every API-initiated GGUF load. Admitting the config -- and only
# the config -- is what makes the promise keepable, and lets the gate refuse up front instead.


def test_the_image_base_file_set_stages_the_transformer_config_but_not_its_shards():
    from core.inference.diffusion import _base_file_downloaded as keep
    assert keep("transformer/config.json", include_transformer = False)
    assert not keep(
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        include_transformer = False,
    )


def _sf_kwargs_keys(module_path: str) -> list[set[str]]:
    """The literal keys of every ``sf_kwargs = {...}`` in *module_path*, one set per assignment.

    Read from the source, like the call-site checks above: reaching this branch needs a real
    multi-GB GGUF plus its base repo, which no unit test can stage. The call itself is
    ``from_single_file(path, **sf_kwargs)``, so the keyword lives in the dict, not the call.
    """
    backend_root = pathlib.Path(diffusion_mod.__file__).resolve().parents[2]
    tree = ast.parse((backend_root / module_path).read_text(encoding = "utf-8"))
    found: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(getattr(t, "id", None) == "sf_kwargs" for t in targets):
            continue
        if isinstance(node.value, ast.Dict):
            found.append({k.value for k in node.value.keys if isinstance(k, ast.Constant)})
    if not found:
        raise AssertionError(f"no sf_kwargs dict literal in {module_path}")
    return found


@pytest.mark.parametrize("module_path", ["core/inference/diffusion.py", "core/inference/video.py"])
def test_every_single_file_build_hands_over_the_flag(module_path):
    for keys in _sf_kwargs_keys(module_path):
        assert "local_files_only" in keys
        # cache_dir is set here too, but diffusers does NOT forward it to the config lookup, so it
        # pins only the checkpoint read. The flag is what keeps the config resolution off the Hub.
        assert "config" in keys
