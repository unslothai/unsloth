"""Pure-CPU, no-network unit tests for the prefetch snapshot scoping in
unsloth/models/_utils.py.

maybe_prefetch_hf_snapshot warms the HF cache before the in-process load so the load is a cache
hit and cannot hang on a stalled Xet transfer. The warm must download AT LEAST what the load
reads (else the missing file falls to an unprotected in-process Xet fetch) but should not pull
weights the load never reads. These tests lock the allow_patterns / ignore_patterns each mode
hands snapshot_download_with_xet_fallback (Codex #6638: adapter-only, weights-at-root, subfolder).
No network, no subprocess: the zoo downloader is monkeypatched to capture its kwargs.
"""

import fnmatch
import sys
import types

import pytest

from unsloth.models import _utils as U


def _filter(names, allow_patterns, ignore_patterns):
    """Mirror Hugging Face filter_repo_objects: keep a name if it matches any allow pattern
    (or allow is None), then drop it if it matches any ignore pattern. fnmatch '*' spans '/'
    exactly as HF's matcher does, so this reproduces the real selection over a sample file list."""
    kept = []
    for name in names:
        if allow_patterns is not None and not any(fnmatch.fnmatch(name, p) for p in allow_patterns):
            continue
        if ignore_patterns and any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
            continue
        kept.append(name)
    return kept


@pytest.fixture
def capture(monkeypatch):
    """Call maybe_prefetch_hf_snapshot with a fake repo id and capture the allow/ignore patterns
    it forwards to the zoo downloader. A fake unsloth_zoo.hf_xet_fallback module is injected into
    sys.modules so the test is independent of the installed unsloth_zoo version (the published
    package may predate the helper, which maybe_prefetch_hf_snapshot then imports lazily). Offline
    env vars are cleared so the warm is not skipped."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    state = {}

    def fake_download(repo_id, **kw):
        state["repo_id"] = repo_id
        state["allow_patterns"] = kw.get("allow_patterns")
        state["ignore_patterns"] = kw.get("ignore_patterns")
        return "/tmp/fake-snapshot"

    fake_module = types.ModuleType("unsloth_zoo.hf_xet_fallback")
    fake_module.snapshot_download_with_xet_fallback = fake_download
    fake_module.DownloadStallError = type("DownloadStallError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "unsloth_zoo.hf_xet_fallback", fake_module)

    # Neutralize the model_info network call (adapter format selection / use_safetensors auto
    # branch) by default so the pure-CPU tests never reach the Hub. A best-effort failure leaves
    # both weight formats eligible; tests that exercise format selection install their own.
    import huggingface_hub

    class _NoNetworkApi:
        def model_info(self, *a, **k):
            raise RuntimeError("no network in test")

    monkeypatch.setattr(huggingface_hub, "HfApi", _NoNetworkApi)

    def run(**call_kwargs):
        state.clear()
        ok = U.maybe_prefetch_hf_snapshot("some-org/some-repo", **call_kwargs)
        return ok, state

    return run


# A representative repo file listing: root weights + tokenizer/config, plus an alternate-precision
# subdir, an adapter, a checkpoint dir, and merged full-model weights an adapter repo might ship.
_SAMPLE_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "fp16/model.safetensors",
    "experimental/model-00001-of-00002.safetensors",
    "checkpoint-500/model.safetensors",
    "adapter_config.json",
    "adapter_model.safetensors",
]


def test_weights_at_root_excludes_subdir_weights(capture):
    """A bare root load reads only root weight files, so weights nested in subdirs (fp16/,
    experimental/, checkpoint-500/) must be ignored while the root weights stay warmed. An
    explicit use_safetensors avoids the auto branch's model_info network call."""
    ok, st = capture(weights_at_root = True, use_safetensors = True)
    assert ok is True
    assert st["allow_patterns"] is None  # the warm stays otherwise unfiltered
    ig = st["ignore_patterns"]
    assert "*/*.safetensors" in ig and "*/*.bin" in ig
    kept = _filter(_SAMPLE_FILES, st["allow_patterns"], ig)
    # Root weights + config/tokenizer survive; subdir weights are dropped.
    assert "model-00001-of-00002.safetensors" in kept
    assert "model.safetensors.index.json" in kept
    assert "config.json" in kept
    assert "fp16/model.safetensors" not in kept
    assert "experimental/model-00001-of-00002.safetensors" not in kept
    assert "checkpoint-500/model.safetensors" not in kept


def test_adapter_only_excludes_merged_weights(capture):
    """An adapter warm reads only adapter_config.json + adapter_model.* (plus root tokenizer /
    config); a repo that also ships merged full-model weights must not pull them."""
    ok, st = capture(adapter_only = True)
    assert ok is True
    assert st["ignore_patterns"] is None  # the exact allowlist makes the format filter moot
    allow = st["allow_patterns"]
    assert "adapter_config.json" in allow and "adapter_model*" in allow
    kept = _filter(_SAMPLE_FILES, allow, st["ignore_patterns"])
    # The adapter's own files + the root aux files are warmed.
    assert "adapter_config.json" in kept
    assert "adapter_model.safetensors" in kept
    assert "config.json" in kept and "tokenizer.json" in kept
    # The merged / full-model weights are NOT pulled.
    assert "model-00001-of-00002.safetensors" not in kept
    assert "pytorch_model.bin" not in kept
    assert "fp16/model.safetensors" not in kept


def test_adapter_only_warms_sharded_adapter(capture):
    """A sharded adapter (adapter_model-00001-of-00002.safetensors) is still covered by the
    adapter_model* glob, so a large adapter is not left to an in-process Xet fetch."""
    _, st = capture(adapter_only = True)
    sharded = [
        "adapter_config.json",
        "adapter_model-00001-of-00002.safetensors",
        "adapter_model-00002-of-00002.safetensors",
        "adapter_model.safetensors.index.json",
    ]
    kept = _filter(sharded, st["allow_patterns"], st["ignore_patterns"])
    assert set(kept) == set(sharded)


def test_tokenizer_only_warms_only_aux_files(capture):
    """A distinct tokenizer repo warms only its tokenizer / config / vocab files, never weights."""
    _, st = capture(tokenizer_only = True)
    assert st["ignore_patterns"] is None
    assert st["allow_patterns"] == list(U._ROOT_AUX_PREFETCH_PATTERNS)
    kept = _filter(_SAMPLE_FILES, st["allow_patterns"], st["ignore_patterns"])
    assert "tokenizer.json" in kept and "config.json" in kept
    assert "model-00001-of-00002.safetensors" not in kept
    assert "adapter_model.safetensors" not in kept


def test_aux_warm_covers_arbitrary_remote_code_modules(capture):
    """A trust_remote_code auto_map can name its module arbitrarily (modeling.py, tokenization.py,
    my_code.py), not just the transformers modeling_*.py convention, so the aux warm must cover any
    *.py -- else the load fetches the code file in-process over Xet (Codex #6638)."""
    _, st = capture(tokenizer_only = True)
    allow = st["allow_patterns"]
    assert "*.py" in allow
    remote_code = [
        "config.json",
        "modeling.py",  # auto_map "modeling.Model" -- no underscore suffix
        "tokenization.py",
        "my_custom_code.py",
        "configuration_foo.py",  # the convention still covered by *.py too
    ]
    kept = _filter(remote_code, allow, st["ignore_patterns"])
    for name in ("modeling.py", "tokenization.py", "my_custom_code.py", "configuration_foo.py"):
        assert name in kept, name


def test_subfolder_warms_subfolder_plus_root_aux(capture):
    """A subfolder load warms that subfolder's weights plus the root tokenizer / config; the
    root weights and OTHER subfolders are skipped."""
    _, st = capture(subfolder = "fp16")
    allow = st["allow_patterns"]
    assert "fp16/*" in allow
    assert all(p in allow for p in U._ROOT_AUX_PREFETCH_PATTERNS)
    kept = _filter(_SAMPLE_FILES, allow, st["ignore_patterns"])
    assert "fp16/model.safetensors" in kept
    assert "config.json" in kept
    assert "experimental/model-00001-of-00002.safetensors" not in kept


def test_subfolder_takes_precedence_over_weights_at_root(capture):
    """weights_at_root is a root-load assertion; when a subfolder IS requested the subfolder
    branch wins (the load reads that subfolder), so the warm is the subfolder, not a
    root-with-subdir-weights-excluded warm."""
    _, st = capture(subfolder = "fp16", weights_at_root = True)
    assert "fp16/*" in st["allow_patterns"]
    kept = _filter(_SAMPLE_FILES, st["allow_patterns"], st["ignore_patterns"])
    assert "fp16/model.safetensors" in kept


def test_local_dir_is_not_warmed(capture, tmp_path):
    """A local directory path has nothing to download: the warm is skipped (returns False)."""
    d = tmp_path / "local-model"
    d.mkdir()
    ok = U.maybe_prefetch_hf_snapshot(str(d), weights_at_root = True)
    assert ok is False


def _install_fake_model_info(monkeypatch, filenames):
    """Make HfApi().model_info(...).siblings report *filenames*, with no network."""
    import huggingface_hub

    class _Sib:
        def __init__(self, name):
            self.rfilename = name

    class _Info:
        def __init__(self, names):
            self.siblings = [_Sib(n) for n in names]

    class _Api:
        def model_info(self, *a, **k):
            return _Info(filenames)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)


# ----- Finding P: variant-aware weight-format selection -----


def test_variant_keeps_bin_when_only_default_safetensors(monkeypatch):
    """With variant='fp16' requested, a DEFAULT model.safetensors must not prove the variant
    pytorch_model.fp16.bin redundant: dropping it would leave the variant load to fetch the .bin
    in-process over Xet. The .bin stays warmed (Codex #6638)."""
    _install_fake_model_info(monkeypatch, ["model.safetensors", "pytorch_model.fp16.bin"])
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16")
    assert "*.bin" not in ig
    # No variant: the default safetensors DOES make .bin redundant (existing behavior).
    ig_default = U._prefetch_ignore_patterns("org/repo")
    assert "*.bin" in ig_default


def test_variant_drops_bin_when_variant_safetensors_present(monkeypatch):
    """When a variant-matching safetensors (model.fp16.safetensors) is shipped, the variant load
    reads it and the variant .bin is redundant, so .bin is dropped from the warm."""
    _install_fake_model_info(monkeypatch, ["model.fp16.safetensors", "pytorch_model.fp16.bin"])
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16")
    assert "*.bin" in ig


def test_filename_has_variant_matches_single_and_sharded():
    """The variant detector matches both the single-file (.fp16.) and SHARDED (.fp16-) infixes and
    rejects the default (non-variant) names (gemini #6638)."""
    assert U._filename_has_variant("model.fp16.safetensors", "fp16") is True
    assert U._filename_has_variant("model.fp16-00001-of-00002.safetensors", "fp16") is True
    assert U._filename_has_variant("diffusion_pytorch_model.fp16.safetensors", "fp16") is True
    assert U._filename_has_variant("model.safetensors", "fp16") is False
    assert U._filename_has_variant("model-00001-of-00002.safetensors", "fp16") is False


def test_variant_drops_bin_for_sharded_variant_safetensors(monkeypatch):
    """A SHARDED variant safetensors (model.fp16-00001-of-00002.safetensors) is recognized, so its
    redundant variant .bin is dropped rather than both formats warmed (gemini #6638)."""
    _install_fake_model_info(
        monkeypatch,
        [
            "model.fp16-00001-of-00002.safetensors",
            "model.fp16-00002-of-00002.safetensors",
            "pytorch_model.fp16-00001-of-00002.bin",
        ],
    )
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16")
    assert "*.bin" in ig


def test_tokenizer_only_warms_extra_vocab_files(capture):
    """tokenizer_only must warm the SentencePiece / vocab / processor files real tokenizers load
    (spm.model, normalizer.json, video_preprocessor_config.json, tokenizer.model.v3, and a named
    additional_chat_templates/<name>.jinja) so a separate-repo tokenizer / processor load does not
    fetch them in-process over Xet (Codex #6638)."""
    _, st = capture(tokenizer_only = True)
    allow = st["allow_patterns"]
    for name in (
        "spm.model",
        "normalizer.json",
        "video_preprocessor_config.json",
        "tokenizer.model.v3",
    ):
        assert name in allow, name
    sample = [
        "spm.model",
        "normalizer.json",
        "video_preprocessor_config.json",
        "tokenizer.model.v3",
        "additional_chat_templates/custom.jinja",
    ]
    kept = _filter(sample, allow, st["ignore_patterns"])
    assert set(kept) == set(sample)


def test_format_probe_runs_even_when_config_cached(capture, monkeypatch):
    """A cached config.json must NOT skip the weight-format probe: AutoConfig caches config.json
    before this helper runs (Llama / diffusion), so a config-based "cached" guess would leave both
    formats eligible and over-fetch both multi-GB weight sets even when no weights are cached. The
    auto branch still consults model_info and drops the redundant .bin for a safetensors repo
    (Codex #6638)."""
    import huggingface_hub

    # Pretend config.json is locally cached (the AutoConfig side effect). This must not gate the probe.
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: "/cache/config.json"
    )
    _install_fake_model_info(monkeypatch, ["model.safetensors", "pytorch_model.bin"])
    _, st = capture(weights_at_root = True)
    ig = st["ignore_patterns"] or []
    assert "*.bin" in ig  # redundant .bin dropped because real model safetensors is present


def test_optimizer_safetensors_does_not_drop_bin(monkeypatch):
    """A training-state optimizer.safetensors sidecar must NOT count as model safetensors: a repo
    whose real weights are pytorch_model.bin alongside an optimizer.safetensors must keep its .bin,
    else the in-process load fetches the only weights over Xet without the fallback (Codex #6638)."""
    _install_fake_model_info(monkeypatch, ["pytorch_model.bin", "optimizer.safetensors"])
    ig = U._prefetch_ignore_patterns("org/repo")
    assert "*.bin" not in ig  # .bin is the only real weight -> not dropped


def test_model_safetensors_still_drops_bin(monkeypatch):
    """Control for the optimizer case: a real model.safetensors next to pytorch_model.bin still
    drops the redundant .bin (the sidecar exclusion must not over-trigger) (Codex #6638)."""
    _install_fake_model_info(
        monkeypatch, ["model.safetensors", "pytorch_model.bin", "optimizer.safetensors"]
    )
    ig = U._prefetch_ignore_patterns("org/repo")
    assert "*.bin" in ig


def test_is_model_weight_safetensors_classification():
    """Direct unit coverage: real model weights count, adapter / trainer-state sidecars do not."""
    assert U._is_model_weight_safetensors("model.safetensors") is True
    assert U._is_model_weight_safetensors("model-00001-of-00002.safetensors") is True
    assert U._is_model_weight_safetensors("model.safetensors.index.json") is True
    assert U._is_model_weight_safetensors("consolidated.safetensors") is True
    assert U._is_model_weight_safetensors("adapter_model.safetensors") is False
    assert U._is_model_weight_safetensors("optimizer.safetensors") is False
    assert U._is_model_weight_safetensors("scheduler.safetensors") is False
    assert U._is_model_weight_safetensors("rng_state_0.safetensors") is False


def test_tokenizer_only_warms_slow_sentencepiece_vocab(capture):
    """tokenizer_only must warm the slow-tokenizer SentencePiece / BPE vocab files AutoTokenizer
    fetches first (sentencepiece.bpe.model for XLM-R / mBART, source.spm / target.spm for Marian,
    bpe.codes / vocab.bpe), so they are not left to an in-process Xet fetch (Codex #6638)."""
    _, st = capture(tokenizer_only = True)
    allow = st["allow_patterns"]
    for name in ("sentencepiece.bpe.model", "source.spm", "target.spm", "bpe.codes", "vocab.bpe"):
        assert name in allow, name


def test_adapter_safetensors_check_scoped_to_root(monkeypatch):
    """_adapter_repo_has_safetensors must only count a ROOT adapter_model*.safetensors: a repo with
    a root adapter_model.bin plus an unrelated checkpoint-5/adapter_model.safetensors must NOT drop
    the root .bin (the adapter warm only pulls root adapter_model*) (Codex #6638)."""
    import huggingface_hub

    class _Sib:
        def __init__(self, name):
            self.rfilename = name

    class _Api:
        def __init__(self, names):
            self._names = names

        def model_info(self, *a, **k):
            return type("MI", (), {"siblings": [_Sib(n) for n in self._names]})()

    # Subdir safetensors only -> not at root -> must NOT report safetensors present.
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: _Api(
            ["adapter_config.json", "adapter_model.bin", "checkpoint-5/adapter_model.safetensors"]
        ),
    )
    assert U._adapter_repo_has_safetensors("org/repo") is False
    # Root safetensors -> reported present.
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: _Api(["adapter_config.json", "adapter_model.safetensors"]),
    )
    assert U._adapter_repo_has_safetensors("org/repo") is True


def test_gguf_file_warm_keeps_gguf(capture):
    """A gguf_file load reads exactly that GGUF, so the warm must allow-list it (not drop *.gguf via
    the static ignore list) while not pulling other quants the repo may publish (Codex #6638)."""
    _, st = capture(weights_at_root = True, gguf_file = "model-Q4_K_M.gguf")
    allow = st["allow_patterns"]
    ig = st["ignore_patterns"]
    assert allow is not None and "model-Q4_K_M.gguf" in allow
    sample = [
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",  # a different quant the load does not read
        "config.json",
        "tokenizer.json",
    ]
    kept = _filter(sample, allow, ig)
    assert "model-Q4_K_M.gguf" in kept  # the requested GGUF is warmed
    assert "config.json" in kept  # root aux warmed
    assert "model-Q8_0.gguf" not in kept  # other quants are not pulled


# ----- Finding Q: adapter weight-format selection -----


def test_adapter_only_prefers_safetensors_over_bin(capture, monkeypatch):
    """A mixed-format adapter repo (adapter_model.safetensors AND adapter_model.bin) warms only
    the safetensors PeftModel.from_pretrained reads, not both formats (Codex #6638)."""
    _install_fake_model_info(
        monkeypatch, ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
    )
    _, st = capture(adapter_only = True)
    ig = st["ignore_patterns"]
    assert ig is not None and "adapter_model*.bin" in ig
    kept = _filter(
        ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"],
        st["allow_patterns"],
        ig,
    )
    assert "adapter_model.safetensors" in kept
    assert "adapter_model.bin" not in kept


def test_adapter_only_bin_only_keeps_bin(capture, monkeypatch):
    """A .bin-only adapter repo must keep adapter_model.bin -- never under-warm it into an
    in-process Xet fetch (best-effort: no safetensors found -> both formats eligible)."""
    _install_fake_model_info(monkeypatch, ["adapter_config.json", "adapter_model.bin"])
    _, st = capture(adapter_only = True)
    kept = _filter(
        ["adapter_config.json", "adapter_model.bin"], st["allow_patterns"], st["ignore_patterns"]
    )
    assert "adapter_model.bin" in kept


def test_adapter_only_explicit_use_safetensors_false_keeps_bin(capture):
    """An explicit use_safetensors=False forces the .bin form without a model_info call."""
    _, st = capture(adapter_only = True, use_safetensors = False)
    ig = st["ignore_patterns"]
    assert ig is not None and "adapter_model*.safetensors" in ig
    kept = _filter(
        ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"],
        st["allow_patterns"],
        ig,
    )
    assert "adapter_model.bin" in kept
    assert "adapter_model.safetensors" not in kept


def test_gguf_file_with_subfolder_warms_subfolder_path(capture):
    """gguf_file + subfolder: the load resolves <subfolder>/<gguf_file>, so the warm must allow-list
    that subfolder path, not the bare root name (Codex #6638)."""
    _, st = capture(weights_at_root = True, gguf_file = "model-Q4_K_M.gguf", subfolder = "gguf")
    allow = st["allow_patterns"]
    assert "gguf/model-Q4_K_M.gguf" in allow
    kept = _filter(["gguf/model-Q4_K_M.gguf", "config.json"], allow, st["ignore_patterns"])
    assert "gguf/model-Q4_K_M.gguf" in kept and "config.json" in kept


def test_from_tf_root_load_ignores_nested_h5(capture):
    """A from_tf root load reads the ROOT .h5; nested .h5 / .msgpack checkpoints under subdirs are
    unread, so the root-only subdir ignore must drop them (it covers every weight format, not only
    safetensors / bin) (Codex #6638)."""
    _, st = capture(weights_at_root = True, from_tf = True)
    ig = st["ignore_patterns"]
    assert "*/*.h5" in ig and "*/*.msgpack" in ig
    kept = _filter(["model.h5", "checkpoint-1/model.h5", "config.json"], st["allow_patterns"], ig)
    assert "model.h5" in kept  # root TF weight warmed
    assert "checkpoint-1/model.h5" not in kept  # nested TF checkpoint ignored


def test_sentence_transformer_from_pretrained_is_prefetch_wired():
    """FastSentenceTransformer.from_pretrained must warm the repo via maybe_prefetch_hf_snapshot as an
    UNCONDITIONAL top-level statement (so it fires on every load path: for_inference, fast-encoder,
    fallback) and before any top-level return. Static AST guard (importing ST pulls heavy optional
    deps); checking the call is top-level -- not nested in an if/for/try -- catches a dead-branch wire."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastSentenceTransformer"
    )
    fp = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")

    def _is_prefetch_call(node):
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "maybe_prefetch_hf_snapshot"
        )

    prefetch_pos = next((i for i, n in enumerate(fp.body) if _is_prefetch_call(n)), None)
    return_pos = next((i for i, n in enumerate(fp.body) if isinstance(n, ast.Return)), len(fp.body))
    assert (
        prefetch_pos is not None
    ), "from_pretrained must call maybe_prefetch_hf_snapshot at top level"
    assert prefetch_pos < return_pos, "prefetch must run before any top-level return"
