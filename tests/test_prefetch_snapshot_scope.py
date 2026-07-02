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
        state["variant"] = kw.get("variant")
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
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16", weights_at_root = True)
    assert "*.bin" not in ig
    # No variant: the default safetensors DOES make .bin redundant (existing behavior).
    ig_default = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" in ig_default


def test_variant_drops_bin_when_variant_safetensors_present(monkeypatch):
    """When a variant-matching safetensors (model.fp16.safetensors) is shipped, the variant load
    reads it and the variant .bin is redundant, so .bin is dropped from the warm."""
    _install_fake_model_info(monkeypatch, ["model.fp16.safetensors", "pytorch_model.fp16.bin"])
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16", weights_at_root = True)
    assert "*.bin" in ig


def test_no_variant_keeps_bin_when_only_variant_safetensors(monkeypatch):
    """A no-variant load reads pytorch_model.bin; a lone variant safetensors (model.fp16.safetensors)
    must NOT prove the .bin redundant -- only a CANONICAL safetensors does. Else the .bin the load reads
    is dropped from the warm and fetched in-process over Xet (Codex #6638)."""
    _install_fake_model_info(monkeypatch, ["model.fp16.safetensors", "pytorch_model.bin"])
    ig = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)  # variant unset
    assert "*.bin" not in ig
    # A canonical safetensors DOES make the .bin redundant for a no-variant load.
    _install_fake_model_info(monkeypatch, ["model.safetensors", "pytorch_model.bin"])
    ig2 = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" in ig2


def test_variant_keeps_bin_for_noncanonical_sidecar(monkeypatch):
    """With variant='fp16', a NON-canonical sidecar (consolidated.fp16.safetensors) must not prove the
    variant pytorch_model.fp16.bin redundant: a transformers variant load reads model.fp16.safetensors,
    not consolidated.*, so dropping the .bin would leave the only loadable weights to an in-process Xet
    fetch. The .bin stays warmed (Codex #6638)."""
    _install_fake_model_info(
        monkeypatch, ["consolidated.fp16.safetensors", "pytorch_model.fp16.bin"]
    )
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16", weights_at_root = True)
    assert "*.bin" not in ig
    # A canonical variant safetensors DOES make the variant .bin redundant.
    _install_fake_model_info(monkeypatch, ["model.fp16.safetensors", "pytorch_model.fp16.bin"])
    ig2 = U._prefetch_ignore_patterns("org/repo", variant = "fp16", weights_at_root = True)
    assert "*.bin" in ig2


def test_is_canonical_model_weight_safetensors():
    """The canonical detector matches only the non-variant model-weight safetensors names a default
    load reads, and rejects variant / sidecar names (Codex #6638)."""
    assert U._is_canonical_model_weight_safetensors("model.safetensors") is True
    assert U._is_canonical_model_weight_safetensors("model-00001-of-00002.safetensors") is True
    assert U._is_canonical_model_weight_safetensors("model.safetensors.index.json") is True
    assert U._is_canonical_model_weight_safetensors("model.fp16.safetensors") is False
    assert (
        U._is_canonical_model_weight_safetensors("model.fp16-00001-of-00002.safetensors") is False
    )
    assert U._is_canonical_model_weight_safetensors("adapter_model.safetensors") is False


def test_st_prefetch_resolves_env_cache_and_runs_after_validation():
    """The ST prefetch must resolve SENTENCE_TRANSFORMERS_HOME for its cache (so a load relying on that
    env is a cache hit, not an unprotected in-process download) and must run AFTER the mutually-exclusive
    load-mode validation (so a config rejected locally wastes no multi-GB download) (Codex #6638). Static
    guard: importing ST pulls heavy optional deps."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    prefetch_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "maybe_prefetch_hf_snapshot"
    ]
    assert len(prefetch_calls) == 1, "expected exactly one ST prefetch call"
    call = prefetch_calls[0]
    # F3: the cache_dir kwarg resolves SENTENCE_TRANSFORMERS_HOME.
    cache_dir_kw = next((kw for kw in call.keywords if kw.arg == "cache_dir"), None)
    assert cache_dir_kw is not None, "ST prefetch must pass cache_dir"
    assert "SENTENCE_TRANSFORMERS_HOME" in ast.dump(
        cache_dir_kw.value
    ), "ST prefetch cache_dir must resolve SENTENCE_TRANSFORMERS_HOME"
    # F2: the load-mode validation runs before the prefetch (fewer source lines = earlier).
    val_lineno = src[: src.index("Can only load in 4bit or 8bit or 16bit")].count("\n")
    assert val_lineno < call.lineno, "load-mode validation must precede the ST prefetch"


def test_st_cache_resolutions_honor_explicit_hf_cache_dir():
    """Every ST cache resolution (the prefetch and the fallback module loads) that falls back to
    SENTENCE_TRANSFORMERS_HOME must first honor an explicit HF cache_dir. The FastModel fallback load
    forwards kwargs['cache_dir'], so a caller passing cache_dir would otherwise warm one cache and read
    another, missing the warm and fetching in-process over Xet (Codex #6638). Static guard."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    resolutions = [
        kw
        for kw in ast.walk(tree)
        if isinstance(kw, ast.keyword)
        and kw.arg == "cache_dir"
        and "SENTENCE_TRANSFORMERS_HOME" in ast.dump(kw.value)
    ]
    assert resolutions, "expected cache_dir resolutions referencing SENTENCE_TRANSFORMERS_HOME"
    for kw in resolutions:
        assert "'cache_dir'" in ast.dump(
            kw.value
        ), "an ST cache_dir resolution must read an explicit kwargs.get('cache_dir') first"


def test_st_native_loads_map_hf_cache_dir_to_cache_folder():
    """The for_inference and fast-encoder branches construct a native SentenceTransformer, which takes
    cache_folder (not cache_dir). The prefetch warms cache_dir first, so an explicit HF cache_dir must be
    mapped onto cache_folder for those native loads; otherwise the load reads a different cache, misses the
    warm, and starts an unprotected in-process Xet download (Codex #6638). Static guard."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    # Every native SentenceTransformer(...) constructor that forwards cache_folder must read cache_dir.
    st_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "SentenceTransformer"
    ]
    cache_folder_kws = [kw for call in st_calls for kw in call.keywords if kw.arg == "cache_folder"]
    assert cache_folder_kws, "expected a native SentenceTransformer call forwarding cache_folder"
    for kw in cache_folder_kws:
        assert "'cache_dir'" in ast.dump(
            kw.value
        ), "a native SentenceTransformer cache_folder must map the explicit HF cache_dir first"
    # The for_inference branch feeds cache_folder through st_kwargs; it must map cache_dir there too, and
    # both native branches resolve cache_dir -> cache_folder (reformatting-tolerant normalized check).
    normalized = "".join(src.split())
    assert (
        'st_kwargs["cache_folder"]=' in normalized
    ), "for_inference must set st_kwargs cache_folder"
    assert (
        normalized.count('kwargs.get("cache_dir")orkwargs.get("cache_folder")') >= 2
    ), "both native ST branches (for_inference, fast-encoder) must map cache_dir -> cache_folder"


def test_vision_warms_vllm_tokenizer_after_remap():
    """On the vLLM path the base warm is skipped and the tokenizer warm is deferred until after
    fast_inference_setup may remap model_name. The final tokenizer repo must then be warmed (tokenizer
    only) so the in-process processor / tokenizer load is a cache hit, not an unprotected Xet fetch
    (Codex #6638). Static guard: the vLLM-gated tokenizer warm appears after the remap."""
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "vision.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        src = f.read()
    guard = "if _vllm_owns_weights and isinstance(tokenizer_name"
    assert guard in src, "expected a vLLM-gated tokenizer warm"
    assert src.index(guard) > src.index(
        "fast_inference_setup("
    ), "the vLLM tokenizer warm must run after the fast_inference_setup remap"


def test_diffusion_forwards_variant_to_real_load():
    """FastDiffusionModel must forward `variant` to the real model_cls.from_pretrained load, not only to
    the prefetch: without it the pipeline asks for the default weight variant, missing the warmed variant
    weights (wrong precision, or a default weight fetched in-process over Xet) (Codex #6638). Static
    guard."""
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "diffusion.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        src = f.read()
    assert (
        'load_kwargs["variant"] = kwargs["variant"]' in src
    ), "the diffusion load must forward variant to model_cls.from_pretrained"


def test_vision_prefetch_runs_after_load_mode_validation():
    """The FastBaseModel (vision / FastModel) prefetch must run AFTER the mutually-exclusive load-mode
    validation, so an invalid load_in_4bit/8bit/16bit combination fails locally without first downloading
    a multi-GB snapshot (Codex #6638). check_and_disable_bitsandbytes_loading can only resolve after the
    config fetch, so the check cannot move earlier; the prefetch moves after it instead. Static guard:
    importing the loader pulls heavy optional deps."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "vision.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    prefetch_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "maybe_prefetch_hf_snapshot"
    ]
    assert prefetch_calls, "expected a vision prefetch call"
    first_prefetch = min(call.lineno for call in prefetch_calls)
    val_lineno = src[: src.index("Can only load in 4bit or 8bit or 16bit")].count("\n")
    assert val_lineno < first_prefetch, "load-mode validation must precede the vision prefetch"


def test_llama_prefetch_skips_only_real_vllm_loads():
    """A num_labels classification load takes the AutoModelForSequenceClassification branch (an in-process
    download) even under fast_inference=True, so the llama prefetch's fast_inference skip must be gated on
    num_labels is None -- else that load's weights fetch over un-killable Xet (Codex #6638). Static guard:
    the base prefetch's fast_inference kwarg references both fast_inference and num_labels."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "llama.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    gated = False
    for n in ast.walk(tree):
        if not (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "maybe_prefetch_hf_snapshot"
        ):
            continue
        fi_kw = next((kw for kw in n.keywords if kw.arg == "fast_inference"), None)
        if fi_kw is None:
            continue
        dumped = ast.dump(fi_kw.value)
        if "fast_inference" in dumped and "num_labels" in dumped:
            gated = True
    assert gated, "llama prefetch fast_inference must be gated on num_labels is None"


def test_st_fallback_module_loads_resolve_env_cache():
    """The fallback module loads must resolve the SAME cache the prefetch warmed. _module_path /
    _read_pooling_mode call hf_hub_download directly, which does NOT honor SENTENCE_TRANSFORMERS_HOME,
    so any cache_dir derived from cache_folder must also fall back to the env var; otherwise, when a
    caller relies on SENTENCE_TRANSFORMERS_HOME without passing cache_folder, modules.json / module
    files miss the warm and are fetched in-process over Xet (Codex #6638). Static guard: importing ST
    pulls heavy optional deps."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        src = f.read()
    tree = ast.parse(src)

    # Every _module_path / _load_modules call whose cache_dir is derived from cache_folder (i.e. the
    # from_pretrained fallback sites, not the internal `cache_dir = cache_dir` pass-throughs) must also
    # resolve SENTENCE_TRANSFORMERS_HOME so the resolution matches the prefetch above.
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("_module_path", "_load_modules"):
            continue
        cache_dir_kw = next((kw for kw in node.keywords if kw.arg == "cache_dir"), None)
        if cache_dir_kw is None:
            continue
        dumped = ast.dump(cache_dir_kw.value)
        if "cache_folder" not in dumped:
            continue  # internal pass-through (cache_dir = cache_dir): not a resolution site
        checked += 1
        assert (
            "SENTENCE_TRANSFORMERS_HOME" in dumped
        ), f"{node.func.attr} cache_dir resolves cache_folder but not SENTENCE_TRANSFORMERS_HOME"
    assert (
        checked >= 2
    ), "expected the fallback _module_path and _load_modules calls to resolve the env cache"


def test_st_fallback_module_loads_forward_revision():
    """A revision-pinned ST repo loaded via the custom fallback path loads its model WEIGHTS from the
    requested revision (FastModel forwards revision to the weight load), so the module files (modules.json,
    pooling config, per-module dirs) must load from the SAME revision. Otherwise they resolve the repo
    default branch: fetched in-process over Xet (missing the prefetch's revision-pinned warm) and mixed
    with the revision-pinned weights (Codex #6638). Static guard: (a) _module_path / _read_pooling_mode /
    _load_modules accept a revision arg, (b) every hf_hub_download / load_dir_path inside them forwards
    revision, (c) _load_modules threads revision into its internal _module_path / _read_pooling_mode
    calls, (d) the from_pretrained fallback _module_path / _load_modules calls forward revision. Importing
    ST pulls heavy optional deps."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())

    funcs = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_module_path", "_read_pooling_mode", "_load_modules")
    }
    assert set(funcs) == {"_module_path", "_read_pooling_mode", "_load_modules"}

    # (a) each helper takes a revision parameter.
    for name, fn in funcs.items():
        arg_names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        assert "revision" in arg_names, f"{name} must accept a revision argument"

    # (b) every direct download primitive inside the helpers forwards revision.
    downloads = 0
    for name, fn in funcs.items():
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in ("hf_hub_download", "load_dir_path"):
                continue
            downloads += 1
            assert any(
                kw.arg == "revision" for kw in node.keywords
            ), f"{node.func.id} in {name} must forward revision"
    assert downloads >= 3, "expected the module-download primitives to be revision-guarded"

    # (c) _load_modules threads revision into its internal _module_path / _read_pooling_mode calls.
    internal = 0
    for node in ast.walk(funcs["_load_modules"]):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("_module_path", "_read_pooling_mode"):
            continue
        internal += 1
        assert any(
            kw.arg == "revision" for kw in node.keywords
        ), f"_load_modules must forward revision to {node.func.attr}"
    assert internal >= 2, "expected _load_modules to call _module_path and _read_pooling_mode"

    # (d) the from_pretrained fallback _module_path / _load_modules sites forward revision.
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("_module_path", "_load_modules"):
            continue
        cache_dir_kw = next((kw for kw in node.keywords if kw.arg == "cache_dir"), None)
        if cache_dir_kw is None or "cache_folder" not in ast.dump(cache_dir_kw.value):
            continue  # internal pass-through, not a from_pretrained fallback site
        checked += 1
        rev_kw = next((kw for kw in node.keywords if kw.arg == "revision"), None)
        assert rev_kw is not None and "revision" in ast.dump(
            rev_kw.value
        ), f"{node.func.attr} fallback call must forward revision"
    assert (
        checked >= 2
    ), "expected the fallback _module_path and _load_modules calls to forward revision"


def test_st_fallback_model_load_resolves_env_cache():
    """The fallback FastModel weight load resolves its cache from the HF cache_dir, not ST's cache_folder /
    SENTENCE_TRANSFORMERS_HOME. from_pretrained must therefore resolve the SAME cache the prefetch warmed
    into kwargs['cache_dir'] BEFORE the FastModel.from_pretrained call -- else the weights miss the warm
    and start an unprotected in-process Xet download (Codex #6638). Static guard: importing ST pulls heavy
    optional deps."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())

    def _resolves_st_cache(value_node):
        # The resolution may be inline in the assigned value, or in the assignment to the intermediate
        # variable the value references (kwargs['cache_dir'] = _st_cache_dir; _st_cache_dir = ...).
        dumped = ast.dump(value_node)
        if "cache_folder" in dumped and "SENTENCE_TRANSFORMERS_HOME" in dumped:
            return True
        if isinstance(value_node, ast.Name):
            for n in ast.walk(tree):
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == value_node.id for t in n.targets
                ):
                    d = ast.dump(n.value)
                    if "cache_folder" in d and "SENTENCE_TRANSFORMERS_HOME" in d:
                        return True
        return False

    resolved_lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (
                isinstance(tgt, ast.Subscript)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "kwargs"
                and isinstance(tgt.slice, ast.Constant)
                and tgt.slice.value == "cache_dir"
                and _resolves_st_cache(node.value)
            ):
                resolved_lines.append(node.lineno)
    assert resolved_lines, "from_pretrained must resolve the ST cache into kwargs['cache_dir']"

    fastmodel_calls = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "from_pretrained"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "FastModel"
    ]
    assert fastmodel_calls, "expected a FastModel.from_pretrained call"
    assert min(resolved_lines) < min(
        fastmodel_calls
    ), "kwargs['cache_dir'] must be resolved before the fallback FastModel weight load"


def test_canonical_variant_model_weight_matches_transformers_names():
    """The variant safetensors detector matches only CANONICAL model variant names a transformers load
    reads (single, either shard infix, and the variant index) and rejects a non-canonical sidecar
    (consolidated.fp16.safetensors) so its variant .bin is never wrongly dropped, plus the default and
    wrong-variant names (Codex #6638)."""
    f = U._is_canonical_variant_model_weight_safetensors
    assert f("model.fp16.safetensors", "fp16") is True
    assert f("model.fp16-00001-of-00002.safetensors", "fp16") is True
    assert f("model-00001-of-00002.fp16.safetensors", "fp16") is True
    assert f("model.safetensors.index.fp16.json", "fp16") is True
    # A non-canonical sidecar variant does NOT prove the .bin redundant (the M2 hang guard).
    assert f("consolidated.fp16.safetensors", "fp16") is False
    # Default (non-variant) and wrong-variant names are not a match for variant='fp16'.
    assert f("model.safetensors", "fp16") is False
    assert f("model-00001-of-00002.safetensors", "fp16") is False
    assert f("model.bf16.safetensors", "fp16") is False


def test_variant_is_forwarded_to_downloader(capture):
    """maybe_prefetch_hf_snapshot must forward `variant` to snapshot_download_with_xet_fallback so the
    PRE cache-skip gate can defer on a variant load: a cache holding only the default canonical weight
    must not fast-path a variant='fp16' request, else the in-process load fetches the missing variant
    weight over un-killable Xet. Absent a variant, nothing is forwarded (the fast path stays live)."""
    _, st = capture(weights_at_root = True, use_safetensors = True, variant = "fp16")
    assert st["variant"] == "fp16"
    _, st = capture(weights_at_root = True, use_safetensors = True)
    assert st["variant"] is None


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
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16", weights_at_root = True)
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
    ig = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" not in ig  # .bin is the only real weight -> not dropped


def test_model_safetensors_still_drops_bin(monkeypatch):
    """Control for the optimizer case: a real model.safetensors next to pytorch_model.bin still
    drops the redundant .bin (the sidecar exclusion must not over-trigger) (Codex #6638)."""
    _install_fake_model_info(
        monkeypatch, ["model.safetensors", "pytorch_model.bin", "optimizer.safetensors"]
    )
    ig = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" in ig


def test_whole_multi_component_snapshot_keeps_subdir_bin(monkeypatch):
    """A whole multi-component snapshot (weights_at_root=False, no subfolder: a SentenceTransformer /
    diffusers repo) must NOT drop *.bin even when root safetensors exist -- HF's "*" spans "/", so the
    drop would strip a subdir module's only weight (1_Dense/pytorch_model.bin) and leave the module load
    to an in-process Xet fetch. A root-scoped load of the same repo still drops the redundant root .bin
    (Codex #6638)."""
    _install_fake_model_info(monkeypatch, ["model.safetensors", "1_Dense/pytorch_model.bin"])
    ig = U._prefetch_ignore_patterns("org/repo", weights_at_root = False)
    assert "*.bin" not in ig
    ig_root = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" in ig_root


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
    bpe.codes / vocab.bpe, sentencepiece.model for RemBERT, vocab-src.json / vocab-tgt.json for FSMT),
    so they are not left to an in-process Xet fetch (Codex #6638)."""
    _, st = capture(tokenizer_only = True)
    allow = st["allow_patterns"]
    for name in (
        "sentencepiece.bpe.model",
        "source.spm",
        "target.spm",
        "bpe.codes",
        "vocab.bpe",
        "sentencepiece.model",
        "vocab-src.json",
        "vocab-tgt.json",
    ):
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
    # local_files_only must be forwarded so an offline / cache-only load does not start a Hub download
    # via the prefetch before the ST load sees the flag (Codex #6638).
    prefetch_call = fp.body[prefetch_pos].value
    assert "local_files_only" in {
        kw.arg for kw in prefetch_call.keywords
    }, "prefetch must forward local_files_only"


def test_st_module_download_forwards_cache_folder():
    """_load_modules must forward the custom cache_folder into load_dir_path so per-module subdirs are
    read from the warmed cache rather than the default one, avoiding a second in-process Hub/Xet fetch
    (Codex #6638). Static AST guard (importing ST pulls heavy optional deps)."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "load_dir_path"
    ]
    assert calls, "expected a load_dir_path call in sentence_transformer.py"
    assert all(
        "cache_folder" in {kw.arg for kw in c.keywords} for c in calls
    ), "every load_dir_path call must forward cache_folder"


def test_st_native_sentence_transformer_calls_forward_cache_folder():
    """Every native SentenceTransformer(model_name, ...) load (for_inference AND fast-encoder) must
    forward cache_folder, so a custom cache_folder reads the cache the prefetch warmed instead of
    missing it and starting an unprotected in-process Hub/Xet download (Codex #6638). The modules-based
    SentenceTransformer(modules=...) call builds from already-loaded modules and needs no cache_folder.
    Static AST guard (importing ST pulls heavy optional deps)."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    weight_loading_calls = []
    for n in ast.walk(tree):
        if not (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "SentenceTransformer"
        ):
            continue
        kw_names = {kw.arg for kw in n.keywords}
        # A modules-based build (SentenceTransformer(modules=...)) downloads nothing; only a
        # repo-name load (positional model_name, no modules=) reads the cache.
        if "modules" in kw_names:
            continue
        weight_loading_calls.append(n)
    assert (
        weight_loading_calls
    ), "expected a repo-name SentenceTransformer load in sentence_transformer.py"
    # cache_folder is forwarded either explicitly (fast-encoder branch) or via a **kwargs unpacking
    # (for_inference branch builds st_kwargs incl. cache_folder). A ** unpacking has kw.arg == None.
    for c in weight_loading_calls:
        kw_names = {kw.arg for kw in c.keywords}
        forwards = "cache_folder" in kw_names or None in kw_names
        assert forwards, (
            "a repo-name SentenceTransformer load must forward cache_folder "
            f"(explicitly or via **kwargs) at line {c.lineno}"
        )
