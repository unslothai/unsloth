# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Pure-CPU, no-network unit tests for prefetch snapshot scoping in unsloth/models/_utils.py.

maybe_prefetch_hf_snapshot warms the HF cache before the in-process load. The warm must cover at
least what the load reads (else the missing file falls to an unprotected in-process Xet fetch) but
not pull weights the load never reads. These tests lock the allow/ignore patterns each mode hands
snapshot_download_with_xet_fallback. The zoo downloader is monkeypatched to capture its kwargs.
"""

import fnmatch
import sys
import types

import pytest

from unsloth.models import _utils as U


def _filter(names, allow_patterns, ignore_patterns):
    """Mirror HF filter_repo_objects: keep on allow match (or None), drop on ignore match."""
    kept = []
    for name in names:
        if allow_patterns is not None and not any(
            fnmatch.fnmatch(name, p) for p in allow_patterns
        ):
            continue
        if ignore_patterns and any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
            continue
        kept.append(name)
    return kept


@pytest.fixture
def capture(monkeypatch):
    """Run maybe_prefetch_hf_snapshot with a fake repo, capturing the patterns forwarded to a
    fake injected zoo downloader (independent of the installed unsloth_zoo). Offline env cleared."""
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

    # Neutralize the model_info network call by default; tests exercising format selection
    # install their own.
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


# Representative repo listing: root weights + aux, subdir, adapter, checkpoint, merged weights.
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
    """A root load ignores subdir weights (fp16/, experimental/, checkpoint-500/) but keeps root weights."""
    ok, st = capture(weights_at_root = True, use_safetensors = True)
    assert ok is True
    assert st["allow_patterns"] is None
    ig = st["ignore_patterns"]
    assert "*/*.safetensors" in ig and "*/*.bin" in ig
    kept = _filter(_SAMPLE_FILES, st["allow_patterns"], ig)
    assert "model-00001-of-00002.safetensors" in kept
    assert "model.safetensors.index.json" in kept
    assert "config.json" in kept
    assert "fp16/model.safetensors" not in kept
    assert "experimental/model-00001-of-00002.safetensors" not in kept
    assert "checkpoint-500/model.safetensors" not in kept


def test_adapter_only_excludes_merged_weights(capture):
    """An adapter warm keeps adapter files + root aux, not merged full-model weights."""
    ok, st = capture(adapter_only = True)
    assert ok is True
    assert st["ignore_patterns"] is None
    allow = st["allow_patterns"]
    assert "adapter_config.json" in allow and "adapter_model*" in allow
    kept = _filter(_SAMPLE_FILES, allow, st["ignore_patterns"])
    assert "adapter_config.json" in kept
    assert "adapter_model.safetensors" in kept
    assert "config.json" in kept and "tokenizer.json" in kept
    assert "model-00001-of-00002.safetensors" not in kept
    assert "pytorch_model.bin" not in kept
    assert "fp16/model.safetensors" not in kept


def test_adapter_only_warms_sharded_adapter(capture):
    """A sharded adapter is still covered by the adapter_model* glob."""
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
    """A tokenizer-only repo warms tokenizer/config/vocab files, never weights."""
    _, st = capture(tokenizer_only = True)
    assert st["ignore_patterns"] is None
    assert st["allow_patterns"] == list(U._ROOT_AUX_PREFETCH_PATTERNS)
    kept = _filter(_SAMPLE_FILES, st["allow_patterns"], st["ignore_patterns"])
    assert "tokenizer.json" in kept and "config.json" in kept
    assert "model-00001-of-00002.safetensors" not in kept
    assert "adapter_model.safetensors" not in kept


def test_aux_warm_covers_arbitrary_remote_code_modules(capture):
    """The aux warm must cover any *.py, since trust_remote_code auto_map names modules freely."""
    _, st = capture(tokenizer_only = True)
    allow = st["allow_patterns"]
    assert "*.py" in allow
    remote_code = [
        "config.json",
        "modeling.py",
        "tokenization.py",
        "my_custom_code.py",
        "configuration_foo.py",
    ]
    kept = _filter(remote_code, allow, st["ignore_patterns"])
    for name in (
        "modeling.py",
        "tokenization.py",
        "my_custom_code.py",
        "configuration_foo.py",
    ):
        assert name in kept, name


def test_subfolder_warms_subfolder_plus_root_aux(capture):
    """A subfolder load warms that subfolder's weights plus root aux; other subdirs/root weights skipped."""
    _, st = capture(subfolder = "fp16")
    allow = st["allow_patterns"]
    assert "fp16/*" in allow
    assert all(p in allow for p in U._ROOT_AUX_PREFETCH_PATTERNS)
    kept = _filter(_SAMPLE_FILES, allow, st["ignore_patterns"])
    assert "fp16/model.safetensors" in kept
    assert "config.json" in kept
    assert "experimental/model-00001-of-00002.safetensors" not in kept


def test_subfolder_takes_precedence_over_weights_at_root(capture):
    """When a subfolder is requested the subfolder branch wins over weights_at_root."""
    _, st = capture(subfolder = "fp16", weights_at_root = True)
    assert "fp16/*" in st["allow_patterns"]
    kept = _filter(_SAMPLE_FILES, st["allow_patterns"], st["ignore_patterns"])
    assert "fp16/model.safetensors" in kept


def test_local_dir_is_not_warmed(capture, tmp_path):
    """A local directory path skips the warm (returns False)."""
    d = tmp_path / "local-model"
    d.mkdir()
    ok = U.maybe_prefetch_hf_snapshot(str(d), weights_at_root = True)
    assert ok is False


def _install_fake_model_info(monkeypatch, filenames):
    """Make HfApi().model_info(...).siblings report filenames, with no network."""
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
    """A default model.safetensors must not prove a variant .bin redundant; without a variant it does."""
    _install_fake_model_info(
        monkeypatch, ["model.safetensors", "pytorch_model.fp16.bin"]
    )
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16", weights_at_root = True)
    assert "*.bin" not in ig
    ig_default = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" in ig_default


def test_variant_drops_bin_when_variant_safetensors_present(monkeypatch):
    """A variant-matching safetensors makes the variant .bin redundant, so .bin is dropped."""
    _install_fake_model_info(
        monkeypatch, ["model.fp16.safetensors", "pytorch_model.fp16.bin"]
    )
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16", weights_at_root = True)
    assert "*.bin" in ig


def test_no_variant_keeps_bin_when_only_variant_safetensors(monkeypatch):
    """For a no-variant load, only a canonical safetensors (not a lone variant) makes .bin redundant."""
    _install_fake_model_info(
        monkeypatch, ["model.fp16.safetensors", "pytorch_model.bin"]
    )
    ig = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" not in ig
    _install_fake_model_info(monkeypatch, ["model.safetensors", "pytorch_model.bin"])
    ig2 = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" in ig2


def test_variant_keeps_bin_for_noncanonical_sidecar(monkeypatch):
    """A non-canonical variant sidecar must not prove the variant .bin redundant; a canonical one does."""
    _install_fake_model_info(
        monkeypatch, ["consolidated.fp16.safetensors", "pytorch_model.fp16.bin"]
    )
    ig = U._prefetch_ignore_patterns("org/repo", variant = "fp16", weights_at_root = True)
    assert "*.bin" not in ig
    _install_fake_model_info(
        monkeypatch, ["model.fp16.safetensors", "pytorch_model.fp16.bin"]
    )
    ig2 = U._prefetch_ignore_patterns("org/repo", variant = "fp16", weights_at_root = True)
    assert "*.bin" in ig2


def test_is_canonical_model_weight_safetensors():
    """The canonical detector matches only non-variant model-weight safetensors names."""
    assert U._is_canonical_model_weight_safetensors("model.safetensors") is True
    assert (
        U._is_canonical_model_weight_safetensors("model-00001-of-00002.safetensors")
        is True
    )
    assert (
        U._is_canonical_model_weight_safetensors("model.safetensors.index.json") is True
    )
    assert U._is_canonical_model_weight_safetensors("model.fp16.safetensors") is False
    assert (
        U._is_canonical_model_weight_safetensors(
            "model.fp16-00001-of-00002.safetensors"
        )
        is False
    )
    assert (
        U._is_canonical_model_weight_safetensors("adapter_model.safetensors") is False
    )


def test_st_prefetch_resolves_env_cache_and_runs_after_validation():
    """The ST prefetch must resolve SENTENCE_TRANSFORMERS_HOME and run after load-mode validation."""
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
    # cache_dir kwarg resolves SENTENCE_TRANSFORMERS_HOME.
    cache_dir_kw = next((kw for kw in call.keywords if kw.arg == "cache_dir"), None)
    assert cache_dir_kw is not None, "ST prefetch must pass cache_dir"
    assert "SENTENCE_TRANSFORMERS_HOME" in ast.dump(
        cache_dir_kw.value
    ), "ST prefetch cache_dir must resolve SENTENCE_TRANSFORMERS_HOME"
    # Load-mode validation runs before the prefetch (fewer source lines = earlier).
    val_lineno = src[: src.index("Can only load in 4bit or 8bit or 16bit")].count("\n")
    assert val_lineno < call.lineno, "load-mode validation must precede the ST prefetch"


def test_st_cache_resolutions_honor_explicit_hf_cache_dir():
    """Every ST cache resolution falling back to SENTENCE_TRANSFORMERS_HOME must first honor an explicit HF cache_dir."""
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
    assert (
        resolutions
    ), "expected cache_dir resolutions referencing SENTENCE_TRANSFORMERS_HOME"
    for kw in resolutions:
        assert (
            "'cache_dir'" in ast.dump(kw.value)
        ), "an ST cache_dir resolution must read an explicit kwargs.get('cache_dir') first"


def test_st_native_loads_map_hf_cache_dir_to_cache_folder():
    """Native SentenceTransformer loads take cache_folder, so an explicit HF cache_dir must be mapped onto it."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    # Every native SentenceTransformer(...) forwarding cache_folder must read cache_dir.
    st_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "SentenceTransformer"
    ]
    cache_folder_kws = [
        kw for call in st_calls for kw in call.keywords if kw.arg == "cache_folder"
    ]
    assert (
        cache_folder_kws
    ), "expected a native SentenceTransformer call forwarding cache_folder"
    for kw in cache_folder_kws:
        assert (
            "'cache_dir'" in ast.dump(kw.value)
        ), "a native SentenceTransformer cache_folder must map the explicit HF cache_dir first"
    # for_inference feeds cache_folder via st_kwargs; both native branches map cache_dir -> cache_folder.
    normalized = "".join(src.split())
    assert (
        'st_kwargs["cache_folder"]=' in normalized
    ), "for_inference must set st_kwargs cache_folder"
    assert (
        normalized.count('kwargs.get("cache_dir")orkwargs.get("cache_folder")') >= 2
    ), "both native ST branches (for_inference, fast-encoder) must map cache_dir -> cache_folder"


def test_vision_warms_vllm_tokenizer_after_remap():
    """On the vLLM path the tokenizer warm is deferred until after the fast_inference_setup remap."""
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
    """FastDiffusionModel must forward variant to the real model_cls.from_pretrained load, not just the prefetch."""
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "diffusion.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        src = f.read()
    assert (
        'load_kwargs["variant"] = kwargs["variant"]' in src
    ), "the diffusion load must forward variant to model_cls.from_pretrained"


def test_vision_prefetch_runs_after_load_mode_validation():
    """The FastBaseModel (vision) prefetch must run after the load-mode validation."""
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
    assert (
        val_lineno < first_prefetch
    ), "load-mode validation must precede the vision prefetch"


def test_llama_prefetch_skips_only_real_vllm_loads():
    """The llama prefetch's fast_inference skip must be gated on num_labels is None (a classification load still downloads)."""
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
    """Fallback module loads deriving cache_dir from cache_folder must also fall back to SENTENCE_TRANSFORMERS_HOME."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        src = f.read()
    tree = ast.parse(src)

    # Fallback sites (cache_dir derived from cache_folder) must resolve SENTENCE_TRANSFORMERS_HOME.
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
            continue  # internal pass-through, not a resolution site
        checked += 1
        assert (
            "SENTENCE_TRANSFORMERS_HOME" in dumped
        ), f"{node.func.attr} cache_dir resolves cache_folder but not SENTENCE_TRANSFORMERS_HOME"
    assert (
        checked >= 2
    ), "expected the fallback _module_path and _load_modules calls to resolve the env cache"


def test_st_fallback_module_loads_forward_revision():
    """The fallback module loads must forward revision so module files match the revision-pinned weights.
    Guards: (a) helpers accept revision, (b) every download primitive forwards it, (c) _load_modules
    threads it into internal calls, (d) the from_pretrained fallback sites forward it."""
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

    # (b) every download primitive inside the helpers forwards revision.
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
    assert (
        downloads >= 3
    ), "expected the module-download primitives to be revision-guarded"

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
    assert (
        internal >= 2
    ), "expected _load_modules to call _module_path and _read_pooling_mode"

    # (d) the from_pretrained fallback _module_path / _load_modules sites forward revision.
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("_module_path", "_load_modules"):
            continue
        cache_dir_kw = next((kw for kw in node.keywords if kw.arg == "cache_dir"), None)
        if cache_dir_kw is None or "cache_folder" not in ast.dump(cache_dir_kw.value):
            continue  # internal pass-through, not a fallback site
        checked += 1
        rev_kw = next((kw for kw in node.keywords if kw.arg == "revision"), None)
        assert rev_kw is not None and "revision" in ast.dump(
            rev_kw.value
        ), f"{node.func.attr} fallback call must forward revision"
    assert (
        checked >= 2
    ), "expected the fallback _module_path and _load_modules calls to forward revision"


def test_st_fallback_model_load_resolves_env_cache():
    """from_pretrained must resolve the warmed ST cache into kwargs['cache_dir'] before the FastModel weight load."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())

    def _resolves_st_cache(value_node):
        # Resolution may be inline or in the assignment to an intermediate variable the value references.
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
    assert (
        resolved_lines
    ), "from_pretrained must resolve the ST cache into kwargs['cache_dir']"

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
    """The variant safetensors detector matches only canonical variant names, rejecting sidecars and wrong variants."""
    f = U._is_canonical_variant_model_weight_safetensors
    assert f("model.fp16.safetensors", "fp16") is True
    assert f("model.fp16-00001-of-00002.safetensors", "fp16") is True
    assert f("model-00001-of-00002.fp16.safetensors", "fp16") is True
    assert f("model.safetensors.index.fp16.json", "fp16") is True
    assert f("consolidated.fp16.safetensors", "fp16") is False
    assert f("model.safetensors", "fp16") is False
    assert f("model-00001-of-00002.safetensors", "fp16") is False
    assert f("model.bf16.safetensors", "fp16") is False


def test_variant_is_forwarded_to_downloader(capture):
    """maybe_prefetch_hf_snapshot must forward variant to the downloader (absent a variant, nothing is forwarded)."""
    _, st = capture(weights_at_root = True, use_safetensors = True, variant = "fp16")
    assert st["variant"] == "fp16"
    _, st = capture(weights_at_root = True, use_safetensors = True)
    assert st["variant"] is None


def test_variant_drops_bin_for_sharded_variant_safetensors(monkeypatch):
    """A sharded variant safetensors is recognized, so its redundant variant .bin is dropped."""
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
    """tokenizer_only must warm SentencePiece / vocab / processor files, including a named jinja template."""
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
    """A cached config.json must not skip the weight-format probe; model_info still drops the redundant .bin."""
    import huggingface_hub

    # Pretend config.json is cached (the AutoConfig side effect); this must not gate the probe.
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: "/cache/config.json"
    )
    _install_fake_model_info(monkeypatch, ["model.safetensors", "pytorch_model.bin"])
    _, st = capture(weights_at_root = True)
    ig = st["ignore_patterns"] or []
    assert "*.bin" in ig


def test_optimizer_safetensors_does_not_drop_bin(monkeypatch):
    """An optimizer.safetensors sidecar must not count as model safetensors, so the real .bin weights are kept."""
    _install_fake_model_info(
        monkeypatch, ["pytorch_model.bin", "optimizer.safetensors"]
    )
    ig = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" not in ig


def test_model_safetensors_still_drops_bin(monkeypatch):
    """Control for the optimizer case: a real model.safetensors next to pytorch_model.bin still drops the .bin."""
    _install_fake_model_info(
        monkeypatch, ["model.safetensors", "pytorch_model.bin", "optimizer.safetensors"]
    )
    ig = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" in ig


def test_whole_multi_component_snapshot_keeps_subdir_bin(monkeypatch):
    """A whole multi-component snapshot must not drop *.bin (it would strip a subdir module's weight); a root load still does."""
    _install_fake_model_info(
        monkeypatch, ["model.safetensors", "1_Dense/pytorch_model.bin"]
    )
    ig = U._prefetch_ignore_patterns("org/repo", weights_at_root = False)
    assert "*.bin" not in ig
    ig_root = U._prefetch_ignore_patterns("org/repo", weights_at_root = True)
    assert "*.bin" in ig_root


def test_is_model_weight_safetensors_classification():
    """Real model weights count; adapter / trainer-state sidecars do not."""
    assert U._is_model_weight_safetensors("model.safetensors") is True
    assert U._is_model_weight_safetensors("model-00001-of-00002.safetensors") is True
    assert U._is_model_weight_safetensors("model.safetensors.index.json") is True
    assert U._is_model_weight_safetensors("consolidated.safetensors") is True
    assert U._is_model_weight_safetensors("adapter_model.safetensors") is False
    assert U._is_model_weight_safetensors("optimizer.safetensors") is False
    assert U._is_model_weight_safetensors("scheduler.safetensors") is False
    assert U._is_model_weight_safetensors("rng_state_0.safetensors") is False


def test_tokenizer_only_warms_slow_sentencepiece_vocab(capture):
    """tokenizer_only must warm the slow-tokenizer SentencePiece / BPE vocab files AutoTokenizer fetches first."""
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
    """_adapter_repo_has_safetensors must only count a root adapter_model*.safetensors, not a subdir one."""
    import huggingface_hub

    class _Sib:
        def __init__(self, name):
            self.rfilename = name

    class _Api:
        def __init__(self, names):
            self._names = names

        def model_info(self, *a, **k):
            return type("MI", (), {"siblings": [_Sib(n) for n in self._names]})()

    # Subdir safetensors only -> not reported present.
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: _Api(
            [
                "adapter_config.json",
                "adapter_model.bin",
                "checkpoint-5/adapter_model.safetensors",
            ]
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
    """A gguf_file load allow-lists that GGUF while not pulling other quants the repo publishes."""
    _, st = capture(weights_at_root = True, gguf_file = "model-Q4_K_M.gguf")
    allow = st["allow_patterns"]
    ig = st["ignore_patterns"]
    assert allow is not None and "model-Q4_K_M.gguf" in allow
    sample = [
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
        "config.json",
        "tokenizer.json",
    ]
    kept = _filter(sample, allow, ig)
    assert "model-Q4_K_M.gguf" in kept
    assert "config.json" in kept
    assert "model-Q8_0.gguf" not in kept


# ----- Finding Q: adapter weight-format selection -----


def test_adapter_only_prefers_safetensors_over_bin(capture, monkeypatch):
    """A mixed-format adapter repo warms only the safetensors PeftModel reads, not both formats."""
    _install_fake_model_info(
        monkeypatch,
        ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"],
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
    """A .bin-only adapter repo must keep adapter_model.bin (no safetensors found -> both formats eligible)."""
    _install_fake_model_info(monkeypatch, ["adapter_config.json", "adapter_model.bin"])
    _, st = capture(adapter_only = True)
    kept = _filter(
        ["adapter_config.json", "adapter_model.bin"],
        st["allow_patterns"],
        st["ignore_patterns"],
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
    """gguf_file + subfolder: the warm allow-lists <subfolder>/<gguf_file>, not the bare root name."""
    _, st = capture(
        weights_at_root = True, gguf_file = "model-Q4_K_M.gguf", subfolder = "gguf"
    )
    allow = st["allow_patterns"]
    assert "gguf/model-Q4_K_M.gguf" in allow
    kept = _filter(
        ["gguf/model-Q4_K_M.gguf", "config.json"], allow, st["ignore_patterns"]
    )
    assert "gguf/model-Q4_K_M.gguf" in kept and "config.json" in kept


def test_from_tf_root_load_ignores_nested_h5(capture):
    """A from_tf root load keeps the root .h5 but drops nested .h5 / .msgpack checkpoints."""
    _, st = capture(weights_at_root = True, from_tf = True)
    ig = st["ignore_patterns"]
    assert "*/*.h5" in ig and "*/*.msgpack" in ig
    kept = _filter(
        ["model.h5", "checkpoint-1/model.h5", "config.json"], st["allow_patterns"], ig
    )
    assert "model.h5" in kept
    assert "checkpoint-1/model.h5" not in kept


def test_sentence_transformer_from_pretrained_is_prefetch_wired():
    """from_pretrained must call maybe_prefetch_hf_snapshot as an unconditional top-level statement before any return."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FastSentenceTransformer"
    )
    fp = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
    )

    def _prefetch_call(node):
        # a bare call statement, or one whose return is captured (e.g. _st_prefetched = ...)
        value = node.value if isinstance(node, (ast.Expr, ast.Assign)) else None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "maybe_prefetch_hf_snapshot"
        ):
            return value
        return None

    prefetch_pos = next((i for i, n in enumerate(fp.body) if _prefetch_call(n)), None)
    return_pos = next(
        (i for i, n in enumerate(fp.body) if isinstance(n, ast.Return)), len(fp.body)
    )
    assert (
        prefetch_pos is not None
    ), "from_pretrained must call maybe_prefetch_hf_snapshot at top level"
    assert prefetch_pos < return_pos, "prefetch must run before any top-level return"
    # local_files_only must be forwarded so an offline load does not start a Hub download.
    prefetch_call = _prefetch_call(fp.body[prefetch_pos])
    assert "local_files_only" in {
        kw.arg for kw in prefetch_call.keywords
    }, "prefetch must forward local_files_only"


def test_st_module_download_forwards_cache_folder():
    """_load_modules must forward the custom cache_folder into load_dir_path so per-module subdirs read the warmed cache."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(U.__file__), "sentence_transformer.py")
    with open(src_path, "r", encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "load_dir_path"
    ]
    assert calls, "expected a load_dir_path call in sentence_transformer.py"
    assert all(
        "cache_folder" in {kw.arg for kw in c.keywords} for c in calls
    ), "every load_dir_path call must forward cache_folder"


def test_st_native_sentence_transformer_calls_forward_cache_folder():
    """Every native SentenceTransformer(model_name, ...) load must forward cache_folder; a modules-based build needs none."""
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
        # A modules-based build downloads nothing; only a repo-name load reads the cache.
        if "modules" in kw_names:
            continue
        weight_loading_calls.append(n)
    assert (
        weight_loading_calls
    ), "expected a repo-name SentenceTransformer load in sentence_transformer.py"
    # cache_folder is forwarded explicitly or via a **kwargs unpacking (kw.arg == None).
    for c in weight_loading_calls:
        kw_names = {kw.arg for kw in c.keywords}
        forwards = "cache_folder" in kw_names or None in kw_names
        assert forwards, (
            "a repo-name SentenceTransformer load must forward cache_folder "
            f"(explicitly or via **kwargs) at line {c.lineno}"
        )
