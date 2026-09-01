# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A sentence-transformers model can have weights at the root AND in subfolders.

`weights_at_root` splits repos two ways, root weights or per-subfolder weights,
and `unsloth/embeddinggemma-300m` is both: a root `model.safetensors` plus
`2_Dense/model.safetensors` and `3_Dense/model.safetensors`, which the ST load
reads as part of the model. `_SUBDIR_WEIGHT_IGNORE_PATTERNS` pruned those two,
unsloth_zoo's post-download gate correctly flagged the missing weights, and the
retry excluded the same files again and raised DownloadStallError, blaming the
network for a request that could never have been satisfied.

Offline: the hub call is stubbed, since a test that depends on the network
eventually reports a bug that is not there.
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unsloth.models._utils as U  # noqa: E402


ROOT_ONLY = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
]

EMBEDDINGGEMMA = ROOT_ONLY + [
    {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
    {"idx": 3, "name": "3", "path": "3_Dense", "type": "sentence_transformers.models.Dense"},
    {"idx": 4, "name": "4", "path": "", "type": "sentence_transformers.models.Normalize"},
]


@pytest.fixture
def modules_json(tmp_path, monkeypatch):
    """Stub hf_hub_download so it hands back a modules.json we control."""

    def _install(payload):
        if payload is None:  # repo ships no modules.json

            def boom(*a, **k):
                raise OSError("404 modules.json")

            monkeypatch.setattr(U, "hf_hub_download", boom, raising = False)
            import huggingface_hub

            monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
            return
        p = tmp_path / "modules.json"
        p.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding = "utf-8")
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(p))

    return _install




def test_embeddinggemma_layout_is_detected(modules_json):
    modules_json(EMBEDDINGGEMMA)
    assert U._repo_has_weighted_st_subfolders("unsloth/embeddinggemma-300m") is True


def test_a_root_only_st_model_is_not(modules_json):
    """Pooling lives in a subfolder but holds no weight, so nothing is at risk
    and the existing subdir pruning should stay in force."""
    modules_json(ROOT_ONLY)
    assert U._repo_has_weighted_st_subfolders("org/plain-st") is False


def test_a_repo_without_modules_json_is_not(modules_json):
    """The overwhelming majority. A plain causal LM must keep the old
    behaviour exactly, so a fix for one notebook does not enlarge 400 other
    downloads."""
    modules_json(None)
    assert U._repo_has_weighted_st_subfolders("unsloth/Qwen3-0.6B") is False


@pytest.mark.parametrize(
    "payload",
    [
        "{ not json",
        json.dumps({"not": "a list"}),
        json.dumps([None, 3, "x"]),
        json.dumps([{"path": "2_Dense"}]),
        json.dumps([{"type": "...Dense"}]),
        json.dumps([{"path": "  ", "type": "...Dense"}]),
        json.dumps([{"path": "/", "type": "...Dense"}]),
    ],
)
def test_malformed_modules_json_falls_back_to_the_old_behaviour(modules_json, payload):
    """Best-effort by design: anything unreadable must not start failing loads
    that work today."""
    modules_json(payload)
    assert U._repo_has_weighted_st_subfolders("org/whatever") is False


def test_an_unknown_subfolder_module_type_is_not_assumed_weighted(modules_json):
    modules_json([{"path": "2_Custom", "type": "mypkg.WeirdModule"}])
    assert U._repo_has_weighted_st_subfolders("org/custom") is False


@pytest.mark.parametrize("leaf", ["Dense", "CNN", "LSTM", "dense"])
def test_every_weight_bearing_type_counts(modules_json, leaf):
    modules_json([{"path": f"2_{leaf}", "type": f"sentence_transformers.models.{leaf}"}])
    assert U._repo_has_weighted_st_subfolders("org/x") is True


def test_the_taxonomy_is_shared_with_unsloth_zoo_not_restated():
    """If these two ever disagree, unsloth would fetch a module the gate then
    rejects, or prune one it demands -- the exact shape of the original bug."""
    src = (Path(U.__file__)).read_text(encoding = "utf-8")
    assert "_ST_WEIGHTED_MODULE_TYPES" in src
    assert '"dense"' not in src.split("_repo_has_weighted_st_subfolders")[1][:2000]




# the behaviour that actually changed ---------------------------------
def _ignores(
    model_name,
    monkeypatch,
    siblings = None,
    **kw,
):
    """The ignore_patterns `maybe_prefetch_hf_snapshot` actually sends.

    Driven through the real function with the downloader stubbed, not through
    `_prefetch_ignore_patterns`, which knows nothing about the subdir branch and
    would have passed either way.
    """
    seen = {}

    def fake_download(name, **kwargs):
        seen.update(kwargs)
        return "/nonexistent/snapshot"

    # The prefetch is a no-op in offline mode, so clear it:
    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(flag, raising = False)

    import unsloth_zoo.hf_xet_fallback as XF

    monkeypatch.setattr(XF, "snapshot_download_with_xet_fallback", fake_download)
    # The auto format branch calls model_info;
    import huggingface_hub

    class _Api:
        def model_info(self, *a, **k):
            if siblings is None:
                raise RuntimeError("no network in test")
            return types.SimpleNamespace(
                siblings = [types.SimpleNamespace(rfilename = f) for f in siblings]
            )

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    U.maybe_prefetch_hf_snapshot(model_name, weights_at_root = True, **kw)
    assert seen, "the downloader was never reached; the call bailed out early"
    return list(seen.get("ignore_patterns") or [])


def test_the_subdir_weight_patterns_are_dropped_for_such_a_repo(modules_json, monkeypatch):
    modules_json(EMBEDDINGGEMMA)
    got = _ignores("unsloth/embeddinggemma-300m", monkeypatch)
    assert "*/*.safetensors" not in got, got


def test_the_subdir_weight_patterns_are_kept_for_everything_else(modules_json, monkeypatch):
    """The other half of the claim. Without this, the test above would pass
    just as well if the patterns had been deleted outright."""
    modules_json(None)
    got = _ignores("unsloth/Qwen3-0.6B", monkeypatch)
    assert "*/*.safetensors" in got, got


def test_only_the_subdir_weight_patterns_differ(modules_json, monkeypatch):
    """The fix must not quietly change anything else about the request."""
    modules_json(None)
    plain = set(_ignores("unsloth/Qwen3-0.6B", monkeypatch))
    modules_json(EMBEDDINGGEMMA)
    st = set(_ignores("unsloth/embeddinggemma-300m", monkeypatch))
    assert plain - st == set(U._SUBDIR_WEIGHT_IGNORE_PATTERNS)
    assert st - plain == set()


def test_the_patterns_still_exist(modules_json):
    """They are correct for the case they were written for -- an fp16/ or
    experimental/ directory a root load never reads. This fix narrows where
    they apply, it does not retire them."""
    assert "*/*.safetensors" in U._SUBDIR_WEIGHT_IGNORE_PATTERNS
    assert "*/*.bin" in U._SUBDIR_WEIGHT_IGNORE_PATTERNS


def test_an_older_unsloth_zoo_degrades_instead_of_crashing(modules_json):
    """`_ST_WEIGHTED_MODULE_TYPES` is private, so a user on an older zoo must
    fall back to today's pruning rather than get an ImportError on every load."""
    import unsloth_zoo.hf_cache_state as HCS

    modules_json(EMBEDDINGGEMMA)
    saved = HCS._ST_WEIGHTED_MODULE_TYPES
    del HCS._ST_WEIGHTED_MODULE_TYPES
    try:
        assert U._repo_has_weighted_st_subfolders("unsloth/embeddinggemma-300m") is False
    finally:
        HCS._ST_WEIGHTED_MODULE_TYPES = saved
    # ...and the taxonomy being back restores the fix, so the assertion above is about the missing name and not about a
    assert U._repo_has_weighted_st_subfolders("unsloth/embeddinggemma-300m") is True


def test_both_weights_at_root_call_sites_go_through_the_check():
    """`weights_at_root = True` is passed from exactly two places (vision.py and
    llama.py), both reaching the prune through maybe_prefetch_hf_snapshot, so one
    carve-out covers both. A third call site, or an inlined copy of the patterns,
    fails here instead of leaving half the loaders pruning ST weights."""
    root = Path(U.__file__).resolve().parents[1]
    sites = []
    for p in root.rglob("*.py"):
        if "tests" in p.parts:
            continue
        for n, line in enumerate(p.read_text(encoding = "utf-8").splitlines(), 1):
            if "weights_at_root = True" in line:
                sites.append(f"{p.name}:{n}")
    assert sorted(s.split(":")[0] for s in sites) == ["llama.py", "vision.py"], sites

    # AST, not grep: the name also appears in prose inside a docstring, and a text count would police the documentation
    import ast

    tree = ast.parse(Path(U.__file__).read_text(encoding = "utf-8"))
    loads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name)
        and n.id == "_SUBDIR_WEIGHT_IGNORE_PATTERNS"
        and isinstance(n.ctx, ast.Load)
    ]
    assert len(loads) == 1, [n.lineno for n in loads]


def test_a_hub_failure_keeps_the_patterns(monkeypatch):
    """Network trouble must not silently enlarge every download."""
    import huggingface_hub

    def boom(*a, **k):
        raise RuntimeError("hub down")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    got = _ignores("org/anything", monkeypatch)
    assert "*/*.safetensors" in got


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))



# mixed weight formats: root safetensors, subfolder .bin ---------------
BIN_DENSE_FILES = [
    "config.json",
    "modules.json",
    "model.safetensors",
    "pytorch_model.bin",
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/pytorch_model.bin",
]


def _kept(files, patterns):
    """What snapshot_download would actually fetch, using its own matcher."""
    import fnmatch
    return [f for f in files if not any(fnmatch.fnmatch(f, p) for p in patterns)]


def test_a_bin_only_dense_module_keeps_its_only_weight(modules_json, monkeypatch):
    """Root model.safetensors plus 2_Dense/pytorch_model.bin, the legacy ST head shape. The redundant
    format prune adds a bare "*.bin", and "*" spans "/" in the Hub's fnmatch, so the glob would strip
    the Dense module's only weight: the same unsatisfiable request, one branch further along."""
    modules_json(EMBEDDINGGEMMA)
    patterns = _ignores("org/st-bin-dense", monkeypatch, siblings = BIN_DENSE_FILES)
    kept = _kept(BIN_DENSE_FILES, patterns)
    assert "2_Dense/pytorch_model.bin" in kept, patterns
    assert "pytorch_model.bin" not in kept, (
        "the redundant ROOT .bin must still be pruned",
        patterns,
    )
    assert "model.safetensors" in kept, patterns


def test_the_bin_prune_is_untouched_without_st_modules(modules_json, monkeypatch):
    """A plain repo still gets the cheap glob, not an enumeration."""
    modules_json(None)
    patterns = _ignores("org/plain", monkeypatch, siblings = BIN_DENSE_FILES)
    assert "*.bin" in patterns
    assert "pytorch_model.bin" not in _kept(BIN_DENSE_FILES, patterns)


def test_an_explicit_format_request_keeps_both_for_such_a_repo(modules_json, monkeypatch):
    """use_safetensors fetches no repo listing, so the glob cannot be scoped and pruning it would
    drop the module weight. Keeping both formats is the trade the multi-component case already makes."""
    modules_json(EMBEDDINGGEMMA)
    patterns = _ignores("org/st-bin-dense", monkeypatch, use_safetensors = True)
    assert "*.bin" not in patterns
    modules_json(None)
    assert "*.bin" in _ignores("org/plain", monkeypatch, use_safetensors = True)


def test_a_module_path_is_not_read_as_a_glob(modules_json, monkeypatch):
    """Repo filenames go into ignore_patterns verbatim, so a "[" in a name would silently become a
    character class and stop matching itself."""
    modules_json(EMBEDDINGGEMMA)
    files = ["model.safetensors", "weird[1].bin", "2_Dense/pytorch_model.bin"]
    patterns = _ignores("org/st-bin-dense", monkeypatch, siblings = files)
    kept = _kept(files, patterns)
    assert "weird[1].bin" not in kept, patterns
    assert "2_Dense/pytorch_model.bin" in kept, patterns
