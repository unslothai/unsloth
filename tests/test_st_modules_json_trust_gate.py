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

"""A modules.json "type" is untrusted input, not an import target.

`_module_path` fetches modules.json with `hf_hub_download`, so publishing a Hub repo is
enough; no local access required. See `_resolve_module_class` for the gate being tested.

Offline and CPU-only: the marker package is inert and only appends to a file, which is how
these tests observe whether an import happened at all.
"""

import json
import os
import sys

import pytest

# sentence-transformers is an extra, so skip rather than error in core-only shards.
pytest.importorskip("sentence_transformers")

from unsloth import FastSentenceTransformer  # noqa: E402


MARKER = "unsloth_st_gate_marker"


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """A model folder whose single modules.json entry names the marker package."""
    witness = tmp_path / "witness.txt"

    pkg = tmp_path / "site" / MARKER
    pkg.mkdir(parents = True)
    (pkg / "__init__.py").write_text(
        f"open({str(witness)!r}, 'a').write('imported\\n')\n"
        "class Thing:\n"
        "    @staticmethod\n"
        "    def load(path, **kwargs):\n"
        "        return Thing()\n",
        encoding = "utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))

    model = tmp_path / "model"
    (model / "0_Sub").mkdir(parents = True)
    (model / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "0_Sub", "type": f"{MARKER}.Thing"}]),
        encoding = "utf-8",
    )

    yield model, witness

    for name in [m for m in sys.modules if m.split(".")[0] == MARKER]:
        del sys.modules[name]


def _load_modules(model, trust_remote_code):
    return FastSentenceTransformer._load_modules(
        str(model),
        None,
        None,
        None,
        64,
        "mean",
        trust_remote_code = trust_remote_code,
    )


def test_an_untrusted_module_type_is_never_imported(model_dir):
    """The refusal must land BEFORE the import, so the witness must not exist."""
    model, witness = model_dir

    with pytest.raises(ValueError, match = "not part of Sentence Transformers"):
        _load_modules(model, False)

    assert not witness.exists(), "modules.json type was imported without consent"


def test_the_refusal_fails_the_load_rather_than_skipping_the_module(model_dir):
    """Fail closed, rather than degrading to "load the other modules and carry on"."""
    model, witness = model_dir
    (model / "1_Pooling").mkdir()
    (model / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "name": "0", "path": "0_Sub", "type": f"{MARKER}.Thing"},
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
            ]
        ),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError):
        _load_modules(model, False)

    assert not witness.exists()


def test_consent_still_loads_a_custom_module_type(model_dir):
    model, witness = model_dir

    modules, no_modules_json = _load_modules(model, True)

    assert witness.exists(), "consented module type was not imported"
    assert no_modules_json is False
    assert [type(m).__name__ for m in modules.values()] == ["Thing"]


def test_a_non_string_module_type_is_refused(model_dir):
    model, witness = model_dir
    (model / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "0_Sub", "type": {"bad": 1}}]),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError):
        _load_modules(model, False)

    assert not witness.exists()


@pytest.mark.parametrize(
    "class_ref",
    [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
        "sentence_transformers.models.Dense",
    ],
)
def test_the_stock_module_types_resolve_without_consent(class_ref, tmp_path):
    """These are every type real embedders ship: bge, MiniLM, e5, gte, bge-m3 and
    Qwen3-Embedding use the first three, embeddinggemma-300m adds the two Dense entries."""
    module_class = FastSentenceTransformer._resolve_module_class(class_ref, str(tmp_path), False)

    assert module_class.__module__.startswith("sentence_transformers")


@pytest.mark.parametrize(
    "class_ref",
    [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.transformer.Transformer",
        "sentence_transformers.base.modules.transformer.Transformer",
    ],
)
def test_the_fast_encoder_path_still_recognises_the_transformer_refs(class_ref):
    """These three refs name ST's Transformer across the 5.x/6.x layouts."""
    assert FastSentenceTransformer._is_transformer_module_ref(class_ref) is True


def test_an_untrusted_type_is_not_probed_as_a_transformer_ref(model_dir):
    """It used to import the ref just to compare it against ST's Transformer, which is the same
    execution by another door. It is now a pure string test."""
    model, witness = model_dir

    assert FastSentenceTransformer._is_transformer_module_ref(f"{MARKER}.Thing") is False
    assert not witness.exists()


def test_a_module_config_may_not_smuggle_a_class_path_past_the_gate(tmp_path):
    """Dense imports AND CALLS activation_function, on whichever version is installed."""
    from sentence_transformers.models import Dense

    load_path = tmp_path / "2_Dense"
    load_path.mkdir()
    (load_path / "config.json").write_text(
        json.dumps({"in_features": 8, "out_features": 8, "activation_function": f"{MARKER}.Thing"}),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError, match = "executes third-party code"):
        FastSentenceTransformer._check_module_config_class_refs(
            str(load_path), "sentence_transformers.models.Dense", "some/repo", False, Dense
        )


@pytest.mark.parametrize(
    "module_name, config_name, config",
    [
        # Each module names its own config file; the dotted-path loaders are why this exists.
        ("Router", "router_config.json", {"types": {"query": MARKER + ".Thing"}}),
        ("WordEmbeddings", "wordembedding_config.json", {"tokenizer_class": MARKER + ".Thing"}),
    ],
)
def test_an_in_namespace_type_cannot_smuggle_a_path_via_its_own_config_file(
    module_name, config_name, config, model_dir
):
    """Through _load_modules, not the helper: the type is an allowed sentence_transformers.* class,
    so the refusal has to come from reading the config file that module's loader reads."""
    from sentence_transformers import models as st_models

    if getattr(st_models, module_name, None) is None:
        pytest.skip(f"installed sentence-transformers has no {module_name}")

    model, witness = model_dir
    (model / f"1_{module_name}").mkdir()
    (model / f"1_{module_name}" / config_name).write_text(json.dumps(config), encoding = "utf-8")
    (model / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": f"1_{module_name}",
                    "type": f"sentence_transformers.models.{module_name}",
                }
            ]
        ),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError, match = "executes third-party code"):
        _load_modules(model, False)

    assert not witness.exists()


def test_a_real_dense_config_is_untouched_by_that_check(tmp_path):
    """embeddinggemma-300m's Dense modules declare torch.nn.modules.linear.Identity, exactly what
    ST >= 6 allows unconsented."""
    from sentence_transformers.models import Dense

    load_path = tmp_path / "2_Dense"
    load_path.mkdir()
    (load_path / "config.json").write_text(
        json.dumps(
            {
                "in_features": 768,
                "out_features": 3072,
                "bias": False,
                "activation_function": "torch.nn.modules.linear.Identity",
            }
        ),
        encoding = "utf-8",
    )

    FastSentenceTransformer._check_module_config_class_refs(
        str(load_path),
        "sentence_transformers.models.Dense",
        "unsloth/embeddinggemma-300m",
        False,
        Dense,
    )


def test_the_delegating_routes_gate_the_module_types_too(model_dir):
    """These routes never reach _load_modules, and below 6.0 sentence-transformers has no gate."""
    model, witness = model_dir

    with pytest.raises(ValueError, match = "not part of Sentence Transformers"):
        FastSentenceTransformer._check_modules_json_types(str(model), None, False)

    assert not witness.exists()


def test_that_validation_passes_a_stock_module_list_untouched(tmp_path):
    """A no-op for a normal repo: it runs on every delegated load."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Normalize",
                    "type": "sentence_transformers.models.Normalize",
                },
            ]
        ),
        encoding = "utf-8",
    )

    FastSentenceTransformer._check_modules_json_types(str(model), None, False)


def test_that_validation_is_silent_when_there_is_no_modules_json(tmp_path):
    """A transformers-native encoder has no modules.json."""
    model = tmp_path / "model"
    model.mkdir()

    FastSentenceTransformer._check_modules_json_types(str(model), None, False)


def test_validation_reads_the_sentence_transformers_cache(tmp_path, monkeypatch):
    """The delegated loads honour SENTENCE_TRANSFORMERS_HOME while hf_hub_download does not, so a
    model cached only there must still be read rather than becoming a silent pass."""
    st_home = tmp_path / "st_home"
    snapshot = st_home / "models--acme--embedder" / "snapshots" / "deadbeef"
    snapshot.mkdir(parents = True)
    (snapshot / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "0_Sub", "type": f"{MARKER}.Thing"}]),
        encoding = "utf-8",
    )
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_home))

    seen = {}

    def fake_module_path(
        model_name,
        token = None,
        cache_dir = None,
        revision = None,
    ):
        seen["cache_dir"] = cache_dir
        return str(snapshot / "modules.json") if cache_dir == str(st_home) else None

    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(fake_module_path))

    with pytest.raises(ValueError, match = "not part of Sentence Transformers"):
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False)

    assert seen["cache_dir"] == str(st_home)


def test_consent_skips_validation_so_a_repo_local_class_still_loads(model_dir, monkeypatch):
    """Below 6 the delegated loader fetches a repo-local class with get_class_from_dynamic_module,
    which this helper cannot, so validating a consented load would break it."""
    model, witness = model_dir

    called = []
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_module_path",
        staticmethod(lambda *a, **k: called.append(1) or None),
    )

    FastSentenceTransformer._check_modules_json_types(str(model), None, True)

    assert not called, "a consented load must not be validated or resolved here"
    assert not witness.exists()


def test_an_unverifiable_modules_json_fails_closed_where_upstream_would_not_gate(monkeypatch):
    """_module_path turns every failure into absent, so a transient failure would read as
    "nothing to check" while the load that follows could still import the type."""
    import sentence_transformers

    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_modules_json_for_gating",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(ValueError("Unsloth: Could not read"))),
    )

    with pytest.raises(ValueError, match = "Could not read"):
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False)


def test_a_local_directory_without_modules_json_is_a_confirmed_absence(tmp_path):
    """A transformers-native encoder has no modules.json and must not be refused for it."""
    model = tmp_path / "model"
    model.mkdir()

    assert FastSentenceTransformer._modules_json_for_gating(str(model), None) is None


def test_unreachable_is_not_read_as_absent(tmp_path, monkeypatch):
    """LocalEntryNotFoundError subclasses EntryNotFoundError, so catching the latter first turned
    "could not reach it" into "the repo has no modules.json", which is the opposite answer."""
    import sentence_transformers
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _patch_download(
        monkeypatch,
        lambda *a, **k: (_ for _ in ()).throw(LocalEntryNotFoundError("offline")),
    )
    monkeypatch.setattr(sentence_transformers, "__version__", "5.2.0", raising = False)

    with pytest.raises(ValueError, match = "Could not read modules.json"):
        FastSentenceTransformer._modules_json_for_gating("acme/embedder", None)


def test_unreachable_is_tolerated_where_upstream_gates_it_anyway(tmp_path, monkeypatch):
    """From 6.0 sentence-transformers refuses the type itself, so an unreachable check there must
    not turn a working load into a failure."""
    import sentence_transformers
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _patch_download(
        monkeypatch,
        lambda *a, **k: (_ for _ in ()).throw(LocalEntryNotFoundError("offline")),
    )
    monkeypatch.setattr(sentence_transformers, "__version__", "6.1.0", raising = False)

    assert FastSentenceTransformer._modules_json_for_gating("acme/embedder", None) is None


def _patch_download(monkeypatch, replacement):
    """Replace hf_hub_download in the namespace the method under test actually reads.

    Not by dotted string, and not on whatever sys.modules currently holds. Both of those
    resolve the module afresh, and tests/vllm_compat/test_extended_module_imports.py pops
    unsloth.models.sentence_transformer out of sys.modules and re-imports it, so by the
    time these tests run the name points at a second module object with its own globals
    dict. The class imported at the top of this file still closes over the first one.
    Patching the later object left the real hf_hub_download in place, and three tests
    reached out to the Hub and asserted against a 404 instead of against the fake.

    __globals__ is that first dict by definition, whatever else has been re-imported.
    """
    namespace = FastSentenceTransformer._check_delegated_module_config.__globals__
    monkeypatch.setitem(namespace, "hf_hub_download", replacement)


def _simulate_pre_six(monkeypatch):
    """Report sentence-transformers 5.x. The version is the whole simulation.

    This used to delete util.import_module_class as well, because the config check keyed
    off that attribute. #12444 replaced it with a version test, for the reason the
    attribute was never a good one: 5.5 exports the helper while its Dense and Router
    loaders still resolve config class references ungated. Deleting it here meant these
    tests passed for a reason production did not have, and hid that exact gap. Setting
    only the version is what makes them exercise the real condition.
    """
    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "__version__", "5.2.0", raising = False)


def _delegated_model(tmp_path, activation_function):
    """A local model whose Dense module config names a class, as the Hub form would."""
    model = tmp_path / "model"
    (model / "1_Dense").mkdir(parents = True)
    (model / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense",
                }
            ]
        ),
        encoding = "utf-8",
    )
    (model / "1_Dense" / "config.json").write_text(
        json.dumps(
            {"in_features": 8, "out_features": 8, "activation_function": activation_function}
        ),
        encoding = "utf-8",
    )
    return model


def test_a_delegated_route_also_checks_the_module_config_class_ref(tmp_path, monkeypatch):
    """An allowed type is not the whole check.

    Dense is a permitted sentence_transformers class, and below 6.0 its loader resolves and
    calls whatever activation_function the config names. _load_modules checks this from the
    files it downloaded; the delegated routes hand the load straight to
    sentence-transformers, so without this they validated the type and nothing else.

    Hiding import_module_class simulates sentence-transformers < 6, which is where the
    ungated loader lives; on this branch the config check still keys off that attribute,
    and #12444 removes the fork so it runs on every version.
    """
    _simulate_pre_six(monkeypatch)

    model = _delegated_model(tmp_path, f"{MARKER}.Thing")

    with pytest.raises(ValueError, match = "executes third-party code"):
        FastSentenceTransformer._check_modules_json_types(str(model), None, False)


def test_a_delegated_route_leaves_a_real_dense_config_alone(tmp_path, monkeypatch):
    """embeddinggemma-300m's Dense modules name torch.nn.modules.linear.Identity."""
    _simulate_pre_six(monkeypatch)

    model = _delegated_model(tmp_path, "torch.nn.modules.linear.Identity")

    FastSentenceTransformer._check_modules_json_types(str(model), None, False)


def test_an_ordinary_embedder_fetches_no_module_configs(tmp_path, monkeypatch):
    """The cost rule: Transformer, Pooling and Normalize read no class ref out of their
    configs, so gating them must not add a request per module to every load."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Normalize",
                    "type": "sentence_transformers.models.Normalize",
                },
            ]
        ),
        encoding = "utf-8",
    )

    requested = []

    def fake_download(repo_id, filename, **kwargs):
        requested.append(filename)
        raise AssertionError(f"unexpected fetch of {filename}")

    _patch_download(monkeypatch, fake_download)

    FastSentenceTransformer._check_modules_json_types(str(model), None, False)

    assert requested == []


def test_a_recorded_absence_pins_the_commit_the_cache_holds(tmp_path, monkeypatch):
    """Unreachable, with a recorded absence: pin to the snapshot that absence belongs to.

    "The load can only use the cache too" was the reasoning for returning success with no
    pin, and it is wrong: this request failed, not every request, so the delegated load
    can recover and fetch the current branch, which may have gained a module type since
    the cached answer. The commit is recovered from a file in the same snapshot, so the
    load matches what was checked, and offline it is what would have been served anyway.

    The lookup also has to read the cache hf_hub_download reads: HUGGINGFACE_HUB_CACHE is
    the legacy constant and does not follow HF_HUB_CACHE, so naming one of our own
    searched a different cache and a recorded absence there was missed.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    commit = "e" * 40
    snapshot = tmp_path / "active" / "models--acme--embedder" / "snapshots" / commit
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}", encoding = "utf8")

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "active"))
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising = False)
    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _simulate_pre_six(monkeypatch)
    _patch_download(
        monkeypatch,
        lambda *a, **k: (_ for _ in ()).throw(LocalEntryNotFoundError("offline")),
    )

    seen = []

    def fake_try_to_load_from_cache(
        repo_id,
        filename,
        cache_dir = None,
        revision = None,
    ):
        seen.append((filename, cache_dir))
        if filename == "config.json":
            return str(snapshot / "config.json")
        # The sentinel the hub returns when it has recorded that the file does not exist.
        return object()

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", fake_try_to_load_from_cache)

    assert FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False) == commit
    assert all(
        cache_dir is None for _, cache_dir in seen
    ), "a cache of our own choosing is not the cache hf_hub_download reads"


def test_a_recorded_absence_with_no_recoverable_commit_refuses(tmp_path, monkeypatch):
    """No commit to pin to means no way to make the load match what was checked.

    Fail closed below 6.0, the same rule as every other branch that could not establish an
    answer.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _simulate_pre_six(monkeypatch)
    _patch_download(
        monkeypatch,
        lambda *a, **k: (_ for _ in ()).throw(LocalEntryNotFoundError("offline")),
    )
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda *a, **k: object(),
    )

    with pytest.raises(ValueError, match = "names no commit"):
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False)


def test_a_recorded_absence_is_tolerated_where_upstream_gates_it(tmp_path, monkeypatch):
    """From 6.0 upstream refuses the type itself, so refusing here would only break a load
    that is already safe."""
    import sentence_transformers
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(sentence_transformers, "__version__", "6.1.0", raising = False)
    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _patch_download(
        monkeypatch,
        lambda *a, **k: (_ for _ in ()).throw(LocalEntryNotFoundError("offline")),
    )
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda *a, **k: object(),
    )

    assert FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False) == ""


def test_a_module_at_the_repository_root_is_checked_too(tmp_path, monkeypatch):
    """ "path": "" is the repository root, which sentence-transformers accepts.

    Skipping an empty path left the whole module-config check bypassable by declaring
    Dense at the root and putting the class ref in the root config.json.
    """
    _simulate_pre_six(monkeypatch)

    model = tmp_path / "model"
    model.mkdir()
    (model / "modules.json").write_text(
        json.dumps(
            [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Dense"}]
        ),
        encoding = "utf-8",
    )
    (model / "config.json").write_text(
        json.dumps({"in_features": 8, "out_features": 8, "activation_function": f"{MARKER}.Thing"}),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError, match = "executes third-party code"):
        FastSentenceTransformer._check_modules_json_types(str(model), None, False)


def test_an_unreadable_module_config_refuses_rather_than_passing(tmp_path, monkeypatch):
    """A fetch that fails is not a module without a config.

    Swallowing the error meant the gate recorded "nothing to check" while the load that
    follows makes its own request, which can succeed and resolve the unchecked name.
    """
    _simulate_pre_six(monkeypatch)
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_modules_json_for_gating",
        staticmethod(lambda *a, **k: str(tmp_path / "modules.json")),
    )
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense",
                }
            ]
        ),
        encoding = "utf-8",
    )

    def unreachable(*args, **kwargs):
        raise OSError("hub unreachable")

    _patch_download(monkeypatch, unreachable)

    with pytest.raises(ValueError, match = "refuses rather than loading unchecked"):
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False)


def test_a_module_that_ships_no_config_is_not_a_refusal(tmp_path, monkeypatch):
    """A genuine 404 means there is no class ref to check, which is not an error."""
    from huggingface_hub.errors import EntryNotFoundError

    _simulate_pre_six(monkeypatch)
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_modules_json_for_gating",
        staticmethod(lambda *a, **k: str(tmp_path / "modules.json")),
    )
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense",
                }
            ]
        ),
        encoding = "utf-8",
    )

    def absent(*args, **kwargs):
        raise EntryNotFoundError("no such file")

    _patch_download(monkeypatch, absent)

    FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False)


def test_the_module_config_check_is_skipped_where_upstream_gates_it(tmp_path, monkeypatch):
    """On 6.0 and later there is nothing to add, so it must not spend a request."""
    model = tmp_path / "model"
    (model / "1_Dense").mkdir(parents = True)
    (model / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense",
                }
            ]
        ),
        encoding = "utf-8",
    )
    (model / "1_Dense" / "config.json").write_text(
        json.dumps({"activation_function": f"{MARKER}.Thing"}), encoding = "utf-8"
    )

    def refuse(*args, **kwargs):
        raise AssertionError("no fetch expected where upstream gates the name")

    _patch_download(monkeypatch, refuse)

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "__version__", "6.1.0", raising = False)
    FastSentenceTransformer._check_modules_json_types(str(model), None, False)


def test_only_the_config_the_loader_reads_is_requested(tmp_path, monkeypatch):
    """WordEmbeddings.load never opens config.json, so asking for it is a liability.

    With the fetch failing closed, an unnecessary request against a cache that has no
    recorded 404 for the unused name turned a model the delegated loader could load
    entirely from cache into a refusal.
    """
    _simulate_pre_six(monkeypatch)
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_modules_json_for_gating",
        staticmethod(lambda *a, **k: str(tmp_path / "modules.json")),
    )
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "0_WordEmbeddings",
                    "type": "sentence_transformers.models.WordEmbeddings",
                }
            ]
        ),
        encoding = "utf-8",
    )

    requested = []
    config_dir = tmp_path / "cached" / "0_WordEmbeddings"
    config_dir.mkdir(parents = True)
    (config_dir / "wordembedding_config.json").write_text(
        json.dumps(
            {"tokenizer_class": "sentence_transformers.models.tokenizer.WhitespaceTokenizer"}
        ),
        encoding = "utf-8",
    )

    def fake_download(repo_id, filename, **kwargs):
        requested.append(filename)
        name = filename.rsplit("/", 1)[-1]
        target = config_dir / name
        if not target.is_file():
            raise OSError("not cached, and the loader would never ask for this one")
        return str(target)

    _patch_download(monkeypatch, fake_download)

    FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False)

    assert requested == ["0_WordEmbeddings/wordembedding_config.json"]


def test_router_still_falls_back_to_config_json(tmp_path, monkeypatch):
    """Router is the one class that does read config.json when its own file is absent."""
    from huggingface_hub.errors import EntryNotFoundError
    from sentence_transformers import models as st_models

    if getattr(st_models, "Router", None) is None:
        pytest.skip("installed sentence-transformers has no Router")

    _simulate_pre_six(monkeypatch)
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_modules_json_for_gating",
        staticmethod(lambda *a, **k: str(tmp_path / "modules.json")),
    )
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "1_Router",
                    "type": "sentence_transformers.models.Router",
                }
            ]
        ),
        encoding = "utf-8",
    )

    config_dir = tmp_path / "cached" / "1_Router"
    config_dir.mkdir(parents = True)
    (config_dir / "config.json").write_text(
        json.dumps({"types": {"query": f"{MARKER}.Thing"}}), encoding = "utf-8"
    )

    requested = []

    def fake_download(repo_id, filename, **kwargs):
        requested.append(filename)
        name = filename.rsplit("/", 1)[-1]
        target = config_dir / name
        if not target.is_file():
            raise EntryNotFoundError("absent")
        return str(target)

    _patch_download(monkeypatch, fake_download)

    with pytest.raises(ValueError, match = "executes third-party code"):
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False)

    assert requested[-1] == "1_Router/config.json"


def test_an_empty_router_config_still_checks_the_fallback(tmp_path, monkeypatch):
    """Router.load falls back on falsey content, not only on a missing file.

    sentence-transformers 5.x Router.load reads its own file and then does
    `if not config: config = cls.load_config(config_filename = "config.json")`, so an
    existing but empty router_config.json sends it to config.json and it resolves the
    `types` it finds there. Stopping at the first file that downloaded meant the gate
    validated the empty file, never fetched the fallback, and passed on a repository whose
    config.json names an arbitrary class.
    """
    import sentence_transformers

    st_models = pytest.importorskip("sentence_transformers.models")
    if getattr(st_models, "Router", None) is None:
        pytest.skip("this sentence-transformers has no Router")

    _simulate_pre_six(monkeypatch)

    # Two dirs on purpose: on a delegated route the cache must start empty.
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "router_config.json").write_text("{}", encoding = "utf8")
    (remote / "config.json").write_text(
        json.dumps({"types": {"query": "evil_pkg.Thing"}}),
        encoding = "utf8",
    )
    cache = tmp_path / "cache" / "1_Router"
    cache.mkdir(parents = True)

    requested = []

    def fake_download(repo, filename, **kwargs):
        requested.append(os.path.basename(filename))
        name = os.path.basename(filename)
        source = remote / name
        if not source.exists():
            from huggingface_hub.errors import EntryNotFoundError
            raise EntryNotFoundError(filename)
        target = cache / name
        target.write_text(source.read_text(encoding = "utf8"), encoding = "utf8")
        return str(target)

    _patch_download(monkeypatch, fake_download)

    with pytest.raises(ValueError, match = "evil_pkg.Thing"):
        FastSentenceTransformer._check_delegated_module_config(
            "acme/embedder",
            {
                "idx": 0,
                "name": "0",
                "path": "1_Router",
                "type": "sentence_transformers.models.Router",
            },
            "sentence_transformers.models.Router",
            st_models.Router,
        )
    assert requested == ["router_config.json", "config.json"]


def test_a_non_empty_config_asks_for_nothing_extra(tmp_path, monkeypatch):
    """The fallback costs a request only where upstream would take it.

    A real Router config is a non-empty dict, upstream never reaches `if not config`, and
    the check must not start fetching a second file on every ordinary load.
    """
    st_models = pytest.importorskip("sentence_transformers.models")
    if getattr(st_models, "Router", None) is None:
        pytest.skip("this sentence-transformers has no Router")

    _simulate_pre_six(monkeypatch)

    folder = tmp_path / "1_Router"
    folder.mkdir()
    (folder / "router_config.json").write_text(
        json.dumps({"types": {"query": "sentence_transformers.models.Transformer"}}),
        encoding = "utf8",
    )
    (folder / "config.json").write_text(
        json.dumps({"types": {"query": "evil_pkg.Thing"}}),
        encoding = "utf8",
    )

    requested = []

    def fake_download(repo, filename, **kwargs):
        requested.append(os.path.basename(filename))
        return str(folder / os.path.basename(filename))

    _patch_download(monkeypatch, fake_download)

    FastSentenceTransformer._check_delegated_module_config(
        "acme/embedder",
        {"idx": 0, "name": "0", "path": "1_Router", "type": "sentence_transformers.models.Router"},
        "sentence_transformers.models.Router",
        st_models.Router,
    )
    assert requested == ["router_config.json"]


def test_the_validated_commit_is_reported(tmp_path, monkeypatch):
    """The gate hands back the commit it read, so the load can be pinned to it.

    Validation and the load resolve the branch separately, so a repository that advances
    between the two is checked on one snapshot and loaded from another.
    """
    snapshot = tmp_path / "models--acme--embedder" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                }
            ]
        ),
        encoding = "utf8",
    )
    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _patch_download(monkeypatch, lambda *a, **k: str(snapshot / "modules.json"))

    assert (
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False) == "a" * 40
    )


def test_the_commit_comes_from_the_resolution_that_already_happened(tmp_path, monkeypatch):
    """_module_path resolves through hf_hub_download, so its snapshot is the live one.

    That is the branch an ordinary load takes, and taking the commit from the path it
    returns is what makes the pin cost nothing: no second request, and no separate
    resolution to disagree with.
    """
    snapshot = tmp_path / "models--acme--embedder" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                }
            ]
        ),
        encoding = "utf8",
    )
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_module_path",
        staticmethod(lambda *a, **k: str(snapshot / "modules.json")),
    )
    _patch_download(
        monkeypatch,
        lambda *a, **k: pytest.fail("_module_path already resolved it; nothing else should fetch"),
    )

    assert (
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False) == "b" * 40
    )


def test_a_local_directory_has_no_commit_to_pin(tmp_path, monkeypatch):
    """A folder on disk is not a snapshot, so the load must be left exactly as it was."""
    local = tmp_path / "my-embedder"
    local.mkdir()
    (local / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                }
            ]
        ),
        encoding = "utf8",
    )
    assert FastSentenceTransformer._check_modules_json_types(str(local), None, False) == ""


def test_consent_reports_no_commit(monkeypatch):
    """With trust_remote_code there is nothing to gate, so there is nothing to pin."""
    assert FastSentenceTransformer._check_modules_json_types("acme/embedder", None, True) == ""


@pytest.mark.parametrize(
    "path, expected",
    [
        (f"/c/models--a--b/snapshots/{'c' * 40}/modules.json", "c" * 40),
        (f"/c/models--a--b/snapshots/{'C' * 40}/modules.json", "c" * 40),
        ("/c/models--a--b/snapshots/main/modules.json", ""),
        ("/home/me/my-model/modules.json", ""),
        (None, ""),
    ],
)
def test_only_an_immutable_snapshot_is_a_commit(path, expected):
    """A branch name in the snapshot slot is not something to pin to."""
    assert FastSentenceTransformer._snapshot_revision(path) == expected


def test_the_module_configs_are_read_from_the_validated_snapshot(tmp_path, monkeypatch):
    """Every config read has to come from the commit modules.json came out of.

    Passing the caller's revision through meant modules.json could resolve commit A while
    each module config resolved commit B. The caller then pins the load to A, so a clean
    config in B was validated while A's unchecked value was the one that ran. That inverts
    the race rather than closing it.
    """
    _simulate_pre_six(monkeypatch)

    commit = "c" * 40
    snapshot = tmp_path / "models--acme--embedder" / "snapshots" / commit
    snapshot.mkdir(parents = True)
    (snapshot / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense",
                }
            ]
        ),
        encoding = "utf8",
    )
    dense = snapshot / "1_Dense"
    dense.mkdir()
    (dense / "config.json").write_text(
        json.dumps({"activation_function": "torch.nn.Tanh"}),
        encoding = "utf8",
    )

    revisions = []

    def fake_download(repo, filename, **kwargs):
        revisions.append((os.path.basename(filename), kwargs.get("revision")))
        return str(snapshot / filename)

    _patch_download(monkeypatch, fake_download)

    assert (
        FastSentenceTransformer._check_modules_json_types(
            "acme/embedder", None, False, revision = "main"
        )
        == commit
    )

    # modules.json uses the caller's revision; everything after uses the resolved commit.
    assert revisions[0] == ("modules.json", "main")
    assert [r for name, r in revisions[1:]] == [commit] * len(revisions[1:])
    assert len(revisions) > 1


def test_a_named_branch_still_resolves_to_a_commit(tmp_path, monkeypatch):
    """The gate reports a commit even when the caller named a branch.

    That is the input the pin needs, and it held on the previous head too: the bug was the
    condition at the call site, which discarded this value whenever a revision was given
    and so left `revision = "main"` racing exactly as a missing revision did. That half is
    not unit-testable without a full load, and is covered by a traced base-against-head
    run instead, where base passes "main" to SentenceTransformer and head passes the
    resolved commit. This test guards the contract the call site depends on.
    """
    commit = "d" * 40
    snapshot = tmp_path / "models--acme--embedder" / "snapshots" / commit
    snapshot.mkdir(parents = True)
    (snapshot / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                }
            ]
        ),
        encoding = "utf8",
    )
    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _patch_download(monkeypatch, lambda *a, **k: str(snapshot / "modules.json"))

    assert (
        FastSentenceTransformer._check_modules_json_types(
            "acme/embedder", None, False, revision = "main"
        )
        == commit
    )


def test_an_immutable_revision_resolves_to_itself(tmp_path, monkeypatch):
    """A caller who already named a commit gets a pin that changes nothing."""
    commit = "e" * 40
    snapshot = tmp_path / "models--acme--embedder" / "snapshots" / commit
    snapshot.mkdir(parents = True)
    (snapshot / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                }
            ]
        ),
        encoding = "utf8",
    )
    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _patch_download(monkeypatch, lambda *a, **k: str(snapshot / "modules.json"))

    assert (
        FastSentenceTransformer._check_modules_json_types(
            "acme/embedder", None, False, revision = commit
        )
        == commit
    )


def test_a_confirmed_absent_modules_json_still_pins_the_load(tmp_path, monkeypatch):
    """Absence is the thing being validated, so it has to be pinned like a presence.

    And pinned to the commit that answered the 404, read off that response, not from a
    second resolution of the branch. A second lookup can return a newer commit than the
    one whose answer was "no modules.json", which pins the load to a snapshot nothing
    checked: the same inversion as validating one commit and loading another, moved one
    step along. The hub sends x-repo-commit on the 404, so the exact commit is already in
    hand and costs no request.
    """
    from huggingface_hub.errors import EntryNotFoundError

    commit = "f" * 40
    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))

    class Response:
        headers = {"x-repo-commit": commit}

    def absent(*args, **kwargs):
        error = EntryNotFoundError("no modules.json")
        error.response = Response()
        raise error

    _patch_download(monkeypatch, absent)

    def unexpected(*args, **kwargs):
        pytest.fail("the 404 already named the commit; nothing should resolve it again")

    monkeypatch.setattr("huggingface_hub.HfApi", unexpected)

    assert (
        FastSentenceTransformer._check_modules_json_types(
            "acme/embedder", None, False, revision = "main"
        )
        == commit
    )


def test_a_404_without_a_commit_header_is_tolerated_where_upstream_gates_it(tmp_path, monkeypatch):
    """From 6.0 upstream refuses the type itself, so an unpinnable absence is not fatal.

    Below 6.0 it is: see the refusal test beside this one. A proxy that strips
    x-repo-commit, or an older hub whose error carries no response at all, leaves nothing
    to pin to, and on a version where nothing else checks the type that has to refuse.
    """
    import sentence_transformers
    from huggingface_hub.errors import EntryNotFoundError

    # The floor lane installs sentence-transformers 5.x, so force the version under test.
    monkeypatch.setattr(sentence_transformers, "__version__", "6.1.0", raising = False)
    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _patch_download(
        monkeypatch,
        lambda *a, **k: (_ for _ in ()).throw(EntryNotFoundError("no modules.json")),
    )

    assert FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False) == ""


@pytest.mark.parametrize("helper", ["present", "absent", "none"])
def test_the_check_does_not_key_off_the_import_module_class_attribute(
    helper, tmp_path, monkeypatch
):
    """Not a version test: 5.5 exports it while its WordEmbeddings.load still calls
    import_from_string on tokenizer_class, so the refusal holds whatever the attribute is."""
    import sentence_transformers.util as st_util
    from sentence_transformers.models import Dense

    if helper == "absent":
        monkeypatch.delattr(st_util, "import_module_class", raising = False)
    elif helper == "none":
        monkeypatch.setattr(st_util, "import_module_class", None, raising = False)

    load_path = tmp_path / "2_Dense"
    load_path.mkdir()
    (load_path / "config.json").write_text(
        json.dumps({"activation_function": f"{MARKER}.Thing"}), encoding = "utf-8"
    )

    with pytest.raises(ValueError, match = "executes third-party code"):
        FastSentenceTransformer._check_module_config_class_refs(
            str(load_path), "sentence_transformers.models.Dense", "some/repo", False, Dense
        )


def test_a_legacy_module_class_without_config_file_name_is_still_read(tmp_path):
    """3.x/4.x WordEmbeddings has no config_file_name and opens its file by name."""

    class WordEmbeddings:  # no config_file_name, like the 3.x/4.x class
        pass

    load_path = tmp_path / "0_WordEmbeddings"
    load_path.mkdir()
    (load_path / "wordembedding_config.json").write_text(
        json.dumps({"tokenizer_class": f"{MARKER}.Thing"}), encoding = "utf-8"
    )

    with pytest.raises(ValueError, match = "executes third-party code"):
        FastSentenceTransformer._check_module_config_class_refs(
            str(load_path),
            "sentence_transformers.models.WordEmbeddings",
            "some/repo",
            False,
            WordEmbeddings,
        )


@pytest.mark.parametrize("trusted, expect_refusal", [(False, True), (True, False)])
def test_the_checkpoint_reload_honours_the_consent_the_model_was_loaded_with(
    trusted, expect_refusal, tmp_path
):
    """The reload takes only a checkpoint path, so hard-coding False left no way to consent."""
    from sentence_transformers.models import Dense

    load_path = tmp_path / "2_Dense"
    load_path.mkdir()
    (load_path / "config.json").write_text(
        json.dumps({"activation_function": f"{MARKER}.Thing"}), encoding = "utf-8"
    )

    class FakeModel:
        _unsloth_trust_remote_code = trusted

    trust = getattr(FakeModel, "_unsloth_trust_remote_code", False)
    if expect_refusal:
        with pytest.raises(ValueError, match = "executes third-party code"):
            FastSentenceTransformer._check_module_config_class_refs(
                str(load_path), "sentence_transformers.models.Dense", str(tmp_path), trust, Dense
            )
    else:
        FastSentenceTransformer._check_module_config_class_refs(
            str(load_path), "sentence_transformers.models.Dense", str(tmp_path), trust, Dense
        )


def test_every_load_route_records_the_consent_it_used():
    """The reload reads the flag off the model, so every route must set it."""
    import inspect

    source = inspect.getsource(FastSentenceTransformer.from_pretrained)
    assert source.count("_unsloth_trust_remote_code = trust_remote_code") == source.count(
        "return st_model"
    )


def _write_module_config(folder, name, payload):
    folder.mkdir(parents = True, exist_ok = True)
    (folder / name).write_text(json.dumps(payload), encoding = "utf8")
    return folder


def test_a_literal_activation_on_a_pooling_module_is_not_a_class_ref(tmp_path):
    """SpladePooling's activation_function is an enum it compares, not a path it imports.

    It lists activation_function in its own config_keys and saves "relu" or "log1p_relu"
    there, then does `if self.activation_function == "log1p_relu"`. Keying the check on
    the key name alone refused both values for not starting with "torch.", so no SPLADE
    sparse model could load without the user turning on remote code for no reason.
    """

    class SpladePooling:
        config_file_name = "config.json"

    folder = _write_module_config(
        tmp_path / "1_SpladePooling",
        "config.json",
        {"pooling_strategy": "max", "activation_function": "log1p_relu"},
    )
    FastSentenceTransformer._check_module_config_class_refs(
        str(folder),
        "sentence_transformers.sparse_encoder.models.SpladePooling",
        "acme/sparse",
        False,
        SpladePooling,
    )


def test_dense_still_refuses_an_activation_outside_torch(tmp_path):
    """Dense.load does `import_from_string(config["activation_function"])()`, so this one
    is a real import and the scoping must not have let it through."""
    st_models = pytest.importorskip("sentence_transformers.models")

    folder = _write_module_config(
        tmp_path / "2_Dense",
        "config.json",
        {"in_features": 8, "out_features": 8, "activation_function": "evil_pkg.Boom"},
    )
    with pytest.raises(ValueError, match = "evil_pkg.Boom"):
        FastSentenceTransformer._check_module_config_class_refs(
            str(folder),
            "sentence_transformers.models.Dense",
            "acme/embedder",
            False,
            st_models.Dense,
        )


def test_word_embeddings_still_refuses_a_foreign_tokenizer_class(tmp_path):
    """WordEmbeddings.load does `import_from_string(config.pop("tokenizer_class"))`."""
    st_models = pytest.importorskip("sentence_transformers.models")
    if getattr(st_models, "WordEmbeddings", None) is None:
        pytest.skip("this sentence-transformers has no WordEmbeddings")

    folder = _write_module_config(
        tmp_path / "0_WordEmbeddings",
        "config.json",
        {"tokenizer_class": "evil_pkg.Tok", "max_seq_length": 8},
    )
    with pytest.raises(ValueError, match = "evil_pkg.Tok"):
        FastSentenceTransformer._check_module_config_class_refs(
            str(folder),
            "sentence_transformers.models.WordEmbeddings",
            "acme/embedder",
            False,
            st_models.WordEmbeddings,
        )


def test_a_subclass_inherits_its_parents_rules(tmp_path):
    """The rules follow the bases, so a version that subclasses Dense is still gated."""
    st_models = pytest.importorskip("sentence_transformers.models")

    class TunedDense(st_models.Dense):
        pass

    assert ("activation_function", "torch.") in FastSentenceTransformer._class_ref_rules(TunedDense)


def test_every_config_driven_import_in_sentence_transformers_has_a_rule():
    """The scoping is only safe if the list of loaders is complete.

    Asym is an alias of Router from 5.0, so it reports Router's name and is covered by
    that entry rather than its own. This asserts the set of classes, so a version that
    adds another config-driven import shows up here as a missing key rather than as a
    silently unchecked path.
    """
    covered = set(FastSentenceTransformer._MODULE_CONFIG_CLASS_REFS)
    assert covered == {"Dense", "WordEmbeddings", "Router", "Asym"}

    st_models = pytest.importorskip("sentence_transformers.models")
    for name in ("Dense", "WordEmbeddings", "Router"):
        klass = getattr(st_models, name, None)
        if klass is None:
            continue
        assert FastSentenceTransformer._class_ref_rules(klass), name


def test_the_delegated_check_fetches_the_legacy_config_filename(tmp_path, monkeypatch):
    """A 3.x/4.x WordEmbeddings has no config_file_name, and its loader still reads
    wordembedding_config.json.

    Asking only for config.json got a 404, left `folders` empty, and passed, after which
    the delegated loader read the legacy file and imported its tokenizer_class with
    trust_remote_code=False. The local checker already consults
    _LEGACY_MODULE_CONFIG_FILES; the delegated one did not.
    """
    _simulate_pre_six(monkeypatch)

    class LegacyWordEmbeddings:
        """No config_file_name, which is what 3.x and 4.x look like."""

        __name__ = "WordEmbeddings"

    LegacyWordEmbeddings.__name__ = "WordEmbeddings"

    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "wordembedding_config.json").write_text(
        json.dumps({"tokenizer_class": "evil_pkg.Tok"}), encoding = "utf8"
    )
    cache = tmp_path / "cache" / "0_WordEmbeddings"
    cache.mkdir(parents = True)

    requested = []

    def fake_download(repo, filename, **kwargs):
        name = os.path.basename(filename)
        requested.append(name)
        source = remote / name
        if not source.exists():
            from huggingface_hub.errors import EntryNotFoundError
            raise EntryNotFoundError(filename)
        target = cache / name
        target.write_text(source.read_text(encoding = "utf8"), encoding = "utf8")
        return str(target)

    _patch_download(monkeypatch, fake_download)

    with pytest.raises(ValueError, match = "evil_pkg.Tok"):
        FastSentenceTransformer._check_delegated_module_config(
            "acme/embedder",
            {
                "idx": 0,
                "name": "0",
                "path": "0_WordEmbeddings",
                "type": "sentence_transformers.models.WordEmbeddings",
            },
            "sentence_transformers.models.WordEmbeddings",
            LegacyWordEmbeddings,
        )
    assert "wordembedding_config.json" in requested


def test_an_unparseable_manifest_refuses_rather_than_passing(tmp_path, monkeypatch):
    """The file is in hand and still unreadable, which is not "nothing to check".

    The delegated loader opens the very same path straight afterwards, so a transient read
    error here followed by a retry that succeeds there would import a type nothing ever
    looked at. Below 6.0 nothing else checks it, so this refuses.
    """
    _simulate_pre_six(monkeypatch)

    broken = tmp_path / "models--acme--embedder" / "snapshots" / ("a" * 40)
    broken.mkdir(parents = True)
    (broken / "modules.json").write_text("{not json", encoding = "utf8")
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_module_path",
        staticmethod(lambda *a, **k: str(broken / "modules.json")),
    )

    with pytest.raises(ValueError, match = "Could not read the modules.json"):
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False)


def test_an_unparseable_manifest_is_tolerated_where_upstream_gates_it(tmp_path, monkeypatch):
    """From 6.0 upstream refuses the type itself, so refusing here would only break a
    load that is already safe."""
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "__version__", "6.1.0", raising = False)

    broken = tmp_path / "models--acme--embedder" / "snapshots" / ("b" * 40)
    broken.mkdir(parents = True)
    (broken / "modules.json").write_text("{not json", encoding = "utf8")
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_module_path",
        staticmethod(lambda *a, **k: str(broken / "modules.json")),
    )

    assert (
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False) == "b" * 40
    )


def test_a_manifest_that_is_not_a_list_is_not_a_refusal(tmp_path, monkeypatch):
    """A modules.json that is not a list names no modules, and the loader iterates it too,
    so it cannot build a module from this either. Refusing would only break a load that
    upstream fails on its own."""
    _simulate_pre_six(monkeypatch)

    odd = tmp_path / "models--acme--embedder" / "snapshots" / ("c" * 40)
    odd.mkdir(parents = True)
    (odd / "modules.json").write_text(json.dumps({"not": "a list"}), encoding = "utf8")
    monkeypatch.setattr(
        FastSentenceTransformer,
        "_module_path",
        staticmethod(lambda *a, **k: str(odd / "modules.json")),
    )

    assert (
        FastSentenceTransformer._check_modules_json_types("acme/embedder", None, False) == "c" * 40
    )


def test_a_404_without_a_commit_header_refuses_below_six(tmp_path, monkeypatch):
    """No commit from the response and none from the caller means no pin is possible.

    The delegated request resolves the branch again and can get one that has since gained
    a modules.json, so accepting the absence unpinned let that through.
    """
    from huggingface_hub.errors import EntryNotFoundError

    _simulate_pre_six(monkeypatch)
    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _patch_download(
        monkeypatch,
        lambda *a, **k: (_ for _ in ()).throw(EntryNotFoundError("no modules.json")),
    )

    with pytest.raises(ValueError, match = "named no commit"):
        FastSentenceTransformer._check_modules_json_types(
            "acme/embedder", None, False, revision = "main"
        )


def test_an_explicit_commit_needs_no_commit_header(tmp_path, monkeypatch):
    """A caller who named a commit has already answered the question the header answers.

    The 404 was answered at that commit and there is nothing for a second resolution to
    disagree with, so this must not refuse for want of a header.
    """
    from huggingface_hub.errors import EntryNotFoundError

    commit = "a" * 40
    _simulate_pre_six(monkeypatch)
    monkeypatch.setattr(FastSentenceTransformer, "_module_path", staticmethod(lambda *a, **k: None))
    _patch_download(
        monkeypatch,
        lambda *a, **k: (_ for _ in ()).throw(EntryNotFoundError("no modules.json")),
    )

    assert (
        FastSentenceTransformer._check_modules_json_types(
            "acme/embedder", None, False, revision = commit
        )
        == commit
    )


def test_an_unreadable_module_config_refuses_rather_than_skipping(tmp_path, monkeypatch):
    """A config that is present and unreadable is unverifiable, not absent.

    The loader opens this same path next and its read may succeed, so skipping meant a
    Dense activation could be imported with nothing having checked it. The refusal the
    fetch path promised was only ever about the fetch; this is the read.
    """
    st_models = pytest.importorskip("sentence_transformers.models")
    _simulate_pre_six(monkeypatch)

    folder = tmp_path / "2_Dense"
    folder.mkdir()
    (folder / "config.json").write_text("{not json", encoding = "utf8")

    with pytest.raises(ValueError, match = "Could not read"):
        FastSentenceTransformer._check_module_config_class_refs(
            str(folder),
            "sentence_transformers.models.Dense",
            "acme/embedder",
            False,
            st_models.Dense,
        )


def test_an_unreadable_module_config_is_tolerated_where_upstream_gates_it(tmp_path, monkeypatch):
    """From 6.0 upstream resolves these through its own gate, so refusing would only break
    a load that is already safe."""
    import sentence_transformers

    st_models = pytest.importorskip("sentence_transformers.models")
    monkeypatch.setattr(sentence_transformers, "__version__", "6.1.0", raising = False)

    folder = tmp_path / "2_Dense"
    folder.mkdir()
    (folder / "config.json").write_text("{not json", encoding = "utf8")

    FastSentenceTransformer._check_module_config_class_refs(
        str(folder),
        "sentence_transformers.models.Dense",
        "acme/embedder",
        False,
        st_models.Dense,
    )
