# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A trained MTP head attached as `_unsloth_mtp_head` must reach every export: the merge
copies the STOCK head through from the original shards, so without the stitch a fine-tuned
model ships a head that was never aligned to it."""

import ast
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth" / "save.py"


@pytest.fixture(scope = "module")
def tree():
    return ast.parse(SAVE_PY.read_text(encoding = "utf-8"))


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in save.py")


def _calls(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            out.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", None))
    return out


@pytest.mark.parametrize(
    "function, expected",
    [
        (
            "unsloth_save_model",
            {
                "_stitch_mtp_head",
                "_save_mtp_head_beside_adapter",
                "_push_with_mtp_head",
                "_mtp_head_tensor_names",
            },
        ),
        (
            "unsloth_generic_save",
            {"_stitch_mtp_head", "_attached_mtp_head", "_push_merged_to_hub_revision"},
        ),
        ("unsloth_save_pretrained_gguf", {"_stitch_mtp_head"}),
        ("_gguf_reuses_loaded_checkpoint", {"_attached_mtp_head"}),
        # torchao's reload drops native MTP tensors: say so instead of exporting it silently.
        ("_unsloth_save_torchao", {"_attached_mtp_head"}),
    ],
)
def test_save_paths_are_wired(tree, function, expected):
    assert expected <= _calls(_func(tree, function))


@pytest.fixture(scope = "module")
def save_module():
    pytest.importorskip(
        "unsloth_zoo.speculative.stitch", reason = "unsloth_zoo has no speculative.stitch"
    )
    pytest.importorskip("unsloth", reason = "unsloth is not importable on this runner")
    try:
        import unsloth.save as module
    except ImportError as error:
        pytest.skip(f"unsloth.save is not importable on this runner: {error}")
    return module


def _head(torch):
    head = torch.nn.Module()
    head.fc = torch.nn.Linear(16, 8, bias = False)
    head.norm = torch.nn.Module()
    head.norm.weight = torch.nn.Parameter(torch.randn(8))
    object.__setattr__(head, "_unsloth_mtp_info", {"prefix": "mtp.", "source": "tiny"})
    return head


def _wrapped(torch, head, registered):
    base = torch.nn.Module()
    base.config = None
    if registered:
        base._unsloth_mtp_head = head
    else:
        object.__setattr__(base, "_unsloth_mtp_head", head)
    lora = torch.nn.Module()
    lora.model = base
    peft = torch.nn.Module()
    peft.base_model = lora
    return peft


@pytest.mark.parametrize("registered", [True, False])
def test_head_is_found_through_wrappers(save_module, registered):
    import torch

    head = _head(torch)
    assert save_module._attached_mtp_head(_wrapped(torch, head, registered)) is head
    assert save_module._attached_mtp_head(torch.nn.Linear(2, 2)) is None
    assert save_module._mtp_head_tensor_names(torch.nn.Linear(2, 2)) == []
    assert sorted(save_module._mtp_head_tensor_names(_wrapped(torch, head, registered))) == [
        "mtp.fc.weight",
        "mtp.norm.weight",
    ]


def test_no_head_is_a_no_op(save_module, tmp_path):
    import torch

    model = torch.nn.Linear(2, 2)
    assert save_module._stitch_mtp_head(model, tmp_path) is None
    assert save_module._save_mtp_head_beside_adapter(model, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_stitch_appends_and_declares(save_module, tmp_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    save_file(
        {"model.norm.weight": torch.ones(8, dtype = torch.bfloat16)}, tmp_path / "model.safetensors"
    )
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 2}}))
    head = _head(torch)
    head.layers = torch.nn.ModuleList([torch.nn.Module()])
    head.layers[0].up = torch.nn.Linear(8, 8, bias = False)
    report = save_module._stitch_mtp_head(_wrapped(torch, head, registered = False), tmp_path)
    assert report["mode"] == "appended"
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert {"mtp.fc.weight", "mtp.norm.weight", "mtp.layers.0.up.weight"} <= set(
        index["weight_map"]
    )
    assert (
        json.loads((tmp_path / "config.json").read_text())["text_config"]["mtp_num_hidden_layers"]
        == 1
    )
    with safe_open(str(tmp_path / index["weight_map"]["mtp.fc.weight"]), "pt") as f:
        assert torch.equal(
            f.get_tensor("mtp.fc.weight"), head.fc.weight.detach().to(torch.bfloat16)
        )


def test_adapter_save_writes_head_file(save_module, tmp_path):
    import torch
    from unsloth_zoo.speculative.stitch import load_saved_mtp_head

    head = _head(torch)
    save_module._save_mtp_head_beside_adapter(_wrapped(torch, head, registered = True), tmp_path)
    loaded, info = load_saved_mtp_head(tmp_path)
    assert set(loaded) == {"fc.weight", "norm.weight"} and info["source"] == "tiny"


def test_gguf_does_not_reuse_checkpoint_with_head(save_module, tmp_path):
    import torch

    model = torch.nn.Module()
    model.config = type("C", (), {"_name_or_path": str(tmp_path)})()
    assert save_module._gguf_reuses_loaded_checkpoint(model) is True
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    assert save_module._gguf_reuses_loaded_checkpoint(model) is False


def test_plain_save_pretrained_keeps_the_head(save_module, tmp_path):
    import torch
    from unsloth.models.speculative import _wrap_save_pretrained
    from unsloth_zoo.speculative.stitch import load_saved_mtp_head

    calls = []

    class Adapter(torch.nn.Module):
        def save_pretrained(self, save_directory, **kwargs):
            calls.append(save_directory)
            (Path(save_directory) / "adapter_config.json").write_text("{}")

    model = Adapter()
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    _wrap_save_pretrained(model)
    _wrap_save_pretrained(model)
    model.save_pretrained(str(tmp_path))
    assert calls == [str(tmp_path)]
    loaded, _ = load_saved_mtp_head(tmp_path)
    assert set(loaded) == {"fc.weight", "norm.weight"}


def test_internal_savers_stitch_once(save_module, tmp_path, monkeypatch):
    """Inside `_internal_mtp_save` the plain-save wrapper does not stitch, even under an outer wrapper
    (an export reload wraps `save_pretrained` again), so one save writes the head once."""
    import torch
    from unsloth.models.speculative import _wrap_save_pretrained

    stitched = []
    monkeypatch.setattr(save_module, "_stitch_mtp_head", lambda m, d: stitched.append(d))

    class Plain(torch.nn.Module):
        def save_pretrained(self, save_directory, **kwargs):
            (Path(save_directory) / "config.json").write_text("{}")

    model = Plain()
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    _wrap_save_pretrained(model)
    inner = model.save_pretrained
    object.__setattr__(model, "save_pretrained", lambda *a, **k: inner(*a, **k))
    with save_module._internal_mtp_save():
        model.save_pretrained(str(tmp_path))
    assert stitched == []
    model.save_pretrained(str(tmp_path))
    assert stitched == [str(tmp_path)]


def test_reattach_is_a_no_op_without_a_saved_head(tmp_path):
    loader_tree = ast.parse((SAVE_PY.parent / "models" / "loader.py").read_text(encoding = "utf-8"))
    calls = [
        _calls(n)
        for n in ast.walk(loader_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_reattach_saved_mtp_head"
    ]
    assert calls, "loader.py lost _reattach_saved_mtp_head"
    sites = [
        n
        for n in ast.walk(loader_tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_reattach_saved_mtp_head"
    ]
    assert len(sites) == 2
    for site in sites:
        assert {k.arg for k in site.keywords} == {
            "token",
            "revision",
            "cache_dir",
            "local_files_only",
        }


def _tiny_glm(tmp_path, torch):
    pytest.importorskip("transformers.models.glm4_moe")
    from transformers import Glm4MoeForCausalLM
    from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

    torch.manual_seed(0)
    config = Glm4MoeConfig(
        vocab_size = 97,
        hidden_size = 64,
        intermediate_size = 128,
        moe_intermediate_size = 32,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_hidden_layers = 2,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        n_shared_experts = 1,
        first_k_dense_replace = 1,
        num_nextn_predict_layers = 1,
        partial_rotary_factor = 0.5,
    )
    return Glm4MoeForCausalLM(config).eval()


def test_nextn_head_round_trips_through_save_pretrained(save_module, tmp_path):
    """A GLM-style head (model.layers.{N}.*) is appended on a plain save and reloads identically."""
    import torch
    from safetensors.torch import load_file

    pytest.importorskip("unsloth_zoo.speculative.nextn")
    from unsloth.models.speculative import get_mtp_head

    model = _tiny_glm(tmp_path, torch)
    head = get_mtp_head(model, random_init = True)
    assert type(head).__name__ == "NextNHead"
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.01)
    expected = {k: v.detach().clone() for k, v in head.mtp_state_dict().items()}
    assert all(name.startswith("model.layers.2.") for name in expected)
    assert sorted(save_module._mtp_head_tensor_names(model)) == sorted(expected)

    out = tmp_path / "saved"
    model.save_pretrained(str(out))
    config = json.loads((out / "config.json").read_text())
    assert config["num_nextn_predict_layers"] == 1 and "mtp_num_hidden_layers" not in config
    written = {}
    for shard in out.glob("*.safetensors"):
        written.update(load_file(str(shard)))
    for name, tensor in expected.items():
        assert torch.equal(written[name], tensor.to(written[name].dtype)), name

    fresh = _tiny_glm(tmp_path, torch)
    reloaded = get_mtp_head(fresh, source = str(out))
    assert all(torch.equal(reloaded.mtp_state_dict()[k], v) for k, v in expected.items())

    report = save_module._stitch_mtp_head(fresh, str(out))
    assert report["mode"] == "replaced"


class _AdapterSaver:
    """A model whose save_pretrained writes an adapter folder, like PEFT's."""

    def __init__(self, torch):
        self.module = torch.nn.Module()
        self.saved = []

    def save_pretrained(self, save_directory, **kwargs):
        self.saved.append(kwargs)
        Path(save_directory).mkdir(parents = True, exist_ok = True)
        (Path(save_directory) / "adapter_config.json").write_text("{}")


def _wrapped_adapter(torch, pushes = True):
    """`pushes`: transformers' save_pretrained (takes push_to_hub), else PEFT's (does not)."""
    from unsloth.models.speculative import _wrap_save_pretrained

    model = torch.nn.Module()
    calls = []

    def transformers_save(
        save_directory,
        push_to_hub = False,
        **kwargs,
    ):
        calls.append(dict(kwargs, push_to_hub = push_to_hub))
        Path(save_directory).mkdir(parents = True, exist_ok = True)
        (Path(save_directory) / "adapter_config.json").write_text("{}")

    def save_pretrained(save_directory, **kwargs):
        calls.append(kwargs)
        Path(save_directory).mkdir(parents = True, exist_ok = True)
        (Path(save_directory) / "adapter_config.json").write_text("{}")

    object.__setattr__(model, "save_pretrained", transformers_save if pushes else save_pretrained)
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    _wrap_save_pretrained(model)
    return model, calls


def test_plain_save_refuses_a_variant_before_writing(save_module, tmp_path):
    import torch

    model, calls = _wrapped_adapter(torch)
    with pytest.raises(NotImplementedError):
        model.save_pretrained(str(tmp_path / "v"), variant = "foo")
    assert calls == []


def test_plain_save_push_uploads_the_head(save_module, tmp_path, monkeypatch):
    import huggingface_hub
    import torch

    uploads = []

    class Api:
        def __init__(self, token = None):
            self.token = token

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    model, _ = _wrapped_adapter(torch)
    out = tmp_path / "me-adapter"
    model.save_pretrained(str(out), push_to_hub = True, repo_id = "me/adapter", token = "t")
    assert (out / "mtp_head.safetensors").is_file()
    assert (
        uploads
        and uploads[0]["repo_id"] == "me/adapter"
        and uploads[0]["allow_patterns"] == ["mtp_head.safetensors"]
    )


def test_plain_save_push_skips_peft_and_refuses_create_pr(save_module, tmp_path, monkeypatch):
    import huggingface_hub
    import torch

    uploads = []

    class Api:
        def __init__(self, token = None):
            pass

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    peft_model, _ = _wrapped_adapter(torch, pushes = False)
    peft_model.save_pretrained(str(tmp_path / "a"), push_to_hub = True, repo_id = "me/adapter")
    assert (tmp_path / "a" / "mtp_head.safetensors").is_file() and uploads == []
    model, calls = _wrapped_adapter(torch)
    with pytest.raises(ValueError, match = "create_pr"):
        model.save_pretrained(str(tmp_path / "b"), push_to_hub = True, repo_id = "me/m", create_pr = True)
    assert calls == [] and uploads == []


def test_adapter_save_without_a_head_drops_a_stale_one(save_module, tmp_path):
    import torch

    (tmp_path / "mtp_head.safetensors").write_bytes(b"old")
    save_module._save_mtp_head_beside_adapter(torch.nn.Module(), str(tmp_path))
    assert not (tmp_path / "mtp_head.safetensors").exists()


def test_modelscope_snapshot_is_the_head_source(tmp_path):
    import torch
    from unsloth.models.speculative import _checkpoint_source

    model = torch.nn.Module()
    model.config = type("C", (), {"_name_or_path": "Org/Model"})()
    assert _checkpoint_source(model) == "Org/Model"
    model._unsloth_modelscope_snapshot = str(tmp_path)
    assert _checkpoint_source(model) == str(tmp_path)


def test_hub_adapter_reattaches_its_head(tmp_path, monkeypatch):
    import huggingface_hub
    from unsloth.models import loader, speculative

    head_file = tmp_path / "snap" / "mtp_head.safetensors"
    head_file.parent.mkdir()
    head_file.write_bytes(b"h")
    seen = []
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo, name, **k: seen.append(k) or str(head_file)
    )
    attached = []
    monkeypatch.setattr(speculative, "attach_saved_mtp_head", lambda model, d: attached.append(d))
    loader._reattach_saved_mtp_head(
        object(), "me/adapter", token = "t", cache_dir = "/c", local_files_only = True
    )
    assert attached == [str(head_file.parent)]
    # The adapter's own cache and offline mode, else an offline load reads the default cache.
    assert seen[0]["cache_dir"] == "/c" and seen[0]["local_files_only"] is True

    def missing(*a, **k):
        raise FileNotFoundError("no head")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    loader._reattach_saved_mtp_head(object(), "me/other")
    assert attached == [str(head_file.parent)]


def test_mtp_push_keeps_the_datasets_metadata(save_module, monkeypatch):
    """The attribution is staged in the README, so it lands in the same commit / revision / PR."""
    import os

    import huggingface_hub

    calls = []

    class Api:
        def __init__(self, token = None):
            pass

        def create_repo(self, **kwargs):
            pass

        def create_branch(self, **kwargs):
            pass

        def upload_folder(self, **kwargs):
            readme = os.path.join(kwargs["folder_path"], "README.md")
            calls.append((kwargs["revision"], kwargs["create_pr"], open(readme).read()))

    def no_card(*args, **kwargs):
        raise huggingface_hub.errors.EntryNotFoundError("no README")

    monkeypatch.setattr(save_module, "HfApi", Api)
    monkeypatch.setattr(save_module, "_determine_username", lambda d, u, t: ("me/m", "me"))
    monkeypatch.setattr(save_module, "unsloth_save_model", lambda **kwargs: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_card)
    monkeypatch.setattr(
        huggingface_hub, "metadata_update", lambda *a, **k: pytest.fail("edits the default branch")
    )
    save_module._push_with_mtp_head(
        dict(
            is_main_process = True,
            save_directory = "m",
            token = "t",
            private = False,
            revision = "dev",
            create_pr = True,
            commit_message = None,
            datasets = ["org/data"],
            tags = ["my-tag"],
            model = SimpleNamespace(
                config = SimpleNamespace(_name_or_path = "org/base", model_type = "qwen3_5")
            ),
        )
    )
    assert len(calls) == 1 and calls[0][:2] == ("dev", True) and "org/data" in calls[0][2]
    assert "unsloth" in calls[0][2] and "my-tag" in calls[0][2] and "org/base" in calls[0][2]


def test_unsloth_wrapped_save_still_pushes_the_head(save_module, tmp_path, monkeypatch):
    """Unsloth's loaders wrap save_pretrained as (self, *args, **kwargs), hiding push_to_hub."""
    from unsloth.models.speculative import _takes_push_to_hub

    class Model:
        def original_model_save_pretrained(
            self,
            save_directory,
            push_to_hub = False,
            **kwargs,
        ):
            pass

        def unsloth_model_save_pretrained(self, *args, **kwargs):
            pass

    assert _takes_push_to_hub(Model().unsloth_model_save_pretrained)


def test_merged_save_with_a_head_refuses_pickles(tree):
    fn = ast.get_source_segment(
        SAVE_PY.read_text(encoding = "utf-8"), _func(tree, "unsloth_save_model")
    )
    assert "if _keeps_mtp_head and not safe_serialization:" in fn
    assert "safe_serialization and (n_cpus <= 2) and not _keeps_mtp_head" in fn


def test_get_mtp_head_returns_the_attached_head(save_module):
    import torch
    from unsloth.models.speculative import get_mtp_head

    head = _head(torch)
    model = _wrapped(torch, head, registered = False)
    assert get_mtp_head(model) is head


def test_plain_save_refuses_pickles_before_writing(save_module, tmp_path):
    import torch

    model, calls = _wrapped_adapter(torch)
    with pytest.raises(ValueError, match = "safetensors"):
        model.save_pretrained(str(tmp_path / "p"), safe_serialization = False)
    assert calls == []


def test_modelscope_snapshot_wins_under_a_peft_wrapper(tmp_path):
    import torch
    from unsloth.models.speculative import _checkpoint_source

    inner = torch.nn.Module()
    inner.config = SimpleNamespace(_name_or_path = "modelscope/only-id")
    object.__setattr__(inner, "_unsloth_modelscope_snapshot", str(tmp_path))
    peft = torch.nn.Module()
    peft.config = SimpleNamespace(_name_or_path = "modelscope/only-id")
    peft.base_model = torch.nn.Module()
    peft.base_model.model = inner
    assert _checkpoint_source(peft) == str(tmp_path)


def test_mtp_push_never_replaces_a_card_it_could_not_read(save_module, monkeypatch):
    import huggingface_hub

    uploads = []

    class Api:
        def __init__(self, token = None):
            pass

        def create_repo(self, **kwargs):
            pass

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

    def unreachable(*args, **kwargs):
        raise huggingface_hub.errors.LocalEntryNotFoundError("offline")

    monkeypatch.setattr(save_module, "HfApi", Api)
    monkeypatch.setattr(save_module, "_determine_username", lambda d, u, t: ("me/m", "me"))
    monkeypatch.setattr(save_module, "unsloth_save_model", lambda **kwargs: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unreachable)
    with pytest.raises(huggingface_hub.errors.LocalEntryNotFoundError):
        save_module._push_with_mtp_head(
            dict(
                is_main_process = True,
                save_directory = "m",
                token = "t",
                private = False,
                revision = None,
                create_pr = False,
                commit_message = None,
                datasets = None,
                tags = None,
                model = SimpleNamespace(
                    config = SimpleNamespace(_name_or_path = "org/base", model_type = "qwen3_5")
                ),
            )
        )
    assert uploads == []


def test_stale_adapter_config_does_not_turn_a_full_save_into_a_sidecar(
    save_module, tmp_path, monkeypatch
):
    import os

    import torch
    from unsloth.models.speculative import _wrap_save_pretrained

    out = tmp_path / "reused"
    out.mkdir()
    (out / "adapter_config.json").write_text("{}")
    os.utime(out / "adapter_config.json", (1, 1))
    stitched, sidecars = [], []
    monkeypatch.setattr(save_module, "_stitch_mtp_head", lambda m, d: stitched.append(d))
    monkeypatch.setattr(
        save_module, "_save_mtp_head_beside_adapter", lambda m, d: sidecars.append(d)
    )
    model = torch.nn.Module()
    object.__setattr__(model, "save_pretrained", lambda save_directory, **kwargs: None)
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    _wrap_save_pretrained(model)
    model.save_pretrained(str(out))
    assert stitched == [str(out)] and sidecars == []


def test_get_mtp_head_reads_the_loaded_commit(tmp_path):
    from unsloth.models.speculative import _loaded_revision

    model = SimpleNamespace(config = SimpleNamespace(_name_or_path = "org/m", _commit_hash = "abc123"))
    assert _loaded_revision(model, "org/m", None) == "abc123"
    assert _loaded_revision(model, "org/m", "dev") == "dev"
    assert _loaded_revision(model, str(tmp_path), None) is None


@pytest.mark.parametrize("adapter_name", ["default", "task"])
def test_named_peft_adapter_save_keeps_the_head(save_module, tmp_path, adapter_name):
    """PEFT writes a non-default adapter under `<out>/<adapter_name>/`; the head goes beside it."""
    import torch
    from peft import LoraConfig, get_peft_model

    pytest.importorskip("unsloth_zoo.speculative.nextn")
    from unsloth.models.speculative import attach_saved_mtp_head, get_mtp_head

    model = get_peft_model(
        _tiny_glm(tmp_path, torch),
        LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"]),
        adapter_name = adapter_name,
    )
    head = get_mtp_head(model, random_init = True)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.01)
    expected = {k: v.detach().clone() for k, v in head.mtp_state_dict().items()}

    out = tmp_path / "saved"
    model.save_pretrained(str(out))
    folder = out if adapter_name == "default" else out / adapter_name
    assert (folder / "adapter_config.json").is_file()
    assert (folder / "mtp_head.safetensors").is_file()
    assert not list(out.glob("model*.safetensors"))

    reloaded = attach_saved_mtp_head(_tiny_glm(tmp_path, torch), str(folder))
    assert reloaded is not None
    assert all(torch.equal(reloaded.mtp_state_dict()[k], v) for k, v in expected.items())


def test_every_saved_adapter_folder_gets_the_head(save_module, tmp_path, monkeypatch):
    """A save of several adapters writes the head beside each folder and pushes each one."""
    import torch
    from unsloth.models.speculative import _wrap_save_pretrained
    from unsloth_zoo.speculative.stitch import load_saved_mtp_head

    model = torch.nn.Module()
    object.__setattr__(model, "peft_config", {"default": None, "task": None})

    def transformers_save(
        save_directory,
        push_to_hub = False,
        **kwargs,
    ):
        for name in ("default", "task"):
            folder = Path(save_directory) / ("" if name == "default" else name)
            folder.mkdir(parents = True, exist_ok = True)
            (folder / "adapter_config.json").write_text("{}")

    object.__setattr__(model, "save_pretrained", transformers_save)
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    _wrap_save_pretrained(model)
    uploads = []

    class Api:
        def __init__(self, **kwargs):
            pass

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    model.save_pretrained(str(tmp_path), push_to_hub = True, repo_id = "me/x")
    for folder in (tmp_path, tmp_path / "task"):
        loaded, _ = load_saved_mtp_head(folder)
        assert set(loaded) == {"fc.weight", "norm.weight"}
    assert len(uploads) == 1
    assert sorted(uploads[0]["allow_patterns"]) == [
        "mtp_head.safetensors",
        "task/mtp_head.safetensors",
    ]


def _two_adapter_model(torch, active):
    """PEFT-shaped model: `selected_adapters` picks which folders a save writes."""
    from unsloth.models.speculative import _wrap_save_pretrained

    model = torch.nn.Module()
    object.__setattr__(model, "peft_config", {"default": None, "task": None})
    object.__setattr__(model, "active_adapter", active)

    def peft_save(
        save_directory,
        safe_serialization = True,
        selected_adapters = None,
        **kwargs,
    ):
        for name in ("default", "task") if selected_adapters is None else selected_adapters:
            folder = Path(save_directory) / ("" if name == "default" else name)
            folder.mkdir(parents = True, exist_ok = True)
            (folder / "adapter_config.json").write_text(json.dumps({"name": name}))

    object.__setattr__(model, "save_pretrained", peft_save)
    object.__setattr__(model, "_unsloth_mtp_head", _head(torch))
    _wrap_save_pretrained(model)
    return model


@pytest.mark.parametrize("positional", [False, True])
def test_selected_adapters_save_leaves_other_folders_alone(save_module, tmp_path, positional):
    """`default` saved a moment earlier is not rewritten by a save of `task` only."""
    import torch

    model = _two_adapter_model(torch, active = "task")
    (tmp_path / "adapter_config.json").write_text("{}")
    if positional:
        model.save_pretrained(str(tmp_path), True, ["task"])
    else:
        model.save_pretrained(str(tmp_path), selected_adapters = ["task"])
    assert (tmp_path / "task" / "mtp_head.safetensors").is_file()
    assert not (tmp_path / "mtp_head.safetensors").exists()


def test_head_is_written_only_beside_the_active_adapter(save_module, tmp_path):
    """The head was trained against the active adapter; an inactive one must not inherit it."""
    import torch

    model = _two_adapter_model(torch, active = "task")
    model.save_pretrained(str(tmp_path))
    assert (tmp_path / "task" / "mtp_head.safetensors").is_file()
    assert not (tmp_path / "mtp_head.safetensors").exists()
    out = tmp_path / "inactive"
    model.save_pretrained(str(out), selected_adapters = ["default"])
    assert (out / "adapter_config.json").is_file()
    assert not list(out.rglob("mtp_head.safetensors"))


@pytest.mark.parametrize("adapter_name", ["default", "task"])
def test_transformers_native_adapter_head_goes_to_the_root(
    save_module, tmp_path, monkeypatch, adapter_name
):
    """`PreTrainedModel.add_adapter` saves the active adapter into `<out>/` whatever its name."""
    import torch
    from peft import LoraConfig

    pytest.importorskip("unsloth_zoo.speculative.nextn")
    import unsloth.save as save
    from unsloth.models.speculative import attach_saved_mtp_head, get_mtp_head

    model = _tiny_glm(tmp_path, torch)
    model.add_adapter(
        LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"]),
        adapter_name = adapter_name,
    )
    model.set_adapter(adapter_name)
    head = get_mtp_head(model, random_init = True)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.01)
    expected = {k: v.detach().clone() for k, v in head.mtp_state_dict().items()}

    out = tmp_path / "saved"
    monkeypatch.setattr(
        save, "_stitch_mtp_head", lambda *a, **k: pytest.fail("stitched an adapter save")
    )
    model.save_pretrained(str(out))
    assert (out / "adapter_config.json").is_file()
    assert (out / "mtp_head.safetensors").is_file()
    assert not (out / adapter_name).exists()

    reloaded = attach_saved_mtp_head(_tiny_glm(tmp_path, torch), str(out))
    assert reloaded is not None
    assert all(torch.equal(reloaded.mtp_state_dict()[k], v) for k, v in expected.items())


def test_an_empty_adapter_selection_saves_no_head(save_module, tmp_path, monkeypatch):
    """`selected_adapters = []` saves nothing: no head, no `default` fallback, no checkpoint stitch."""
    import torch
    import unsloth.save as save

    model = _two_adapter_model(torch, active = "default")
    # Fresh `default` adapter config from a save a moment earlier, not part of this one.
    (tmp_path / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(
        save, "_stitch_mtp_head", lambda *a, **k: pytest.fail("stitched an adapter save")
    )
    model.save_pretrained(str(tmp_path), selected_adapters = [])
    assert not list(tmp_path.rglob("mtp_head.safetensors"))
    assert not (tmp_path / "task").exists()


def test_a_combined_adapter_head_is_not_split_into_sidecars(save_module, tmp_path, capsys):
    """Trained with several adapters active at once, the head belongs to none of them alone."""
    import torch

    model = _two_adapter_model(torch, active = ["default", "task"])
    # A head left in the folder by an earlier single-adapter save must not survive either.
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "mtp_head.safetensors").write_bytes(b"stale")
    model.save_pretrained(str(tmp_path))
    assert (tmp_path / "task" / "adapter_config.json").is_file()
    assert not list(tmp_path.rglob("mtp_head.safetensors"))
    assert "active together" in capsys.readouterr().out
    object.__setattr__(model, "active_adapter", "task")
    model.save_pretrained(str(tmp_path / "single"))
    assert (tmp_path / "single" / "task" / "mtp_head.safetensors").is_file()
    assert not (tmp_path / "single" / "mtp_head.safetensors").exists()
