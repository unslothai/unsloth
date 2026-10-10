# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Context parallelism plumbing that runs without a process group (the ring itself needs >= 2 GPUs)."""

import contextlib

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad


def _manager(size = 2):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    return manager


def _fake_context_parallel(sharded):
    @contextlib.contextmanager
    def fake(mesh, buffers, buffer_seq_dims, no_restore_buffers):
        sharded.extend(buffers)
        yield

    return fake


def test_labels_shift_before_sharding_and_pad_to_load_balancer_chunks():
    manager = _manager(size = 2)
    labels = torch.tensor([[-100, 5, 6, 7, 8, 9]])
    inputs = {"input_ids": torch.tensor([[1, 5, 6, 7, 8, 9]]), "labels": labels.clone()}
    manager._prepare_inputs(inputs)
    # 6 tokens -> 8 (2 * size chunks); the target of token t is label t + 1, even across a shard seam.
    assert inputs["shift_labels"].tolist() == [[5, 6, 7, 8, 9, -100, -100, -100]]
    assert inputs["labels"].tolist() == [[-100, 5, 6, 7, 8, 9, -100, -100]]
    assert inputs["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert inputs["input_ids"].shape == (1, 8)


@pytest.mark.parametrize("mask", [[[0, 1, 1, 1]], [[1, 0, 1, 1]], [[[1, 1, 1, 1]] * 4]])
def test_left_padded_or_holed_masks_are_refused(mask):
    inputs = {"input_ids": torch.ones(1, 4, dtype = torch.long), "attention_mask": torch.tensor(mask)}
    with pytest.raises(ValueError, match = "right-padded"):
        _manager()._prepare_inputs(inputs)
    right = {
        "input_ids": torch.ones(1, 4, dtype = torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
    }
    _manager()._prepare_inputs(right)


def test_inputs_embeds_batches_are_refused():
    with pytest.raises(ValueError, match = "input_ids"):
        _manager()._prepare_inputs(
            {"inputs_embeds": torch.zeros(1, 4, 8), "labels": torch.ones(1, 4)}
        )


def test_active_manager_resets_when_the_step_raises(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    manager = _manager()
    manager.mesh = None
    inputs = {"input_ids": torch.ones(1, 4, dtype = torch.long)}
    with pytest.raises(RuntimeError, match = "boom"):
        with manager.apply(inputs):
            assert cp.get_cp_manager() is manager
            raise RuntimeError("boom")
    assert cp.get_cp_manager() is None


def test_attention_hook_only_drops_the_mask_inside_a_cp_step(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    mask = torch.ones(1, 4)
    _, kwargs = cp._self_attn_pre_forward_hook(None, (), {"attention_mask": mask})
    assert kwargs["attention_mask"] is mask
    manager = _manager()
    manager.mesh = None
    with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
        _, kwargs = cp._self_attn_pre_forward_hook(None, (), {"attention_mask": mask})
    assert kwargs["attention_mask"] is None


def test_backend_is_sdpa_under_cp_and_packing_is_refused(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    manager = _manager()
    manager.mesh = None
    with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
        assert ad.select_attention_backend(False) == ad.SDPA
        with pytest.raises(ValueError, match = "packing"):
            ad.select_attention_backend(True)


def test_sdpa_is_looked_up_per_call(monkeypatch):
    # context_parallel swaps torch.nn.functional.scaled_dot_product_attention for ring attention;
    # a name bound at import would silently keep attending to the local shard only.
    calls = []
    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", lambda *a, **k: calls.append(1)
    )
    ad.scaled_dot_product_attention(None, None, None)
    assert calls == [1]


def test_training_step_divides_the_pre_shard_token_count_by_cp_size(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    seen = []

    class Trainer:
        def __init__(self):
            pass

        def training_step(
            self,
            model,
            inputs,
            num_items_in_batch = None,
        ):
            seen.append(("train", num_items_in_batch))

        def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            num_items_in_batch = None,
        ):
            seen.append(("eval", num_items_in_batch))
            return (None, None, None)

        def train(self):
            pass

        def evaluate(self):
            pass

    import trl

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    trainer = Trainer()
    manager = _manager(size = 4)
    manager.mesh = None
    trainer._context_parallel_manager = manager
    batch = lambda: {"input_ids": torch.ones(1, 8, dtype = torch.long)}
    trainer.training_step(None, batch(), torch.tensor(12.0))
    trainer.training_step(None, batch(), num_items_in_batch = torch.tensor(12.0))
    # Eval counts tokens after sharding: the gathered count is already the global one.
    trainer.prediction_step(None, batch(), True, num_items_in_batch = torch.tensor(12.0))
    assert [(k, float(n)) for k, n in seen] == [("train", 3.0), ("train", 3.0), ("eval", 12.0)]


def test_shift_labels_is_not_an_eval_label_name():
    # find_labels treats every forward parameter containing "label" as required: a named
    # shift_labels made plain eval batches look label-less (KeyError 'prompt' in prediction_step).
    from transformers.utils import find_labels
    from unsloth.models.llama import CausalLM_fast_forward, LlamaModel_fast_forward_inference

    class LM(torch.nn.Module):
        forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)

    assert find_labels(LM) == ["labels"]


def test_gqa_is_expanded_under_cp(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    seen = []
    monkeypatch.setattr(
        ad,
        "scaled_dot_product_attention",
        lambda Q, K, V, **k: (seen.append((K.shape[1], k.get("enable_gqa"))), Q)[1],
    )
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = 2, n_groups = 2)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = 4,
        kv_seq_len = 4,
        n_heads = 4,
        head_dim = 8,
        requires_grad = True,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
    )
    Q, K = torch.zeros(1, 4, 4, 8), torch.zeros(1, 2, 4, 8)
    manager = _manager()
    manager.mesh = None
    with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
        ad.run_attention(config = config, context = context, Q = Q, K = K, V = K)
    assert seen == [(4, None)]


def test_every_rank_builds_the_same_global_mesh(monkeypatch):
    # DeviceMesh is SPMD; per-group rank lists ([0, 1] vs [2, 3]) hang process group creation.
    built = []

    class FakeMesh:
        def __init__(
            self,
            device_type,
            mesh,
            mesh_dim_names = None,
        ):
            built.append((mesh.tolist(), mesh_dim_names))

        def __getitem__(self, name):
            return ("submesh", name)

    monkeypatch.setattr(cp, "DeviceMesh", FakeMesh)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: 4)
    for rank in range(4):
        monkeypatch.setattr(cp.dist, "get_rank", lambda rank = rank: rank)
        manager = cp.ContextParallelManager(2)
        assert manager.mesh == ("submesh", "cp")
    assert built == [([[0, 1], [2, 3]], ("dp_replicate", "cp"))] * 4


def _patched_trainer(monkeypatch, **init_attrs):
    import trl

    class Trainer:
        def __init__(self, *a, **k):
            for key, value in init_attrs.items():
                setattr(self, key, value)

        def training_step(
            self,
            model,
            inputs,
            num_items_in_batch = None,
        ):
            pass

        def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys = None,
        ):
            return (torch.tensor(1.0), None, None)

        def train(self, *a, **k):
            return "trained"

        def evaluate(self, *a, **k):
            return "evaluated"

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    return Trainer


def test_predictions_are_refused_but_loss_only_eval_runs(monkeypatch):
    import types

    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    Trainer = _patched_trainer(monkeypatch)
    trainer = object.__new__(Trainer)
    manager = _manager()
    manager.mesh = None
    trainer._context_parallel_manager = manager
    batch = lambda: {"input_ids": torch.ones(1, 4, dtype = torch.long)}
    reduced = []
    monkeypatch.setattr(
        cp.dist, "all_reduce", lambda t, group = None: (reduced.append(t), t.mul_(2))
    )
    manager.mesh = types.SimpleNamespace(get_group = lambda: None)
    loss, *_ = trainer.prediction_step(None, batch(), True)
    # Every CP rank reports the group's mean loss (here 2 ranks of 1.0 each).
    assert reduced and loss.item() == 1.0
    with pytest.raises(NotImplementedError, match = "loss-only"):
        trainer.prediction_step(None, batch(), False)


@pytest.mark.parametrize(
    "attrs",
    [
        {"label_smoothing_factor": 0.1},
        {"compute_loss_func": lambda *a, **k: 0},
        {"loss_type": "chunked_nll"},
    ],
)
def test_label_dropping_loss_paths_are_refused(monkeypatch, attrs):
    import types

    monkeypatch.setattr(cp.dist, "is_available", lambda: True)
    monkeypatch.setattr(cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = types.SimpleNamespace(context_parallel_size = 2, label_smoothing_factor = 0.0)
    for key in ("label_smoothing_factor", "loss_type"):
        if key in attrs:
            setattr(args, key, attrs[key])
    init = {"args": args}
    if "compute_loss_func" in attrs:
        init["compute_loss_func"] = attrs["compute_loss_func"]
    Trainer = _patched_trainer(monkeypatch, **init)
    with pytest.raises(NotImplementedError, match = "loss_type = 'nll'"):
        Trainer()


def test_unaveraged_token_counts_are_refused(monkeypatch):
    import types

    monkeypatch.setattr(cp.dist, "is_available", lambda: True)
    monkeypatch.setattr(cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = types.SimpleNamespace(
        context_parallel_size = 2,
        label_smoothing_factor = 0.0,
        average_tokens_across_devices = False,
    )
    Trainer = _patched_trainer(monkeypatch, args = args)
    with pytest.raises(NotImplementedError, match = "average_tokens_across_devices"):
        Trainer()


def test_old_accelerate_is_refused(monkeypatch):
    import types, accelerate

    monkeypatch.setattr(cp.dist, "is_available", lambda: True)
    monkeypatch.setattr(cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(accelerate, "__version__", "1.9.0")
    args = types.SimpleNamespace(context_parallel_size = 2, label_smoothing_factor = 0.0)
    Trainer = _patched_trainer(monkeypatch, args = args)
    with pytest.raises(NotImplementedError, match = "accelerate >= 1.10.0"):
        Trainer()


def test_fp32_qkv_is_downcast_under_cp(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    seen = []
    monkeypatch.setattr(
        ad, "scaled_dot_product_attention", lambda Q, K, V, **k: (seen.append(Q.dtype), Q)[1]
    )
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = 4, n_groups = 1)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = 4,
        kv_seq_len = 4,
        n_heads = 4,
        head_dim = 8,
        requires_grad = True,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
    )
    Q = torch.zeros(1, 4, 4, 8, dtype = torch.float32)
    ad.run_attention(config = config, context = context, Q = Q, K = Q, V = Q)
    manager = _manager()
    manager.mesh = None
    with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
        out = ad.run_attention(config = config, context = context, Q = Q, K = Q, V = Q)
    assert seen[0] == torch.float32 and seen[1] in (torch.bfloat16, torch.float16)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("distributed_type", ["DEEPSPEED", "FSDP"])
def test_deepspeed_and_fsdp_are_refused(monkeypatch, distributed_type):
    import types

    monkeypatch.setattr(cp.dist, "is_available", lambda: True)
    monkeypatch.setattr(cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cp, "_supports_context_parallel", lambda model: True)
    args = types.SimpleNamespace(context_parallel_size = 2, label_smoothing_factor = 0.0)
    accelerator = types.SimpleNamespace(
        distributed_type = types.SimpleNamespace(name = distributed_type)
    )
    Trainer = _patched_trainer(monkeypatch, args = args, accelerator = accelerator, model = None)
    with pytest.raises(NotImplementedError, match = "DDP only"):
        Trainer()


def _cp_env(monkeypatch):
    monkeypatch.setattr(cp.dist, "is_available", lambda: True)
    monkeypatch.setattr(cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cp, "_supports_context_parallel", lambda model: True)


@pytest.mark.parametrize("case", ["iterable", "dispatch", "eval_dict"])
def test_dispatched_or_iterable_loaders_are_refused(monkeypatch, case):
    import types

    _cp_env(monkeypatch)

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())

    args = types.SimpleNamespace(context_parallel_size = 2, label_smoothing_factor = 0.0)
    accelerator = types.SimpleNamespace(
        distributed_type = types.SimpleNamespace(name = "MULTI_GPU"),
        dispatch_batches = case == "dispatch",
    )
    Trainer = _patched_trainer(
        monkeypatch,
        args = args,
        accelerator = accelerator,
        model = None,
        train_dataset = Stream() if case == "iterable" else [1],
        eval_dataset = {"a": [1], "b": Stream()} if case == "eval_dict" else None,
    )
    with pytest.raises(NotImplementedError, match = "iterable datasets or dispatch_batches"):
        Trainer()


def test_a_later_non_cp_trainer_drops_the_installed_mesh(monkeypatch):
    import types

    mesh = object()
    monkeypatch.setattr(cp, "_INSTALLED_MESH", [mesh])
    state = types.SimpleNamespace(device_mesh = mesh)
    args = types.SimpleNamespace(context_parallel_size = 1)
    Trainer = _patched_trainer(
        monkeypatch, args = args, accelerator = types.SimpleNamespace(state = state)
    )
    Trainer()
    assert state.device_mesh is None
    other = object()
    state.device_mesh = other  # someone else's mesh (e.g. accelerate's own) is left alone
    Trainer()
    assert state.device_mesh is other


def test_every_attention_layer_must_be_the_llama_forward():
    from unsloth.models.llama import LlamaAttention_fast_forward

    class Llama(torch.nn.Module):
        forward = LlamaAttention_fast_forward

    class Other(torch.nn.Module):
        def forward(self, x):
            return x

    def model(*attns):
        root = torch.nn.Module()
        root.layers = torch.nn.ModuleList()
        for attn in attns:
            layer = torch.nn.Module()
            layer.self_attn = attn()
            root.layers.append(layer)
        return root

    assert cp._supports_context_parallel(model(Llama, Llama))
    assert not cp._supports_context_parallel(model(Llama, Other))
    assert not cp._supports_context_parallel(model())


@pytest.mark.parametrize("user_value", [None, True])
def test_find_unused_parameters_is_off_under_cp_unless_the_user_set_it(monkeypatch, user_value):
    import types

    _cp_env(monkeypatch)
    monkeypatch.setattr(
        cp,
        "ContextParallelManager",
        lambda size: types.SimpleNamespace(
            size = size, device_mesh = object(), attach_attention_hooks = lambda model: None
        ),
    )
    args = types.SimpleNamespace(
        context_parallel_size = 2,
        label_smoothing_factor = 0.0,
        ddp_find_unused_parameters = user_value,
    )
    # What the Trainer builds for a PeftModel when the user leaves the argument as None.
    handler = types.SimpleNamespace(find_unused_parameters = True if user_value is None else user_value)
    accelerator = types.SimpleNamespace(
        distributed_type = types.SimpleNamespace(name = "MULTI_GPU"),
        state = types.SimpleNamespace(device_mesh = None),
        ddp_handler = handler,
    )
    Trainer = _patched_trainer(
        monkeypatch, args = args, accelerator = accelerator, model = None, train_dataset = [1]
    )
    Trainer()
    expected = False if user_value is None else True
    # 5.x reads the handler built at init; 4.x rebuilds it from the argument at train time.
    assert handler.find_unused_parameters is expected
    assert args.ddp_find_unused_parameters is expected


def test_active_manager_is_visible_from_the_autograd_device_thread(monkeypatch):
    # On CUDA the reentrant checkpoint recompute runs on autograd's device thread, not this one.
    import threading

    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    manager = _manager()
    manager.mesh = None
    seen = []
    with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
        thread = threading.Thread(target = lambda: seen.append(cp.get_cp_manager()))
        thread.start()
        thread.join()
    assert seen == [manager]
    assert cp.get_cp_manager() is None


def test_transformers_cp_size_is_refused(monkeypatch):
    import types

    args = types.SimpleNamespace(
        context_parallel_size = 1, parallelism_config = types.SimpleNamespace(cp_size = 2)
    )
    Trainer = _patched_trainer(monkeypatch, args = args)
    with pytest.raises(NotImplementedError, match = "context_parallel_size"):
        Trainer()


def test_cp_state_is_reinstalled_at_train_after_another_trainer(monkeypatch):
    import types

    mesh = object()
    state = types.SimpleNamespace(device_mesh = None)
    Trainer = _patched_trainer(monkeypatch, args = types.SimpleNamespace(context_parallel_size = 1))
    trainer = object.__new__(Trainer)
    trainer.accelerator = types.SimpleNamespace(state = state)
    trainer.args = types.SimpleNamespace(ddp_find_unused_parameters = None)
    trainer._context_parallel_manager = types.SimpleNamespace(size = 2, device_mesh = mesh)
    # A later non-CP trainer cleared the shared state; train() must put the CP mesh back.
    assert trainer.train() == "trained"
    assert state.device_mesh is mesh
    assert trainer.args.ddp_find_unused_parameters is False


def test_attention_hooks_are_not_stacked_by_a_second_manager():
    model = torch.nn.Module()
    model.self_attn = torch.nn.Linear(2, 2)
    for _ in range(2):
        _manager().attach_attention_hooks(model)
    assert len(model.self_attn._forward_pre_hooks) == 1


def test_sdpa_is_restored_when_the_cp_step_raises(monkeypatch):
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention

    @contextlib.contextmanager
    def leaky(mesh, buffers, buffer_seq_dims, no_restore_buffers):
        F.scaled_dot_product_attention = lambda *a, **k: None  # torch < 2.13 has no finally here
        yield
        F.scaled_dot_product_attention = original

    monkeypatch.setattr(cp, "context_parallel", leaky)
    manager = _manager()
    manager.mesh = None
    with pytest.raises(RuntimeError):
        with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
            raise RuntimeError("OOM")
    assert F.scaled_dot_product_attention is original
