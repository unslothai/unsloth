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


def test_none_position_ids_and_shift_labels_are_rebuilt():
    inputs = {
        "input_ids": torch.ones(1, 4, dtype = torch.long),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "position_ids": None,
        "shift_labels": None,
    }
    _manager()._prepare_inputs(inputs)
    assert inputs["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert inputs["shift_labels"].tolist() == [[2, 3, 4, -100]]


def test_labels_shift_before_sharding_and_pad_to_load_balancer_chunks():
    manager = _manager(size = 2)
    labels = torch.tensor([[-100, 5, 6, 7, 8, 9]])
    inputs = {"input_ids": torch.tensor([[1, 5, 6, 7, 8, 9]]), "labels": labels.clone()}
    manager._prepare_inputs(inputs)
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
    # A name bound at import would keep attending to the local shard only.
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

        def prediction_step(self, model, inputs, prediction_loss_only):
            return (None, None, None)

        def train(self):
            pass

        def evaluate(self):
            pass

        def evaluation_loop(self, *a, **k):
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
    trainer.training_step(None, batch(), torch.tensor(12))
    assert [(k, float(n)) for k, n in seen] == [("train", 3.0)] * 3
    assert seen[-1][1].dtype == torch.int64


def test_shift_labels_is_not_an_eval_label_name():
    # A named shift_labels made find_labels treat it as a required eval label.
    from transformers.utils import find_labels
    from unsloth.models.llama import CausalLM_fast_forward, LlamaModel_fast_forward_inference

    class LM(torch.nn.Module):
        forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)

    assert find_labels(LM) == ["labels"]


@pytest.mark.parametrize("opt_out", [False, True])
def test_gqa_stays_native_under_cp_unless_opted_out(monkeypatch, opt_out):
    if opt_out:
        monkeypatch.setenv("UNSLOTH_CP_GQA", "0")
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    monkeypatch.setattr(ad, "SDPA_HAS_GQA", True)
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
    assert seen == ([(4, None)] if opt_out else [(2, True)])


def test_every_rank_builds_the_same_global_mesh(monkeypatch):
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

        def _prepare_inputs(self, inputs):
            return inputs

        def compute_loss_context_manager(self):
            return contextlib.nullcontext()

        def compute_loss(self, model, inputs):
            assert inputs["use_cache"] is False
            targets = inputs["shift_labels"].ne(-100).sum()
            return targets.float() / inputs["num_items_in_batch"]

        def train(self, *a, **k):
            return "trained"

        def evaluate(self, *a, **k):
            return "evaluated"

        def predict(self, *a, **k):
            return "predicted"

        def evaluation_loop(self, *a, **k):
            return self.gather_function(torch.arange(1.0, 3.0))

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    return Trainer


@pytest.mark.parametrize("world_size", [2, 4])
def test_eval_loss_is_the_token_mean_even_with_an_empty_shard(monkeypatch, world_size):
    import types

    @contextlib.contextmanager
    def first_half(mesh, buffers, buffer_seq_dims, no_restore_buffers):
        for buffer in buffers:
            buffer.data = buffer[:, : buffer.shape[1] // 2].clone()
        yield

    monkeypatch.setattr(cp, "context_parallel", first_half)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: world_size)
    Trainer = _patched_trainer(monkeypatch)
    trainer = object.__new__(Trainer)
    manager = _manager(size = 2)
    manager.mesh = types.SimpleNamespace(get_group = lambda: "cp")
    trainer._context_parallel_manager = manager
    counts = []

    def all_reduce(tensor, group = None):
        assert group == "cp"
        counts.append(group)
        tensor.add_(2 / 2)

    monkeypatch.setattr(cp.dist, "all_reduce", all_reduce)
    labels = torch.tensor([[-100] * 7 + [1, 1]])
    batch = lambda: {"input_ids": torch.ones(1, 9, dtype = torch.long), "labels": labels.clone()}
    loss, logits, labels_out = trainer.prediction_step(None, batch(), True)
    assert counts == ["cp"] and loss.item() == pytest.approx(1.0)
    assert logits is None and labels_out is None
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
    monkeypatch.setattr(cp, "_supports_context_parallel", lambda model, backend = "ring": True)
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
    monkeypatch.setattr(cp, "_supports_context_parallel", lambda model, backend = "ring": True)


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
    with pytest.raises(NotImplementedError, match = "iterable datasets|dispatch_batches"):
        Trainer()


def test_iterable_eval_dataset_passed_to_evaluate_is_refused(monkeypatch):
    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())

    Trainer = _patched_trainer(monkeypatch)
    trainer = object.__new__(Trainer)
    trainer._context_parallel_manager = _manager()
    with pytest.raises(NotImplementedError, match = "iterable datasets"):
        trainer.evaluate(Stream())
    with pytest.raises(NotImplementedError, match = "iterable datasets"):
        trainer.evaluate(eval_dataset = {"a": Stream()})
    trainer._context_parallel_manager = None
    assert trainer.evaluate(Stream()) == "evaluated"


def test_iterable_predict_dataset_is_refused(monkeypatch):
    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())

    Trainer = _patched_trainer(monkeypatch)
    trainer = object.__new__(Trainer)
    trainer._context_parallel_manager = _manager()
    with pytest.raises(NotImplementedError, match = "iterable datasets"):
        trainer.predict(Stream())
    trainer._context_parallel_manager = None
    assert trainer.predict(Stream()) == "predicted"


def test_sliding_window_models_are_refused(monkeypatch):
    import types

    _cp_env(monkeypatch)
    args = types.SimpleNamespace(context_parallel_size = 2, label_smoothing_factor = 0.0)
    accelerator = types.SimpleNamespace(distributed_type = types.SimpleNamespace(name = "MULTI_GPU"))
    model = types.SimpleNamespace(config = types.SimpleNamespace(sliding_window = 4096))
    Trainer = _patched_trainer(monkeypatch, args = args, accelerator = accelerator, model = model)
    with pytest.raises(NotImplementedError, match = "sliding window"):
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
    state.device_mesh = other
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
        lambda size, *rest: types.SimpleNamespace(
            size = size,
            backend = "ring",
            device_mesh = object(),
            attach_attention_hooks = lambda model: None,
        ),
    )
    args = types.SimpleNamespace(
        context_parallel_size = 2,
        label_smoothing_factor = 0.0,
        ddp_find_unused_parameters = user_value,
    )
    handler = types.SimpleNamespace(
        find_unused_parameters = True if user_value is None else user_value
    )
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
    trainer._context_parallel_manager = types.SimpleNamespace(
        size = 2, backend = "ring", device_mesh = mesh
    )
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
        F.scaled_dot_product_attention = lambda *a, **k: None  # torch < 2.13
        yield
        F.scaled_dot_product_attention = original

    monkeypatch.setattr(cp, "context_parallel", leaky)
    manager = _manager()
    manager.mesh = None
    with pytest.raises(RuntimeError):
        with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
            raise RuntimeError("OOM")
    assert F.scaled_dot_product_attention is original


@pytest.mark.parametrize("remainder", [0, 3])
def test_eval_gather_keeps_one_loss_per_cp_group(monkeypatch, remainder):
    import types

    per_rank = [torch.tensor([10.0 * r + 1, 10.0 * r + 2]) for r in (0, 0, 1, 1, 2, 2)]
    accelerator = types.SimpleNamespace(
        gather = lambda t: torch.cat(per_rank),
        gather_for_metrics = lambda t, *a, **k: t,
        gradient_state = types.SimpleNamespace(end_of_dataloader = True, remainder = remainder),
    )
    Trainer = _patched_trainer(monkeypatch)
    trainer = object.__new__(Trainer)
    trainer.accelerator = accelerator
    trainer._context_parallel_manager = _manager(size = 2)
    out = trainer.evaluation_loop()
    expected = [1.0, 2.0, 11.0, 12.0, 21.0, 22.0]
    assert out.tolist() == (expected[:remainder] if remainder else expected)


def test_replaced_iterable_train_dataset_is_refused_at_train(monkeypatch):
    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())

    Trainer = _patched_trainer(monkeypatch)
    trainer = object.__new__(Trainer)
    trainer._context_parallel_manager = _manager()
    trainer.train_dataset = Stream()
    with pytest.raises(NotImplementedError, match = "iterable datasets"):
        trainer.train()


def test_null_inputs_embeds_placeholder_is_allowed():
    inputs = {"input_ids": torch.ones(1, 4, dtype = torch.long), "inputs_embeds": None}
    _manager()._prepare_inputs(inputs)
    assert inputs["position_ids"].shape == (1, 4)


def test_replaced_iterable_eval_dataset_is_refused_at_evaluate(monkeypatch):
    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())

    Trainer = _patched_trainer(monkeypatch)
    trainer = object.__new__(Trainer)
    trainer._context_parallel_manager = _manager()
    trainer.eval_dataset = Stream()
    with pytest.raises(NotImplementedError, match = "iterable datasets"):
        trainer.evaluate()
    trainer.eval_dataset = {"a": [1], "b": Stream()}
    with pytest.raises(NotImplementedError, match = "iterable datasets"):
        trainer.evaluate("b")
    assert trainer.evaluate("a") == "evaluated"


def test_model_init_is_refused(monkeypatch):
    import types

    _cp_env(monkeypatch)
    args = types.SimpleNamespace(context_parallel_size = 2, label_smoothing_factor = 0.0)
    Trainer = _patched_trainer(monkeypatch, args = args, model_init = lambda: None)
    with pytest.raises(NotImplementedError, match = "model_init"):
        Trainer()


def test_cp_dtensor_dispatcher_is_torn_down_when_the_step_raises(monkeypatch):
    dispatcher = cp.DTensor._op_dispatcher
    original = dispatcher._custom_op_handlers
    unregistered = []
    monkeypatch.setattr(cp, "_unregister_cp_sharding_rules", lambda: unregistered.append(1))

    @contextlib.contextmanager
    def leaky(mesh, buffers, buffer_seq_dims, no_restore_buffers):
        dispatcher._custom_op_handlers = {**original, "cp": None}  # torch < 2.13: no finally
        yield
        dispatcher._custom_op_handlers = original

    monkeypatch.setattr(cp, "context_parallel", leaky)
    manager = _manager()
    manager.mesh = None
    with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
        pass
    assert dispatcher._custom_op_handlers is original and unregistered == []
    with pytest.raises(RuntimeError):
        with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
            raise RuntimeError("OOM")
    assert dispatcher._custom_op_handlers is original and unregistered == [1]


def _gated_trainer(
    monkeypatch,
    model = None,
    **args_overrides,
):
    """A trainer that clears every CP gate, with the manager stubbed (no process group)."""
    import types

    _cp_env(monkeypatch)
    built = []

    def manager(
        size,
        backend = "ring",
        rotate_method = "allgather",
    ):
        built.append((size, backend, rotate_method))
        return types.SimpleNamespace(
            size = size,
            backend = backend,
            device_mesh = object(),
            attach_attention_hooks = lambda model: None,
        )

    monkeypatch.setattr(cp, "ContextParallelManager", manager)
    args = types.SimpleNamespace(context_parallel_size = 2, label_smoothing_factor = 0.0)
    for key, value in args_overrides.items():
        setattr(args, key, value)
    accelerator = types.SimpleNamespace(
        distributed_type = types.SimpleNamespace(name = "MULTI_GPU"),
        state = types.SimpleNamespace(device_mesh = None),
    )
    init = {"args": args, "accelerator": accelerator, "model": model, "train_dataset": [1]}
    return _patched_trainer(monkeypatch, **init), args, built


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"context_parallel_size": True}, ValueError),
        ({"context_parallel_size": 0}, ValueError),
        ({"context_parallel_size": -2}, ValueError),
        ({"context_parallel_size": 2.0}, ValueError),
        ({"context_parallel_size": "2"}, ValueError),
        ({"context_parallel_backend": "tree"}, ValueError),
        ({"context_parallel_rotate_method": "ring"}, ValueError),
    ],
)
def test_bad_cp_options_fail_before_the_trainer_builds(monkeypatch, overrides, error):
    import types, trl

    seen = []

    class Trainer:
        def __init__(self, *a, **k):
            seen.append(1)

        training_step = prediction_step = train = evaluate = evaluation_loop = None

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    args = types.SimpleNamespace(**{"context_parallel_size": 2, **overrides})
    with pytest.raises(error):
        Trainer(None, args)
    with pytest.raises(error):
        Trainer(None, args = args)
    assert seen == []


def test_cp_option_defaults_and_size_one():
    import types

    assert cp.context_parallel_options(None) == (1, "ring", "allgather")
    config = types.SimpleNamespace(context_parallel_size = 1, context_parallel_backend = "ulysses")
    assert cp.context_parallel_options(config) == (1, "ulysses", "allgather")
    config = types.SimpleNamespace(
        context_parallel_size = 4, context_parallel_rotate_method = "alltoall"
    )
    assert cp.context_parallel_options(config) == (4, "ring", "alltoall")


def test_manager_gets_the_backend_and_rotate_method(monkeypatch):
    Trainer, _, built = _gated_trainer(monkeypatch, context_parallel_rotate_method = "alltoall")
    Trainer()
    assert built == [(2, "ring", "alltoall")]


@pytest.mark.parametrize("version", ["2.6.0", "2.6.0+cu124", "2.5.1"])
def test_torch_older_than_2_7_is_refused(monkeypatch, version):
    Trainer, _, _ = _gated_trainer(monkeypatch)
    monkeypatch.setattr(cp.torch, "__version__", version)
    with pytest.raises(RuntimeError, match = "PyTorch >= 2.7"):
        Trainer()


def test_torch_2_7_dev_builds_pass_the_gate(monkeypatch):
    monkeypatch.setattr(cp.torch, "__version__", "2.7.0a0+git1234")
    assert cp._torch_supports_context_parallel()


@pytest.mark.parametrize(
    "config, reason",
    [
        ({"layer_types": ["full_attention", "sliding_attention"]}, "sliding_attention"),
        ({"layer_types": ["chunked_attention", "full_attention"]}, "chunked_attention"),
        ({"attn_logit_softcapping": 50.0}, "softcapping"),
        ({"attention_chunk_size": 8192}, "chunked attention"),
        ({"use_sliding_window": True}, "sliding window"),
        ({"text_config": {"layer_types": ["sliding_attention"]}}, "sliding_attention"),
    ],
)
def test_windowed_or_softcapped_attention_is_refused_for_ring(monkeypatch, config, reason):
    import types

    if "text_config" in config:
        config = {"text_config": types.SimpleNamespace(**config["text_config"])}
    model = types.SimpleNamespace(config = types.SimpleNamespace(**config))
    Trainer, _, _ = _gated_trainer(monkeypatch, model = model)
    with pytest.raises(NotImplementedError, match = reason):
        Trainer()


def test_full_attention_layer_types_pass(monkeypatch):
    import types

    config = types.SimpleNamespace(
        layer_types = ["full_attention"] * 2, sliding_window = None, use_sliding_window = False
    )
    Trainer, _, built = _gated_trainer(monkeypatch, model = types.SimpleNamespace(config = config))
    Trainer()
    assert built


def test_legacy_prediction_loop_is_refused(monkeypatch):
    Trainer, _, built = _gated_trainer(monkeypatch, use_legacy_prediction_loop = True)
    with pytest.raises(NotImplementedError, match = "use_legacy_prediction_loop"):
        Trainer()
    assert built == []


def test_model_without_loss_kwargs_is_refused(monkeypatch):
    Trainer, _, _ = _gated_trainer(monkeypatch)
    monkeypatch.setattr(Trainer, "model_accepts_loss_kwargs", False, raising = False)
    with pytest.raises(NotImplementedError, match = "loss kwargs"):
        Trainer()


def test_flash_attention_impl_is_scoped_to_train_without_packing(monkeypatch):
    from unsloth.utils import packing

    scoped = []
    monkeypatch.setattr(
        packing, "_scope_flash_attention_impl", lambda model, trainer: scoped.append(trainer)
    )
    Trainer, _, _ = _gated_trainer(monkeypatch)
    trainer = Trainer()
    assert scoped == [trainer]


def test_training_step_refuses_a_missing_token_count_with_labels(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    Trainer = _patched_trainer(monkeypatch)
    trainer = object.__new__(Trainer)
    manager = _manager()
    manager.mesh = None
    trainer._context_parallel_manager = manager
    batch = {"input_ids": torch.ones(1, 4, dtype = torch.long), "labels": torch.ones(1, 4)}
    with pytest.raises(RuntimeError, match = "num_items_in_batch"):
        trainer.training_step(None, dict(batch))
    with pytest.raises(RuntimeError, match = "num_items_in_batch"):
        trainer.training_step(None, dict(batch), None)
    shifted = {"input_ids": torch.ones(1, 4, dtype = torch.long), "shift_labels": torch.ones(1, 4)}
    with pytest.raises(RuntimeError, match = "num_items_in_batch"):
        trainer.training_step(None, shifted)
    assert cp.get_cp_manager() is None
    trainer.training_step(None, {"input_ids": torch.ones(1, 4, dtype = torch.long)})
    trainer.training_step(None, dict(batch), torch.tensor(4.0))


@pytest.mark.parametrize("raises", [False, True])
def test_rotate_method_is_set_for_the_step_and_restored(monkeypatch, raises):
    if cp._cp_options is None:
        pytest.skip("this torch has no ring rotate option")
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    manager = _manager()
    manager.mesh = None
    manager.rotate_method = "alltoall"
    before = cp._cp_options.rotate_method
    seen = []
    with contextlib.suppress(RuntimeError):
        with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
            seen.append(cp._cp_options.rotate_method.name)
            if raises:
                raise RuntimeError("boom")
    assert seen == ["ALL_TO_ALL"]
    assert cp._cp_options.rotate_method == before


@pytest.mark.parametrize("raises", [False, True])
def test_ring_pins_sdpa_to_kernels_the_dispatcher_rings(monkeypatch, raises):
    if cp.sdpa_kernel is None:
        pytest.skip("no torch.nn.attention.sdpa_kernel")
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    manager = _manager()
    manager.mesh = None
    cuda = torch.backends.cuda
    before = (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.math_sdp_enabled())
    seen = []
    with contextlib.suppress(RuntimeError):
        with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
            seen.append(
                (
                    cuda.flash_sdp_enabled(),
                    cuda.mem_efficient_sdp_enabled(),
                    cuda.math_sdp_enabled(),
                )
            )
            if raises:
                raise RuntimeError("boom")
    assert seen == [(True, True, False)]
    assert (
        cuda.flash_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
    ) == before


def test_cudnn_is_pinned_only_where_the_ring_handles_it():
    backends = cp._ring_sdpa_backends()
    if backends is None:
        pytest.skip("no SDPBackend")
    from torch.nn.attention import SDPBackend

    assert backends[:2] == [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    from packaging.version import Version

    if Version(Version(torch.__version__).base_version) >= Version("2.13"):
        assert SDPBackend.CUDNN_ATTENTION in backends


def test_unsloth_training_arguments_keep_the_cp_options():
    from unsloth.trainer import UnslothTrainingArguments
    args = UnslothTrainingArguments(
        output_dir = "unused",
        context_parallel_size = 2,
        context_parallel_rotate_method = "alltoall",
        report_to = "none",
    )
    assert cp.context_parallel_options(args) == (2, "ring", "alltoall")


def test_cli_forwards_the_cp_options():
    import importlib.util, pathlib, types

    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth-cli.py"
    spec = importlib.util.spec_from_file_location("unsloth_cli_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = types.SimpleNamespace(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 0,
        max_steps = 1,
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 0,
        output_dir = "unused",
        report_to = "none",
        max_seq_length = 64,
        packing = False,
        per_device_eval_batch_size = 4,
        context_parallel_size = 2,
        context_parallel_backend = "ring",
        context_parallel_rotate_method = "alltoall",
    )
    config = module._build_sft_config(lambda **kwargs: kwargs, args, False, True)
    assert config["context_parallel_size"] == 2
    assert config["context_parallel_backend"] == "ring"
    assert config["context_parallel_rotate_method"] == "alltoall"
    assert config["average_tokens_across_devices"] is True


def test_generated_sft_config_carries_the_cp_options(tmp_path):
    import dataclasses
    import trl.trainer.sft_config as config_module

    names = {f.name for f in dataclasses.fields(config_module.SFTConfig)}
    if "context_parallel_size" not in names:
        pytest.skip("SFTConfig is not the Unsloth-generated class in this process")
    assert {"context_parallel_backend", "context_parallel_rotate_method"} <= names
    config = config_module.SFTConfig(
        output_dir = str(tmp_path),
        context_parallel_size = 2,
        context_parallel_rotate_method = "alltoall",
        report_to = "none",
    )
    assert cp.context_parallel_options(config) == (2, "ring", "alltoall")
    default = config_module.SFTConfig(output_dir = str(tmp_path), report_to = "none")
    assert cp.context_parallel_options(default) == (1, "ring", "allgather")


class _FakeCPMesh:
    def __init__(self, rank):
        self.rank = rank

    def get_local_rank(self):
        return self.rank

    def get_group(self):
        return ("group", self.rank)


def _ulysses_manager(rank, size = 2):
    manager = _manager(size)
    manager.backend = "ulysses"
    manager.mesh = _FakeCPMesh(rank)
    return manager


def test_ulysses_shards_contiguously_after_global_shift():
    labels = torch.tensor([[-100, 5, 6, 7, 8, 9, 10]])
    shards = []
    for rank in range(2):
        inputs = {"input_ids": torch.arange(1, 8)[None], "labels": labels.clone()}
        manager = _ulysses_manager(rank)
        with manager.apply(inputs):
            assert cp.get_cp_manager() is manager
            shards.append(inputs)
        assert cp.get_cp_manager() is None
    assert [s["input_ids"].tolist() for s in shards] == [[[1, 2, 3, 4]], [[5, 6, 7, 0]]]
    assert [s["shift_labels"].tolist() for s in shards] == [[[5, 6, 7, 8]], [[9, 10, -100, -100]]]
    assert [s["position_ids"].tolist() for s in shards] == [[[0, 1, 2, 3]], [[4, 5, 6, 7]]]


def test_ulysses_packed_batch_keeps_global_lengths_and_masks_document_starts():
    lengths = torch.tensor([3, 4], dtype = torch.int32)
    labels = torch.tensor([[10, 11, 12, 20, 21, 22, 23]])
    shards = []
    for rank in range(2):
        inputs = {
            "input_ids": torch.arange(7)[None],
            "labels": labels.clone(),
            "packed_seq_lengths": lengths,
        }
        with _ulysses_manager(rank).apply(inputs):
            shards.append(inputs)
    assert all(s["packed_seq_lengths"] is lengths for s in shards)
    assert torch.cat([s["position_ids"] for s in shards], 1).tolist() == [[0, 1, 2, 0, 1, 2, 3, 4]]
    shift = torch.cat([s["shift_labels"] for s in shards], 1).tolist()
    assert shift == [[11, 12, -100, 21, 22, 23, -100, -100]]


def test_ulysses_refuses_packed_lengths_longer_than_the_row():
    inputs = {
        "input_ids": torch.ones(1, 4, dtype = torch.long),
        "position_ids": torch.arange(4)[None],
        "packed_seq_lengths": torch.tensor([3, 10**9]),
    }
    with pytest.raises(ValueError, match = "packed row has 4 tokens"):
        _ulysses_manager(0)._prepare_inputs(inputs)


def test_ring_refuses_packed_batches():
    inputs = {
        "input_ids": torch.ones(1, 4, dtype = torch.long),
        "packed_seq_lengths": torch.tensor([4]),
    }
    manager = _manager()
    manager.backend = "ring"
    with pytest.raises(ValueError, match = "ring context parallelism does not support packing"):
        manager._prepare_inputs(inputs)


def test_ulysses_keeps_varlen_and_routes_attention_through_the_exchange(monkeypatch):
    calls = []

    def fake_ulysses(run_attention_fn, config, context, Q, K, V, group):
        calls.append((run_attention_fn, group))
        return "exchanged"

    monkeypatch.setattr(ad, "ulysses_attention", fake_ulysses)
    manager = _ulysses_manager(1)
    monkeypatch.setattr(cp, "_ACTIVE_MANAGER", manager)
    if ad.HAS_FLASH_ATTENTION:
        assert ad.select_attention_backend(True) == ad.FLASH_VARLEN
    else:
        assert ad.select_attention_backend(True) != ad.FLASH_VARLEN
    assert ad.run_attention(config = None, context = None, Q = 1, K = 2, V = 3) == "exchanged"
    assert calls == [(ad._run_attention, ("group", 1))]
    manager.backend = "ring"
    with pytest.raises(ValueError, match = "context_parallel_backend = 'ulysses'"):
        ad.select_attention_backend(True)


def test_inherited_trainer_methods_are_resolved_per_call(monkeypatch):
    import trl

    class Base:
        def train(self):
            return "base"

        def training_step(self, *a, **k):
            pass

        evaluate = evaluation_loop = prediction_step = training_step

    class Trainer(Base):
        def __init__(self):
            pass

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    monkeypatch.setattr(Base, "train", lambda self: "patched later")
    assert Trainer().train() == "patched later"


def test_ulysses_head_counts_are_checked_at_init(monkeypatch):
    import types

    config = types.SimpleNamespace(num_attention_heads = 3, num_key_value_heads = 3)
    model = types.SimpleNamespace(config = config)
    Trainer, _, built = _gated_trainer(monkeypatch, model = model, context_parallel_backend = "ulysses")
    with pytest.raises(ValueError, match = "divisible by context_parallel_size"):
        Trainer()
    assert built == []
    config.num_attention_heads = config.num_key_value_heads = 4
    Trainer()
    assert built == [(2, "ulysses", "allgather")]


@pytest.mark.parametrize("backend", ["ring", "ulysses"])
def test_only_ring_needs_sm80(monkeypatch, backend):
    import types

    config = types.SimpleNamespace(num_attention_heads = 4, num_key_value_heads = 4)
    Trainer, _, built = _gated_trainer(
        monkeypatch, model = types.SimpleNamespace(config = config), context_parallel_backend = backend
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    if backend == "ring":
        with pytest.raises(NotImplementedError, match = "compute capability"):
            Trainer()
    else:
        Trainer()
        assert built == [(2, "ulysses", "allgather")]
