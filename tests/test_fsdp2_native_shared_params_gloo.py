# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 must keep parameters shared across layers (Zamba2's shared transformer) tied."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Zamba2Config"):
    pytest.skip("transformers without Zamba2", allow_module_level = True)

KINDS = ["mamba", "hybrid", "mamba", "hybrid", "mamba", "hybrid"]


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _zamba2(device = None):
    config = transformers.Zamba2Config(
        vocab_size = 128,
        hidden_size = 32,
        num_hidden_layers = len(KINDS),
        layers_block_type = KINDS,
        num_mem_blocks = 1,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        attention_head_dim = 8,
        mamba_d_state = 8,
        mamba_headdim = 8,
        n_mamba_heads = 8,
        intermediate_size = 64,
        use_shared_attention_adapter = False,
        adapter_rank = 4,
        chunk_size = 16,
    )
    torch.manual_seed(0)
    if device == "meta":
        with torch.device("meta"):
            return transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    return transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)


def _shared_ids(model):
    owners = {}
    # Per path: transformers 4.x reuses one shared_transformer instance, which modules() yields once.
    for name, module in model.named_modules(remove_duplicate = False):
        for p in module.parameters(recurse = False):
            owners.setdefault(id(p), set()).add(name)
    return {k for k, v in owners.items() if len(v) > 1}


def test_zamba2_has_shared_parameters_across_layers():
    model = _zamba2("meta")
    assert _shared_ids(model), "the fixture no longer ties the hybrid layers"


class _Block(torch.nn.Module):
    def __init__(self, shared = None):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        if shared is not None:
            self.shared = shared


def _reused_module_model():
    # transformers 4.x Zamba2 layout: one module instance reused by several layers.
    shared = torch.nn.Linear(4, 4)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([_Block(shared), _Block(), _Block(shared), _Block()])
    return model


def test_reused_module_instance_counts_as_shared():
    native = _load_native()
    model = _reused_module_model()
    units = native.find_shard_units(model)
    assert units == [model.layers[1], model.layers[3]]
    with pytest.raises(RuntimeError, match = "tied / shared"):
        native.check_shard_units(model, list(model.layers))


def test_embedding_inside_a_dropped_block_is_not_an_extra_unit():
    native = _load_native()
    model = _reused_module_model()
    model.layers[0].table = torch.nn.Embedding(512, 4)
    units = native.find_shard_units(model)
    assert model.layers[0] not in units
    assert native._large_root_embeddings(model, units, 0.1) == []


def test_explicitly_ignored_shared_params_pass_the_check():
    native = _load_native()
    model = _reused_module_model()
    ignored = set(model.layers[0].shared.parameters())
    assert native.check_shard_units(model, list(model.layers), ignored)
    assert native.find_shard_units(model, ignored) == list(model.layers)


def test_find_shard_units_keeps_sharing_blocks_in_the_root():
    native = _load_native()
    model = _zamba2("meta")
    units = native.find_shard_units(model)
    assert [type(u).__name__ for u in units] == ["Zamba2MambaDecoderLayer"] * 3
    shared = _shared_ids(model)
    for unit in units:
        assert not any(id(p) in shared for p in unit.parameters())


def test_explicit_units_splitting_a_shared_parameter_raise():
    native = _load_native()
    model = _zamba2("meta")
    with pytest.raises(RuntimeError, match = "tied / shared"):
        native.check_shard_units(model, list(model.model.layers))


def _worker(
    rank,
    world,
    port,
    out,
    frozen = None,
):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
    )
    try:
        from torch.distributed.tensor import DTensor

        native = _load_native()
        ref, model = _zamba2(), _zamba2()
        layers = model.model.layers
        hybrid = [i for i, k in enumerate(KINDS) if k == "hybrid"]
        config = native.FSDP2Config(param_dtype = torch.float32, device = "cpu")
        if frozen:
            # Frozen shared weights replicated (shard_frozen=False): per-layer units are fine.
            for m in (ref, model):
                for p in m.model.layers[hybrid[0]].shared_transformer.parameters():
                    p.requires_grad_(False)
            if frozen == "explicit":
                config.units = list(layers)
            if frozen == "fit":
                # The fit layout replicates frozen weights on its own; keep per-block trainable units.
                config.fit_layout = True
                os.environ["UNSLOTH_FSDP2_TRAINABLE_UNITS"] = "blocks"
            else:
                config.shard_frozen = False
        native.shard_model(model, config)
        first = dict(layers[hybrid[0]].shared_transformer.named_parameters())
        tied = all(
            dict(layers[i].shared_transformer.named_parameters())[n] is p
            for i in hybrid[1:]
            for n, p in first.items()
        )
        ids = torch.randint(0, 128, (1, 16), generator = torch.Generator().manual_seed(0))
        loss = model(input_ids = ids, labels = ids, use_cache = False).loss
        loss.backward()
        ref_loss = ref(input_ids = ids, labels = ids, use_cache = False).loss
        ref_loss.backward()
        ref_params = dict(ref.named_parameters())
        diff = 0.0
        missing = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                missing.append(name)
                continue
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            diff = max(diff, float((g - ref_params[name].grad).abs().max()))
        res = {
            "units": [type(u).__name__.removeprefix("FSDP") for u in model._unsloth_fsdp2_units],
            "tied": tied,
            "loss_diff": abs(float(loss) - float(ref_loss)),
            "grad_diff": diff,
            "missing": missing,
        }
        out.put(res)
    except Exception as exc:
        import traceback
        out.put({"error": f"rank {rank} {type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("frozen", [None, "explicit", "auto", "fit"])
def test_zamba2_shards_with_shared_transformer_tied(frozen):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker,
        args = (2, _free_port(), out, frozen),
        nprocs = 2,
        join = True,
        start_method = "spawn",
    )
    results = [out.get(timeout = 5) for _ in range(2)]
    expected = (
        [type(layer).__name__ for layer in _zamba2("meta").model.layers]
        if frozen
        else (["Zamba2MambaDecoderLayer"] * 3)
    )
    for r in results:
        assert "error" not in r, r["error"]
        assert r["units"] == expected, r
        assert r["tied"] is True, r
        assert not r["missing"], r
        assert r["loss_diff"] < 1e-5 and r["grad_diff"] < 1e-5, r
