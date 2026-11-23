# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_BLOCK_CUDAGRAPHS=1, native FSDP2 (gloo, world 2): MoE blocks run without graphs, so their
gathered weights must still be freed at reshard; dense blocks keep theirs."""

import importlib.util
import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_block_cudagraphs_gloo import _gathered, _mark_blocks  # noqa: E402
from test_fsdp2_native_layout_agree_gloo import _init, _spawn  # noqa: E402
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402

STEPS = 2


def _load_block_compile():
    # by file path: a CPU-only `import unsloth` is refused
    name = "unsloth.models._block_compile"
    if name not in sys.modules:
        path = os.path.join(REPO, "unsloth", "models", "_block_compile.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _build_moe():
    # layer 0 dense (mlp_only_layers), layer 1 MoE
    import transformers

    config = transformers.Qwen3MoeConfig(
        vocab_size = 256,
        hidden_size = 32,
        intermediate_size = 64,
        moe_intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        num_experts = 4,
        num_experts_per_tok = 2,
        mlp_only_layers = [0],
        max_position_embeddings = 64,
    )
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)


def _block_index(model):
    out = {}
    for i, layer in enumerate(model.model.layers):
        for m in layer.modules():
            out[id(m)] = i
    return out


def _worker(rank, world, port, out, path):
    dist = _init(rank, world, port)
    os.environ["UNSLOTH_BLOCK_CUDAGRAPHS"] = "1"
    try:
        native = _load_native()
        _load_block_compile()
        model = _build_moe()
        restore = _mark_blocks(model)
        try:
            index = _block_index(model)
            native.shard_model(
                model, native.FSDP2Config(param_dtype = torch.float32, fit_layout = False)
            )
            model.train()
            opt = torch.optim.SGD(model.parameters(), lr = 0.1)
            for step in range(STEPS):
                ids = torch.randint(
                    3, 256, (2, 16), generator = torch.Generator().manual_seed(10 * rank + step)
                )
                model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
                opt.step()
                opt.zero_grad(set_to_none = True)
            from torch.distributed.fsdp import FSDPModule

            sizes = {0: [], 1: []}
            for module in model.modules():
                if not isinstance(module, FSDPModule):
                    continue
                group = getattr(module._get_fsdp_state(), "_fsdp_param_group", None)
                for fsdp_param in getattr(group, "fsdp_params", None) or ():
                    block = index.get(id(fsdp_param._module_info.module))
                    if block is None:
                        continue
                    for t in getattr(fsdp_param, "all_gather_outputs", None) or []:
                        sizes[block].append(t.untyped_storage().size())
            assert _gathered(model)
        finally:
            for cls, original in restore.items():
                cls.forward = original
        torch.save(sizes, f"{path}.rank{rank}")
        out.put("ok")
    except Exception as exc:
        import traceback
        out.put({"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        os.environ.pop("UNSLOTH_BLOCK_CUDAGRAPHS", None)
        dist.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3MoeConfig"):
        pytest.skip("transformers without Qwen3-MoE")
    path = str(tmp_path_factory.mktemp("cudagraphs_moe") / "res.pt")
    payloads = _spawn(_worker, path)
    assert payloads == ["ok", "ok"], payloads
    return [torch.load(f"{path}.rank{r}") for r in range(2)]


def test_moe_block_weights_are_freed_under_block_cudagraphs(ranks):
    for sizes in ranks:
        assert sizes[1], "no FSDP-managed parameters in the MoE block"
        assert all(size == 0 for size in sizes[1]), sizes[1]


def test_dense_block_weights_stay_kept_under_block_cudagraphs(ranks):
    for sizes in ranks:
        assert sizes[0], "no FSDP-managed parameters in the dense block"
        assert all(size > 0 for size in sizes[0]), sizes[0]
