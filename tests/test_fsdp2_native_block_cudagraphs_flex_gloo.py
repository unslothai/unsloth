# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_BLOCK_CUDAGRAPHS=1, native FSDP2 (gloo, world 2): flex BlockMask blocks free their weights."""

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
from test_fsdp2_native_layout_agree_gloo import _init, _spawn  # noqa: E402
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402

STEPS = 3


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


def _mark_blocks(model, bc):
    """Stand-in for _block_compile's CUDA-only wrapper; `_test_no_mask` blocks count as non-BlockMask."""
    note = getattr(bc, "_note_block_mask_call", None)
    restore = {}
    for module in model.modules():
        cls = type(module)
        if cls.__name__.endswith("DecoderLayer") and cls not in restore:
            original = cls.forward

            def forward(
                self,
                *args,
                _original = original,
                **kwargs,
            ):
                seen = ((), {}) if self.__dict__.get("_test_no_mask") else (args[1:], kwargs)
                if note is not None:
                    note(self, *seen)
                else:
                    bc._flex_block_mask_call(*seen)
                return _original(self, *args, **kwargs)

            forward._unsloth_block_compiled = True
            restore[cls] = original
            cls.forward = forward
    return restore


def _dense_flex_on_cpu():
    """CPU flex attention has no backward: keep the BlockMask but attend through SDPA."""
    from torch.nn.attention.flex_attention import create_mask
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def flex_attention_forward(module, query, key, value, attention_mask, **kwargs):
        if type(attention_mask).__name__ == "BlockMask":
            q, kv = query.shape[-2], key.shape[-2]
            attention_mask = create_mask(attention_mask.mask_mod, None, None, q, kv, query.device)
        kwargs.pop("is_causal", None)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex_attention_forward


def _build(attn):
    import transformers

    config = transformers.Gemma3TextConfig(
        vocab_size = 256,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        sliding_window = 8,
        max_position_embeddings = 64,
        attn_implementation = attn,
    )
    torch.manual_seed(0)
    return transformers.Gemma3ForCausalLM(config).float()


def _sizes(model):
    """{block index: [all-gather output storage sizes]} after the step."""
    from torch.distributed.fsdp import FSDPModule

    index = {id(m): i for i, layer in enumerate(model.model.layers) for m in layer.modules()}
    sizes = {i: [] for i in range(len(model.model.layers))}
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        group = getattr(module._get_fsdp_state(), "_fsdp_param_group", None)
        for fsdp_param in getattr(group, "fsdp_params", None) or ():
            block = index.get(id(fsdp_param._module_info.module))
            if block is not None:
                for t in getattr(fsdp_param, "all_gather_outputs", None) or []:
                    sizes[block].append(t.untyped_storage().size())
    return sizes


def _run(
    native,
    bc,
    rank,
    attn,
    graphs,
    switch_to = None,
    no_mask = (),
    budget_blocks = None,
):
    os.environ["UNSLOTH_BLOCK_CUDAGRAPHS"] = "1" if graphs else "0"
    bc._COMPILE_OPTIONS["triton.cudagraphs"] = graphs  # read at import, as _block_compile does
    model = _build(attn)
    for i in no_mask:
        model.model.layers[i].__dict__["_test_no_mask"] = True
    if budget_blocks is not None:
        block = sum(p.numel() * 4 for p in model.model.layers[0].parameters())
        os.environ["UNSLOTH_BLOCK_CUDAGRAPHS_FSDP2_KEEP_GB"] = str(block * budget_blocks / 2**30)
    restore = _mark_blocks(model, bc)
    try:
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, fit_layout = False))
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr = 0.1)
        losses, sizes = [], []
        for step in range(STEPS):
            if switch_to is not None and step == STEPS - 1:
                model.config._attn_implementation = switch_to
            ids = torch.randint(
                3, 256, (2, 16), generator = torch.Generator().manual_seed(10 * rank + step)
            )
            loss = model(input_ids = ids, labels = ids, use_cache = False).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none = True)
            losses.append(float(loss.detach()))
            sizes.append(_sizes(model))
        off = [
            bool(layer.__dict__.get("_unsloth_block_graphs_off")) for layer in model.model.layers
        ]
        return {"losses": losses, "sizes": sizes, "graphs_off": off}
    finally:
        for cls, original in restore.items():
            cls.forward = original
        os.environ.pop("UNSLOTH_BLOCK_CUDAGRAPHS_FSDP2_KEEP_GB", None)


def _worker(rank, world, port, out, path):
    dist = _init(rank, world, port)
    try:
        native = _load_native()
        bc = _load_block_compile()
        _dense_flex_on_cpu()
        res = {
            "flex": _run(native, bc, rank, "flex_attention", True),
            "flex_no_graphs": _run(native, bc, rank, "flex_attention", False),
            "sdpa": _run(native, bc, rank, "sdpa", True),
            "flex_then_sdpa": _run(native, bc, rank, "flex_attention", True, switch_to = "sdpa"),
            # layer 0 called without a BlockMask; the budget holds one block, not two
            "mixed_budget": _run(
                native, bc, rank, "flex_attention", True, no_mask = (0,), budget_blocks = 1.5
            ),
        }
        torch.save(res, f"{path}.rank{rank}")
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
    if not hasattr(transformers, "Gemma3TextConfig"):
        pytest.skip("transformers without Gemma3")
    path = str(tmp_path_factory.mktemp("cudagraphs_flex") / "res.pt")
    payloads = _spawn(_worker, path)
    assert payloads == ["ok", "ok"], payloads
    res = [torch.load(f"{path}.rank{r}") for r in range(2)]
    for r, run in enumerate(res):
        for name, data in run.items():
            print("FLEX_KEEP", r, name, data["losses"], data["sizes"][-1], data["graphs_off"])
    return res


def test_flex_block_weights_are_freed_under_block_cudagraphs(ranks):
    for res in ranks:
        for step in res["flex"]["sizes"]:
            for sizes in step.values():
                assert sizes and all(size == 0 for size in sizes), step


def test_dense_block_weights_stay_kept_under_block_cudagraphs(ranks):
    for res in ranks:
        for sizes in res["sdpa"]["sizes"][-1].values():
            assert sizes and all(size > 0 for size in sizes), res["sdpa"]["sizes"]


def test_a_call_without_block_mask_keeps_the_weights_again(ranks):
    # Sometimes flex, sometimes not: the graphed calls need one address, so the block keeps.
    for res in ranks:
        run = res["flex_then_sdpa"]
        assert all(s == 0 for sizes in run["sizes"][0].values() for s in sizes), run["sizes"]
        for sizes in run["sizes"][-1].values():
            assert sizes and all(size > 0 for size in sizes), run["sizes"]


def test_flex_blocks_leave_the_keep_budget(ranks):
    # Two blocks over the budget; the flex one drops out, so the other keeps its weights and graphs.
    for res in ranks:
        run = res["mixed_budget"]
        last = run["sizes"][-1]
        assert last[0] and all(size > 0 for size in last[0]), run
        assert last[1] and all(size == 0 for size in last[1]), run
        assert run["graphs_off"][0] is False, run


def test_losses_are_unchanged(ranks):
    for res in ranks:
        assert res["flex"]["losses"] == res["flex_no_graphs"]["losses"]
        assert len(set(res["flex"]["losses"])) == STEPS


def test_flex_note_does_not_count_only_the_first_block(capsys):
    # Regression: the note once counted 1 block when every block takes a BlockMask.
    import types

    native = _load_native()
    blocks = [(torch.nn.Module(), [types.SimpleNamespace()], 10) for _ in range(3)]
    keeper = native._BlockWeightKeep(blocks, 1000)
    keeper.apply()
    for block, _, _ in blocks:
        block.__dict__["_unsloth_block_flex_only"] = True
        keeper.observe(block)
    out = capsys.readouterr().out
    assert out.count("flex attention blocks") == 1, out
    assert "1 flex attention blocks" not in out, out
    assert all(not getattr(p[0], "_unsloth_keep_unsharded", True) for _, p, _ in blocks)
