"""Per-layer CP vs no-CP hidden-state diff on a real checkpoint.

PYTHONPATH=<repo> torchrun --nproc_per_node=2 cp_layer_diff.py <model_dir> <seq_len>
  DIFF_LOADER=unsloth (default, same path as align_sft) | hf (plain transformers, unsloth never imported)
  DIFF_DTYPE=bf16 (default) | fp32
Each rank: reference forward on the full sequence (no CP), then apply_cp and forward its own
contiguous shard; compare text decoder-layer outputs on that shard.
"""
import importlib
import importlib.util
import os
import pathlib
import sys

import torch
import torch.distributed as dist

LOADER = os.environ.get("DIFF_LOADER", "unsloth")
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("DIFF_DTYPE", "bf16")]
ROOT = pathlib.Path(os.environ.get("UNSLOTH_ROOT", pathlib.Path(__file__).resolve().parents[2]))


def _parallel():
    if LOADER == "unsloth":
        return importlib.import_module("unsloth.distributed.parallel")
    # Load unsloth/distributed standalone so unsloth's global model patches never run.
    name = "_unsloth_distributed_standalone"
    if name not in sys.modules:
        root = ROOT / "unsloth" / "distributed"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations = [str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.parallel")


def _load(model_dir, seq, local):
    if LOADER == "unsloth":
        from unsloth.distributed.align_sft import load_model

        model = load_model(model_dir, seq, {"": local}, 16)
        return model.get_base_model()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype = DTYPE, attn_implementation = "sdpa", trust_remote_code = True
    ).to(torch.device("cuda", local))
    model.config.use_cache = False
    return model


def _layers(base):
    # Multimodal shells also hold vision / audio encoders with .layers; take the text decoder.
    for module in base.modules():
        if type(module).__name__.endswith("TextModel") and isinstance(getattr(module, "layers", None), torch.nn.ModuleList):
            return module.layers
    for module in base.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 1 and hasattr(layers[0], "self_attn"):
            return layers
    raise RuntimeError("no decoder layers found")


def main():
    model_dir, seq = sys.argv[1], int(sys.argv[2])
    P = _parallel()
    _device, _, cp_group = P.build_meshes(False, True)
    local = int(os.environ["LOCAL_RANK"])
    rank, cp = dist.get_rank(cp_group), dist.get_world_size(cp_group)
    base = _load(model_dir, seq, local)
    if not hasattr(base, "model") and hasattr(base, "language_model"):
        base = base.language_model
    layers = _layers(base)

    caps = {}

    def hook(idx):
        def fn(_m, _i, out):
            out = out[0] if isinstance(out, tuple) else out
            caps[idx] = out.detach().float().clone()
        return fn

    for i, layer in enumerate(layers):
        layer.register_forward_hook(hook(i))

    generator = torch.Generator()
    generator.manual_seed(3407)
    ids = torch.randint(10, 2000, (1, seq), generator = generator).to(torch.device("cuda", local))
    with torch.no_grad():
        base.model(input_ids = ids, use_cache = False)
    ref = dict(caps)
    caps.clear()

    P.apply_cp(base, cp_group)
    length = seq // cp
    start = rank * length
    pos = torch.arange(start, start + length, device = ids.device).unsqueeze(0)
    with torch.no_grad():
        base.model(input_ids = ids[:, start : start + length], position_ids = pos, use_cache = False)

    first_bad = None
    lines = []
    for i in sorted(ref):
        r = ref[i][:, start : start + length]
        c = caps[i]
        diff = (r - c).abs().max().item()
        rel = diff / (r.abs().max().item() + 1e-6)
        attn = layers[i].self_attn
        kind = getattr(attn, "layer_type", type(attn).__name__)
        shared = getattr(attn, "is_kv_shared_layer", None)
        store = getattr(attn, "store_full_length_kv", None)
        lines.append(f"rank{rank} L{i:02d} {kind:18s} shared={shared} store={store} max_diff={diff:.3e} rel={rel:.3e}")
        if first_bad is None and rel > 2e-2:
            first_bad = i
    for rk in range(cp):
        if rk == rank:
            print(f"[{LOADER} {os.environ.get('DIFF_DTYPE', 'bf16')}]", flush = True)
            print("\n".join(lines), flush = True)
            print(f"rank{rank} FIRST_BAD_LAYER={first_bad}", flush = True)
        dist.barrier()


if __name__ == "__main__":
    main()
