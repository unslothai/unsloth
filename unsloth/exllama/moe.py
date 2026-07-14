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

"""Mixture-of-Experts support: reconstruct fused transformers-5 experts for EXL3."""

from __future__ import annotations

import os
from typing import Optional

import torch

from .utils import is_exllama_available, require_exllama


# Fused-expert attribute -> alternative per-expert sub-tensor names, by arch
# (Mixtral: w1/w3/w2; Qwen3: gate/up/down). gate_up_proj = concat(gate, up).
_FUSED_EXPERT_LAYOUTS = {
    "gate_up_proj": (("w1", "gate_proj"), ("w3", "up_proj")),
    "down_proj": (("w2", "down_proj"),),
}


def _reconstruct_exl3_weight(stc, key: str, device: torch.device) -> Optional[torch.Tensor]:
    """Reconstruct a dense ``[out, in]`` fp16 weight for one EXL3 tensor group.

    ``key`` is the module prefix, e.g.
    ``model.layers.0.block_sparse_moe.experts.0.w1``. Returns None if the group
    is absent.
    """
    from exllamav3.modules.quant.exl3 import LinearEXL3

    if not stc.has_tensor_group(key, [["sv", "svh"], ["su", "suh"], "trellis"]):
        return None

    tensors = stc.get_tensors(key, device = device)
    trellis = tensors.get(f"{key}.trellis")
    suh = tensors.get(f"{key}.suh")
    svh = tensors.get(f"{key}.svh")
    su = tensors.get(f"{key}.su")
    sv = tensors.get(f"{key}.sv")
    mcg = tensors.get(f"{key}.mcg")
    mul1 = tensors.get(f"{key}.mul1")

    if suh is not None:
        suh = suh.half()
    if svh is not None:
        svh = svh.half()

    in_features = suh.shape[0] if suh is not None else None
    out_features = svh.shape[0] if svh is not None else None

    linear = LinearEXL3(
        config = None,
        in_features = in_features,
        out_features = out_features,
        trellis = trellis,
        suh = suh,
        svh = svh,
        su = su,
        sv = sv,
        mcg = mcg,
        mul1 = mul1,
        bias = None,
        out_dtype = torch.float16,
        transformers_fix = True,
    )
    # get_weight_tensor() returns [in, out]; transpose to the [out, in]
    # convention used by fused expert parameters and nn.functional.linear.
    w = linear.get_weight_tensor().t().contiguous()
    return w


def _build_exl3_linear(stc, key: str, device: torch.device):
    """Build (and return) a quantized ``LinearEXL3`` for one tensor group.

    Unlike :func:`_reconstruct_exl3_weight` this keeps the weight *quantized*
    (the trellis stays in VRAM at the target bitrate); the dense weight is only
    materialized transiently when the layer is used. Returns ``(linear, out, in)``
    or ``None``.
    """
    from exllamav3.modules.quant.exl3 import LinearEXL3

    if not stc.has_tensor_group(key, [["sv", "svh"], ["su", "suh"], "trellis"]):
        return None
    tensors = stc.get_tensors(key, device = device)
    trellis = tensors.get(f"{key}.trellis")
    suh = tensors.get(f"{key}.suh")
    svh = tensors.get(f"{key}.svh")
    su = tensors.get(f"{key}.su")
    sv = tensors.get(f"{key}.sv")
    mcg = tensors.get(f"{key}.mcg")
    mul1 = tensors.get(f"{key}.mul1")
    if suh is not None:
        suh = suh.half()
    if svh is not None:
        svh = svh.half()
    in_features = suh.shape[0] if suh is not None else None
    out_features = svh.shape[0] if svh is not None else None
    linear = LinearEXL3(
        config = None,
        in_features = in_features,
        out_features = out_features,
        trellis = trellis,
        suh = suh,
        svh = svh,
        su = su,
        sv = sv,
        mcg = mcg,
        mul1 = mul1,
        bias = None,
        out_dtype = torch.float16,
        transformers_fix = True,
    )
    return linear, out_features, in_features


def _find_experts_modules(model):
    """Yield (module, prefix) for every fused experts module in the model.

    ``prefix`` is the dotted path used in the checkpoint's tensor keys, e.g.
    ``model.layers.0.block_sparse_moe.experts``. transformers-5 module names may
    differ from checkpoint keys (``mlp.experts`` vs ``block_sparse_moe.experts``),
    so we return both the module and its module-path and resolve the checkpoint
    prefix separately.
    """
    for name, module in model.named_modules():
        has_fused = any(
            isinstance(getattr(module, attr, None), torch.nn.Parameter)
            and getattr(module, attr).dim() == 3
            for attr in _FUSED_EXPERT_LAYOUTS
        )
        if has_fused:
            yield module, name


def _candidate_checkpoint_prefixes(module_path: str):
    """Map a transformers module path to likely checkpoint tensor prefixes.

    e.g. ``model.layers.0.mlp.experts`` -> also try
    ``model.layers.0.block_sparse_moe.experts`` (Mixtral's stored name).
    """
    prefixes = [module_path]
    if ".mlp.experts" in module_path:
        prefixes.append(module_path.replace(".mlp.experts", ".block_sparse_moe.experts"))
    if ".block_sparse_moe.experts" in module_path:
        prefixes.append(module_path.replace(".block_sparse_moe.experts", ".mlp.experts"))
    # Qwen3.5 (and other multimodal-derived) checkpoints store the decoder under
    # a ``model.language_model.`` prefix even though the loaded HF module path is
    # ``model.layers...``. Add both directions of that mapping.
    extra = []
    for p in list(prefixes):
        if p.startswith("model.") and not p.startswith("model.language_model."):
            extra.append("model.language_model." + p[len("model.") :])
        if p.startswith("model.language_model."):
            extra.append("model." + p[len("model.language_model.") :])
            extra.append("language_model.model." + p[len("model.language_model.") :])
    prefixes.extend(extra)
    # De-duplicate while preserving order.
    seen = set()
    uniq = []
    for p in prefixes:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


@torch.no_grad()
def reload_exl3_experts(
    model,
    checkpoint_dir: str,
    compute_dtype: torch.dtype = torch.float16,
) -> int:
    """Reconstruct fused MoE expert weights from a checkpoint's EXL3 tensors.

    Returns the number of expert *matrices* reconstructed (across all experts
    and all fused attributes). A return of 0 means no fused experts were found
    (e.g. a dense model) - this is a safe no-op.
    """
    if not is_exllama_available():
        return 0
    if not os.path.isdir(checkpoint_dir):
        return 0

    require_exllama()
    from exllamav3.loader import SafetensorsCollection

    experts_modules = list(_find_experts_modules(model))
    if not experts_modules:
        return 0

    stc = SafetensorsCollection(checkpoint_dir)
    count = 0
    try:
        for module, module_path in experts_modules:
            num_experts = int(getattr(module, "num_experts", 0)) or _infer_num_experts(module)
            if num_experts == 0:
                continue
            prefixes = _candidate_checkpoint_prefixes(module_path)

            for attr, subkeys in _FUSED_EXPERT_LAYOUTS.items():
                param = getattr(module, attr, None)
                if not (isinstance(param, torch.nn.Parameter) and param.dim() == 3):
                    continue
                device = param.device if param.device.type != "meta" else _model_device(model)

                stacked_experts = []
                ok = True
                for e in range(num_experts):
                    per_subkey = []
                    for sub in subkeys:
                        w = _resolve_expert_weight(stc, prefixes, e, sub, device)
                        if w is None:
                            ok = False
                            break
                        per_subkey.append(w)
                    if not ok:
                        break
                    # Concatenate sub-weights along the output dim (e.g. gate|up).
                    if len(per_subkey) == 1:
                        w_e = per_subkey[0]
                    else:
                        w_e = torch.cat(per_subkey, dim = 0)
                    stacked_experts.append(w_e)

                if not ok or len(stacked_experts) != num_experts:
                    continue

                fused = torch.stack(stacked_experts, dim = 0).to(device = device, dtype = compute_dtype)
                # The fused parameter is frozen (expert base weights are not
                # trained; LoRA runs on attention / router / other layers).
                new_param = torch.nn.Parameter(fused, requires_grad = False)
                setattr(module, attr, new_param)
                count += num_experts
    finally:
        stc.close()

    # transformers' `grouped` MoE kernel needs Hopper (CC >= 9.0); force the
    # portable `eager` forward so EXL3 MoEs run on consumer GPUs.
    if count > 0:
        _force_eager_experts(model)

    return count


def _force_eager_experts(model) -> None:
    """Set every experts module's config to the portable ``eager`` forward."""
    config = getattr(model, "config", None)
    for cfg in _iter_configs(config):
        try:
            cfg._experts_implementation = "eager"
        except Exception:
            pass
    # Stamp each experts module's own config too (dispatch reads it from there).
    for module in model.modules():
        mcfg = getattr(module, "config", None)
        if mcfg is not None and hasattr(type(module), "forward"):
            try:
                mcfg._experts_implementation = "eager"
            except Exception:
                pass


def _iter_configs(config):
    """Yield the top-level config and any nested sub-configs."""
    if config is None:
        return
    yield config
    for attr in ("text_config", "get_text_config"):
        sub = getattr(config, attr, None)
        if callable(sub):
            try:
                sub = sub()
            except Exception:
                sub = None
        if sub is not None and sub is not config:
            yield sub


def _resolve_expert_weight(stc, prefixes, expert_idx: int, subkey, device):
    """Reconstruct one expert sub-weight.

    ``subkey`` may be a single name or a tuple of alternative names (e.g.
    ``("w1", "gate_proj")`` to cover Mixtral vs Qwen naming). Each candidate
    prefix x each alternative name is tried.
    """
    alternatives = (subkey,) if isinstance(subkey, str) else tuple(subkey)
    for prefix in prefixes:
        for name in alternatives:
            key = f"{prefix}.{expert_idx}.{name}"
            w = _reconstruct_exl3_weight(stc, key, device)
            if w is not None:
                return w
    return None


def _infer_num_experts(module) -> int:
    for attr in _FUSED_EXPERT_LAYOUTS:
        param = getattr(module, attr, None)
        if isinstance(param, torch.nn.Parameter) and param.dim() == 3:
            return int(param.shape[0])
    return 0


def _model_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _experts_have_bias(module) -> bool:
    """True if a fused-experts module carries per-expert bias (e.g. gpt_oss)."""
    for attr in (
        "gate_up_proj_bias",
        "down_proj_bias",
        "gate_proj_bias",
        "up_proj_bias",
        "w1_bias",
        "w2_bias",
        "w3_bias",
    ):
        if getattr(module, attr, None) is not None:
            return True
    return False


# Keeps each expert quantized and reconstructs only the routed top-k on the fly,
# so a large MoE stays at its quantized footprint instead of OOMing on dense
# experts.
class Exl3QuantizedExperts(torch.nn.Module):
    """EXL3-quantized drop-in for a fused MoE experts module."""

    def __init__(self, num_experts, hidden_dim, intermediate_dim, act_fn):
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.act_fn = act_fn
        # Per-expert quantized linears, stored OUTSIDE the nn.Module tree (so
        # they neither appear in state_dict nor get moved/cast by accelerate).
        object.__setattr__(self, "_gate_up", [None] * self.num_experts)
        object.__setattr__(self, "_down", [None] * self.num_experts)
        # LRU cache of reconstructed dense experts (frozen base, so always
        # valid); bounds VRAM while skipping re-reconstruction of hot experts.
        import os as _os

        cap = int(_os.environ.get("UNSLOTH_EXL3_EXPERT_CACHE", "64"))
        object.__setattr__(self, "_cache_cap", max(0, cap))
        object.__setattr__(self, "_gu_cache", {})  # expert idx -> dense gate_up
        object.__setattr__(self, "_dn_cache", {})  # expert idx -> dense down
        object.__setattr__(self, "_lru", [])  # recency order of idxs

    def set_expert(self, idx, gate_up_linear, down_linear):
        self._gate_up[idx] = gate_up_linear
        self._down[idx] = down_linear

    @staticmethod
    @torch.no_grad()
    def _dense(
        linear,
        dtype,
        device = None,
    ):
        # get_weight_tensor() -> [in, out]; transpose to [out, in] for F.linear.
        # Cast to the activation's dtype and device (multi-GPU device_map).
        w = linear.get_weight_tensor().t().contiguous().to(dtype)
        if device is not None and w.device != device:
            w = w.to(device)
        return w

    def _get_dense(
        self,
        e,
        dtype,
        device = None,
    ):
        """Return (gate_up_dense, down_dense) for expert e, using the LRU cache."""
        if self._cache_cap <= 0:
            return (
                self._dense(self._gate_up[e], dtype, device),
                self._dense(self._down[e], dtype, device),
            )
        if e in self._gu_cache:
            # Refresh recency.
            try:
                self._lru.remove(e)
            except ValueError:
                pass
            self._lru.append(e)
            return self._gu_cache[e], self._dn_cache[e]
        gu = self._dense(self._gate_up[e], dtype, device)
        dn = self._dense(self._down[e], dtype, device)
        self._gu_cache[e] = gu
        self._dn_cache[e] = dn
        self._lru.append(e)
        # Evict least-recently-used beyond the cap.
        while len(self._lru) > self._cache_cap:
            old = self._lru.pop(0)
            self._gu_cache.pop(old, None)
            self._dn_cache.pop(old, None)
        return gu, dn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes = self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim = (-1, -2)), 0).nonzero()
        dtype = hidden_states.dtype
        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            if e >= self.num_experts or self._gate_up[e] is None:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current = hidden_states[token_idx]
            gu_w, down_w = self._get_dense(e, dtype, current.device)  # cached dense weights
            gate, up = torch.nn.functional.linear(current, gu_w).chunk(2, dim = -1)
            h = (self.act_fn(gate) if self.act_fn is not None else gate) * up
            out = torch.nn.functional.linear(h, down_w)
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final


@torch.no_grad()
def reload_exl3_experts_quantized(
    model,
    checkpoint_dir,
    compute_dtype = torch.bfloat16,
) -> int:
    """Replace fused MoE experts with :class:`Exl3QuantizedExperts` (quantized).

    Memory-efficient alternative to :func:`reload_exl3_experts`: keeps experts at
    their EXL3 (e.g. 2-bit) footprint instead of reconstructing them dense.
    Returns the number of expert *matrices* loaded. No-op for dense models.
    """
    if not is_exllama_available() or not os.path.isdir(checkpoint_dir):
        return 0
    require_exllama()
    from exllamav3.loader import SafetensorsCollection

    experts_modules = list(_find_experts_modules(model))
    if not experts_modules:
        return 0

    # gate_up alternatives: build fused [gate|up] from separate w1/w3 or
    # gate_proj/up_proj; down from w2/down_proj.
    gate_alts = ("w1", "gate_proj")
    up_alts = ("w3", "up_proj")
    down_alts = ("w2", "down_proj")

    stc = SafetensorsCollection(checkpoint_dir)
    count = 0
    device = _model_device(model)
    try:
        for module, module_path in experts_modules:
            num_experts = int(getattr(module, "num_experts", 0)) or _infer_num_experts(module)
            hidden = int(getattr(module, "hidden_dim", 0)) or _infer_hidden(module)
            inter = int(getattr(module, "intermediate_dim", 0)) or _infer_inter(module)
            act_fn = getattr(module, "act_fn", None)
            prefixes = _candidate_checkpoint_prefixes(module_path)
            # Skip bias-bearing experts (the quantized path can't represent bias).
            if _experts_have_bias(module):
                continue

            new = Exl3QuantizedExperts(num_experts, hidden, inter, act_fn)
            ok_all = True
            for e in range(num_experts):
                gate = _resolve_quant_linear(stc, prefixes, e, gate_alts, device)
                up = _resolve_quant_linear(stc, prefixes, e, up_alts, device)
                down = _resolve_quant_linear(stc, prefixes, e, down_alts, device)
                if gate is None or up is None or down is None:
                    ok_all = False
                    break
                gate_up = _FusedGateUp(gate, up)
                new.set_expert(e, gate_up, down)
            if not ok_all:
                continue
            # Swap the fused module in-place (find parent by walking).
            _replace_module(model, module, new)
            count += num_experts * 2
    finally:
        stc.close()

    if count > 0:
        _force_eager_experts(model)
    return count


class _FusedGateUp:
    """Presents concat(gate, up) along the output dim as one 'linear' with a
    ``get_weight_tensor()`` returning the fused [in, 2*inter] tensor (so the
    reconstruct+transpose in Exl3QuantizedExperts yields [2*inter, in])."""

    def __init__(self, gate_linear, up_linear):
        self.gate = gate_linear
        self.up = up_linear

    def get_weight_tensor(self):
        # each get_weight_tensor() -> [in, out]; cat along out (dim=1).
        return torch.cat([self.gate.get_weight_tensor(), self.up.get_weight_tensor()], dim = 1)


def _resolve_quant_linear(stc, prefixes, expert_idx, alts, device):
    for prefix in prefixes:
        for name in alts:
            key = f"{prefix}.{expert_idx}.{name}"
            built = _build_exl3_linear(stc, key, device)
            if built is not None:
                return built[0]
    return None


def _infer_hidden(module):
    for attr in ("gate_up_proj", "down_proj"):
        p = getattr(module, attr, None)
        if isinstance(p, torch.nn.Parameter) and p.dim() == 3:
            return p.shape[2] if attr == "gate_up_proj" else p.shape[1]
    return 0


def _infer_inter(module):
    p = getattr(module, "down_proj", None)
    if isinstance(p, torch.nn.Parameter) and p.dim() == 3:
        return p.shape[2]
    p = getattr(module, "gate_up_proj", None)
    if isinstance(p, torch.nn.Parameter) and p.dim() == 3:
        return p.shape[1] // 2
    return 0


def _replace_module(model, target, new_module):
    for parent in model.modules():
        for attr_name, child in list(parent.named_children()):
            if child is target:
                setattr(parent, attr_name, new_module)
                return True
    return False


@torch.no_grad()
def install_quantized_experts_before_load(model, checkpoint_dir) -> int:
    """Swap fused MoE experts for quantized experts BEFORE transformers loads
    weights, so the dense expert params are never allocated/loaded on GPU.

    Called from the EXL3 quantizer's ``_process_model_before_weight_loading``
    (see patcher). At this point the fused ``gate_up_proj``/``down_proj`` are
    still meta/uninitialised, so replacing the whole experts module is cheap.
    The per-expert trellis is read from the checkpoint here and lives on GPU at
    its 2-bit footprint. Returns the number of expert matrices installed.
    """
    if not is_exllama_available() or not os.path.isdir(checkpoint_dir):
        return 0
    require_exllama()
    from exllamav3.loader import SafetensorsCollection

    experts_modules = list(_find_experts_modules(model))
    if not experts_modules:
        return 0

    # Only take over LARGE MoEs here; small ones use the (simpler, exact) dense
    # post-load path unless the user forces quantized experts.
    mode = os.environ.get("UNSLOTH_EXL3_QUANTIZED_EXPERTS", "auto").strip().lower()
    max_experts = max(
        (int(getattr(m, "num_experts", 0)) or _infer_num_experts(m)) for m, _ in experts_modules
    )
    if mode in ("0", "false", "no", "dense"):
        return 0
    if mode in ("auto", "") and max_experts <= 32:
        return 0

    gate_alts, up_alts, down_alts = ("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stc = SafetensorsCollection(checkpoint_dir)
    count = 0
    try:
        for module, module_path in experts_modules:
            num_experts = int(getattr(module, "num_experts", 0)) or _infer_num_experts(module)
            hidden = int(getattr(module, "hidden_dim", 0)) or _infer_hidden(module)
            inter = int(getattr(module, "intermediate_dim", 0)) or _infer_inter(module)
            act_fn = getattr(module, "act_fn", None)
            prefixes = _candidate_checkpoint_prefixes(module_path)
            # Skip bias-bearing experts (the quantized path can't represent bias).
            if _experts_have_bias(module):
                continue
            new = Exl3QuantizedExperts(num_experts, hidden, inter, act_fn)
            ok_all = True
            for e in range(num_experts):
                gate = _resolve_quant_linear(stc, prefixes, e, gate_alts, device)
                up = _resolve_quant_linear(stc, prefixes, e, up_alts, device)
                down = _resolve_quant_linear(stc, prefixes, e, down_alts, device)
                if gate is None or up is None or down is None:
                    ok_all = False
                    break
                new.set_expert(e, _FusedGateUp(gate, up), down)
            if not ok_all:
                continue
            _replace_module(model, module, new)
            count += num_experts * 2
    finally:
        stc.close()
    if count:
        _force_eager_experts(model)
        print(
            f"Unsloth: installed {count} EXL3-quantized MoE expert matrices "
            f"pre-load (reconstruct-on-forward) - dense expert params never "
            f"allocated, so the large MoE fits in VRAM.",
            flush = True,
        )
    return count
