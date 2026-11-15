# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Version-independent loader for the released DeepSeek-V4.1 checkpoint layout.

transformers 4.57 has no ``core_model_loading`` (renames / expert merges at load), and the FP8 quantizer of 4.57 and
early 5.x rejects the release's ue8m0 32x32 blocks with fp4 experts. For those versions the released files are
converted once, streamed one layer at a time, into this implementation's own layout:

* DeepSeek key names -> module names (the same table ``register_deepseek_v41`` gives transformers 5);
* per-expert ``w1`` / ``w3`` / ``w2`` -> stacked ``gate_up_proj`` [E, 2I, H] / ``down_proj`` [E, H, I];
* fp8 e4m3 32x32-block (or 128x128) linears dequantized to bf16 exactly as inference/convert.py reads them (fp32
  product, one rounding);
* int8-packed fp4 experts (e2m1, low nibble first, one e8m0 scale per 32: inference/convert.py:28-36) kept packed.
  That is the MXFP4 layout, so they are stored as ``gate_up_blocks`` [E, 2I, H/32, 16] / ``gate_up_scales``
  [E, 2I, H/32] (uint8 biased exponents) and ``down_blocks`` / ``down_scales``, which the native experts load as
  unsloth_zoo ``Mxfp4ExpertParam`` stacks (``"unsloth_experts_format": "mxfp4"`` in config.json), a quarter of
  their bf16 size. ``UNSLOTH_DSV41_FP4_EXPERTS=bf16`` dequantizes them to bf16 instead (full finetuning);
* the engram tables kept as fp8 rows (their e8m0 scales widened to fp32, exact), for the frozen table store to bind;
* ``mtp.*``, ``vision.*``, ``aligner.*``, ``image_*`` dropped; ``quantization_config`` removed from config.json.
"""

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import PurePosixPath, PureWindowsPath

import torch

__all__ = [
    "keep_fp4_experts_packed",
    "native_key",
    "convert_released_checkpoint",
    "released_needs_conversion",
]

_RENAMES = [
    (r"^layers\.(\d+)\.engram\.embed\.weight$", r"engram_tables.\1.weight"),
    (r"^layers\.(\d+)\.engram\.embed\.scale$", r"engram_tables.\1.weight_scale_inv"),
    (r"\.attn\.", ".self_attn."),
    (r"\.ffn\.", ".mlp."),
    (r"^embed\.weight$", "embed_tokens.weight"),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.ffn_norm\.", ".post_attention_layernorm."),
    (r"\.hc_attn_fn$", ".attn_hc.fn"),
    (r"\.hc_attn_base$", ".attn_hc.base"),
    (r"\.hc_attn_scale$", ".attn_hc.scale"),
    (r"\.hc_ffn_fn$", ".ffn_hc.fn"),
    (r"\.hc_ffn_base$", ".ffn_hc.base"),
    (r"\.hc_ffn_scale$", ".ffn_hc.scale"),
    (r"\.attn_sink$", ".sinks"),
    (r"\.wq_a\.", ".q_a_proj."),
    (r"\.q_norm\.", ".q_a_norm."),
    (r"\.wq_b\.", ".q_b_proj."),
    (r"\.self_attn\.wkv\.", ".self_attn.kv_proj."),
    (r"\.compressor\.wkv\.", ".compressor.kv_proj."),
    (r"\.compressor\.wgate\.", ".compressor.gate_proj."),
    (r"\.compressor\.norm\.", ".compressor.kv_norm."),
    (r"\.indexer\.wk\.", ".indexer.k_proj."),
    (r"\.wo_a\.", ".o_a_proj."),
    (r"\.wo_b\.", ".o_b_proj."),
    (r"\.gate\.bias_vl$", ".gate.e_score_correction_bias_vl"),
    (r"\.gate\.bias$", ".gate.e_score_correction_bias"),
    (r"\.shared_experts\.w1\.", ".shared_experts.gate_proj."),
    (r"\.shared_experts\.w2\.", ".shared_experts.down_proj."),
    (r"\.shared_experts\.w3\.", ".shared_experts.up_proj."),
]
_DROPPED = re.compile(r"^(mtp\.|vision\.|aligner\.|image_(start|end|newline)$)")
_EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def native_key(name):
    """Released tensor name -> this implementation's state-dict name (with the ``model.`` / ``lm_head`` split);
    None for tensors the text model does not hold. Routed experts are handled by the converter (stacked)."""
    if _DROPPED.match(name):
        return None
    if name == "head.weight":
        return "lm_head.weight"
    for pattern, target in _RENAMES:
        name = re.sub(pattern, target, name)
    return "model." + name


def _dequant_fp8(weight, scale):
    out_dim, in_dim = weight.shape
    bo, bi = out_dim // scale.shape[0], in_dim // scale.shape[1]
    w = weight.float().view(out_dim // bo, bo, in_dim // bi, bi) * scale.float()[:, None, :, None]
    return w.view(out_dim, in_dim).to(torch.bfloat16)


def _dequant_fp4(
    packed,
    scale,
    block = 32,
):
    """inference/convert.py:28-36: two e2m1 values per byte, low nibble first, times the e8m0 scale per 32."""
    lut = torch.tensor(_FP4, dtype = torch.float32)
    u = packed.view(torch.uint8)
    x = torch.stack([lut[(u & 15).long()], lut[(u >> 4).long()]], dim = -1).flatten(-2)
    rows = x.shape[0]
    x = x.view(rows, -1, block) * scale.float().view(rows, -1, 1)
    return x.view(rows, -1).to(torch.bfloat16)


def keep_fp4_experts_packed():
    """fp4 experts stay MXFP4-packed unless UNSLOTH_DSV41_FP4_EXPERTS=bf16 (or 0) asks for a bf16 copy."""
    return os.environ.get("UNSLOTH_DSV41_FP4_EXPERTS", "mxfp4").strip().lower() not in (
        "bf16",
        "0",
        "dense",
    )


def _e8m0_bytes(scale):
    """The release's per-32 e8m0 scales as uint8 biased exponents (2^(b - 127)), the MXFP4 scale encoding."""
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None and scale.dtype == e8m0:
        return scale.view(torch.uint8)
    if scale.dtype == torch.uint8:
        return scale
    s = scale.float()
    exponent = torch.log2(s)
    if not torch.equal(exponent, exponent.round()):
        raise ValueError("Unsloth: DeepSeek-V4.1 fp4 expert scales are not powers of two")
    return (exponent.round() + 127).clamp(0, 254).to(torch.uint8)


def _pack_fp4(
    packed,
    scale,
    block = 32,
):
    """int8 [out, in / 2] + scale [out, in / 32] -> MXFP4 blocks [out, in / 32, 16] and uint8 scales [out, in / 32]."""
    u = packed.view(torch.uint8)
    rows = u.shape[0]
    return u.reshape(rows, -1, block // 2).contiguous(), _e8m0_bytes(scale).reshape(
        rows, -1
    ).contiguous()


def _transformers_major():
    try:
        import transformers
        return int(transformers.__version__.split(".")[0])
    except Exception:
        return 0


def _checked_shard(name):
    """An index shard name must be a plain relative path inside the checkpoint directory (lexical check, so the
    symlinked blobs of a Hub snapshot keep working)."""
    bad = not isinstance(name, str) or not name
    if not bad:
        for pure in (PurePosixPath(name), PureWindowsPath(name)):
            if pure.is_absolute() or pure.drive or pure.anchor or ".." in pure.parts:
                bad = True
                break
    if bad:
        raise ValueError(
            f"Unsloth: DeepSeek-V4.1 checkpoint index names an unsafe shard path {name!r}"
        )
    return name


def _read_index(src):
    index = os.path.join(src, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding = "utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        for shard in weight_map.values():
            _checked_shard(shard)
        return weight_map
    from safetensors import safe_open

    single = os.path.join(src, "model.safetensors")
    with safe_open(single, "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def released_needs_conversion():
    """True when transformers itself cannot load the released layout (no core_model_loading, or an FP8 quantizer
    without fp4 experts / ue8m0 scales). UNSLOTH_DSV41_CONVERT=1 / 0 forces it."""
    forced = os.environ.get("UNSLOTH_DSV41_CONVERT")
    if forced in ("0", "1"):
        return forced == "1"
    try:
        import transformers.core_model_loading as core
        if not hasattr(core, "WeightConverter"):
            return True
    except Exception:
        return True
    try:
        import inspect
        from transformers import FineGrainedFP8Config

        params = inspect.signature(FineGrainedFP8Config.__init__).parameters
        return not ("expert_dtype" in params and "scale_fmt" in params)
    except Exception:
        return True


def _source_dir(
    path,
    subfolder = None,
    **hub_kwargs,
):
    """<local dir or Hub snapshot>/<subfolder> holding the released files."""
    sub = str(subfolder or "").replace("\\", "/").strip("/")
    if sub:
        _checked_shard(sub)
    if os.path.isdir(str(path)):
        base = str(path)
    else:
        from huggingface_hub import snapshot_download

        allowed = {
            k: v
            for k, v in hub_kwargs.items()
            if k in ("cache_dir", "token", "revision", "local_files_only")
        }
        patterns = ["*.safetensors", "*.json", "tokenizer*", "*.jinja"]
        if sub:
            patterns = [sub + "/" + p for p in patterns]
        base = snapshot_download(str(path), allow_patterns = patterns, **allowed)
    return os.path.join(base, *sub.split("/")) if sub else base


def _target_dir(src):
    root = os.environ.get("UNSLOTH_DSV41_CONVERTED_DIR")
    if not root:
        home = os.environ.get("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface"
        )
        root = os.path.join(home, "unsloth_converted", "deepseek_v41")
    stamp = []
    for f in sorted(os.listdir(src)):
        if f.endswith((".safetensors", ".json")):
            st = os.stat(os.path.join(src, f))
            stamp.append(f"{f}:{st.st_size}:{st.st_mtime_ns}:{st.st_ino}")
    mode = "fp4packed" if keep_fp4_experts_packed() else "bf16"
    digest = hashlib.sha256(
        (os.path.realpath(src) + "|" + "|".join(stamp) + "|" + mode).encode()
    ).hexdigest()[:16]
    return os.path.join(root, digest)


@contextlib.contextmanager
def _dst_lock(dst):
    """Inter-process lock guarding the completion check, conversion and publication of ``dst``."""
    lock_path = dst + ".lock"
    try:
        from filelock import FileLock
    except Exception:
        FileLock = None
    if FileLock is not None:
        with FileLock(lock_path):
            yield
        return
    try:
        import fcntl
    except Exception:
        fcntl = None
    if fcntl is None:
        yield
        return
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _drop_lock_file(dst):
    """Remove ``dst``.lock once ``dst`` is published. Safe while held: every later critical section, even one that
    still waits on the unlinked inode, only re-reads the completion marker, which is never removed again."""
    try:
        os.unlink(dst + ".lock")
    except OSError:
        pass


def convert_released_checkpoint(path, **hub_kwargs):
    """Convert (once, cached) and return the directory of the native-layout checkpoint."""
    src = _source_dir(path, **hub_kwargs)
    dst = _target_dir(src)
    if _transformers_major() < 5:
        print(
            "Unsloth: recent transformers 5.x releases load the release directly (fp8 linears and fp4 experts stay "
            "quantized, no converted copy on disk). `pip install -U transformers` is recommended for DeepSeek-V4.1."
        )
    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok = True)
    done = os.path.join(dst, ".unsloth_converted")
    with _dst_lock(dst):
        if os.path.isfile(done) and not os.path.islink(dst):
            _drop_lock_file(dst)
            return dst
        # A private staging dir per call (never a predictable path another user could pre-plant), published with
        # one atomic rename while the lock is held.
        tmp = tempfile.mkdtemp(prefix = os.path.basename(dst) + ".partial-", dir = parent)
        published = False
        try:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(
                tmp, 0o777 & ~umask
            )  # published with the usual permissions, not mkdtemp's 0700
            _convert_into(src, tmp, dst)
            if os.path.islink(dst):
                os.unlink(dst)
            elif os.path.isdir(dst):
                shutil.rmtree(dst)
            os.replace(tmp, dst)
            published = True
            _drop_lock_file(dst)
        finally:
            if not published:
                shutil.rmtree(tmp, ignore_errors = True)
    return dst


def _convert_into(src, tmp, dst):
    from safetensors import safe_open
    from safetensors.torch import save_file

    weight_map = _read_index(src)
    handles = {}

    def get(name):
        f = weight_map[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(src, _checked_shard(f)), "pt")
        return handles[f].get_tensor(name)

    pack_fp4 = keep_fp4_experts_packed()
    has_fp4 = any(
        _EXPERT.match(n) and n.endswith(".scale") and n[: -len("scale")] + "weight" in weight_map
        for n in weight_map
    )
    print(
        "Unsloth: converting the released DeepSeek-V4.1 checkpoint once into Unsloth's layout"
        + (
            "; fp4 experts stay MXFP4-packed, fp8 linears become bf16"
            if pack_fp4 and has_fp4
            else "; fp4 / fp8 weights become bf16"
        )
        + f" ({dst})."
    )
    packed_experts = False
    groups = {}
    for name in weight_map:
        m = re.match(r"^layers\.(\d+)\.", name)
        groups.setdefault(int(m.group(1)) if m else -1, []).append(name)
    new_map = {}
    try:
        for gi, (group, names) in enumerate(sorted(groups.items())):
            out, experts = {}, {}
            for name in names:
                if name.endswith(".scale") and ".engram.embed." not in name:
                    continue  # consumed with its weight
                em = _EXPERT.match(name)
                if em:
                    if em.group(4) == "scale":
                        continue
                    layer, e, which = int(em.group(1)), int(em.group(2)), em.group(3)
                    w = get(name)
                    scale_name = name[: -len("weight")] + "scale"
                    if w.dtype == torch.int8 and scale_name in weight_map and pack_fp4:
                        experts.setdefault(layer, {})[(e, which)] = _pack_fp4(w, get(scale_name))
                        continue
                    if w.dtype == torch.int8 and scale_name in weight_map:
                        w = _dequant_fp4(w, get(scale_name))
                    elif w.dtype == torch.float8_e4m3fn and scale_name in weight_map:
                        w = _dequant_fp8(w, get(scale_name))
                    experts.setdefault(layer, {})[(e, which)] = w.to(torch.bfloat16)
                    continue
                key = native_key(name)
                if key is None:
                    continue
                t = get(name)
                scale_name = name[: -len("weight")] + "scale" if name.endswith(".weight") else None
                if (
                    t.dtype == torch.float8_e4m3fn
                    and scale_name in weight_map
                    and ".engram.embed." not in name
                ):
                    t = _dequant_fp8(t, get(scale_name))
                elif t.dtype == getattr(torch, "float8_e8m0fnu", None):
                    # engram scales: transformers 4.57's safetensors reader has no F8_E8M0; powers of two are exact
                    # in fp32
                    t = t.float()
                out[key] = t.contiguous()
            for layer, parts in experts.items():
                n = 1 + max(e for e, _ in parts)
                if isinstance(parts[(0, "w1")], tuple):
                    # MXFP4 stacks: rows [w1; w3] per expert, like the bf16 gate_up_proj
                    packed_experts = True
                    prefix = f"model.layers.{layer}.mlp.experts."
                    for kind, idx in (("blocks", 0), ("scales", 1)):
                        out[prefix + f"gate_up_{kind}"] = torch.stack(
                            [
                                torch.cat([parts[(e, "w1")][idx], parts[(e, "w3")][idx]], 0)
                                for e in range(n)
                            ]
                        )
                        out[prefix + f"down_{kind}"] = torch.stack(
                            [parts[(e, "w2")][idx] for e in range(n)]
                        )
                    continue
                gate_up = torch.stack(
                    [torch.cat([parts[(e, "w1")], parts[(e, "w3")]], 0) for e in range(n)]
                )
                down = torch.stack([parts[(e, "w2")] for e in range(n)])
                out[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = gate_up
                out[f"model.layers.{layer}.mlp.experts.down_proj"] = down
            if out:
                fname = f"model-{gi:05d}.safetensors"
                save_file(out, os.path.join(tmp, fname), metadata = {"format": "pt"})
                new_map.update({k: fname for k in out})
    finally:
        handles.clear()
    with open(os.path.join(tmp, "model.safetensors.index.json"), "w", encoding = "utf-8") as f:
        json.dump({"metadata": {}, "weight_map": new_map}, f)
    with open(os.path.join(src, "config.json"), encoding = "utf-8") as f:
        config = json.load(f)
    config.pop("quantization_config", None)
    if packed_experts:
        config["unsloth_experts_format"] = "mxfp4"
        if isinstance(config.get("text_config"), dict):
            config["text_config"]["unsloth_experts_format"] = "mxfp4"
    with open(os.path.join(tmp, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 1)
    for f in os.listdir(src):
        if (
            f.startswith("tokenizer")
            or f.endswith(".jinja")
            or f in ("generation_config.json", "special_tokens_map.json")
        ):
            shutil.copy(os.path.join(src, f), os.path.join(tmp, f))
    open(os.path.join(tmp, ".unsloth_converted"), "w", encoding = "utf-8").close()
