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
"""Standalone llm-compressor runner for Unsloth's FP8/FP4 export.

Launched as a subprocess by file path (not `python -m`) so the Unsloth package, which patches
transformers attention, is not imported here; llm-compressor needs an unpatched forward for
calibration (e.g. NVFP4). Reads a merged 16bit checkpoint, writes a compressed-tensors one.
"""

import argparse
import glob
import json
import os
import sys


def _is_moe(config):
    """True if the model config looks like a sparse Mixture-of-Experts model."""
    if config is None:
        return False
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        for attr in (
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "moe_num_experts",
        ):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 1:
                return True
    return "moe" in (getattr(config, "model_type", "") or "").lower()


def _has_mtp(config):
    """True if the model carries MTP / speculative-decoding layers (e.g. Qwen3-Next, DeepSeek)."""
    if config is None:
        return False
    mt = (getattr(config, "model_type", "") or "").lower()
    if "qwen3_next" in mt or "mtp" in mt:
        return True
    for attr in ("num_nextn_predict_layers", "num_mtp_layers", "mtp_num_layers"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 0:
            return True
    return False


def _build_calibration_dataset(tokenizer, kind, value, num_samples, max_seq_length):
    from datasets import DatasetDict, load_dataset, load_from_disk

    _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    if kind == "none":
        print(
            f"Unsloth: NVFP4 needs calibration data. Defaulting to {num_samples} samples of "
            "HuggingFaceH4/ultrachat_200k. For best accuracy pass your own training data via "
            "`calibration_dataset=...`.",
            flush = True,
        )
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k", split = f"train_sft[:{num_samples}]"
        )
        ds = ds.shuffle(seed = 42)
    elif kind == "hfid":
        # Not every dataset has a "train" split (e.g. train_sft only); fall back to the first one.
        try:
            ds = load_dataset(value, split = f"train[:{num_samples}]")
        except (ValueError, KeyError):
            from datasets import get_dataset_split_names
            try:
                # Resolve the first split name so only num_samples rows are fetched, instead of
                # downloading/materializing the whole dataset just to take a small slice.
                split = get_dataset_split_names(value)[0]
                ds = load_dataset(value, split = f"{split}[:{num_samples}]")
            except Exception:
                # Last resort: materialize, then subselect (preserves the original behavior).
                ds = load_dataset(value)
                if isinstance(ds, DatasetDict):
                    ds = ds[next(iter(ds.keys()))]
                if num_samples and len(ds) > num_samples:
                    ds = ds.select(range(num_samples))
        ds = ds.shuffle(seed = 42)
    elif kind == "disk":
        ds = load_from_disk(value)
        if isinstance(ds, DatasetDict):
            if "train" in ds:
                ds = ds["train"]
            elif len(ds) == 1:
                ds = next(iter(ds.values()))
            else:
                raise RuntimeError(
                    "Unsloth: disk calibration_dataset is a DatasetDict with multiple splits; "
                    "pass a single split, e.g. calibration_dataset=dataset['train']."
                )
        if num_samples and len(ds) > num_samples:
            ds = ds.shuffle(seed = 42).select(range(num_samples))
    else:
        raise ValueError(f"Unknown calibration-dataset-kind: {kind}")

    try:
        if len(ds) == 0:
            raise RuntimeError(
                "Unsloth: the calibration dataset is empty after loading/subsampling; "
                "pass a non-empty calibration_dataset."
            )
    except TypeError:
        pass  # streaming / iterable datasets have no len(); let llm-compressor handle them

    cols = set(ds.column_names)
    if "input_ids" in cols:
        # Drop non-model-input columns (e.g. a leftover 'messages' list) so llm-compressor's
        # collator does not try to batch them.
        keep = {"input_ids", "attention_mask", "labels", "position_ids"}
        extra = [c for c in ds.column_names if c not in keep]
        if extra:
            ds = ds.remove_columns(extra)
        return ds
    if "messages" in cols:
        # Base / non-chat tokenizers have no chat template; concatenate message contents instead
        # of calling apply_chat_template (which would raise).
        has_chat_template = bool(getattr(_tok, "chat_template", None))

        def _content_to_text(content):
            # content may be a str, None, or a multimodal list of parts (str or {"text": ...}).
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, (list, tuple)):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        text = part.get("text") or part.get("content")
                        if isinstance(text, str):
                            parts.append(text)
                return " ".join(parts)
            return str(content)

        def _prep(ex):
            msgs = ex["messages"] or []
            if has_chat_template:
                return {"text": _tok.apply_chat_template(msgs, tokenize = False)}
            return {"text": "\n".join(_content_to_text(m.get("content")) for m in msgs)}

        ds = ds.map(_prep)
    elif "text" not in cols:
        raise RuntimeError(
            "Unsloth: calibration_dataset must contain a 'messages', 'text', or 'input_ids' "
            f"column (got: {sorted(cols)})."
        )

    def _tokenize(sample):
        return _tok(
            sample["text"],
            padding = False,
            max_length = max_seq_length,
            truncation = True,
            add_special_tokens = False,
        )

    return ds.map(_tokenize, remove_columns = ds.column_names)


def _from_pretrained(auto_model, model_path, trust_remote_code):
    import torch

    # transformers renamed torch_dtype -> dtype; support both.
    try:
        return auto_model.from_pretrained(
            model_path,
            device_map = "auto",
            low_cpu_mem_usage = True,
            trust_remote_code = trust_remote_code,
            dtype = torch.bfloat16,
        )
    except TypeError:
        return auto_model.from_pretrained(
            model_path,
            device_map = "auto",
            low_cpu_mem_usage = True,
            trust_remote_code = trust_remote_code,
            torch_dtype = torch.bfloat16,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True, help = "merged 16bit HF checkpoint dir")
    ap.add_argument("--scheme", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--needs-calibration", action = "store_true")
    ap.add_argument(
        "--calibration-dataset-kind", default = "none", choices = ["none", "hfid", "disk"]
    )
    ap.add_argument("--calibration-dataset", default = "")
    ap.add_argument("--num-calibration-samples", type = int, default = 512)
    ap.add_argument("--max-seq-length", type = int, default = 2048)
    ap.add_argument("--is-vlm", action = "store_true")
    ap.add_argument("--trust-remote-code", action = "store_true")
    ap.add_argument("--trust-remote-code-tokenizer", action = "store_true")
    ap.add_argument(
        "--variant", default = "", help = "weight-filename variant for the output shards"
    )
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # Import the VLM auto-class only when needed - some transformers versions lack it, and the
    # text path must not fail just because that newer class is unavailable.
    if args.is_vlm:
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _VLMModel
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as _VLMModel
            except ImportError as e:
                raise RuntimeError(
                    "Unsloth: this transformers version has no VLM auto-model class for "
                    "compressed multimodal export. Please upgrade transformers."
                ) from e
        auto_model, auto_proc = _VLMModel, AutoProcessor
    else:
        auto_model, auto_proc = AutoModelForCausalLM, AutoTokenizer

    model = _from_pretrained(auto_model, args.model, args.trust_remote_code)
    model.eval()
    # A tokenizer may be absent if the caller saved it separately; only calibration needs one.
    try:
        # The tokenizer/processor has its own trust flag: consent for one component must not
        # let the other's custom code run.
        tokenizer = auto_proc.from_pretrained(
            args.model, trust_remote_code = args.trust_remote_code_tokenizer
        )
    except Exception:
        if args.needs_calibration:
            raise RuntimeError(
                f"Unsloth: calibration export needs a tokenizer but none was found in {args.model}. "
                "Pass tokenizer=... to save_pretrained_merged."
            )
        tokenizer = None

    # MoE models: keep the router/gate unquantized (it decides expert routing) and calibrate every
    # expert even if the sample set does not route tokens to all of them.
    is_moe = _is_moe(getattr(model, "config", None))
    ignore = ["lm_head"]
    # Skip the same modules RedHatAI/NVIDIA skip for the Qwen3.5 / Qwen3-Next family (these also have
    # shapes not divisible by the grouped-scheme group_size, which would otherwise error). No-ops
    # elsewhere. Hybrid linear attention, VLM vision tower, and the MTP/speculative head.
    ignore += ["re:.*\\.linear_attn\\..*", "re:.*\\.visual\\..*", "re:.*mtp.*"]
    if is_moe:
        # Keep MoE routing layers unquantized: the router gate and (Qwen) shared-expert gate.
        ignore += ["re:.*\\.gate$", "re:.*\\.shared_expert_gate$"]
    moe_kwargs = {"moe_calibrate_all_experts": True} if is_moe else {}

    def _make_recipe():
        return QuantizationModifier(targets = "Linear", scheme = args.scheme, ignore = ignore)

    if args.needs_calibration:
        ds = _build_calibration_dataset(
            tokenizer,
            args.calibration_dataset_kind,
            args.calibration_dataset,
            args.num_calibration_samples,
            args.max_seq_length,
        )
        # Use the sequential pipeline: it onloads layer-by-layer, so models that do not fit in
        # memory at once can still calibrate. Running here in a clean process (Unsloth's attention
        # patches are absent) means tracing works; fall back to the memory-hungry "basic" pipeline
        # only if tracing fails.
        try:
            oneshot(
                model = model,
                dataset = ds,
                recipe = _make_recipe(),
                max_seq_length = args.max_seq_length,
                num_calibration_samples = args.num_calibration_samples,
                pipeline = "sequential",
                **moe_kwargs,
            )
        except Exception as e:
            print(
                f"Unsloth: sequential calibration pipeline failed ({type(e).__name__}: {e}); "
                "retrying with the 'basic' pipeline (needs the full model to fit in memory).",
                flush = True,
            )
            # Free the partially-processed model before loading a fresh copy, so the fallback does
            # not transiently hold two copies on GPU. llm-compressor keeps the model in a global
            # session after a failed run, so reset it first; also drop the traceback frames (e) and
            # the local reference that pin the model.
            import gc as _gc
            import torch as _torch

            try:
                from llmcompressor.core import reset_session
                reset_session()
            except Exception:
                pass
            e = None
            del model
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            model = _from_pretrained(auto_model, args.model, args.trust_remote_code)
            model.eval()
            oneshot(
                model = model,
                dataset = ds,
                recipe = _make_recipe(),
                max_seq_length = args.max_seq_length,
                num_calibration_samples = args.num_calibration_samples,
                pipeline = "basic",
                **moe_kwargs,
            )
    else:
        oneshot(model = model, recipe = _make_recipe())

    os.makedirs(args.out, exist_ok = True)
    save_kwargs = {"variant": args.variant} if args.variant else {}
    model.save_pretrained(args.out, save_compressed = True, **save_kwargs)
    if tokenizer is not None:
        tokenizer.save_pretrained(args.out)

    if _has_mtp(getattr(model, "config", None)):
        print(
            "Unsloth: WARNING - this model has MTP / speculative-decoding tensors that are not "
            "included in the compressed export (only the main model is quantized and saved). Use "
            "the non-compressed save path if you need the MTP weights.",
            flush = True,
        )

    cfg_path = os.path.join(args.out, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding = "utf-8") as f:
            cfg = json.load(f)
    if "quantization_config" not in cfg:
        print(
            f"Unsloth: ERROR - no quantization_config written to {cfg_path}", flush = True
        )
        sys.exit(2)
    shards = glob.glob(os.path.join(args.out, "*.safetensors"))
    qfmt = cfg["quantization_config"].get("format")
    print(
        f"[compressed-quantize] OK scheme={args.scheme} format={qfmt} "
        f"shards={len(shards)} -> {args.out}",
        flush = True,
    )


if __name__ == "__main__":
    main()
