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
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split = f"train_sft[:{num_samples}]")
        ds = ds.shuffle(seed = 42)
    elif kind == "hfid":
        # Not every dataset has a "train" split (e.g. train_sft only); fall back to the first one.
        try:
            ds = load_dataset(value, split = f"train[:{num_samples}]")
        except (ValueError, KeyError):
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

    cols = set(ds.column_names)
    if "input_ids" in cols:
        return ds
    if "messages" in cols:

        def _prep(ex):
            return {"text": _tok.apply_chat_template(ex["messages"], tokenize = False)}

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
    ap.add_argument("--calibration-dataset-kind", default = "none", choices = ["none", "hfid", "disk"])
    ap.add_argument("--calibration-dataset", default = "")
    ap.add_argument("--num-calibration-samples", type = int, default = 512)
    ap.add_argument("--max-seq-length", type = int, default = 2048)
    ap.add_argument("--is-vlm", action = "store_true")
    ap.add_argument("--trust-remote-code", action = "store_true")
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
        tokenizer = auto_proc.from_pretrained(args.model, trust_remote_code = args.trust_remote_code)
    except Exception:
        if args.needs_calibration:
            raise RuntimeError(
                f"Unsloth: calibration export needs a tokenizer but none was found in {args.model}. "
                "Pass tokenizer=... to save_pretrained_merged."
            )
        tokenizer = None

    recipe = QuantizationModifier(targets = "Linear", scheme = args.scheme, ignore = ["lm_head"])
    if args.needs_calibration:
        ds = _build_calibration_dataset(
            tokenizer,
            args.calibration_dataset_kind,
            args.calibration_dataset,
            args.num_calibration_samples,
            args.max_seq_length,
        )
        # "basic" pipeline runs a normal forward (no AST tracing / sequential splitting).
        oneshot(
            model = model,
            dataset = ds,
            recipe = recipe,
            max_seq_length = args.max_seq_length,
            num_calibration_samples = args.num_calibration_samples,
            pipeline = "basic",
        )
    else:
        oneshot(model = model, recipe = recipe)

    os.makedirs(args.out, exist_ok = True)
    model.save_pretrained(args.out, save_compressed = True)
    if tokenizer is not None:
        tokenizer.save_pretrained(args.out)

    cfg_path = os.path.join(args.out, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding = "utf-8") as f:
            cfg = json.load(f)
    if "quantization_config" not in cfg:
        print(f"Unsloth: ERROR - no quantization_config written to {cfg_path}", flush = True)
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
