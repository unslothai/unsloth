# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn an Unsloth fine-tuning notebook into a flow by reading its code, never running it.

Code cells are parsed with `ast` and matched against the calls every Unsloth notebook makes
(`from_pretrained`, `get_peft_model`, `load_dataset`, the trainer config, `save_pretrained_gguf`).
GRPO, DPO and ORPO trainers become the Fine-tune node's objective; reward functions map to library
rewards by name. Anything Studio cannot train is reported, not guessed.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

from core.flows.examples.build_examples import COLAB_NB, notebook_template
from core.flows.nodes.data import _REPO
from utils.notebooks.catalog import _category

MAX_NOTEBOOK_BYTES = 32 * 1024 * 1024

_LOADERS = {
    "FastLanguageModel": "text",
    "FastModel": "text",
    "FastVisionModel": "vision",
    "FastSentenceTransformer": "embedding",
}
_CONFIGS = {
    "SFTConfig",
    "TrainingArguments",
    "UnslothTrainingArguments",
    "SentenceTransformerTrainingArguments",
}
_RL_TRAINERS = {"GRPOTrainer": "grpo", "DPOTrainer": "dpo", "ORPOTrainer": "orpo"}
_RL_CONFIGS = {"GRPOConfig", "DPOConfig", "ORPOConfig"}
_RL_ARGS = {
    "num_generations": "num_generations",
    "max_completion_length": "max_completion_length",
    "temperature": "temperature",
    "beta": "beta",
}
# Notebook reward function name -> library reward. Checked in order, format names first so
# match_format_exactly is a format reward, not an exact answer.
_REWARD_NAMES = (
    (("strict_format", "format_exactly"), "strict-xml-format"),
    (("soft_format", "format_approximately", "xmlcount"), "soft-xml-format"),
    (("correct", "check_answer", "exact"), "exact-answer"),
    (("check_numbers", "numeric", "number"), "numeric-close"),
)
_UNSUPPORTED = {
    "KTOTrainer": "KTO preference training",
    "PPOTrainer": "PPO reinforcement learning",
}
_SFT_TRAINERS = {
    "SFTTrainer",
    "Trainer",
    "UnslothTrainer",
    "SentenceTransformerTrainer",
    "Seq2SeqTrainer",
}
_EMBEDDING_LOSSES = {"MultipleNegativesRankingLoss", "CachedMultipleNegativesRankingLoss"}
_ARGS = {
    "max_steps": "max_steps",
    "learning_rate": "learning_rate",
    "per_device_train_batch_size": "batch_size",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "warmup_steps": "warmup_steps",
    "num_train_epochs": "num_epochs",
    "weight_decay": "weight_decay",
}
_QUANTS = {"q4_k_m", "q5_k_m", "q8_0", "f16"}
_MAGIC = re.compile(r"^\s*[!%]")


_PACKAGED_BUILDERS = frozenset(
    {
        "json",
        "csv",
        "parquet",
        "text",
        "arrow",
        "pandas",
        "imagefolder",
        "audiofolder",
        "videofolder",
        "webdataset",
        "xml",
        "sql",
    }
)


def _local_files(reader: Any, node: ast.Call) -> bool:
    path = reader.kw(node, "path", 0)
    return any(k.arg == "data_files" for k in node.keywords) or (
        isinstance(path, str) and path.lower() in _PACKAGED_BUILDERS
    )


class NotebookImportError(ValueError):
    pass


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    text = "".join(src) if isinstance(src, list) else str(src)
    return "\n".join(
        line[: len(line) - len(line.lstrip())] + "pass" if _MAGIC.match(line) else line
        for line in text.splitlines()
    )


def _name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class _Reader(ast.NodeVisitor):
    def __init__(self) -> None:
        self.consts: dict[str, Any] = {}
        self.found: dict[str, Any] = {}
        self.tune: dict[str, Any] = {}
        self.unsupported: set[str] = set()
        self.approximated: list[str] = []
        self.losses: set[str] = set()
        self.rewards: list[str] = []
        self.format_fns: list[str] = []

    def value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.consts.get(node.id)
        return None

    def kw(
        self,
        call: ast.Call,
        key: str,
        pos: Optional[int] = None,
    ) -> Any:
        for k in call.keywords:
            if k.arg == key:
                return self.value(k.value)
        if pos is not None and len(call.args) > pos:
            return self.value(call.args[pos])
        return None

    def visit_If(self, node: ast.If) -> None:
        if isinstance(node.test, ast.Constant):
            for child in node.body if node.test.value else node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = self.value(node.value)
            if value is not None:
                self.consts[node.targets[0].id] = value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = _name(node.func)
        owner = _name(node.func.value) if isinstance(node.func, ast.Attribute) else None
        if fn == "PatchDPOTrainer":
            pass
        elif fn in _UNSUPPORTED:
            self.unsupported.add(_UNSUPPORTED[fn])
        elif fn in _RL_TRAINERS:
            self.found["objective"] = _RL_TRAINERS[fn]
            self.found.setdefault("objectives", []).append(_RL_TRAINERS[fn])
        elif fn in _SFT_TRAINERS:
            self.found["sft_stage"] = True
        elif fn and fn.endswith("Trainer") and fn not in _SFT_TRAINERS:
            self.unsupported.add(f"{fn} training")
        elif fn and fn.endswith("Loss"):
            self.losses.add(fn)
        elif fn == "from_pretrained" and owner in _LOADERS and "model" not in self.found:
            model = self.kw(node, "model_name", 0)
            if isinstance(model, str):
                self.found["model"] = model
                self.found["task"] = _LOADERS[owner]
                seq = self.kw(node, "max_seq_length")
                if isinstance(seq, int):
                    self.tune["max_seq_length"] = seq
                if self.kw(node, "load_in_4bit") is False:
                    self.tune["load_in_4bit"] = False
                if self.kw(node, "trust_remote_code") is True:
                    self.unsupported.add("a model that needs trust_remote_code")
        elif fn == "train_on_responses_only":
            self.tune["train_on_completions"] = True
        elif fn == "get_peft_model" and any(
            k.arg == "qat_scheme"
            and not (isinstance(k.value, ast.Constant) and k.value.value is None)
            for k in node.keywords
        ):
            self.unsupported.add("quantization-aware training (qat_scheme)")
        elif fn == "get_peft_model":
            self.found["peft"] = True
            r = self.kw(node, "r")
            if isinstance(r, int):
                self.tune["lora_r"] = r
            alpha = self.kw(node, "lora_alpha")
            if isinstance(alpha, int) and not isinstance(alpha, bool):
                self.tune["lora_alpha"] = alpha
        elif fn == "load_dataset" and "dataset" not in self.found and _local_files(self, node):
            self.unsupported.add("a dataset loaded from local files (data_files)")
        elif fn == "load_dataset" and "dataset" in self.found:
            repo = self.kw(node, "path", 0)
            subset = self.kw(node, "name", 1)
            subset = subset if isinstance(subset, str) else ""
            if (
                isinstance(repo, str)
                and _REPO.match(repo)
                and (repo, subset) != (self.found["dataset"], self.found["subset"])
            ):
                self.unsupported.add("training on more than one dataset")
        elif fn == "load_dataset" and "dataset" not in self.found:
            repo = self.kw(node, "path", 0)
            if isinstance(repo, str) and _REPO.match(repo):
                self.found["dataset"] = repo
                subset = self.kw(node, "name", 1)
                self.found["subset"] = subset if isinstance(subset, str) else ""
                split = self.kw(node, "split")
                self.found["split"] = split if isinstance(split, str) else "train"
                if self.kw(node, "streaming") is True:
                    self.tune["dataset_streaming"] = True
        elif fn in _CONFIGS or fn in _RL_CONFIGS:
            args = {**_ARGS, **_RL_ARGS} if fn in _RL_CONFIGS else _ARGS
            for arg, param in args.items():
                value = self.kw(node, arg)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.tune[param] = value
            loss = self.kw(node, "loss_type")
            if fn == "GRPOConfig" and isinstance(loss, str):
                level = self.kw(node, "importance_sampling_level")
                self.tune["grpo_variant"] = "gspo" if level == "sequence" else loss.lower()
        elif fn == "get_chat_template":
            template = self.kw(node, "chat_template")
            if isinstance(template, str):
                self.approximated.append(
                    f"Chat template {template!r}: Studio applies the model's own template automatically"
                )
        elif fn in ("save_pretrained_gguf", "push_to_hub_gguf"):
            quant = self.kw(node, "quantization_method")
            if isinstance(quant, str) and quant.lower() in _QUANTS:
                self.found.setdefault("gguf", quant.lower())
        for k in node.keywords:
            if k.arg == "reward_funcs":
                items = k.value.elts if isinstance(k.value, (ast.List, ast.Tuple)) else [k.value]
                self.rewards += [n for n in map(_name, items) if n]
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if "format" in node.name.lower():
            self.format_fns.append(node.name)
        self.generic_visit(node)


def _cells(data: bytes) -> list[str]:
    try:
        nb = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookImportError("That is not a notebook (.ipynb) file") from exc
    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not isinstance(cells, list):
        raise NotebookImportError("That notebook has no cells")
    return [_source(c) for c in cells if isinstance(c, dict) and c.get("cell_type") == "code"]


def _library_rewards(names: list[str]) -> tuple[list[dict], list[str]]:
    picked: dict[str, dict] = {}
    left_out = []
    for name in names:
        low = name.lower()
        match = next((r for keys, r in _REWARD_NAMES if any(k in low for k in keys)), None)
        if match:
            picked.setdefault(match, {"name": match, "weight": 1.0})
        else:
            left_out.append(name)
    return list(picked.values()), left_out


def _on_mlx() -> bool:
    try:
        from utils.hardware import hardware
        return hardware.DEVICE == hardware.DeviceType.MLX
    except Exception:
        return False


def read_notebook(data: bytes, name: str = "notebook.ipynb") -> tuple[Optional[dict], dict]:
    """(flow document or None, report). None means Studio cannot run this notebook: open it in Colab."""
    if len(data) > MAX_NOTEBOOK_BYTES:
        raise NotebookImportError("That notebook is too large")
    reader = _Reader()
    skipped = 0
    for src in _cells(data):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            skipped += 1
            continue
        reader.visit(tree)
    if _category(name) == "decision":
        reader.unsupported.add("decision fine-tuning (use the Train page)")
    elif _category(name) == "classification":
        reader.unsupported.add("classification fine-tuning (open it in Colab)")
    elif _category(name) == "cpt":
        reader.unsupported.add("continued pretraining on raw text (open it in Colab)")
    rewards: list[dict] = []
    if len(reader.found.get("objectives") or ()) > 1:
        stages = " and ".join(o.upper() for o in reader.found["objectives"])
        reader.unsupported.add(f"more than one RL stage ({stages})")
    if reader.found.get("objective") and reader.found.get("sft_stage"):
        # One Fine-tune node runs one objective; dropping the warm-up would train something else.
        reader.unsupported.add("an SFT stage followed by RL (two training stages)")
    if reader.found.get("objective") == "grpo":
        rewards, left_out = _library_rewards(reader.rewards)
        if not rewards:
            reader.unsupported.add(
                "reward functions with no match in the reward library"
                + (f" ({', '.join(left_out)})" if left_out else "")
            )
        elif left_out:
            reader.approximated.append(
                f"Reward functions with no library match, left out: {', '.join(left_out)}"
            )
    if reader.found.get("task") == "embedding":
        for loss in sorted(reader.losses - _EMBEDDING_LOSSES):
            reader.unsupported.add(f"embedding training with {loss}")
    reader.approximated += [
        f"Formatting function {fn}(): Studio formats the dataset with its own detected format"
        for fn in reader.format_fns
        if fn not in reader.rewards
    ]
    found, tune = reader.found, dict(reader.tune)
    report: dict[str, Any] = {
        "imported": [],
        "approximated": list(dict.fromkeys(reader.approximated)),
        "unsupported": [
            f"Not runnable in Studio: {what}. Open it in Colab."
            for what in sorted(reader.unsupported)
        ],
        "colab_url": f"{COLAB_NB}{name}" if name.endswith(".ipynb") else None,
    }
    if skipped:
        report["approximated"].append(f"{skipped} code cell(s) could not be read and were skipped")
    if not reader.unsupported:
        for key, label in (("model", "a model"), ("dataset", "a Hub dataset")):
            if key not in found:
                report["unsupported"].append(
                    f"Not runnable in Studio: no {label} found in the notebook. Open it in Colab."
                )
    if report["unsupported"]:
        return None, report

    if (tune.get("max_steps") or 0) > 0:
        tune.pop("num_epochs", None)
    else:
        tune.pop("max_steps", None)
        tune.setdefault("num_epochs", 3)
    epochs = tune.get("num_epochs")
    if isinstance(epochs, float) and not epochs.is_integer():
        tune["num_epochs"] = max(1, round(epochs))
        report["approximated"].append(
            f"{epochs} epochs: Studio trains whole epochs, so this flow runs {tune['num_epochs']}"
        )
    elif isinstance(epochs, float):
        tune["num_epochs"] = int(epochs)
    if not found.get("peft"):
        tune["training_type"] = "full"
    if found["task"] == "text" and _category(name) in ("audio", "vision"):
        found["task"] = _category(name)
    if found["task"] != "text" and tune.pop("dataset_streaming", None):
        report["approximated"].append(
            f"Streaming: Studio's {found['task']} training loads the whole dataset instead"
        )
    elif tune.get("dataset_streaming") and (
        tune.get("train_on_completions") or not tune.get("max_steps") or _on_mlx()
    ):
        tune.pop("dataset_streaming")
        why = (
            "responses-only training"
            if tune.get("train_on_completions")
            else "training by epochs"
            if not tune.get("max_steps")
            else "Apple Silicon training"
        )
        report["approximated"].append(f"Streaming: Studio loads the whole dataset for {why}")
    tune["base_model"] = found["model"]
    if found["task"] != "text":
        tune["task"] = found["task"]
    if isinstance(tune.get("learning_rate"), (int, float)):
        tune["learning_rate"] = float(tune["learning_rate"])
    objective = found.get("objective")
    if objective:
        tune["objective"] = objective
    if objective == "grpo":
        tune["rewards"] = rewards
        system = reader.consts.get("SYSTEM_PROMPT")
        # The library rewards read <answer> tags; a prompt asking for other tags would never score.
        if isinstance(system, str) and "<answer>" in system:
            tune["system_prompt"] = system.strip()
        else:
            report["approximated"].append(
                "System prompt: Studio asks for <reasoning>/<answer> tags, which the library rewards check"
            )
        if tune.get("grpo_variant") not in (None, "dapo", "dr_grpo", "bnpo", "grpo", "gspo"):
            report["approximated"].append(f"Loss type {tune.pop('grpo_variant')}: Studio uses DAPO")
    report["imported"] = [f"Model {found['model']}", f"Dataset {found['dataset']}"] + [
        f"{k} = {v}"
        for k, v in tune.items()
        if k not in ("base_model", "rewards", "system_prompt")
    ]
    if rewards:
        report["imported"].append(
            "Rewards " + ", ".join(f"{r['name']} x{r['weight']:g}" for r in rewards)
        )
    if "system_prompt" in tune:
        report["imported"].append("System prompt from the notebook")
    if found.get("gguf"):
        report["imported"].append(f"GGUF export ({found['gguf']})")

    stem = name.removesuffix(".ipynb")
    doc = notebook_template(
        "notebook-import",
        f"Notebook: {stem.replace('_', ' ')}"[:200],
        name,
        f"Imported from the Unsloth notebook {name}.",
        dataset = found["dataset"],
        tune = tune,
        ask = "Hello! What can you do?" if found["task"] == "text" else None,
        export = {"format": "gguf", "quant": found["gguf"]}
        if found.get("gguf")
        else {"format": "lora" if found.get("peft") else "merged"},
        tags = [found["task"]],
    ).doc
    for node in doc["nodes"]:
        if node["type"] == "data.hf_dataset":
            node["params"].update(split = found.get("split", "train"), subset = found.get("subset", ""))
    doc["meta"].pop("example", None)
    return doc, report
