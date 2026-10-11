# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Checkpoint lookup, gold labels and calibration shared by FastDecisionModel's torch and MLX paths.
# Imports torch only inside functions: the MLX path loads this file by path, without unsloth.models.

__all__ = [
    "TRAIN_MAX_LEN",
    "TRAIN_HEAD_MAX_LEN",
    "HOLDOUT_MAX",
    "MIN_CALIBRATION_ITEMS",
    "QUESTION_TYPES",
    "_FILES",
    "_DIRS",
    "_CLEF_HEAD_FILES",
    "_ADAPTER_CONFIG",
    "CLEF_MAX_LEN",
    "CLEF_SERVE_MAX_LEN",
    "_VENDORED_LAYA",
    "HEAD_TEMPERATURE_RANGE",
    "DecisionDataError",
    "_laya",
    "is_decision_checkpoint",
    "is_clef_checkpoint",
    "_is_clef_adapter",
    "_is_clef_repo",
    "_is_plain_lm",
    "_checkpoint_folder",
    "_lm_subfolder",
    "_parsed",
    "_internal",
    "_option_keys",
    "_label",
    "_target",
    "_target_for",
    "_clef_question",
    "_predicted",
    "_metrics",
    "_fit_temperature",
    "_fit_temperatures",
    "_served_temperatures",
    "_calibrate_clef",
    "_served_lengths",
]

import functools
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Optional

TRAIN_MAX_LEN, TRAIN_HEAD_MAX_LEN = 1024, 256
HOLDOUT_MAX = 400
MIN_CALIBRATION_ITEMS = 10
QUESTION_TYPES = ("choice", "score", "noul")
_FILES = ("rl_agent_config.json", "model.safetensors")
_DIRS = ("encoder", "tokenizer")
_CLEF_HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")
_ADAPTER_CONFIG = "adapter_config.json"
CLEF_MAX_LEN = 4096
# predict() and serving read this many tokens whatever the training length: cutting a state drops evidence.
CLEF_SERVE_MAX_LEN = 16384
# laya 0.3.5 ships inside Unsloth for Studio's Decision API (studio/backend/vendor/README.md).
_VENDORED_LAYA = (
    Path(__file__).resolve().parents[2] / "studio" / "backend" / "vendor" / "laya" / "__init__.py"
).resolve()


class DecisionDataError(ValueError):
    pass


@functools.lru_cache(maxsize = None)
def _laya():
    # By path, so a pip installed "laya" is never used or replaced; Studio registers this copy.
    for name in ("laya", "unsloth._laya"):
        module = sys.modules.get(name)
        if getattr(module, "__file__", None) and Path(module.__file__).resolve() == _VENDORED_LAYA:
            return module
    if not _VENDORED_LAYA.is_file():
        raise ImportError(
            f"Unsloth: decision models need laya, which ships with Unsloth at {_VENDORED_LAYA.parent}. "
            "Please reinstall Unsloth."
        )
    spec = importlib.util.spec_from_file_location(
        "unsloth._laya", _VENDORED_LAYA, submodule_search_locations = [str(_VENDORED_LAYA.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # laya's JSON reads and writes use a bare open(), which is the locale encoding on Windows.
    module.agent.open = functools.partial(open, encoding = "utf-8")
    return module


def is_decision_checkpoint(folder) -> bool:
    folder = Path(folder)
    return is_clef_checkpoint(folder) or (
        all((folder / name).is_file() for name in _FILES)
        and all((folder / name).is_dir() for name in _DIRS)
    )


def is_clef_checkpoint(folder) -> bool:
    # Cloudflare's Clef layout: a backbone (merged, or LoRA adapters over a base) plus a joint schema head.
    folder = Path(folder)
    return all((folder / name).is_file() for name in _CLEF_HEAD_FILES) and (
        (folder / "config.json").is_file() or (folder / _ADAPTER_CONFIG).is_file()
    )


def _is_clef_adapter(folder) -> bool:
    folder = Path(folder)
    return not (folder / "config.json").is_file() and (folder / _ADAPTER_CONFIG).is_file()


def _is_clef_repo(model_name, prefix, token, revision) -> Optional[bool]:
    # Asked up front: Unsloth's download wrapper rejects a snapshot that lacks an exact file it was
    # asked for, so a Laya pattern on a Clef repo (or the reverse) would fail as "incomplete".
    from huggingface_hub import HfApi, constants

    if constants.HF_HUB_OFFLINE:
        return None
    try:
        files = HfApi(token = token).list_repo_files(model_name, revision = revision)
    except Exception:
        return None
    return prefix + _CLEF_HEAD_FILES[1] in files


def _is_plain_lm(model_name, subfolder, token, revision, local_files_only) -> bool:
    # Unknown (offline, no access) answers False, so the checkpoint loader names what is missing.
    markers = {_FILES[0], _CLEF_HEAD_FILES[1]}
    if subfolder:
        return False
    root = Path(str(model_name)).expanduser()
    if root.is_dir():
        has = lambda name: (root / name).is_file()
        return (has("config.json") or has(_ADAPTER_CONFIG)) and not any(map(has, markers))
    from huggingface_hub import constants

    if not (local_files_only or constants.HF_HUB_OFFLINE):
        try:
            from huggingface_hub import HfApi
            files = HfApi(token = token).list_repo_files(str(model_name), revision = revision)
        except Exception:
            files = None
        if files is not None:
            names = {name.rsplit("/", 1)[-1] for name in files}
            return bool({"config.json", _ADAPTER_CONFIG} & set(files)) and not (names & markers)
    from huggingface_hub import try_to_load_from_cache

    def cached(name) -> bool:
        try:
            return isinstance(try_to_load_from_cache(str(model_name), name, revision = revision), str)
        except Exception:
            return False

    return (cached("config.json") or cached(_ADAPTER_CONFIG)) and not any(map(cached, markers))


def _checkpoint_folder(model_name, subfolder, token, revision, local_files_only) -> Path:
    root = Path(model_name).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download as cached_snapshot

        try:
            from unsloth_zoo.hf_xet_fallback import (
                snapshot_download_with_xet_fallback as snapshot_download,
            )
        except ImportError:
            snapshot_download = cached_snapshot
        prefix = f"{subfolder}/" if subfolder else ""
        laya = [prefix + name for name in _FILES] + [f"{prefix}{name}/*" for name in _DIRS]
        clef = None if local_files_only else _is_clef_repo(model_name, prefix, token, revision)
        if clef is None:
            # Offline: the cache already holds one layout or the other.
            root = Path(
                cached_snapshot(
                    model_name,
                    token = token,
                    revision = revision,
                    local_files_only = True,
                    allow_patterns = laya + [prefix + "*.json"],
                )
            )
            clef = (root / prefix / _CLEF_HEAD_FILES[1]).is_file()
        # Laya repos hold several checkpoints in subfolders, so only the asked one is fetched.
        root = Path(
            snapshot_download(
                model_name,
                token = token,
                revision = revision,
                local_files_only = local_files_only,
                allow_patterns = [prefix + "*"] if clef else laya,
            )
        )
    folder = root / subfolder if subfolder else root
    if not is_decision_checkpoint(folder):
        raise ValueError(
            f"Unsloth: {folder} is not a decision model checkpoint "
            "(rl_agent_config.json, model.safetensors, encoder/ and tokenizer/, "
            "or a Clef backbone with joint_head.safetensors and joint_head_config.json). "
            'To turn a plain language model into a decision model, pass decision_head = "clef".'
        )
    return folder


def _lm_subfolder(model_name, subfolder, token, revision, local_files_only) -> str:
    if Path(model_name).expanduser().is_dir():
        return str(Path(model_name).expanduser() / subfolder)
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        model_name,
        allow_patterns = [f"{subfolder}/*"],
        token = token,
        revision = revision,
        local_files_only = local_files_only,
    )
    return str(Path(root) / subfolder)


def _parsed(value):
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _internal(question) -> dict:
    if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
        raise DecisionDataError("is not a valid question")
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice" and not (
        isinstance(criteria, (dict, list))
        and criteria
        and all(isinstance(option, str) for option in criteria)
    ):
        raise DecisionDataError("needs criteria naming its options")
    if kind == "score" and not (isinstance(criteria, list) and criteria):
        raise DecisionDataError("needs a list of criteria levels")
    if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
        raise DecisionDataError('criteria may only have "true" and "false"')
    laya_question = {"type": kind, "instructions": question.get("instructions") or ""}
    if criteria is not None:
        laya_question["criteria"] = criteria
    return _laya().agent.Agent._to_internal(laya_question)


def _option_keys(internal: dict) -> list:
    if internal["t"] == "choice":
        return [str(key) for key in internal["crit"]]
    if internal["t"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(internal["crit"]))]


def _label(kind: str, label):
    if kind == "score" and isinstance(label, (int, float, str)) and not isinstance(label, bool):
        try:
            level = float(label)
        except ValueError:
            return None
        return str(int(level)) if level.is_integer() else None
    if label is None:
        return None
    return str(label).strip().lower() if kind == "noul" else str(label)


def _target(internal: dict, gold) -> tuple:
    return _target_for(internal["t"], _option_keys(internal), gold)


def _target_for(kind: str, keys: list, gold) -> tuple:
    if not isinstance(gold, dict):
        gold = {"label": gold}
    label = _label(kind, gold.get("label"))
    probabilities = gold.get("probabilities")
    if kind == "noul" and not isinstance(probabilities, dict):
        noul = gold.get("noul")
        probabilities = {"true": noul} if isinstance(noul, (int, float)) else None
    if isinstance(probabilities, dict):
        try:
            values = {str(key): float(value) for key, value in probabilities.items()}
        except (TypeError, ValueError):
            values = {}
        if not all(math.isfinite(value) for value in values.values()):
            raise DecisionDataError("gold has probabilities that are not finite numbers")
        if kind == "noul" and len(values.keys() & {"false", "true"}) == 1:
            known = "true" if "true" in values else "false"
            values[{"true": "false", "false": "true"}[known]] = 1.0 - values[known]
        target = [max(0.0, values.get(key, 0.0)) for key in keys]
        total = sum(target)
        if 0 < total < math.inf:
            target = [value / total for value in target]
            return target, keys.index(label) if label in keys else target.index(max(target))
    if label in keys:
        return [1.0 if key == label else 0.0 for key in keys], keys.index(label)
    raise DecisionDataError("gold has no usable label or probabilities")


def _clef_question(question) -> dict:
    if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
        raise DecisionDataError("is not a valid question")
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list) and criteria and all(isinstance(c, str) for c in criteria):
            criteria = dict.fromkeys(criteria)
        if not (isinstance(criteria, dict) and criteria):
            raise DecisionDataError("needs criteria naming its options")
        if len({str(key) for key in criteria}) != len(criteria):
            raise DecisionDataError("has repeated options")
    if kind == "score" and not (isinstance(criteria, list) and criteria):
        raise DecisionDataError("needs a list of criteria levels")
    if kind == "noul" and criteria is not None:
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise DecisionDataError('criteria may only have "true" and "false"')
    clef_question = {"type": kind, "instructions": question.get("instructions")}
    if criteria is not None:
        clef_question["criteria"] = criteria
    return clef_question


def _predicted(question: dict, answer: dict, probabilities: dict) -> dict:
    # The Decision API answer plus "answer": the option (choice), True / False (noul) or level number (score).
    kind = question["type"]
    best = max(probabilities, key = probabilities.__getitem__)
    return {
        **answer,
        "answer": int(best) if kind == "score" else best == "true" if kind == "noul" else best,
        "probabilities": answer.get("probabilities")
        or {key: round(float(value), 4) for key, value in probabilities.items()},
    }


def _metrics(logits, items, temperatures) -> dict:
    import numpy as np
    import torch

    conf, correct, loss, records = [], [], [], {}
    for z, item, temperature in zip(logits, items, temperatures):
        log_p = torch.log_softmax(z / temperature, -1)
        conf.append(float(log_p.exp().max()))
        correct.append(float(int(log_p.argmax()) == item["label"]))
        loss.append(float(-(torch.tensor(item["target"]) * log_p).sum()))
        row = item.get("row", ("item", len(correct)))
        records[row] = records.get(row, True) and bool(correct[-1])
    return {
        "accuracy": float(np.mean(correct)),
        "ece": _laya().common.ece_score(np.array(conf), np.array(correct)),
        "loss": float(np.mean(loss)),
        # Every question of a row right, the record-level precision Cloudflare rewards.
        "record_accuracy": float(np.mean(list(records.values()))),
    }


def _fit_temperature(
    logits,
    items,
    line_search = None,
) -> float:
    import torch

    options = max(len(z) for z in logits)
    z = torch.full((len(logits), options), -1e4)
    target = torch.zeros((len(logits), options))
    for i, (row, item) in enumerate(zip(logits, items)):
        z[i, : len(row)] = row
        target[i, : len(row)] = torch.tensor(item["target"])
    log_t = torch.zeros(1, requires_grad = True)
    # Clef's hard-label fit passes "strong_wolfe": plain LBFGS can overshoot on peaked targets.
    optimizer = torch.optim.LBFGS([log_t], lr = 0.1, max_iter = 100, line_search_fn = line_search)

    def closure():
        optimizer.zero_grad()
        loss = -(target * torch.log_softmax(z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def _fit_temperatures(logits, items, indices, fallback: list) -> tuple:
    clamp = _laya().common.clamp_temperature
    temperature, fitted = list(fallback), set()
    for qtype in range(3):
        chosen = [i for i in indices if items[i]["qtype"] == qtype]
        if len(chosen) >= MIN_CALIBRATION_ITEMS:
            temperature[qtype] = clamp(
                _fit_temperature([logits[i] for i in chosen], [items[i] for i in chosen])
            )
            fitted.add(qtype)
    return temperature, fitted


def _served_temperatures(config: dict, logits, items) -> list:
    common = _laya().common
    per_type = [common.clamp_temperature(t) for t in config.get("temperature", [1.0] * 3)]
    buckets = {
        key: common.clamp_temperature(value)
        for key, value in (config.get("temperature_by_options") or {}).items()
    }
    # A Clef temperature not yet folded into the head (_save_clef folds it on save).
    head = config.get("head_temperature", 1.0)
    return [
        head * buckets.get(common.temp_bucket(item["qtype"], len(z)), per_type[item["qtype"]])
        for z, item in zip(logits, items)
    ]


HEAD_TEMPERATURE_RANGE = (0.05, 20.0)


def _calibrate_clef(config: dict, logits, items) -> dict:
    # Calibrated against being right (the gold label), not the soft gold distribution: Clef's
    # confidence is read as the chance the answer is correct, and soft gold targets left a tuned
    # model underconfident (confidence 0.61 at accuracy 0.78 on typed-decisions).
    common = _laya().common
    hard = [
        {**item, "target": [float(j == item["label"]) for j in range(len(z))]}
        for z, item in zip(logits, items)
    ]

    def fit(indices) -> tuple:
        chosen = list(indices)
        head = _fit_temperature(
            [logits[i] for i in chosen], [hard[i] for i in chosen], line_search = "strong_wolfe"
        )
        head = min(max(head, HEAD_TEMPERATURE_RANGE[0]), HEAD_TEMPERATURE_RANGE[1])
        scaled = [z / head for z in logits]
        relative, fitted = _fit_temperatures(scaled, hard, chosen, [1.0] * 3)
        return head, relative, fitted

    everything = range(len(items))
    if len(items) < MIN_CALIBRATION_ITEMS:
        return {
            **_metrics(logits, items, _served_temperatures(config, logits, items)),
            "fitted_types": [],
        }
    head, relative, fitted = fit(everything)
    half = {row: i % 2 for i, row in enumerate(sorted({item["row"] for item in items}))}
    per_item = [head * common.clamp_temperature(relative[item["qtype"]]) for item in items]
    for side in (0, 1) if len(half) > 1 else ():
        side_head, side_relative, _ = fit(i for i in everything if half[items[i]["row"]] != side)
        for i in everything:
            if half[items[i]["row"]] == side:
                per_item[i] = side_head * common.clamp_temperature(side_relative[items[i]["qtype"]])
    config["head_temperature"] = head
    config["temperature"] = relative
    config.pop("temperature_by_options", None)
    return {**_metrics(logits, items, per_item), "fitted_types": sorted(fitted)}


def _served_lengths(
    config: dict,
    positions: int,
    max_seq_length = None,
) -> tuple:
    """(max_len, head_max_len) a Laya checkpoint serves with once loaded."""
    wanted = max_seq_length or max(int(config.get("max_len", 512)), TRAIN_MAX_LEN)
    max_len = min(int(positions), int(wanted))
    return max_len, min(max_len // 2, max(int(config.get("head_max_len", 192)), TRAIN_HEAD_MAX_LEN))
