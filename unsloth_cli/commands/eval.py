# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional, Tuple

import typer
import yaml

_HUB_REPO_RE = re.compile(r"[\w.\-]+/[\w.\-]+")
_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")
_DATASET_SUFFIXES = {".jsonl", ".json", ".csv"}
_YAML_SUFFIXES = {".yaml", ".yml"}


@contextlib.contextmanager
def _silence():
    """Send fd 1/2 to devnull (lm-eval/transformers progress bars); yield a Console on the real stdout."""
    from rich.console import Console

    sys.stdout.flush()
    sys.stderr.flush()
    # keep the entry point's UTF-8 reconfigure (Windows legacy code pages); fdopen would reset it
    real = os.fdopen(
        os.dup(1),
        "w",
        closefd = True,
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8",
        errors = "replace",
    )
    saved_out, saved_err = os.dup(1), os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield Console(file = real)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for fd in (saved_out, saved_err, devnull_fd):
            os.close(fd)
        real.close()


def _is_hub_id(model: str) -> bool:
    return not Path(model).exists() and bool(_HUB_REPO_RE.fullmatch(model))


def resolve_base_model(model: str) -> Optional[str]:
    """base_model_name_or_path of a LoRA adapter (local dir or Hub repo), else None."""
    path = Path(model)
    config = path / "adapter_config.json"
    if not path.is_dir():
        if not _is_hub_id(model):
            return None
        try:
            from huggingface_hub import hf_hub_download
            config = Path(hf_hub_download(model, "adapter_config.json"))
        except Exception:
            return None
    try:
        data = json.loads(config.read_text(encoding = "utf-8"))
    except (ValueError, OSError):
        return None
    return data.get("base_model_name_or_path") if isinstance(data, dict) else None


def _has_tokenizer_files(model: str) -> bool:
    path = Path(model)
    if path.is_dir():
        return any((path / name).exists() for name in _TOKENIZER_FILES)
    if not _is_hub_id(model):
        return False
    try:
        from huggingface_hub import list_repo_files
        files = set(list_repo_files(model))
    except Exception:
        return False
    return any(name in files for name in _TOKENIZER_FILES)


def _hf_device_error(device: str) -> Optional[str]:
    # HFLM silently falls back to its default device on any string outside its device_list
    match = re.fullmatch(r"(cpu|cuda|mps|npu|xpu|hpu)(?::(0|[1-9]\d*))?", device)
    kind, index = (match.group(1), match.group(2)) if match else (None, None)
    if (
        kind is None
        or (kind == "cpu" and index is not None)
        or (kind == "mps" and index not in (None, "0"))
        or (kind in ("npu", "xpu", "hpu") and index is None)
    ):
        return (
            f"invalid --device '{device}': use 'cpu', 'cuda[:<index>]', 'mps', "
            "or '<npu|xpu|hpu>:<index>'."
        )
    if kind == "cpu":
        return None
    import torch

    backend = getattr(torch.backends, "mps", None) if kind == "mps" else getattr(torch, kind, None)
    try:
        available = bool(backend is not None and backend.is_available())
    except Exception:
        available = False
    if not available:
        return f"--device {device} requested but {kind.upper()} is not available."
    if index is not None and kind != "mps":
        try:
            count = int(backend.device_count())
        except Exception:
            count = 0
        if int(index) >= count:
            return f"--device {device} requested but only {count} {kind.upper()} device(s) are available."
    return None


def _hflm_4bit(HFLM):
    # transformers 5 removed from_pretrained(load_in_4bit=...), which HFLM forwards verbatim
    class _HFLM4bit(HFLM):
        def _create_model(
            self,
            *args,
            quantization_config = None,
            load_in_4bit = False,
            **kwargs,
        ):
            if load_in_4bit and quantization_config is None:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_4bit = True)
            return super()._create_model(*args, quantization_config = quantization_config, **kwargs)

    return _HFLM4bit


class _TaskYamlLoader(yaml.SafeLoader):
    """safe_load that tolerates lm-eval's custom tags (!function utils.fn)."""


_TaskYamlLoader.add_multi_constructor(
    "!", lambda loader, suffix, node: getattr(node, "value", None)
)


def _doc_column(key: str) -> str:
    # Jinja stringifies the value but cannot parse non-identifier keys; lm-eval reads a bare column name raw
    import keyword
    if key.isidentifier() and not keyword.iskeyword(key) and key not in ("true", "false", "none"):
        return "{{" + key + "}}"
    return key


def make_dataset_task(
    data_file: Path, input_key: str, target_key: str, out_dir: Path, name: str
) -> None:
    """Write an exact-match generate_until task over a .jsonl/.json/.csv file."""
    builder = "json" if data_file.suffix.lower() in {".json", ".jsonl"} else "csv"
    spec = {
        "task": name,
        "dataset_path": builder,
        "dataset_kwargs": {"data_files": str(data_file.resolve())},
        "test_split": "train",
        "fewshot_split": "train",
        "output_type": "generate_until",
        "doc_to_text": _doc_column(input_key),
        "doc_to_target": _doc_column(target_key),
        "generation_kwargs": {"until": ["\n"]},
        # strip so " 2" matches gold "2"; with one capture group re.findall yields the group text
        "filter_list": [
            {
                "name": "strip",
                "filter": [
                    {"function": "regex", "regex_pattern": r"^\s*(.*?)\s*$", "group_select": 0},
                    {"function": "take_first"},
                ],
            },
        ],
        "metric_list": [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}],
    }
    out_dir.mkdir(parents = True, exist_ok = True)
    (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys = False), encoding = "utf-8")


def _yaml_task_name(path: Path) -> Tuple[str, bool]:
    """(registered name, needs its directory on the include path)."""
    text = path.read_text(encoding = "utf-8")
    try:
        spec = yaml.load(text, Loader = _TaskYamlLoader)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in custom task file '{path}': {e}") from e
    if not isinstance(spec, dict):
        raise ValueError(f"Custom task file '{path}' must define a YAML mapping.")
    is_group = isinstance(spec.get("task"), list)
    name = spec.get("group") if is_group else spec.get("task")
    if not name or not isinstance(name, str):
        raise ValueError(
            f"Custom task file '{path}' needs a top-level 'task:' name ('group:' for a task list)."
        )
    references_siblings = is_group or "include" in spec or "!function" in text
    if references_siblings and path.suffix.lower() == ".yml":
        raise ValueError(
            f"Custom task file '{path}' references sibling files but lm-eval only indexes "
            ".yaml files. Rename it (and the files it references) to .yaml."
        )
    return name, references_siblings


def resolve_tasks(
    tasks: str, input_key: str, target_key: str, tmp_dir: Path
) -> Tuple[List[str], List[str]]:
    """Split --tasks into lm-eval task names plus the include paths custom files need."""
    entries = [e.strip() for e in tasks.split(",") if e.strip()]
    if not entries:
        raise ValueError("No tasks provided. Pass --tasks with at least one task.")
    plain = {e for e in entries if Path(e).suffix.lower() not in _DATASET_SUFFIXES | _YAML_SUFFIXES}
    names: List[str] = []
    include_paths: List[str] = []

    def add(name: str, include: Path) -> None:
        if name in names:
            raise ValueError(f"Duplicate task '{name}' in --tasks.")
        names.append(name)
        if str(include) not in include_paths:
            include_paths.append(str(include))

    for entry in entries:
        path = Path(entry)
        suffix = path.suffix.lower()
        if suffix in _YAML_SUFFIXES:
            if not path.exists():
                raise FileNotFoundError(f"Custom task file not found: {entry}")
            name, needs_dir = _yaml_task_name(path)
            if needs_dir:
                add(name, path.resolve().parent)
            else:
                # copied alone so an unrelated broken yaml beside it cannot break lm-eval's index
                custom_dir = (tmp_dir / "custom").resolve()
                custom_dir.mkdir(parents = True, exist_ok = True)
                shutil.copy2(path, custom_dir / f"{name}.yaml")
                add(name, custom_dir)
        elif suffix in _DATASET_SUFFIXES:
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {entry}")
            # an include-path task overrides a registered one of the same name, so keep clear of
            # names requested alongside (gsm8k,./gsm8k.jsonl)
            name, counter = path.stem, 2
            while name in names or name in plain:
                name, counter = f"{path.stem}_{counter}", counter + 1
            if name != path.stem:
                typer.echo(f"Note: running dataset '{path.name}' as task '{name}'.")
            gen_dir = (tmp_dir / "generated").resolve()
            make_dataset_task(path, input_key, target_key, gen_dir, name)
            add(name, gen_dir)
        else:
            if entry in names:
                raise ValueError(f"Duplicate task '{entry}' in --tasks.")
            names.append(entry)
    return names, include_paths


def _expand_tasks(task_manager, names: List[str]) -> List[str]:
    """Expand globs (mmlu_*) like lm-eval's CLI and reject unknown names."""
    known = set(getattr(task_manager, "all_tasks", None) or [])
    if not known:
        return names
    expanded: List[str] = []
    for name in names:
        if any(ch in name for ch in "*?["):
            matches = task_manager.match_tasks([name])
            if not matches:
                raise ValueError(f"no tasks match pattern '{name}'.")
            expanded.extend(m for m in matches if m not in expanded)
        elif name not in known:
            raise ValueError(
                f"unknown task '{name}'. Pass a built-in task name, a .yaml task file, "
                "or a .jsonl/.csv dataset."
            )
        elif name not in expanded:
            expanded.append(name)
    return expanded


def _metric_number(value):
    # numpy float32/int64 are not int/float subclasses: unwrap scalars via item()
    if not isinstance(value, (int, float)) and callable(getattr(value, "item", None)):
        try:
            value = value.item()
        except Exception:
            return None
    return value if isinstance(value, (int, float)) else None


def _json_default(value):
    try:
        return value.tolist()
    except Exception:
        return str(value)


def _render_results(results: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title = "Evaluation results")
    table.add_column("Task", style = "cyan")
    table.add_column("Metric")
    table.add_column("Value", justify = "right")
    table.add_column("± stderr", justify = "right")
    rows = dict(results.get("results") or {})
    for task, metrics in (results.get("groups") or {}).items():
        rows.setdefault(task, metrics)
    for task, metrics in rows.items():
        for key, raw_value in metrics.items():
            value = _metric_number(raw_value)
            if key == "alias" or "_stderr" in key or value is None:
                continue
            metric, _, flt = key.partition(",")
            stderr = _metric_number(
                metrics.get(f"{metric}_stderr,{flt}" if flt else f"{metric}_stderr")
            )
            table.add_row(task, key, f"{value:.4f}", f"{stderr:.4f}" if stderr is not None else "—")
    Console().print(table)


def _fail(message: str, code: int = 2):
    typer.echo(f"Error: {message}", err = True)
    raise typer.Exit(code = code)


def evaluate(
    model: str = typer.Argument(
        ..., help = "Path to a checkpoint/adapter directory or a HuggingFace model id."
    ),
    tasks: str = typer.Option(
        ...,
        "--tasks",
        "-t",
        help = "Comma-separated built-in task names (e.g. mmlu,gsm8k), or a path to a "
        "custom .yaml task or a .jsonl/.csv dataset.",
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model",
        help = "Base model for a LoRA adapter. Auto-detected from adapter_config.json; "
        "set this to override a moved/renamed base.",
    ),
    num_fewshot: Optional[int] = typer.Option(
        None, "--num-fewshot", "-n", help = "Few-shot examples (default: per-task)."
    ),
    limit: Optional[float] = typer.Option(
        None,
        "--limit",
        help = "Cap examples per task (for quick smoke tests): a whole count, or a "
        "fraction between 0 and 1 for a proportion of each task.",
    ),
    batch_size: str = typer.Option("auto", "--batch-size", "-b", help = "Batch size, or 'auto'."),
    max_seq_length: int = typer.Option(
        2048, "--max-seq-length", help = "Max sequence length for the model."
    ),
    load_in_4bit: bool = typer.Option(
        True, "--load-in-4bit/--no-load-in-4bit", help = "Load the model in 4-bit."
    ),
    backend: str = typer.Option(
        "unsloth",
        "--backend",
        help = "Model backend: 'unsloth' (fast kernels; needs an NVIDIA/AMD/Intel "
        "GPU) or 'hf' (plain transformers; works on CPU/MPS/Mac). "
        "Auto-falls back to 'hf' on Apple Silicon.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help = "Device for the hf backend (e.g. cpu, mps, cuda). Default: auto.",
    ),
    input_key: str = typer.Option(
        "question", "--input-key", help = "Prompt field for a .jsonl/.csv dataset task."
    ),
    target_key: str = typer.Option(
        "answer", "--target-key", help = "Answer field for a .jsonl/.csv dataset task."
    ),
    output_dir: Path = typer.Option(
        Path("./eval_results"), "--output-dir", "-o", help = "Directory for results.json."
    ),
    confirm_run_unsafe_code: bool = typer.Option(
        False,
        "--confirm-run-unsafe-code",
        help = "Allow tasks that lm-eval marks unsafe (e.g. humaneval): they execute "
        "model-generated code on this machine. Off by default.",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN", help = "HuggingFace token if needed."
    ),
):
    """Evaluate a checkpoint or LoRA adapter using lm-eval-harness."""
    bs = batch_size
    if batch_size != "auto":
        bs = int(batch_size) if batch_size.isdigit() else 0
        if bs <= 0:
            _fail("--batch-size must be a positive integer or 'auto'.")
    if backend not in ("unsloth", "hf"):
        _fail(f"--backend must be 'unsloth' or 'hf', got '{backend}'.")
    if num_fewshot is not None and num_fewshot < 0:
        _fail("--num-fewshot must be >= 0.")
    if limit is not None:
        # lm-eval reads < 1 as a fraction and int()s counts, so 2.5 would silently run 2
        if limit <= 0 or (limit >= 1 and not limit.is_integer()):
            _fail("--limit must be a whole count or a fraction between 0 and 1.")
        limit = int(limit) if limit >= 1 else limit
    if max_seq_length <= 0:
        _fail("--max-seq-length must be a positive integer.")
    # find_spec, not import: lm_eval imports transformers, which must come after unsloth
    if "lm_eval" not in sys.modules and find_spec("lm_eval") is None:
        _fail("evaluation requires lm-eval. Install it with `pip install unsloth[eval]`.", code = 1)

    if backend == "unsloth":
        with _silence():
            import unsloth
        if getattr(unsloth, "DEVICE_TYPE", None) == "mlx":
            typer.echo("Note: Apple Silicon (MLX) detected, falling back to --backend hf.")
            backend = "hf"
    # a pre-loaded model makes lm-eval single-process, so every rank would run every task
    if backend == "unsloth" and os.environ.get("WORLD_SIZE", "1") not in ("", "1"):
        _fail(
            "multi-process launches (accelerate/torchrun) are not supported with "
            "--backend unsloth. Use --backend hf for multi-GPU evaluation."
        )

    import inspect

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    effective_base = base_model or resolve_base_model(model)

    tmp_dir = Path(tempfile.mkdtemp(prefix = "unsloth_eval_"))
    try:
        try:
            task_names, include_paths = resolve_tasks(tasks, input_key, target_key, tmp_dir)
            task_manager = TaskManager(include_path = include_paths or None)
            task_names = _expand_tasks(task_manager, task_names)
        except (FileNotFoundError, ValueError) as e:
            _fail(str(e))

        typer.echo(f"Running tasks: {', '.join(task_names)} (backend: {backend})")
        eval_kwargs = dict(
            tasks = task_names,
            num_fewshot = num_fewshot,
            limit = limit,
            task_manager = task_manager,
            log_samples = False,
        )
        # lm-eval < 0.4.8 has no unsafe-code gate
        if "confirm_run_unsafe_code" in inspect.signature(lm_eval.simple_evaluate).parameters:
            eval_kwargs["confirm_run_unsafe_code"] = confirm_run_unsafe_code
        elif confirm_run_unsafe_code:
            typer.echo(
                "Note: this lm-eval version has no unsafe-code gate; the flag has no effect."
            )

        if backend == "hf":
            import torch

            if device is None:
                mps = getattr(torch.backends, "mps", None)
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if mps and mps.is_available()
                    else "cpu"
                )
            elif (device_error := _hf_device_error(device)) is not None:
                _fail(device_error)
            if bs == "auto" and not device.startswith("cuda"):
                typer.echo(
                    "Note: batch size 'auto' is slow on CPU/MPS, using 1 (override with --batch-size)."
                )
                bs = 1
            model_args = {"pretrained": model}
            if effective_base:
                model_args = {"pretrained": effective_base, "peft": model}
                if _has_tokenizer_files(model):
                    model_args["tokenizer"] = model
                typer.echo(f"Evaluating adapter '{model}' on base '{effective_base}'.")
            model_args["max_length"] = max_seq_length
            if load_in_4bit and device.startswith("cuda"):
                if find_spec("bitsandbytes") is not None:
                    model_args["load_in_4bit"] = True
                else:
                    typer.echo("Note: bitsandbytes is not installed, loading in full precision.")
            if model_args.get("load_in_4bit"):
                with _silence():
                    eval_kwargs["model"] = _hflm_4bit(HFLM)(
                        **model_args, batch_size = bs, device = device
                    )
            else:
                eval_kwargs.update(model = "hf", model_args = model_args, batch_size = bs, device = device)
        else:
            from unsloth import FastLanguageModel

            load_kwargs = dict(
                max_seq_length = max_seq_length, load_in_4bit = load_in_4bit, token = hf_token or None
            )
            typer.echo(
                f"Loading base model '{effective_base}' with adapter '{model}'..."
                if effective_base
                else f"Loading model: {model}"
            )
            with _silence():
                lmodel, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = effective_base or model, **load_kwargs
                )
                if effective_base:
                    # resize to the adapter's tokenizer before loading weights, or PEFT size-mismatches
                    if _has_tokenizer_files(model):
                        from transformers import AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained(model)
                        embeddings = lmodel.get_input_embeddings()
                        if embeddings is not None and embeddings.weight.shape[0] != len(tokenizer):
                            lmodel.resize_token_embeddings(len(tokenizer))
                    from peft import PeftModel
                    lmodel = PeftModel.from_pretrained(lmodel, model)
                FastLanguageModel.for_inference(lmodel)
                eval_kwargs["model"] = HFLM(
                    pretrained = lmodel, tokenizer = tokenizer, batch_size = bs, max_length = max_seq_length
                )

        with _silence() as ui:
            from rich.status import Status
            with Status(f"Evaluating {', '.join(task_names)}…", console = ui, spinner = "dots"):
                results = lm_eval.simple_evaluate(**eval_kwargs)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors = True)

    if results is None:
        # lm-eval returns None on non-zero ranks
        if os.environ.get("RANK", "0") != "0" or os.environ.get("LOCAL_RANK", "0") != "0":
            return
        _fail("evaluation returned no results.", code = 1)

    _render_results(results)
    output_dir.mkdir(parents = True, exist_ok = True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent = 2, default = _json_default), encoding = "utf-8")
    typer.echo(f"Saved results to: {results_path}")
