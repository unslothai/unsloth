# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import typer
import yaml


@contextlib.contextmanager
def _spinner(console, text):
    from rich.live import Live
    from rich.spinner import Spinner
    with Live(
        Spinner("dots", text = text, style = "cyan"),
        console = console,
        transient = True,
        refresh_per_second = 12,
        redirect_stdout = False,
        redirect_stderr = False,
    ):
        yield


@contextlib.contextmanager
def _silence():
    from rich.console import Console

    sys.stdout.flush()
    sys.stderr.flush()
    real = os.fdopen(os.dup(1), "w", closefd = True)
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
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull_fd)
        real.close()


def _read_adapter_base(config: Path) -> Optional[str]:
    # ValueError covers both JSONDecodeError and UnicodeDecodeError
    try:
        data = json.loads(config.read_text(encoding = "utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data.get("base_model_name_or_path")


_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")
_HUB_REPO_RE = re.compile(r"[\w.\-]+/[\w.\-]+")


def _has_tokenizer_files(model: str) -> bool:
    path = Path(model)
    if path.is_dir():
        return any((path / name).exists() for name in _TOKENIZER_FILES)
    if path.exists() or not _HUB_REPO_RE.fullmatch(model):
        return False
    try:
        from huggingface_hub import list_repo_files
        files = set(list_repo_files(model))
    except Exception:
        return False
    return any(name in files for name in _TOKENIZER_FILES)


def _bitsandbytes_available() -> bool:
    from importlib.util import find_spec
    return find_spec("bitsandbytes") is not None


def _lm_eval_available() -> bool:
    # probe without importing: on lm-eval 0.4.4 `import lm_eval` pulls in
    # transformers, which must stay unimported until unsloth has loaded
    if "lm_eval" in sys.modules:
        return sys.modules["lm_eval"] is not None
    from importlib.util import find_spec
    try:
        return find_spec("lm_eval") is not None
    except (ImportError, ValueError):
        return False


def _hf_device_error(device: str) -> Optional[str]:
    # lm-eval's HFLM only recognises 'cuda', canonical 'cuda:<i>', 'mps' and
    # 'mps:0'; anything else (cuda0, cuda:, cuda:01, an out-of-range index)
    # silently falls back to its default device, so reject those up front
    if device.startswith("cuda"):
        match = re.fullmatch(r"cuda(?::(0|[1-9]\d*))?", device)
        if not match:
            return f"invalid --device '{device}' — use 'cuda' or 'cuda:<index>'."
        import torch
        if not torch.cuda.is_available():
            return f"--device {device} requested but CUDA is not available."
        if match.group(1) is not None:
            idx = int(match.group(1))
            count = torch.cuda.device_count()
            if idx >= count:
                return f"--device {device} requested but only {count} CUDA device(s) are available."
    elif device.startswith("mps"):
        if not re.fullmatch(r"mps(?::0)?", device):
            return f"invalid --device '{device}' — use 'mps'."
        import torch
        mps = getattr(torch.backends, "mps", None)
        if not (mps and mps.is_available()):
            return f"--device {device} requested but MPS is not available."
    return None


def _registry_names(manager) -> set:
    return (
        set(getattr(manager, "all_tasks", []) or [])
        | set(getattr(manager, "all_groups", []) or [])
        | set(getattr(manager, "all_tags", []) or [])
    )


def resolve_base_model(model: str) -> Optional[str]:
    path = Path(model)
    if path.is_dir():
        config = path / "adapter_config.json"
        return _read_adapter_base(config) if config.exists() else None
    # adapter-only Hub repos carry adapter_config.json but no config.json, so
    # they cannot be passed to lm-eval as `pretrained` — detect them up front
    if path.exists() or not _HUB_REPO_RE.fullmatch(model):
        return None
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model, "adapter_config.json")
    except Exception:
        return None
    return _read_adapter_base(Path(config_path))


def _doc_column(key: str) -> str:
    # a jinja template stringifies the value (needed e.g. for numeric answer
    # columns in few-shot prompts), but jinja can't parse keys that aren't
    # plain identifiers ("prompt-text", "expected answer") or that collide
    # with its keywords/literals — lm-eval treats a raw column name as a
    # direct lookup, so fall back to that for such keys
    import keyword
    if key.isidentifier() and not keyword.iskeyword(key) and key not in ("true", "false", "none"):
        return "{{" + key + "}}"
    return key


def make_jsonl_task(
    data_file: Path,
    input_key: str,
    target_key: str,
    out_dir: Path,
    reserved: frozenset = frozenset(),
) -> str:
    data_file = Path(data_file).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    # a generated task must not shadow a registered task (gsm8k.jsonl vs the
    # gsm8k benchmark) or an earlier dataset with the same stem
    base_name = data_file.stem
    task_name = base_name
    counter = 2
    while task_name in reserved or (out_dir / f"{task_name}.yaml").exists():
        task_name = f"{base_name}_{counter}"
        counter += 1
    if task_name != base_name:
        typer.echo(
            f"Note: task name '{base_name}' is taken — running dataset "
            f"'{data_file.name}' as '{task_name}'."
        )
    builder = "json" if data_file.suffix.lower() in {".json", ".jsonl"} else "csv"
    task_spec = {
        "task": task_name,
        "dataset_path": builder,
        "dataset_kwargs": {"data_files": str(data_file)},
        "test_split": "train",
        # explicit few-shot source so --num-fewshot works on every lm-eval
        # version we support (the file has a single split)
        "fewshot_split": "train",
        "output_type": "generate_until",
        "doc_to_text": _doc_column(input_key),
        "doc_to_target": _doc_column(target_key),
        "generation_kwargs": {"until": ["\n"]},
        # strip surrounding whitespace so " 2" matches gold "2"
        "filter_list": [
            {
                "name": "strip",
                "filter": [
                    {"function": "regex", "regex_pattern": r"^\s*(.*?)\s*$", "group_select": 0},
                    {"function": "take_first"},
                ],
            },
        ],
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True},
        ],
    }
    (out_dir / f"{task_name}.yaml").write_text(
        yaml.safe_dump(task_spec, sort_keys = False), encoding = "utf-8"
    )
    return task_name


def resolve_tasks(
    tasks: str,
    input_key: str,
    target_key: str,
    tmp_dir: Path,
    reserved: frozenset = frozenset(),
) -> Tuple[List[str], List[str]]:
    names: List[str] = []
    include_paths: List[str] = []

    def _add_include(directory: str) -> None:
        if directory not in include_paths:
            include_paths.append(directory)

    for raw in tasks.split(","):
        entry = raw.strip()
        if not entry:
            continue
        suffix = Path(entry).suffix.lower()

        if suffix in {".yaml", ".yml"}:
            path = Path(entry)
            if not path.exists():
                raise FileNotFoundError(f"Custom task file not found: {entry}")
            try:
                spec = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in custom task file '{entry}': {e}") from e
            if not isinstance(spec, dict):
                raise ValueError(f"Custom task file '{entry}' must define a YAML mapping.")
            name = spec.get("task")
            if isinstance(name, list):
                # a group file (group: suite, task: [a, b]) is registered
                # under its group name
                name = spec.get("group")
                if not name:
                    raise ValueError(
                        f"Custom task file '{entry}' defines a task list but no 'group:' name."
                    )
            if not name:
                raise ValueError(f"Custom task file '{entry}' is missing a 'task:' name.")
            name = str(name)
            if name in reserved:
                raise ValueError(
                    f"Custom task file '{entry}' redefines '{name}', which is already a "
                    "registered lm-eval task — the registered one would silently win. "
                    "Rename the task in the YAML."
                )
            if name in names:
                raise ValueError(f"Duplicate task name '{name}' in --tasks.")
            if "include" in spec or isinstance(spec.get("task"), list):
                # include-bearing and group configs reference sibling files,
                # so their directory must stay on the include path
                _add_include(str(path.resolve().parent))
            else:
                # copy just this file into the temp include dir so a broken
                # sibling yaml can't take down TaskManager's include scan
                # (this also normalises .yml, which lm-eval doesn't index)
                custom_dir = Path(tmp_dir) / "custom"
                custom_dir.mkdir(parents = True, exist_ok = True)
                shutil.copy2(path, custom_dir / f"{name}.yaml")
                _add_include(str(custom_dir.resolve()))
            names.append(name)

        elif suffix in {".jsonl", ".json", ".csv"}:
            path = Path(entry)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {entry}")
            gen_dir = Path(tmp_dir) / "generated"
            # names picked so far count as taken too (foo.yaml + foo.jsonl)
            names.append(
                make_jsonl_task(path, input_key, target_key, gen_dir, reserved | frozenset(names))
            )
            _add_include(str(gen_dir.resolve()))

        else:
            if entry in names:
                raise ValueError(f"Duplicate task '{entry}' in --tasks.")
            names.append(entry)

    if not names:
        raise ValueError("No tasks provided. Pass --tasks with at least one task.")
    return names, include_paths


def _render_results(results: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title = "Evaluation results")
    table.add_column("Task", style = "cyan")
    table.add_column("Metric")
    table.add_column("Value", justify = "right")
    table.add_column("± stderr", justify = "right")

    for task, metrics in results.get("results", {}).items():
        for key, value in metrics.items():
            if key == "alias" or "_stderr" in key or not isinstance(value, (int, float)):
                continue
            metric, _, flt = key.partition(",")
            stderr_key = f"{metric}_stderr,{flt}" if flt else f"{metric}_stderr"
            stderr = metrics.get(stderr_key)
            stderr_str = f"{stderr:.4f}" if isinstance(stderr, (int, float)) else "—"
            table.add_row(task, key, f"{value:.4f}", stderr_str)

    Console().print(table)


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
    limit: Optional[int] = typer.Option(
        None, "--limit", help = "Cap examples per task (for quick smoke tests)."
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
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar = "HF_TOKEN", help = "HuggingFace token if needed."
    ),
):
    """Evaluate a checkpoint or LoRA adapter using lm-eval-harness."""
    if batch_size == "auto":
        bs = "auto"
    else:
        try:
            bs = int(batch_size)
            if bs <= 0:
                raise ValueError
        except ValueError:
            typer.echo("Error: --batch-size must be a positive integer or 'auto'.", err = True)
            raise typer.Exit(code = 2)

    if backend not in ("unsloth", "hf"):
        typer.echo(f"Error: --backend must be 'unsloth' or 'hf', got '{backend}'.", err = True)
        raise typer.Exit(code = 2)

    if not _lm_eval_available():
        typer.echo(
            "Error: evaluation requires lm-eval. Install it with `pip install unsloth[eval]`.",
            err = True,
        )
        raise typer.Exit(code = 1)

    if backend == "unsloth":
        # unsloth must be imported before transformers (which lm-eval pulls
        # in) or its patches don't fully apply
        with _silence():
            import unsloth

        if getattr(unsloth, "DEVICE_TYPE", None) == "mlx":
            typer.echo(
                "Note: Apple Silicon (MLX) detected — falling back to "
                "--backend hf (plain transformers)."
            )
            backend = "hf"

    # a pre-loaded model object makes lm-eval single-process (rank 0
    # everywhere), so under accelerate/torchrun every worker would run
    # the full task set and write results
    if backend == "unsloth" and os.environ.get("WORLD_SIZE", "1") not in ("", "1"):
        typer.echo(
            "Error: multi-process launches (accelerate/torchrun) are not "
            "supported with --backend unsloth. Use --backend hf for "
            "multi-GPU evaluation.",
            err = True,
        )
        raise typer.Exit(code = 2)

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
    except ImportError as e:
        typer.echo(
            "Error: evaluation requires lm-eval. Install it with `pip install unsloth[eval]`.",
            err = True,
        )
        raise typer.Exit(code = 1) from e

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token  # both backends read it from the env

    # --base-model => treat <model> as an adapter on this base (and skip the
    # local/Hub adapter_config.json lookup)
    effective_base = base_model or resolve_base_model(model)

    tmp_dir = Path(tempfile.mkdtemp(prefix = "unsloth_eval_"))
    try:
        # a dataset or custom task named after a registered task (gsm8k.jsonl,
        # task: gsm8k) must not be shadowed by the built-in benchmark, so
        # collect registry names first
        base_manager = None
        reserved: frozenset = frozenset()
        if any(
            Path(e.strip()).suffix.lower() in {".jsonl", ".json", ".csv", ".yaml", ".yml"}
            for e in tasks.split(",")
        ):
            base_manager = TaskManager()
            reserved = frozenset(_registry_names(base_manager))

        try:
            task_names, include_paths = resolve_tasks(
                tasks, input_key, target_key, tmp_dir, reserved = reserved
            )
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {e}", err = True)
            raise typer.Exit(code = 2) from e

        # reuse for validation and the eval run
        if include_paths:
            task_manager = TaskManager(include_path = include_paths)
        else:
            task_manager = base_manager or TaskManager()

        registered = getattr(task_manager, "all_tasks", None)
        if registered:
            known = _registry_names(task_manager)
            unknown = [t for t in task_names if t not in known]
            if unknown:
                typer.echo(
                    f"Error: unknown task(s): {', '.join(unknown)}. Pass a built-in "
                    "task name, a .yaml task file, or a .jsonl/.csv dataset.",
                    err = True,
                )
                raise typer.Exit(code = 2)

        if num_fewshot and any(
            (tmp_dir / "generated" / f"{t}.yaml").exists() for t in task_names
        ):
            typer.echo(
                "Note: few-shot examples for a generated task come from the same "
                "file (no held-out split)."
            )

        typer.echo(f"Running tasks: {', '.join(task_names)} (backend: {backend})")

        eval_kwargs = dict(
            tasks = task_names,
            num_fewshot = num_fewshot,
            limit = limit,
            task_manager = task_manager,
            log_samples = False,
        )

        if backend == "hf":
            if device is None:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device_error = _hf_device_error(device)
                if device_error:
                    typer.echo(f"Error: {device_error}", err = True)
                    raise typer.Exit(code = 2)
            if bs == "auto" and not device.startswith("cuda"):
                typer.echo(
                    "Note: batch_size 'auto' is slow on CPU/MPS — using 1 (override with --batch-size)."
                )
                bs = 1
            # dict form: a comma in a path can't corrupt key=value parsing
            if effective_base:
                model_args = {"pretrained": effective_base, "peft": model}
                # adapters that saved their own tokenizer (added tokens etc.)
                # must not be scored with the base tokenizer
                if _has_tokenizer_files(model):
                    model_args["tokenizer"] = model
                typer.echo(f"Evaluating adapter '{model}' on base '{effective_base}'.")
            else:
                model_args = {"pretrained": model}
            model_args["max_length"] = max_seq_length
            if load_in_4bit and device.startswith("cuda"):
                if _bitsandbytes_available():
                    model_args["load_in_4bit"] = True
                else:
                    typer.echo(
                        "Note: bitsandbytes is not installed — loading in full "
                        "precision (`pip install bitsandbytes` to enable 4-bit)."
                    )
            eval_kwargs.update(
                model = "hf",
                model_args = model_args,
                batch_size = bs,
                device = device,
            )
        else:
            from unsloth import FastLanguageModel

            load_kwargs = dict(
                max_seq_length = max_seq_length,
                load_in_4bit = load_in_4bit,
                token = hf_token or None,
            )
            if effective_base:
                typer.echo(f"Loading base model '{effective_base}' with adapter '{model}'...")
                with _silence():
                    lmodel, tokenizer = FastLanguageModel.from_pretrained(
                        model_name = effective_base, **load_kwargs
                    )
                    # adapters that saved their own tokenizer (added tokens
                    # etc.) must not be scored with the base tokenizer, and
                    # the embeddings must match its vocab before the adapter
                    # weights are applied or PEFT fails on a size mismatch
                    if _has_tokenizer_files(model):
                        from transformers import AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained(model)
                        embeddings = lmodel.get_input_embeddings()
                        if embeddings is not None and embeddings.weight.shape[0] != len(tokenizer):
                            lmodel.resize_token_embeddings(len(tokenizer))
                    from peft import PeftModel

                    lmodel = PeftModel.from_pretrained(lmodel, model)
            else:
                typer.echo(f"Loading model: {model}")
                with _silence():
                    lmodel, tokenizer = FastLanguageModel.from_pretrained(
                        model_name = model, **load_kwargs
                    )
            with _silence():
                FastLanguageModel.for_inference(lmodel)
                lm = HFLM(
                    pretrained = lmodel,
                    tokenizer = tokenizer,
                    batch_size = bs,
                    max_length = max_seq_length,
                )
            eval_kwargs["model"] = lm

        with _silence() as ui, _spinner(ui, f"Evaluating {', '.join(task_names)}…"):
            results = lm_eval.simple_evaluate(**eval_kwargs)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors = True)

    if results is None:
        # lm-eval hands results only to rank 0 of a multi-process run
        # (accelerate/torchrun); worker ranks get None and must exit cleanly
        if os.environ.get("RANK", "0") != "0" or os.environ.get("LOCAL_RANK", "0") != "0":
            return
        typer.echo("Error: evaluation returned no results.", err = True)
        raise typer.Exit(code = 1)

    _render_results(results)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent = 2, default = str), encoding = "utf-8")
    typer.echo(f"Saved results to: {results_path}")
