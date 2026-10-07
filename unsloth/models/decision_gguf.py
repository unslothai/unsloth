# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# GGUF export of decision models (Clef and Laya) for llama.cpp's /v1/systemone server. The
# upstream converter (llama.cpp >= b11443) writes the graph; Unsloth then writes the per-type
# temperatures its PyTorch serving applies, which the Clef converter never reads.

__all__ = [
    "DECISION_GGUF_QUANTIZATIONS",
    "gguf_eligibility",
    "effective_temperatures",
    "read_decision_temperatures",
    "write_decision_temperatures",
    "export_decision_gguf",
    "save_pretrained_gguf",
    "push_to_hub_gguf",
]

import functools
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

DECISION_GGUF_QUANTIZATIONS = ("q8_0", "f16", "bf16", "q6_k", "q5_k_m", "q4_k_m")
# Written by convert_hf_to_gguf.py itself; the rest go through llama-quantize.
_OUTTYPES = ("q8_0", "f16", "bf16")
# The first llama.cpp release with the Clef converter, the Decision GGUF keys and /v1/systemone.
DECISION_LLAMA_CPP_TAG = "b11443"
# llama.cpp's Clef graph and converter (conversion/clef.py: ClefModel(Qwen3_5TextModel)).
_CLEF_ARCHITECTURES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
_QTYPES = ("choice", "score", "noul")
# laya.common.temp_bucket's sizes, written as the llama.cpp converter names them.
_BUCKETS = {"2": "2", "3-5": "3_5", "6-10": "6_10", "11+": "11"}
_CONTRACT_FILE = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "core"
    / "systemone"
    / "gguf_export_contract.py"
)


@functools.lru_cache(maxsize = None)
def _contract():
    # By path: it is standard library only, and importing Studio's package would pull in its server.
    spec = importlib.util.spec_from_file_location("unsloth._decision_gguf_contract", _CONTRACT_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quantizations(quantization_method) -> list:
    methods = [quantization_method] if isinstance(quantization_method, str) else quantization_method
    if not methods:
        raise ValueError("Unsloth: quantization_method is empty.")
    chosen = []
    for method in methods:
        method = str(method).strip().lower()
        if method not in DECISION_GGUF_QUANTIZATIONS:
            raise ValueError(
                f"Unsloth: decision models export to GGUF as one of {list(DECISION_GGUF_QUANTIZATIONS)}, "
                f"not {method!r}."
            )
        if method not in chosen:
            chosen.append(method)
    return chosen


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding = "utf-8"))


def _layout(folder: Path) -> Optional[str]:
    from .decision import is_clef_checkpoint, is_decision_checkpoint
    if is_clef_checkpoint(folder):
        return "clef"
    return "laya" if is_decision_checkpoint(folder) else None


def _base_config(base: str) -> Optional[dict]:
    local = Path(base).expanduser()
    if (local / "config.json").is_file():
        return _read_json(local / "config.json")
    try:
        from huggingface_hub import hf_hub_download
        return _read_json(Path(hf_hub_download(base, "config.json")))
    except Exception:
        return None


def _clef_backbone_config(folder: Path) -> Optional[dict]:
    if (folder / "config.json").is_file():
        return _read_json(folder / "config.json")
    # Adapters only: the backbone is the base they were trained on.
    base = None
    if (folder / "unsloth_decision_config.json").is_file():
        base = _read_json(folder / "unsloth_decision_config.json").get("base_model")
    if not base and (folder / "adapter_config.json").is_file():
        base = _read_json(folder / "adapter_config.json").get("base_model_name_or_path")
    return _base_config(str(base)) if base else None


def _clef_reason(architectures) -> Optional[str]:
    architectures = list(architectures or [])
    if any(name in _CLEF_ARCHITECTURES for name in architectures):
        return None
    return (
        f"Unsloth: GGUF export of this decision model is not supported: llama.cpp's decision "
        f"graph is built for Qwen3.5 backbones only, and this one is {', '.join(architectures) or 'unknown'}. "
        "It still serves through PyTorch (FastDecisionModel.predict, or Studio's Decision API)."
    )


def _laya_reason(model_type) -> str:
    return (
        f"Unsloth: GGUF export of Laya decision models needs a ModernBERT encoder "
        f"(llama.cpp's ModernBertDecisionModel), and this one is {model_type or 'unknown'}. "
        "It still serves through PyTorch."
    )


def gguf_eligibility(source) -> dict:
    """{"eligible", "layout", "reason"} for a decision model or an on-disk decision checkpoint."""
    if not isinstance(source, (str, os.PathLike)):
        if getattr(source, "is_clef", False):
            reason = _clef_reason(getattr(source._backbone().config, "architectures", None))
            return {"eligible": reason is None, "layout": "clef", "reason": reason}
        model_type = getattr(getattr(source.encoder, "config", None), "model_type", None)
        reason = None if model_type == "modernbert" else _laya_reason(model_type)
        return {"eligible": reason is None, "layout": "laya", "reason": reason}
    folder = Path(source)
    layout = _layout(folder)
    if layout is None:
        return {
            "eligible": False,
            "layout": None,
            "reason": f"Unsloth: {folder} is not a decision model checkpoint.",
        }
    if layout == "laya":
        encoder = folder / "encoder" / "config.json"
        model_type = _read_json(encoder).get("model_type") if encoder.is_file() else None
        reason = None if model_type == "modernbert" else _laya_reason(model_type)
        return {"eligible": reason is None, "layout": "laya", "reason": reason}
    config = _clef_backbone_config(folder)
    if config is None:
        return {
            "eligible": False,
            "layout": "clef",
            "reason": f"Unsloth: could not read the backbone config of {folder}, so it cannot be exported to GGUF.",
        }
    reason = _clef_reason(config.get("architectures"))
    return {"eligible": reason is None, "layout": "clef", "reason": reason}


def _positive(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Unsloth: {name} = {value!r} is not a number.") from None
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Unsloth: {name} = {value!r} is not a finite positive temperature.")
    return value


def effective_temperatures(config: dict, layout: str) -> dict:
    """The temperature PyTorch serving divides each question type's logits by
    (decision._served_temperatures), keyed as llama.cpp reads them after
    "<arch>.decision.temperature.": "<type>" and "<type>.<option-count bucket>"."""
    from .decision import _laya

    clamp = _laya().common.clamp_temperature
    # Clef only: a head temperature _fold_temperature could not fold into the weights.
    head = (
        _positive(config.get("head_temperature", 1.0), "head_temperature")
        if layout == "clef"
        else 1.0
    )
    per_type = config.get("temperature", [1.0] * 3)
    if not isinstance(per_type, (list, tuple)) or len(per_type) != len(_QTYPES):
        raise ValueError(
            f"Unsloth: temperature = {per_type!r} must hold one value per question type {_QTYPES}."
        )
    out = {name: head * clamp(value) for name, value in zip(_QTYPES, per_type)}
    buckets = config.get("temperature_by_options") or {}
    if not isinstance(buckets, dict):
        raise ValueError(f"Unsloth: temperature_by_options = {buckets!r} is not a mapping.")
    for key, value in buckets.items():
        qtype, _, size = str(key).partition(":")
        if qtype not in _QTYPES or size not in _BUCKETS:
            # laya.common.temp_bucket never produces it, so PyTorch never applies it either.
            continue
        out[f"{qtype}.{_BUCKETS[size]}"] = head * clamp(value)
    for name, value in out.items():
        _positive(value, f"temperature {name}")
    return out


def _import_gguf(gguf_py: Optional[Path] = None):
    if gguf_py is not None and str(gguf_py) not in sys.path:
        try:
            import gguf  # noqa: F401
        except ImportError:
            sys.path.insert(0, str(gguf_py))
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    return gguf, MetadataDetails, copy_with_new_metadata


def _temperature_fields(reader, gguf) -> tuple:
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    prefix = f"{arch}.decision.temperature."
    found = {
        name[len(prefix) :]: float(field.contents())
        for name, field in reader.fields.items()
        if name.startswith(prefix)
    }
    return arch, prefix, found


def read_decision_temperatures(gguf_file, gguf_py = None) -> dict:
    gguf, _, _ = _import_gguf(gguf_py)
    reader = gguf.GGUFReader(str(gguf_file), "r")
    try:
        return _temperature_fields(reader, gguf)[2]
    finally:
        del reader


def _same(found: dict, wanted: dict) -> bool:
    import numpy as np
    return set(found) == set(wanted) and all(
        np.float32(found[name]) == np.float32(wanted[name]) for name in wanted
    )


def write_decision_temperatures(
    gguf_file,
    temperatures: dict,
    gguf_py = None,
) -> bool:
    """Replaces every <arch>.decision.temperature.* key with `temperatures`; everything else is
    copied unchanged. Atomic (temp file in the same folder, then os.replace). False if the file
    already held exactly these values."""
    gguf, MetadataDetails, copy_with_new_metadata = _import_gguf(gguf_py)
    gguf_file = Path(gguf_file)
    wanted = {name: _positive(value, f"temperature {name}") for name, value in temperatures.items()}
    reader = gguf.GGUFReader(str(gguf_file), "r")
    arch, prefix, found = _temperature_fields(reader, gguf)
    if _same(found, wanted):
        del reader
        return False
    remove = [prefix + name for name in found]
    new = {
        prefix + name: MetadataDetails(gguf.GGUFValueType.FLOAT32, value)
        for name, value in sorted(wanted.items())
    }
    fd, tmp = tempfile.mkstemp(prefix = f".{gguf_file.name}.", suffix = ".tmp", dir = gguf_file.parent)
    os.close(fd)
    try:
        writer = gguf.GGUFWriter(tmp, arch = arch, endianess = reader.endianess)
        copy_with_new_metadata(reader, writer, new, remove)
        del reader, writer
        if not _same(read_decision_temperatures(tmp, gguf_py), wanted):
            raise RuntimeError(
                f"Unsloth: the decision temperatures did not survive rewriting {gguf_file}."
            )
        os.replace(tmp, gguf_file)
    except BaseException:
        Path(tmp).unlink(missing_ok = True)
        raise
    return True


def _supports_decision(converter_dir: Path) -> bool:
    constants = converter_dir / "gguf-py" / "gguf" / "constants.py"
    if not (
        (converter_dir / "convert_hf_to_gguf.py").is_file()
        and (converter_dir / "conversion" / "clef.py").is_file()
        and constants.is_file()
    ):
        return False
    text = constants.read_text(encoding = "utf-8", errors = "replace")
    return "{arch}.decision.temperature.{name}" in text and "CLEF" in text


def _llama_cpp_folder() -> Path:
    try:
        from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR
    except ImportError:
        LLAMA_CPP_DEFAULT_DIR = os.environ.get(
            "UNSLOTH_LLAMA_CPP_PATH", os.path.join(str(Path.home()), ".unsloth", "llama.cpp")
        )
    return Path(LLAMA_CPP_DEFAULT_DIR)


def _converter_dir(print_output = False) -> Path:
    """convert_hf_to_gguf.py + conversion/ + gguf-py/ from one llama.cpp revision with decision models."""
    pinned = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", "").strip()
    candidates = [Path(pinned)] if pinned else []
    candidates.append(_llama_cpp_folder())
    for folder in candidates:
        if _supports_decision(folder):
            return folder
    tag = os.environ.get("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "").strip() or DECISION_LLAMA_CPP_TAG
    stage = None
    try:
        from unsloth_zoo.llama_cpp import _stage_converter_sources
        stage = _stage_converter_sources(tag)
    except Exception as error:
        if print_output:
            print(f"Unsloth: could not stage llama.cpp {tag} converter sources: {error}")
    if stage is not None and _supports_decision(Path(stage)):
        print(
            f"Unsloth: the llama.cpp at {candidates[-1]} cannot convert decision models; "
            f"using the llama.cpp {tag} converter sources staged at {stage}."
        )
        return Path(stage)
    raise RuntimeError(
        f"Unsloth: decision model GGUF export needs llama.cpp {DECISION_LLAMA_CPP_TAG} or newer "
        f"(conversion/clef.py and gguf-py's decision keys), and {', '.join(map(str, candidates))} "
        "does not have them. Delete that folder so Unsloth reinstalls llama.cpp, or point "
        "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR at a llama.cpp checkout of that tag."
    )


def _quantizer(print_output = False) -> str:
    from unsloth_zoo.llama_cpp import check_llama_cpp, install_llama_cpp
    folder = _llama_cpp_folder()
    try:
        return check_llama_cpp(llama_cpp_folder = str(folder))[0]
    except Exception:
        print("Unsloth: Installing llama.cpp. This might take 3 minutes...")
        # As save.py: quantizing needs no CUDA build.
        return install_llama_cpp(
            llama_cpp_folder = str(folder), gpu_support = False, print_output = print_output
        )[0]


def _run(
    command: list,
    what: str,
    print_output: bool,
    env = None,
) -> None:
    if print_output:
        print("Unsloth: running", " ".join(map(str, command)))
    result = subprocess.run(
        [str(part) for part in command],
        env = env,
        stdout = None if print_output else subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        encoding = "utf-8",
        errors = "replace",
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout or "").splitlines()[-30:])
        raise RuntimeError(f"Unsloth: {what} failed (exit {result.returncode}).\n{tail}")


def _convert(
    converter: Path, folder: Path, outtype: str, outfile: Path, mmproj: bool, print_output: bool
) -> None:
    env = {k: v for k, v in os.environ.items() if k != "NO_LOCAL_GGUF"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(converter / "gguf-py")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    command = [
        sys.executable,
        converter / "convert_hf_to_gguf.py",
        folder,
        "--outtype",
        outtype,
        "--outfile",
        outfile,
    ]
    if mmproj:
        command.append("--mmproj")
    _run(command, f"converting {folder} to {outfile.name}", print_output, env)
    if not outfile.is_file():
        raise RuntimeError(f"Unsloth: the llama.cpp converter did not write {outfile}.")


def _quantize(quantizer: str, source: Path, target: Path, method: str, print_output: bool) -> None:
    try:
        _run(
            [quantizer, source, target, method.upper()],
            f"quantizing to {method.upper()}",
            print_output,
        )
    except RuntimeError as error:
        if "unknown model architecture" in str(error):
            raise RuntimeError(
                f"Unsloth: {quantizer} predates llama.cpp {DECISION_LLAMA_CPP_TAG}, so it cannot "
                f"quantize decision models to {method}. Update llama.cpp (delete {_llama_cpp_folder()} "
                "so Unsloth reinstalls it), or export as q8_0, f16 or bf16."
            ) from error
        raise


def _has_vision(folder: Path) -> bool:
    config = folder / "config.json"
    return config.is_file() and _read_json(config).get("vision_config") is not None


def _decision_config(folder: Path, layout: str) -> dict:
    name = "unsloth_decision_config.json" if layout == "clef" else "rl_agent_config.json"
    path = folder / name
    return _read_json(path) if path.is_file() else {}


def export_decision_gguf(
    checkpoint_folder,
    quantization_method = "q8_0",
    output_dir = None,
    source_folder = None,
    print_output = False,
) -> dict:
    """Converts a merged decision checkpoint (Clef or Laya layout) to GGUF in output_dir
    (default <checkpoint_folder>/gguf): model-<QUANT>.gguf, mmproj-<QUANT>.gguf for a Clef
    vision tower, and export.json, written last. Returns export.json's content."""
    contract = _contract()
    folder = Path(checkpoint_folder)
    quants = _quantizations(quantization_method)
    eligibility = gguf_eligibility(folder)
    if not eligibility["eligible"]:
        raise ValueError(eligibility["reason"])
    layout = eligibility["layout"]
    if layout == "clef" and not (folder / "config.json").is_file():
        raise ValueError(
            f"Unsloth: {folder} holds LoRA adapters only; load it with FastDecisionModel and call "
            "save_pretrained_gguf, which merges them first."
        )
    temperatures = effective_temperatures(_decision_config(folder, layout), layout)
    source = Path(source_folder) if source_folder is not None else folder
    source_fingerprint = contract.fingerprint(source, layout)
    converter = _converter_dir(print_output)
    gguf_py = converter / "gguf-py"
    kquants = [q for q in quants if q not in _OUTTYPES]
    quantizer = _quantizer(print_output) if kquants else None
    vision = layout == "clef" and _has_vision(folder)
    output = Path(output_dir) if output_dir is not None else folder / contract.EXPORT_DIR
    output.mkdir(parents = True, exist_ok = True)

    def model_name(quant):
        return f"model-{quant.upper()}.gguf"

    def mmproj_name(quant):
        # llama-quantize does not quantize vision towers: a k-quant ships the Q8_0 one.
        return f"mmproj-{(quant if quant in _OUTTYPES else 'q8_0').upper()}.gguf"

    files = {}
    staging = Path(tempfile.mkdtemp(prefix = ".unsloth-gguf-", dir = output.parent))
    try:
        # Laya weights are float16; Clef's Qwen3.5 backbone is bfloat16.
        intermediate = "bf16" if layout == "clef" else "f16"
        outtypes = [q for q in quants if q in _OUTTYPES]
        if kquants and intermediate not in outtypes:
            outtypes.append(intermediate)
        for outtype in outtypes:
            target = staging / model_name(outtype)
            print(f"Unsloth: converting the decision model to {outtype.upper()} GGUF...")
            _convert(converter, folder, outtype, target, False, print_output)
            write_decision_temperatures(target, temperatures, gguf_py)
        for quant in kquants:
            target = staging / model_name(quant)
            print(f"Unsloth: quantizing the decision model to {quant.upper()}...")
            _quantize(quantizer, staging / model_name(intermediate), target, quant, print_output)
            write_decision_temperatures(target, temperatures, gguf_py)
        if vision:
            for name in sorted({mmproj_name(q) for q in quants}):
                outtype = name[len("mmproj-") : -len(".gguf")].lower()
                print(f"Unsloth: converting the vision tower to {name}...")
                _convert(converter, folder, outtype, staging / name, True, print_output)
        for quant in quants:
            found = read_decision_temperatures(staging / model_name(quant), gguf_py)
            if not _same(found, temperatures):
                raise RuntimeError(
                    f"Unsloth: {model_name(quant)} lost its decision temperatures ({found} != {temperatures})."
                )
            files[quant.upper()] = {
                "model": model_name(quant),
                "mmproj": mmproj_name(quant) if vision else None,
            }

        # An earlier export of the same weights keeps its other quantizations.
        previous = contract.read_export(output)
        keep = {}
        if (
            previous is not None
            and previous["source_fingerprint"] == source_fingerprint
            and previous["layout"] == layout
        ):
            for quant, entry in previous["files"].items():
                if quant in files:
                    continue
                names = [entry["model"]] + ([entry["mmproj"]] if entry.get("mmproj") else [])
                if all((output / name).is_file() for name in names):
                    keep[quant] = entry
        merged = {**files, **keep}
        wanted = {
            name
            for entry in merged.values()
            for name in (entry["model"], entry.get("mmproj"))
            if name
        }
        # Readers see no export while files move, never a half-written one.
        (output / contract.EXPORT_FILE).unlink(missing_ok = True)
        if previous is not None:
            for entry in previous["files"].values():
                for name in (entry["model"], entry.get("mmproj")):
                    if name and name not in wanted and Path(name).name == name:
                        (output / name).unlink(missing_ok = True)
        for entry in files.values():
            for name in (entry["model"], entry["mmproj"]):
                if name and (staging / name).is_file():
                    os.replace(staging / name, output / name)
        data = contract.write_export(output, layout, merged, source_fingerprint)
    finally:
        shutil.rmtree(staging, ignore_errors = True)
    print(f"Unsloth: saved decision model GGUF ({', '.join(files)}) to {output}")
    return data


def save_pretrained_gguf(
    self,
    save_directory,
    tokenizer = None,
    quantization_method = "q8_0",
    source_folder = None,
    print_output = False,
    **kwargs,
) -> dict:
    """GGUF for llama.cpp's decision server in <save_directory>/gguf: merges into a temporary
    folder, converts, and writes the calibrated temperatures. quantization_method is one of
    DECISION_GGUF_QUANTIZATIONS or a list of them."""
    from .decision import is_decision_checkpoint

    quants = _quantizations(quantization_method)
    eligibility = gguf_eligibility(self)
    if not eligibility["eligible"]:
        raise ValueError(eligibility["reason"])
    # Fail before the merge when llama.cpp cannot convert or quantize.
    _converter_dir(print_output)
    if any(q not in _OUTTYPES for q in quants):
        _quantizer(print_output)
    output = Path(save_directory)
    output.mkdir(parents = True, exist_ok = True)
    if source_folder is None and is_decision_checkpoint(output):
        source_folder = output
    with tempfile.TemporaryDirectory(prefix = ".unsloth-merged-", dir = output) as merged:
        self.save_pretrained_merged(merged, tokenizer)
        if source_folder is not None and _layout(Path(source_folder)) != eligibility["layout"]:
            source_folder = None
        return export_decision_gguf(
            merged,
            quants,
            output_dir = output / _contract().EXPORT_DIR,
            source_folder = source_folder,
            print_output = print_output,
        )


def push_to_hub_gguf(
    self,
    repo_id,
    tokenizer = None,
    quantization_method = "q8_0",
    token = None,
    private = None,
    print_output = False,
    **kwargs,
) -> dict:
    from huggingface_hub import HfApi

    api = HfApi(token = token)
    with tempfile.TemporaryDirectory() as folder:
        data = self.save_pretrained_gguf(
            folder, tokenizer, quantization_method = quantization_method, print_output = print_output
        )
        repo_id = api.create_repo(repo_id, private = private, exist_ok = True).repo_id
        api.upload_folder(folder_path = str(Path(folder) / _contract().EXPORT_DIR), repo_id = repo_id)
    print(f"Unsloth: Saved the decision model GGUF to https://huggingface.co/{repo_id}")
    return data
