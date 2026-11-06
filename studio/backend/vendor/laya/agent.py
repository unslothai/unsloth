"""High-level inference runtime for laya System 1 decision models."""
import json
import os
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    amp_dtype,
    build_model,
    build_sequence,
    clamp_temperature,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
)


def _fix_tokenizer_config(path: str):
    """Ensure tokenizer_config.json can be loaded across all transformers versions."""
    cfg_file = os.path.join(path, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return
    try:
        with open(cfg_file) as f:
            tcfg = json.load(f)
        changed = False
        if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            changed = True
        # Checkpoints built on the mmBERT/Gemma tokenizer store extra_special_tokens as a list;
        # transformers expects a mapping and raises "'list' object has no attribute 'keys'",
        # which makes AutoTokenizer -- and so the whole model -- fail to load.
        extra = tcfg.get("extra_special_tokens")
        if isinstance(extra, list):
            tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
            changed = True
        if changed:
            with open(cfg_file, "w") as f:
                json.dump(tcfg, f, indent=2)
    except Exception:
        pass


def _verify_compatibility(model: torch.nn.Module, cfg: Dict, weights: Dict[str, torch.Tensor], model_id: str):
    """Verify that the loaded checkpoint weights and config strictly match the expected architecture."""
    # 1. Verify required configuration attributes
    required_cfg = ["encoder", "head_layers"]
    missing_cfg = [k for k in required_cfg if k not in cfg]
    if missing_cfg:
        raise ValueError(
            f"Incompatible model config for {model_id!r}: missing configuration keys {missing_cfg}. "
            f"Ensure this is a valid RL Agent decision model."
        )

    # 2. Check for required component prefixes
    required_prefixes = ("encoder.", "type_emb.", "scorer.", "act_head.")
    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in weights.keys()):
            raise ValueError(
                f"Incompatible model weights for {model_id!r}: checkpoint is missing '{prefix}' parameters. "
                f"Expected an RL Agent decision model with encoder and decision heads."
            )

    # 3. Check for parameter shape mismatches
    model_sd = model.state_dict()
    shape_mismatches = []
    missing_keys = []

    for name, param in model.named_parameters():
        if name not in weights:
            missing_keys.append(name)
        elif tuple(weights[name].shape) != tuple(param.shape):
            shape_mismatches.append(f"  - {name}: expected {tuple(param.shape)}, found {tuple(weights[name].shape)}")

    if shape_mismatches:
        err_details = "\n".join(shape_mismatches[:5])
        if len(shape_mismatches) > 5:
            err_details += f"\n  ... and {len(shape_mismatches) - 5} more mismatched layers."
        raise ValueError(
            f"Model architecture mismatch for {model_id!r}:\n{err_details}\n"
            f"The checkpoint weights do not match the configured model architecture."
        )

    if missing_keys:
        raise ValueError(
            f"Model weights incomplete for {model_id!r}: missing {len(missing_keys)} parameter tensors "
            f"(e.g. {missing_keys[:3]})."
        )


class Agent:
    """System 1 decision model runtime: fast, non-autoregressive, calibrated decisions."""

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
    ):
        """Load a Laya checkpoint.

        `subfolder` selects one checkpoint from a repo that bundles several, e.g.
        `Agent("convaiinnovations/laya", subfolder="multilingual")`. Only that subfolder is
        downloaded, so bundling does not cost every user the whole family.
        """
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        model_dir = model_id_or_path
        if not os.path.exists(model_dir):
            if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
                raise FileNotFoundError(
                    f"Local model path not found: {model_id_or_path!r}. "
                    f"Check that the directory exists and that training saved the model successfully."
                )
            from huggingface_hub import snapshot_download

            # Restrict root checkpoints too: the default repo also contains sibling
            # checkpoints, which an unfiltered snapshot would unnecessarily download.
            prefix = f"{subfolder}/" if subfolder else ""
            kw = {
                "token": token or os.environ.get("HF_TOKEN"),
                "allow_patterns": [prefix + name for name in (
                    "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*",
                )],
            }
            model_dir = snapshot_download(model_id_or_path, **kw)

        if subfolder:
            model_dir = os.path.join(model_dir, subfolder)
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(
                    f"Subfolder {subfolder!r} not found in {model_id_or_path!r}."
                )

        _fix_tokenizer_config(model_dir)

        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Incompatible model: {model_id_or_path!r} does not contain 'rl_agent_config.json'. "
                f"That file ships with the weights of a Laya checkpoint, so load one of those "
                f"(e.g. 'convaiinnovations/laya') or a directory your own training run wrote."
            )

        with open(cfg_path) as f:
            self.cfg = json.load(f)

        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Incompatible model: 'model.safetensors' not found in {model_id_or_path!r}."
            )

        # 1. Device resolution with automatic fallback
        if device is not None:
            target_device = torch.device(device)
            if target_device.type == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            elif target_device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                print("Warning: MPS requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = target_device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        tok_dir = os.path.join(model_dir, "tokenizer")
        self.tok = AutoTokenizer.from_pretrained(tok_dir if os.path.exists(tok_dir) else self.cfg.get("encoder"))

        enc_dir = os.path.join(model_dir, "encoder")
        self.model = build_model(self.cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)

        # Load weights and verify architectural compatibility
        weights = load_file(weights_path)
        _verify_compatibility(self.model, self.cfg, weights, model_id_or_path)

        self.model.load_state_dict(weights, strict=True)

        # ModernBERT's reference_compile defaults to "auto" and will torch.compile the encoder.
        # That is a loss for the batch sizes Laya runs (a handful of questions per call) and can
        # hang on some platforms, so keep the eager path.
        try:
            self.model.encoder.config.reference_compile = False
        except Exception:
            pass

        # Keep what the checkpoint shipped for inspection, but only ever apply clamped values:
        # some buckets are fitted to sharpen rather than soften (see clamp_temperature).
        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {k: clamp_temperature(v)
                                       for k, v in self.temperature_by_options_raw.items()}
        rejected = ["%s=%.4g" % (k, float(v)) for k, v in self.temperature_by_options_raw.items()
                    if clamp_temperature(v) != float(v)]
        rejected += ["temperature[%d]=%.4g" % (i, float(t)) for i, t in enumerate(self.temperature_raw)
                     if clamp_temperature(t) != float(t)]
        if rejected:
            warnings.warn(
                "laya: this checkpoint ships temperatures outside [%g, %g] which would distort "
                "confidence; clamping %s. Treat confidence from the affected buckets as uncalibrated."
                % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
                RuntimeWarning, stacklevel=2)
        self.dtype = amp_dtype(self.cfg.get("amp_dtype", "fp16"))

        if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] < 8:
            self.dtype = torch.float16
        elif self.device.type in ("cpu", "mps"):
            self.dtype = torch.float32

        # 2. Place on device with graceful fallback to CPU on memory error
        fell_back_from = fell_back_why = None
        try:
            self.model.to(self.device).eval()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if self.device.type != "cpu":
                # Record what actually went wrong: the reason matters more than the symptom,
                # and it is the only place the underlying exception is ever surfaced.
                fell_back_from, fell_back_why = self.device, e
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.model.to(self.device).eval()
            else:
                raise e

        if fell_back_from is not None:
            print(
                "\n[laya] Warning: could not place the model on %s, so it is running on CPU.\n"
                "  Reason: %s\n"
                "  Inference will be roughly 10-15x slower (~200-500 ms rather than ~35 ms).\n"
                "  If this is a newer NVIDIA GPU (Blackwell / RTX 50-series), your PyTorch build\n"
                "  may not support its CUDA architecture:\n"
                "    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128\n"
                "  See https://pytorch.org/get-started/locally/\n"
                % (fell_back_from, fell_back_why), flush=True)

    @staticmethod
    def _to_internal(qdef: Dict) -> Dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            ins = json.dumps(ins)
        return {"t": t, "ins": ins, "crit": crit}

    @torch.no_grad()
    def system_one(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate typed questions across state in a single, parallel forward pass.

        Args:
            state: Text string, JSON dict, or conversation turn list.
            questions: Dictionary mapping question_id -> question definition.
                - choice: {"type": "choice", "instructions": "...", "criteria": {"optA": "...", ...}}
                - score:  {"type": "score",  "instructions": "...", "criteria": ["lvl0", "lvl1", ...]}
                - noul:   {"type": "noul",   "instructions": "..."}

        Returns:
            Dictionary with answers, probabilities, calibrated confidence, and token usage.
        """
        ids = list(questions.keys())
        items = []
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)

        for qid in ids:
            q = self._to_internal(questions[qid])
            seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        b = collate_items([items], self.tok.pad_token_id)
        use_amp = self.device.type == "cuda"

        try:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=use_amp):
                logits, act = self.model(
                    b["input_ids"].to(self.device),
                    b["attention_mask"].to(self.device),
                    b["marker_pos"].to(self.device),
                    b["marker_mask"].to(self.device),
                    b["qtype"].to(self.device),
                )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if self.device.type != "cpu" and ("memory" in str(e).lower() or "cuda" in str(e).lower()):
                print("Warning: GPU memory exceeded during inference. Falling back to CPU...")
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.model.to(self.device)
                logits, act = self.model(
                    b["input_ids"].to(self.device),
                    b["attention_mask"].to(self.device),
                    b["marker_pos"].to(self.device),
                    b["marker_mask"].to(self.device),
                    b["qtype"].to(self.device),
                )
            else:
                raise e

        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()

        answers = {}
        n_tokens = int(b["attention_mask"].sum())

        for r, qid in enumerate(ids):
            q = self._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()

            conf_score = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            elif q["t"] == "score":
                exp_score = float((np.arange(k) * p).sum())
                answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    "action": ext,
                }

        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": n_tokens, "output_tokens": 0},
        }

    predict = system_one


RLAgent = Agent


def load(model_id_or_path: str = "convaiinnovations/laya", device: Optional[str] = None,
         token: Optional[str] = None, subfolder: Optional[str] = None) -> Agent:
    """Load a Laya agent.

    `subfolder` picks one checkpoint out of a repo that bundles several:

        laya.load("convaiinnovations/laya")                           # English (repo root)
        laya.load("convaiinnovations/laya", subfolder="multilingual")
    """
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder)
