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

"""EXL3 quantization configuration (the EXL3 analogue of BitsAndBytesConfig)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

# Sensible defaults. 4.0 bpw decoder / 6-bit head roughly matches the footprint
# and quality that Unsloth users expect from the old 4-bit QLoRA default, while
# being noticeably higher quality thanks to the trellis codebook.
DEFAULT_EXL3_BITS: float = 4.0
DEFAULT_EXL3_HEAD_BITS: int = 6
DEFAULT_EXL3_MTP_BITS: int = 4

QUANT_METHOD_EXL3 = "exl3"

# Named convenience presets mapping human-friendly words to a decoder bitrate.
# Users can pass e.g. ``load_in_exl3 = "3bit"`` or ``Exl3Config(preset="2.5bit")``.
_PRESETS = {
    "2bit": 2.0,
    "2.5bit": 2.5,
    "3bit": 3.0,
    "3.5bit": 3.5,
    "4bit": 4.0,
    "5bit": 5.0,
    "6bit": 6.0,
    "8bit": 8.0,
}


def _coerce_bits(value: Any, *, name: str) -> Optional[float]:
    """Coerce a user-supplied bit value into a float, honoring presets/strings."""
    if value is None:
        return None
    if isinstance(value, bool):
        # ``True`` means "use the default", ``False`` means "not set".
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _PRESETS:
            return _PRESETS[key]
        # Accept plain numeric strings and forms like "3bpw" / "3.25 bpw".
        cleaned = key.replace("bpw", "").replace("bits", "").replace("bit", "").strip()
        try:
            return float(cleaned)
        except ValueError as exc:
            raise ValueError(
                f"Unsloth: could not interpret EXL3 {name} value {value!r}. "
                f"Use a number between 1 and 8 (fractional allowed) or one of "
                f"{sorted(_PRESETS)}."
            ) from exc
    raise ValueError(f"Unsloth: unsupported EXL3 {name} value {value!r} of type {type(value)}.")


@dataclass
class Exl3Config:
    """Describes an EXL3 quantization target.

    :param bits:
        Bits per weight for the transformer decoder layers. May be fractional
        (e.g. ``2.5``). Must lie in ``[1, 8]``.
    :param head_bits:
        Bits per weight for the output / ``lm_head`` layer. Integer in
        ``[1, 8]`` or ``16`` to keep the head unquantized. Higher is usually
        beneficial because the head is quality-sensitive and cheap.
    :param mtp_bits:
        Bits per weight for MTP (multi-token-prediction) layers, when present.
    :param hq:
        Enable high-quality mode, which raises the bitrate of select sensitive
        layers (mainly MoE routers / shared experts).
    :param codebook:
        Trellis codebook: ``"mcg"`` (default), ``"mul1"`` or ``"3inst"``.
    :param parallel_mode:
        Use ExLlamaV3's parallel quantization path for many small tensors
        (helps MoE models with hundreds of expert matrices).
    :param cal_rows / cal_cols:
        Calibration dataset size (rows x tokens-per-row) used during
        quantization. Larger = slower but slightly better.
    :param skip_modules:
        Module name fragments to leave unquantized (e.g. task heads).
    :param compute_dtype:
        Preferred compute dtype for dequantized matmuls / LoRA. String or
        ``torch.dtype``; defaults to bfloat16 where supported.
    :param calibrate:
        Whether to run EXL3's data-driven (calibrated) trellis fit. ``True``
        (default) gives the best quality-per-bit and is what you want for
        inference or merging. ``False`` performs an *uncalibrated* (data-free)
        quantization that skips the calibration forward passes / reference
        state - much cheaper to produce, which is often the right tradeoff for
        **QLoRA**: the base weights are frozen and the LoRA adapters absorb
        quantization error during training, so paying for a high-quality base
        fit you are about to adapt anyway is wasteful. (Note: for MoE models the
        dominant cost is the per-expert trellis search itself, which runs in
        both modes; ``calibrate=False`` mainly saves the calibration pass.)
    """

    bits: float = DEFAULT_EXL3_BITS
    head_bits: int = DEFAULT_EXL3_HEAD_BITS
    mtp_bits: int = DEFAULT_EXL3_MTP_BITS
    hq: bool = False
    codebook: str = "mcg"
    parallel_mode: bool = False
    cal_rows: Optional[int] = None
    cal_cols: Optional[int] = None
    skip_modules: list = field(default_factory = list)
    compute_dtype: Any = None
    calibrate: bool = True
    quant_method: str = QUANT_METHOD_EXL3

    def __post_init__(self):
        bits = _coerce_bits(self.bits, name = "bits")
        self.bits = DEFAULT_EXL3_BITS if bits is None else bits
        if not (1.0 <= self.bits <= 8.0):
            raise ValueError(f"Unsloth: EXL3 bits must be in [1, 8], got {self.bits}.")

        head = _coerce_bits(self.head_bits, name = "head_bits")
        self.head_bits = DEFAULT_EXL3_HEAD_BITS if head is None else int(round(head))
        if self.head_bits != 16 and not (1 <= self.head_bits <= 8):
            raise ValueError(
                f"Unsloth: EXL3 head_bits must be in [1, 8] or 16, got {self.head_bits}."
            )

        mtp = _coerce_bits(self.mtp_bits, name = "mtp_bits")
        self.mtp_bits = DEFAULT_EXL3_MTP_BITS if mtp is None else int(round(mtp))

        if self.codebook not in ("mcg", "mul1", "3inst"):
            raise ValueError(
                f"Unsloth: EXL3 codebook must be one of 'mcg', 'mul1', '3inst', "
                f"got {self.codebook!r}."
            )
        if self.skip_modules is None:
            self.skip_modules = []
        self.quant_method = QUANT_METHOD_EXL3

    # bitsandbytes-compatible surface (load_in_4bit / load_in_8bit probes).
    @property
    def load_in_4bit(self) -> bool:
        return False

    @property
    def load_in_8bit(self) -> bool:
        return False

    def get_loading_attributes(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # Represent compute_dtype as a short string for JSON friendliness.
        cd = d.get("compute_dtype")
        if cd is not None and not isinstance(cd, str):
            d["compute_dtype"] = str(cd).replace("torch.", "")
        return d

    def to_diff_dict(self) -> dict:
        return self.to_dict()

    def to_json_string(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent = 2)

    @classmethod
    def from_dict(cls, data: dict) -> "Exl3Config":
        data = dict(data or {})
        data.pop("quant_method", None)
        # Tolerate unknown keys from newer checkpoints.
        allowed = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    def label(self) -> str:
        """Short human/dir-friendly label, e.g. ``4.0bpw_H6`` or ``2.5bpw_H4_hq``."""
        bits = f"{self.bits:g}"
        parts = [f"{bits}bpw", f"H{self.head_bits}"]
        if self.hq:
            parts.append("hq")
        return "_".join(parts)


def normalize_exl3_config(config: Any, *, default_bits: float = DEFAULT_EXL3_BITS) -> Exl3Config:
    """Turn any user input into a concrete :class:`Exl3Config`.

    Accepts:

    * ``None`` / ``True``            -> default config
    * an :class:`Exl3Config`         -> returned as-is
    * a number / preset string       -> config with that decoder bitrate
    * a dict                         -> constructed via ``from_dict``
    """
    if isinstance(config, Exl3Config):
        return config
    if config is None or config is True:
        return Exl3Config(bits = default_bits)
    if isinstance(config, dict):
        return Exl3Config.from_dict(config)
    if isinstance(config, (int, float, str)):
        return Exl3Config(bits = config)
    raise ValueError(
        f"Unsloth: cannot build an Exl3Config from {config!r} " f"of type {type(config)}."
    )
