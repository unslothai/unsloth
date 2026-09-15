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
"""Turn a text source into a list of token-id sequences to measure acceptance over."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import torch


def builtin_prompts() -> list[str]:
    """A small bundled prompt set so the harness runs with no downloads."""
    text = (
        resources.files("unsloth.spec_decoding")
        .joinpath("assets", "sample_prompts.txt")
        .read_text("utf-8")
    )
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _read_text_file(path: Path) -> list[str]:
    raw = path.read_text("utf-8").strip("\n")
    if "\n\n" in raw:  # a real blank line between records -> blank-line separated
        chunks = [c.strip() for c in raw.split("\n\n")]
    else:  # one record per line
        chunks = [c.strip() for c in raw.splitlines()]
    return [c for c in chunks if c]


def load_texts(source: str | None) -> list[str]:
    """Resolve ``source`` to a list of strings.

    - ``None`` or ``"builtin"``  -> the bundled prompt set
    - a path to a ``.txt`` file  -> line- or blank-line-separated records
    - ``"hf:<dataset>:<split>:<column>"`` -> a HuggingFace dataset column (needs `datasets`)
    """
    if source in (None, "builtin"):
        return builtin_prompts()

    if source.startswith("hf:"):
        try:
            from datasets import load_dataset  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError("hf: sources need `pip install datasets`") from e
        _, name, split, column = source.split(":", 3)
        ds = load_dataset(name, split = split)
        return [str(x) for x in ds[column]]

    path = Path(source)
    if path.is_file():
        return _read_text_file(path)

    raise ValueError(f"unrecognised data source: {source!r}")


def _to_ids(out) -> torch.Tensor:
    """Coerce a tokenizer / chat-template result to a 1-D LongTensor of token ids.

    Return types differ across versions and call styles: ``apply_chat_template`` returns a
    plain tensor on transformers 4.x but a ``BatchEncoding`` on 5.x, and either may hand
    back a nested list rather than a tensor.
    """
    if hasattr(out, "keys"):  # BatchEncoding / dict
        out = out["input_ids"]
    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out)
    if out.dim() == 2:  # batch of one
        out = out[0]
    if out.dim() != 1:
        raise ValueError(f"expected 1-D token ids, got shape {tuple(out.shape)}")
    return out.long()


def build_sequences(
    texts: list[str],
    tokenizer,
    *,
    max_samples: int = 32,
    max_length: int = 256,
    min_length: int = 8,
    chat_template: bool = False,
) -> list[torch.Tensor]:
    """Tokenise ``texts`` into 1-D LongTensors, filtered by length."""
    seqs: list[torch.Tensor] = []
    for text in texts:
        if len(seqs) >= max_samples:
            break
        if chat_template and getattr(tokenizer, "chat_template", None):
            ids = _to_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt = True,
                    return_tensors = "pt",
                )
            )
        else:
            ids = _to_ids(tokenizer(text, return_tensors = "pt"))
        ids = ids[:max_length]
        if ids.numel() >= min_length:
            seqs.append(ids.long())
    if not seqs:
        raise ValueError(
            f"no sequences survived filtering (min_length={min_length}); "
            "check the data source or lower --min-length"
        )
    return seqs
