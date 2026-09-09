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

import os
import re
import json
import csv
from typing import List, Dict, Any, Union, Optional
from datasets import Dataset
from pathlib import Path

__all__ = [
    "RawTextDataLoader",
    "TextPreprocessor",
]

SUPPORTED_FORMATS = {
    ".txt": "plain_text",
    ".md": "markdown",
    ".json": "json_lines",
    ".jsonl": "json_lines",
    ".csv": "csv_text_column",
}


class RawTextDataLoader:
    def __init__(
        self,
        tokenizer,
        chunk_size = 2048,
        stride = 512,
        return_tokenized = True,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if stride < 0:
            raise ValueError(f"stride must be non-negative, got {stride}")
        if stride >= chunk_size:
            raise ValueError(f"stride ({stride}) must be smaller than chunk_size ({chunk_size})")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride
        self.return_tokenized = return_tokenized

    def detect_format(self, file_path):
        """Auto-detect file format and parse accordingly"""
        extension = Path(file_path).suffix.lower()
        return SUPPORTED_FORMATS.get(extension, "plain_text")

    def load_from_file(
        self,
        file_path,
        return_tokenized = None,
    ):
        """Load raw text and convert to dataset"""
        if return_tokenized is None:
            return_tokenized = self.return_tokenized
        file_format = self.detect_format(file_path)
        text_content = self._read_file_by_format(file_path, file_format)
        if not text_content or not text_content.strip():
            raise ValueError(f"File '{file_path}' is empty or contains only whitespace")
        chunks = self.smart_chunk_text(text_content, self.chunk_size, self.stride, return_tokenized)
        return self.create_causal_dataset(chunks)

    def load_from_files(
        self,
        file_paths,
        return_tokenized = None,
    ):
        """Load multiple text files"""
        if return_tokenized is None:
            return_tokenized = self.return_tokenized
        all_chunks = []
        for file_path in file_paths:
            file_format = self.detect_format(file_path)
            text_content = self._read_file_by_format(file_path, file_format)
            chunks = self.smart_chunk_text(
                text_content, self.chunk_size, self.stride, return_tokenized
            )
            all_chunks.extend(chunks)
        if not all_chunks:
            # All files empty/whitespace: raise like load_from_file instead of create_causal_dataset([])
            # returning a 0-row text-column dataset.
            raise ValueError("All files are empty or contain only whitespace")
        return self.create_causal_dataset(all_chunks)

    def chunk_text(
        self,
        text,
        return_tokenized = None,
    ):
        """Split text into overlapping chunks"""
        if return_tokenized is None:
            return_tokenized = self.return_tokenized
        return self.smart_chunk_text(text, self.chunk_size, self.stride, return_tokenized)

    def create_causal_dataset(self, chunks):
        """Create dataset for causal language modeling"""
        if chunks and isinstance(chunks[0], dict):
            input_ids = [chunk["input_ids"] for chunk in chunks]
            attention_mask = [chunk["attention_mask"] for chunk in chunks]
            # Labels == input_ids for causal LM
            labels = [list(ids) for ids in input_ids]
            return Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        else:
            return Dataset.from_dict({"text": chunks})

    def smart_chunk_text(
        self,
        text,
        chunk_size,
        stride,
        return_tokenized = True,
    ):
        """
        Intelligent chunking that:
        1. Respects sentence/paragraph boundaries
        2. Handles various text formats (.txt, .md, .json, etc.)
        3. Maintains context with stride overlap
        4. Returns tokenized chunks directly (more efficient) or text chunks
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if stride < 0:
            raise ValueError(
                f"stride must be non-negative, got {stride}: a negative stride advances the loop by "
                f"more than chunk_size and silently skips the tokens in between"
            )
        if stride >= chunk_size:
            raise ValueError(
                f"stride ({stride}) must be smaller than chunk_size ({chunk_size}) to progress the chunking loop"
            )

        # Skip empty/whitespace text before tokenizing: BPE/SentencePiece emit real tokens for
        # spaces/newlines, so a len(tokens)==0 check misses it and would yield a degenerate lone-EOS sample.
        if not text or not text.strip():
            return []

        # Tokenize the whole text once for accurate token counts
        tokenized = self.tokenizer(text, return_tensors = "pt", add_special_tokens = False)
        tokens = tokenized["input_ids"]

        # Normalise tokenizer return formats
        if hasattr(tokens, "__len__") and len(tokens) > 0:
            if hasattr(tokens[0], "__len__"):
                tokens = tokens[0]
        elif isinstance(tokens, int):
            # Tokenizer returned a count; build a range
            tokens = list(range(tokens))

        if len(tokens) <= chunk_size:
            # Fits in a single chunk
            if return_tokenized:
                tokens = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
                eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                    tokens.append(eos_token_id)

                attention_mask = [1] * len(tokens)
                return [{"input_ids": tokens, "attention_mask": attention_mask}]
            else:
                eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
                return [text + eos_token]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            if return_tokenized:
                chunk_tokens_list = (
                    chunk_tokens.tolist() if hasattr(chunk_tokens, "tolist") else list(chunk_tokens)
                )

                # Append EOS on the last or a full chunk.
                if end_idx == len(tokens) or len(chunk_tokens_list) == chunk_size:
                    eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                    if eos_token_id is not None:
                        chunk_tokens_list.append(eos_token_id)

                attention_mask = [1] * len(chunk_tokens_list)

                chunks.append({"input_ids": chunk_tokens_list, "attention_mask": attention_mask})
            else:
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens = True)

                # Append EOS on the last or a full chunk.
                if end_idx == len(tokens) or len(chunk_tokens) == chunk_size:
                    eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
                    chunk_text += eos_token

                chunks.append(chunk_text)

            # Advance with stride overlap
            if end_idx == len(tokens):
                break
            start_idx += chunk_size - stride

        return chunks

    def _read_file_by_format(self, file_path, file_format):
        """Read file content based on detected format."""
        # utf-8-sig: Windows tooling (PowerShell Out-File, Excel "CSV UTF-8") prepends a BOM that plain
        # utf-8 keeps as a leading character. Without a BOM it decodes exactly like utf-8.
        with open(file_path, "r", encoding = "utf-8-sig") as f:
            if file_format == "plain_text" or file_format == "markdown":
                return f.read()
            elif file_format == "json_lines":
                if Path(file_path).suffix.lower() == ".json":
                    # A .json file is a single JSON document (commonly a list of records), so parsing it per line drops
                    # the whole file.
                    try:
                        parsed = json.load(f)
                        records = parsed if isinstance(parsed, list) else [parsed]
                    except json.JSONDecodeError:
                        # Some files carry JSON Lines under a .json name.
                        f.seek(0)
                        records = self._iter_json_lines(f)
                else:
                    # A .jsonl file is one JSON value per line: stay streaming so a large file is never held in memory
                    # at once.
                    records = self._iter_json_lines(f)
                lines = []
                for data in records:
                    text = self._extract_text_from_json(data)
                    if text:
                        lines.append(text)
                return "\n\n".join(lines)
            elif file_format == "csv_text_column":
                reader = csv.DictReader(f)
                texts = []
                for row in reader:
                    text = self._extract_text_from_csv_row(row)
                    if text:
                        texts.append(text)
                return "\n\n".join(texts)
        return ""

    _TEXT_FIELDS = ("text", "content", "message", "body", "description", "prompt")
    _TEXT_COLUMNS = _TEXT_FIELDS

    def _iter_json_lines(self, handle):
        """Yield one parsed JSON value per line, skipping blank and malformed lines."""
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def _extract_text_from_json(self, data):
        """Extract text from JSON object using common field names."""
        # Skip non-object lines (str/list/number): `field in data` would be a substring/membership test, not
        # a key lookup, and `data[field]` raises.
        if not isinstance(data, dict):
            return ""
        for field in self._TEXT_FIELDS:
            if field in data and isinstance(data[field], str):
                return data[field]
        return ""

    def _extract_text_from_csv_row(self, row):
        """Extract text from CSV row using common column names."""
        for column in self._TEXT_COLUMNS:
            if column in row and row[column]:
                return row[column]
        return ""


def _iter_column(dataset, column):
    """Yield one column value at a time.

    `dataset[column]` copies the whole column into Python objects, which for token
    ids dominates validate_dataset's peak memory on datasets<4 (4.x returns a lazy
    Column). Dataset.iter() streams Arrow batches instead; anything without it
    (DataFrames, dicts, custom __getitem__) falls back to plain indexing.
    """
    batched = getattr(dataset, "iter", None)
    if callable(batched):
        for batch in batched(batch_size = 256):
            yield from batch[column]
    else:
        yield from dataset[column]


class TextPreprocessor:
    _WHITESPACE_PATTERN = re.compile(r"[^\S\n]+")
    _INVALID_CHARS_PATTERN = re.compile(r"[^\x20-\x7E\n]")
    _MULTIPLE_SPACES_PATTERN = re.compile(r"[ ]{2,}")
    _NEWLINE_SPACES_PATTERN = re.compile(r" *\n *")
    _MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n{3,}")
    _CHAPTER_PATTERN = re.compile(r"^# (.+)$", re.MULTILINE)
    _SECTION_PATTERN = re.compile(r"^## (.+)$", re.MULTILINE)
    _SUBSECTION_PATTERN = re.compile(r"^### (.+)$", re.MULTILINE)
    _CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)

    def clean_text(self, text):
        """Remove unwanted characters, normalize whitespace"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._WHITESPACE_PATTERN.sub(" ", text)
        text = self._INVALID_CHARS_PATTERN.sub("", text)
        text = self._MULTIPLE_SPACES_PATTERN.sub(" ", text)
        text = self._NEWLINE_SPACES_PATTERN.sub("\n", text)
        text = self._MULTIPLE_NEWLINES_PATTERN.sub("\n\n", text)
        return text.strip()

    def extract_sections(self, text, patterns):
        """Extract specific sections (e.g., code blocks, quotes)"""
        sections = []
        for pattern in patterns:
            # Compile pattern on first use and cache? Well, patterns are user-provided, so just use re.findall
            # with compiled flags
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            sections.extend(matches)
        return sections

    def add_structure_tokens(self, text):
        """Add special tokens for structure (chapters, sections)"""
        text = self._CHAPTER_PATTERN.sub(r"<|chapter|>\1<|/chapter|>", text)
        text = self._SECTION_PATTERN.sub(r"<|section|>\1<|/section|>", text)
        text = self._SUBSECTION_PATTERN.sub(r"<|subsection|>\1<|/subsection|>", text)
        text = self._CODE_BLOCK_PATTERN.sub(r"<|code|\1|>\2<|/code|>", text)
        return text

    def validate_dataset(
        self,
        dataset,
        tokenizer = None,
    ):
        """
        Check for:
        - Minimum/maximum sequence lengths
        - Character encoding issues
        - Repeated content
        - Empty chunks
        """
        stats = {
            "total_samples": len(dataset),
            "empty_samples": 0,
            "min_length": float("inf"),
            "max_length": 0,
            "avg_length": 0,
            "repeated_content": 0,
            "encoding_issues": 0,
            "warnings": [],
        }

        # `column_names` is HF Dataset-only; None means a mapping-like (DataFrame, dict, custom
        # __getitem__), which this method has always accepted.
        column_names = getattr(dataset, "column_names", None)
        if column_names is None or "text" in column_names:
            texts = _iter_column(dataset, "text")

        elif "input_ids" in column_names:
            if tokenizer is None:
                raise ValueError(
                    "Dataset has 'input_ids' but no 'text' column; "
                    "pass `tokenizer=` to validate_dataset() to decode it for validation."
                )
            # Generator, not a list: decoded text is consumed once by the loop below.
            texts = (
                tokenizer.decode(ids, skip_special_tokens = True)
                for ids in _iter_column(dataset, "input_ids")
            )

        else:
            raise ValueError("Dataset must have either 'text' or 'input_ids' column")

        text_lengths = []
        seen_texts = set()

        for i, text in enumerate(texts):
            if not text or len(text.strip()) == 0:
                stats["empty_samples"] += 1
                continue

            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                stats["encoding_issues"] += 1

            length = len(text)
            text_lengths.append(length)
            stats["min_length"] = min(stats["min_length"], length)
            stats["max_length"] = max(stats["max_length"], length)

            text_hash = hash(text.strip())
            if text_hash in seen_texts:
                stats["repeated_content"] += 1
            else:
                seen_texts.add(text_hash)

        if text_lengths:
            stats["avg_length"] = sum(text_lengths) / len(text_lengths)

        # No sample had content, so min_length is still its float("inf") seed; report 0 like max_length and
        # avg_length beside it.
        if stats["min_length"] == float("inf"):
            stats["min_length"] = 0

        if stats["empty_samples"] > 0:
            stats["warnings"].append(f"Found {stats['empty_samples']} empty samples")

        if stats["repeated_content"] > 0:
            stats["warnings"].append(f"Found {stats['repeated_content']} repeated samples")

        if stats["encoding_issues"] > 0:
            stats["warnings"].append(f"Found {stats['encoding_issues']} encoding issues")

        # Guard on text_lengths, not on min_length: with nothing measured it is now 0, and zero measured
        # samples is not "some samples are very short".
        if text_lengths and stats["min_length"] < 10:
            stats["warnings"].append("Some samples are very short (< 10 characters)")

        return stats
