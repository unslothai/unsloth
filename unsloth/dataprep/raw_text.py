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
    def __init__(self, tokenizer, chunk_size = 2048, stride = 512, return_tokenized = True):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride
        self.return_tokenized = return_tokenized

    def detect_format(self, file_path):
        """Auto-detect file format and parse accordingly"""
        extension = Path(file_path).suffix.lower()
        return SUPPORTED_FORMATS.get(extension, "plain_text")

    def load_from_file(self, file_path, return_tokenized = None):
        """Load raw text and convert to dataset"""
        if return_tokenized is None:
            return_tokenized = self.return_tokenized
        file_format = self.detect_format(file_path)
        text_content = self._read_file_by_format(file_path, file_format)
        chunks = self.smart_chunk_text(
            text_content, self.chunk_size, self.stride, return_tokenized
        )
        return self.create_causal_dataset(chunks)

    def load_from_files(self, file_paths, return_tokenized = None):
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
        return self.create_causal_dataset(all_chunks)

    def chunk_text(self, text, return_tokenized = None):
        """Split text into overlapping chunks"""
        if return_tokenized is None:
            return_tokenized = self.return_tokenized
        return self.smart_chunk_text(
            text, self.chunk_size, self.stride, return_tokenized
        )

    def create_causal_dataset(self, chunks):
        """Create dataset for causal language modeling"""
        if chunks and isinstance(chunks[0], dict):
            # If chunks are already tokenized (dict with input_ids, attention_mask)
            # Reorganize the data structure for Dataset.from_dict
            input_ids = [chunk["input_ids"] for chunk in chunks]
            attention_mask = [chunk["attention_mask"] for chunk in chunks]
            return Dataset.from_dict(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
        else:
            # If chunks are text strings (backward compatibility)
            return Dataset.from_dict({"text": chunks})

    def smart_chunk_text(self, text, chunk_size, stride, return_tokenized = True):
        """
        Intelligent chunking that:
        1. Respects sentence/paragraph boundaries
        2. Handles various text formats (.txt, .md, .json, etc.)
        3. Maintains context with stride overlap
        4. Returns tokenized chunks directly (more efficient) or text chunks
        """
        # First pass: tokenize the entire text to get accurate token counts
        tokenized = self.tokenizer(text, return_tensors = "pt", add_special_tokens = False)
        tokens = tokenized["input_ids"]

        # Handle different tokenizer return formats
        if hasattr(tokens, "__len__") and len(tokens) > 0:
            # If it's a nested structure, get the first element
            if hasattr(tokens[0], "__len__"):
                tokens = tokens[0]
        elif isinstance(tokens, int):
            # If tokenizer returns just a count, create a simple range
            tokens = list(range(tokens))

        if len(tokens) <= chunk_size:
            # Text is small enough to fit in one chunk
            if return_tokenized:
                # Add EOS token to the tokens if available
                eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                    tokens = (
                        tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
                    )
                    tokens.append(eos_token_id)

                # Create attention mask
                attention_mask = [1] * len(tokens)
                return [{"input_ids": tokens, "attention_mask": attention_mask}]
            else:
                eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
                return [text + eos_token]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + chunk_size, len(tokens))

            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]

            if return_tokenized:
                # Convert to list if it's a tensor
                chunk_tokens_list = (
                    chunk_tokens.tolist()
                    if hasattr(chunk_tokens, "tolist")
                    else list(chunk_tokens)
                )

                # Add EOS token if it's the last chunk or chunk is complete
                if end_idx == len(tokens) or len(chunk_tokens_list) == chunk_size:
                    eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                    if eos_token_id is not None:
                        chunk_tokens_list.append(eos_token_id)

                # Create attention mask (all tokens are attended to)
                attention_mask = [1] * len(chunk_tokens_list)

                chunks.append(
                    {"input_ids": chunk_tokens_list, "attention_mask": attention_mask}
                )
            else:
                # Decode back to text (backward compatibility)
                chunk_text = self.tokenizer.decode(
                    chunk_tokens, skip_special_tokens = True
                )

                # Add EOS token if it's the last chunk or chunk is complete
                if end_idx == len(tokens) or len(chunk_tokens) == chunk_size:
                    eos_token = (
                        self.tokenizer.eos_token if self.tokenizer.eos_token else ""
                    )
                    chunk_text += eos_token

                chunks.append(chunk_text)

            # Move to next chunk with stride overlap
            if end_idx == len(tokens):
                break
            start_idx += chunk_size - stride

        return chunks

    def _read_file_by_format(self, file_path, file_format):
        """Read file content based on detected format."""
        with open(file_path, "r", encoding = "utf-8") as f:
            if file_format == "plain_text" or file_format == "markdown":
                return f.read()
            elif file_format == "json_lines":
                lines = []
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = self._extract_text_from_json(data)
                        if text:
                            lines.append(text)
                    except json.JSONDecodeError:
                        continue
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

    def _extract_text_from_json(self, data):
        """Extract text from JSON object using common field names."""
        text_fields = ["text", "content", "message", "body", "description", "prompt"]
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                return data[field]
        return ""

    def _extract_text_from_csv_row(self, row):
        """Extract text from CSV row using common column names."""
        text_columns = ["text", "content", "message", "body", "description", "prompt"]
        for column in text_columns:
            if column in row and row[column]:
                return row[column]
        return ""


class TextPreprocessor:
    def clean_text(self, text):
        """Remove unwanted characters, normalize whitespace"""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x20-\x7E\n\t]", "", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_sections(self, text, patterns):
        """Extract specific sections (e.g., code blocks, quotes)"""
        sections = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            sections.extend(matches)
        return sections

    def add_structure_tokens(self, text):
        """Add special tokens for structure (chapters, sections)"""
        text = re.sub(
            r"^# (.+)$", r"<|chapter|>\1<|/chapter|>", text, flags = re.MULTILINE
        )
        text = re.sub(
            r"^## (.+)$", r"<|section|>\1<|/section|>", text, flags = re.MULTILINE
        )
        text = re.sub(
            r"^### (.+)$", r"<|subsection|>\1<|/subsection|>", text, flags = re.MULTILINE
        )
        text = re.sub(
            r"```(\w*)\n(.*?)\n```", r"<|code|\1|>\2<|/code|>", text, flags = re.DOTALL
        )
        return text

    def validate_dataset(self, dataset):
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

        texts = dataset["text"]
        text_lengths = []
        seen_texts = set()

        for i, text in enumerate(texts):
            if not text or len(text.strip()) == 0:
                stats["empty_samples"] += 1
                continue

            # Check for encoding issues
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                stats["encoding_issues"] += 1

            # Calculate lengths
            length = len(text)
            text_lengths.append(length)
            stats["min_length"] = min(stats["min_length"], length)
            stats["max_length"] = max(stats["max_length"], length)

            # Check for repeated content
            text_hash = hash(text.strip())
            if text_hash in seen_texts:
                stats["repeated_content"] += 1
            else:
                seen_texts.add(text_hash)

        # Calculate average length
        if text_lengths:
            stats["avg_length"] = sum(text_lengths) / len(text_lengths)
            stats["min_length"] = (
                stats["min_length"] if stats["min_length"] != float("inf") else 0
            )

        # Generate warnings
        if stats["empty_samples"] > 0:
            stats["warnings"].append(f"Found {stats['empty_samples']} empty samples")

        if stats["repeated_content"] > 0:
            stats["warnings"].append(
                f"Found {stats['repeated_content']} repeated samples"
            )

        if stats["encoding_issues"] > 0:
            stats["warnings"].append(
                f"Found {stats['encoding_issues']} encoding issues"
            )

        if stats["min_length"] < 10:
            stats["warnings"].append("Some samples are very short (< 10 characters)")

        return stats
