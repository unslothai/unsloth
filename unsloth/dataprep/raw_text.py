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
    '.txt': 'plain_text',
    '.md': 'markdown', 
    '.json': 'json_lines',
    '.jsonl': 'json_lines',
    '.csv': 'csv_text_column'
}

class RawTextDataLoader:
    def __init__(self, tokenizer, chunk_size=2048, stride=512):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size  
        self.stride = stride
    
    def detect_format(self, file_path):
        """Auto-detect file format and parse accordingly"""
        extension = Path(file_path).suffix.lower()
        return SUPPORTED_FORMATS.get(extension, 'plain_text')

    def load_from_file(self, file_path):
        """Load raw text and convert to dataset"""
        file_format = self.detect_format(file_path)
        text_content = self._read_file_by_format(file_path, file_format)
        chunks = self.smart_chunk_text(text_content, self.chunk_size, self.stride)
        return self.create_causal_dataset(chunks)
        
    def load_from_files(self, file_paths):
        """Load multiple text files"""
        all_chunks = []
        for file_path in file_paths:
            file_format = self.detect_format(file_path)
            text_content = self._read_file_by_format(file_path, file_format)
            chunks = self.smart_chunk_text(text_content, self.chunk_size, self.stride)
            all_chunks.extend(chunks)
        return self.create_causal_dataset(all_chunks)

        
    def chunk_text(self, text):
        """Split text into overlapping chunks"""
        
    def create_causal_dataset(self, chunks):
        """Create dataset for causal language modeling"""

    def smart_chunk_text(self, text, chunk_size, stride):
        """
        Intelligent chunking that:
        1. Respects sentence/paragraph boundaries
        2. Handles various text formats (.txt, .md, .json, etc.)
        3. Maintains context with stride overlap
        4. Adds proper EOS tokens
        """
    
    def tokenize_and_chunk(self, text):
        """
        Tokenize first, then chunk by token count:
        1. More precise length control
        2. Avoids mid-token splits
        3. Handles different languages better
        """

    def _read_file_by_format(self, file_path, file_format):
        """Read file content based on detected format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_format == 'plain_text' or file_format == 'markdown':
                return f.read()
            elif file_format == 'json_lines':
                lines = []
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = self._extract_text_from_json(data)
                        if text:
                            lines.append(text)
                    except json.JSONDecodeError:
                        continue
                return '\n\n'.join(lines)
            elif file_format == 'csv_text_column':
                reader = csv.DictReader(f)
                texts = []
                for row in reader:
                    text = self._extract_text_from_csv_row(row)
                    if text:
                        texts.append(text)
                return '\n\n'.join(texts)
        return ""
    
    def _extract_text_from_json(self, data):
        """Extract text from JSON object using common field names."""
        text_fields = ['text', 'content', 'message', 'body', 'description', 'prompt']
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                return data[field]
        return ""
    
    def _extract_text_from_csv_row(self, row):
        """Extract text from CSV row using common column names."""
        text_columns = ['text', 'content', 'message', 'body', 'description', 'prompt']
        for column in text_columns:
            if column in row and row[column]:
                return row[column]
        return ""

class TextPreprocessor:
    def clean_text(self, text):
        """Remove unwanted characters, normalize whitespace"""
        
    def extract_sections(self, text, patterns):
        """Extract specific sections (e.g., code blocks, quotes)"""
        
    def add_structure_tokens(self, text):
        """Add special tokens for structure (chapters, sections)"""
    
    def validate_dataset(self, dataset):
        """
        Check for:
        - Minimum/maximum sequence lengths
        - Character encoding issues
        - Repeated content
        - Empty chunks
        """

def validate_dataset(self, dataset):
    """
    Check for:
    - Minimum/maximum sequence lengths
    - Character encoding issues
    - Repeated content
    - Empty chunks
    """

