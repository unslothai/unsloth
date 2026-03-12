# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
VLM (Vision-Language Model) processing utilities.

This module contains functions for generating smart instructions
for VLM datasets based on content analysis and heuristics.
"""

import re
from itertools import islice


def generate_smart_vlm_instruction(
    dataset,
    text_column = "text",
    image_column = "image",
    dataset_name = None,
):
    """
    Generate smart, context-aware instruction for VLM datasets using heuristics.

    Strategy:
    1. Check for explicit question/instruction columns → use that
    2. Infer from text column name + sample content
    3. Analyze dataset name for task hints
    4. Fall back to generic instruction

    Returns:
        dict: {
            "instruction": str or None,  # None means use column content
            "instruction_type": "explicit" | "inferred" | "generic",
            "uses_dynamic_instruction": bool,  # True if instruction varies per sample
            "confidence": float,  # 0.0 to 1.0
        }
    """
    column_names = set(next(iter(dataset)).keys())
    sample = next(iter(dataset))

    # ===== LEVEL 1: Explicit Instruction Columns =====
    # Check for columns that contain per-sample instructions
    question_columns = ["question", "query", "prompt", "instruction", "user_prompt"]

    for col in question_columns:
        if col in column_names:
            # Check if this column has varied content (not just empty/same)
            sample_content = sample[col]
            if sample_content and str(sample_content).strip():
                return {
                    "instruction": None,  # Signal to use column content
                    "instruction_column": col,
                    "instruction_type": "explicit",
                    "uses_dynamic_instruction": True,
                    "confidence": 1.0,
                }

    # ===== LEVEL 2: Infer from Column Names + Content =====
    text_col_lower = text_column.lower()

    # Sample the text content to detect patterns
    text_sample = str(sample.get(text_column, ""))[:500]  # First 500 chars

    # Task-specific keywords and their instructions
    task_patterns = {
        # OCR / Transcription
        "ocr": {
            "keywords": ["ocr", "transcribe", "transcript"],
            "content_hints": [
                r"[A-Za-z\u0600-\u06FF]{10,}"
            ],  # Long text passages (Latin/Arabic)
            "instruction": "Transcribe all the text shown in this image.",
            "confidence": 0.9,
        },
        # LaTeX / Math
        "latex": {
            "keywords": ["latex", "math", "formula", "equation"],
            "content_hints": [r"\\[a-z]+\{", r"\^", r"_", r"\\frac"],  # LaTeX commands
            "instruction": "Convert this image to LaTeX notation.",
            "confidence": 0.95,
        },
        # Caption / Description
        "caption": {
            "keywords": ["caption", "description", "describe"],
            "content_hints": [],
            "instruction": "Provide a detailed description of this image.",
            "confidence": 0.85,
        },
        # Medical / Radiology
        "medical": {
            "keywords": [
                "medical",
                "radiology",
                "xray",
                "ct",
                "mri",
                "scan",
                "diagnosis",
            ],
            "content_hints": [r"\b(lesion|radiograph|patient|diagnosis|findings)\b"],
            "instruction": "Analyze this medical image and describe the key findings.",
            "confidence": 0.9,
        },
        # Code / Programming
        "code": {
            "keywords": ["code", "program", "function", "algorithm"],
            "content_hints": [r"def |class |function|import |return "],
            "instruction": "Explain what this code visualization shows.",
            "confidence": 0.85,
        },
        # Chart / Graph
        "chart": {
            "keywords": ["chart", "graph", "plot", "visualization", "diagram"],
            "content_hints": [r"\b(axis|legend|bar|line|pie|scatter)\b"],
            "instruction": "Describe this chart or graph, including key data points and trends.",
            "confidence": 0.85,
        },
        # Document / Text Recognition
        "document": {
            "keywords": ["document", "page", "paragraph", "article"],
            "content_hints": [r"\n.*\n.*\n"],  # Multi-line text
            "instruction": "Extract and transcribe the text from this document image.",
            "confidence": 0.85,
        },
    }

    # Check column name matches
    best_match = None
    best_score = 0.0

    for task_name, task_info in task_patterns.items():
        score = 0.0

        # Check column name
        if any(keyword in text_col_lower for keyword in task_info["keywords"]):
            score += 0.5

        # Check dataset name if provided
        if dataset_name and any(
            keyword in dataset_name.lower() for keyword in task_info["keywords"]
        ):
            score += 0.3

        # Check content patterns
        for pattern in task_info["content_hints"]:
            if re.search(pattern, text_sample, re.IGNORECASE):
                score += 0.4
                break

        if score > best_score:
            best_score = score
            best_match = task_info

    if best_match and best_score > 0.5:  # Confidence threshold
        return {
            "instruction": best_match["instruction"],
            "instruction_column": None,
            "instruction_type": "inferred",
            "uses_dynamic_instruction": False,
            "confidence": min(best_score, best_match["confidence"]),
        }

    # ===== LEVEL 3: Analyze Dataset Name =====
    if dataset_name:
        name_lower = dataset_name.lower()

        # Common dataset name patterns
        if "vqa" in name_lower or "question" in name_lower:
            return {
                "instruction": "Answer the question about this image.",
                "instruction_column": None,
                "instruction_type": "inferred",
                "uses_dynamic_instruction": False,
                "confidence": 0.75,
            }

        if "coco" in name_lower or "flickr" in name_lower:
            return {
                "instruction": "Provide a detailed caption for this image.",
                "instruction_column": None,
                "instruction_type": "inferred",
                "uses_dynamic_instruction": False,
                "confidence": 0.75,
            }

    # ===== LEVEL 4: LLM-Assisted Instruction Generation =====
    try:
        from .llm_assist import llm_generate_vlm_instruction

        sample_rows = []
        for s in islice(dataset, 5):
            row = {}
            for col in s:
                val = s[col]
                if hasattr(val, "size") and hasattr(val, "mode"):  # PIL Image
                    row[col] = "<image>"
                elif isinstance(val, list):
                    row[col] = str(val)[:300]
                else:
                    row[col] = str(val)[:300]
            sample_rows.append(row)

        llm_result = llm_generate_vlm_instruction(
            column_names = list(column_names),
            samples = sample_rows,
            dataset_name = dataset_name,
        )
        if llm_result and llm_result.get("instruction"):
            print(
                f"\n[DEBUG] LLM-assisted VLM instruction generated: "
                f"'{llm_result['instruction']}' (confidence={llm_result.get('confidence', 'N/A')})\n",
                flush = True,
            )
            return {
                "instruction": llm_result["instruction"],
                "instruction_column": None,
                "instruction_type": "llm_assisted",
                "uses_dynamic_instruction": False,
                "confidence": llm_result.get("confidence", 0.85),
            }
    except Exception as e:
        import logging

        logging.getLogger(__name__).debug(f"LLM-assisted instruction skipped: {e}")

    # ===== LEVEL 5: Generic Fallback =====
    return {
        "instruction": "Describe this image in detail.",
        "instruction_column": None,
        "instruction_type": "generic",
        "uses_dynamic_instruction": False,
        "confidence": 0.5,
    }
