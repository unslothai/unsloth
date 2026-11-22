# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VLM dataset utilities that generate instructions from content analysis and heuristics."""

import re
from itertools import islice


def generate_smart_vlm_instruction(
    dataset,
    text_column = "text",
    image_column = "image",
    dataset_name = None,
):
    """selects a VLM instruction from explicit columns, heuristics, an LLM, or the default."""
    sample = next(iter(dataset))
    column_names = set(sample.keys())

    question_columns = [
        "question",
        "query",
        "prompt",
        "instruction",
        "user_prompt",
        "problem",
        "input",
        "inputs",
    ]
    columns = [col for col in sample if col not in (text_column, image_column)]

    for name in question_columns:
        matches = ([name] if name in columns else []) + [
            col for col in columns if col != name and col.lower() == name
        ]
        for col in matches:
            has_text = any(
                isinstance(row.get(col), str) and row[col].strip() for row in islice(dataset, 100)
            )
            if has_text:
                return {
                    "instruction": None,
                    "instruction_column": col,
                    "instruction_type": "explicit",
                    "uses_dynamic_instruction": True,
                    "confidence": 1.0,
                }

    text_col_lower = text_column.lower()

    # Hints read the rows the question scan reads, not row 0: a seeded row bound reorders train before formatting.
    text_rows = dataset
    fmt = getattr(dataset, "format", None)
    # A set_transform / with_transform callback may read the columns a projection drops: keep whole rows then.
    if hasattr(dataset, "select_columns") and not (
        isinstance(fmt, dict) and fmt.get("type") == "custom"
    ):
        try:
            # Only the text column: full rows decode 100 images, about a second per split on Radiology_mini.
            text_rows = dataset.select_columns([text_column])
        except Exception:
            text_rows = dataset
    text_samples = [str(row.get(text_column, ""))[:500] for row in islice(text_rows, 100)]

    # Task-specific keywords and their instructions
    task_patterns = {
        # OCR / Transcription
        "ocr": {
            "keywords": ["ocr", "transcribe", "transcript"],
            "content_hints": [r"[A-Za-z\u0600-\u06FF]{10,}"],
            "instruction": "Transcribe all the text shown in this image.",
            "confidence": 0.9,
        },
        # LaTeX / Math
        "latex": {
            "keywords": ["latex", "math", "formula", "equation"],
            "content_hints": [r"\\[a-z]+\{", r"\^", r"_", r"\\frac"],
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
            "content_hints": [r"\n.*\n.*\n"],
            "instruction": "Extract and transcribe the text from this document image.",
            "confidence": 0.85,
        },
    }

    # Score each task by column/dataset name and content matches
    best_match = None
    best_score = 0.0
    best_hits = 0

    for task_name, task_info in task_patterns.items():
        score = 0.0

        if any(keyword in text_col_lower for keyword in task_info["keywords"]):
            score += 0.5

        if dataset_name and any(
            keyword in dataset_name.lower() for keyword in task_info["keywords"]
        ):
            score += 0.3

        hits = sum(
            any(re.search(pattern, text, re.IGNORECASE) for pattern in task_info["content_hints"])
            for text in text_samples
        )
        if hits:
            score += 0.4

        # A tie goes to the task whose hints match more rows (LaTeX rows can also hold a long word).
        if (score, hits) > (best_score, best_hits):
            best_score, best_hits = score, hits
            best_match = task_info

    if best_match and best_score > 0.5:
        return {
            "instruction": best_match["instruction"],
            "instruction_column": None,
            "instruction_type": "inferred",
            "uses_dynamic_instruction": False,
            "confidence": min(best_score, best_match["confidence"]),
        }

    if dataset_name:
        name_lower = dataset_name.lower()

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

    try:
        from .llm_assist import llm_generate_vlm_instruction

        sample_rows = []
        for s in islice(dataset, 5):
            row = {}
            for col in s:
                val = s[col]
                if hasattr(val, "size") and hasattr(val, "mode"):
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

    return {
        "instruction": "Describe this image in detail.",
        "instruction_column": None,
        "instruction_type": "generic",
        "uses_dynamic_instruction": False,
        "confidence": 0.5,
    }
