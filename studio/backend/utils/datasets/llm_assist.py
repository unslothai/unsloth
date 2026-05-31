# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
LLM-assisted dataset analysis using an ephemeral GGUF helper model.

Complements heuristic-based detection in format_detection.py and
vlm_processing.py.  Only invoked when heuristics are uncertain.

Architecture:
  - Instantiates LlamaCppBackend, loads model, runs completion(s), unloads.
  - Not kept warm — VRAM is freed immediately after use.
  - Gracefully degrades: returns None when unavailable (no binary, OOM, disabled).
"""

import json
import logging
import os
import re
import textwrap
import time
from itertools import islice
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

DEFAULT_HELPER_MODEL_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_HELPER_MODEL_VARIANT = "UD-Q4_K_XL"

README_MAX_CHARS = 1500


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks emitted by some models.

    If the model places its actual answer OUTSIDE the think block, we
    discard the think block and keep the rest.  If the entire response
    is INSIDE a think block (nothing useful outside), we extract and
    return the inner content instead of discarding everything.
    """
    if "<think>" not in text:
        return text

    # Try stripping think blocks — keep content outside them
    stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags = re.DOTALL).strip()
    if stripped:
        return stripped

    # Everything was inside <think> tags — extract the inner content of the last block
    matches = re.findall(r"<think>(.*?)</think>", text, flags = re.DOTALL)
    if matches:
        return matches[-1].strip()

    return text


def precache_helper_gguf():
    """
    Pre-download the helper GGUF to HF cache.

    Called on FastAPI startup in a background thread so subsequent
    ``_run_with_helper()`` calls skip the download and only pay for
    llama-server startup.  No-op if already cached or disabled.
    """
    if os.environ.get("UNSLOTH_HELPER_MODEL_DISABLE", "").strip() in ("1", "true"):
        return

    repo = os.environ.get("UNSLOTH_HELPER_MODEL_REPO", DEFAULT_HELPER_MODEL_REPO)
    variant = os.environ.get(
        "UNSLOTH_HELPER_MODEL_VARIANT", DEFAULT_HELPER_MODEL_VARIANT
    )

    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

        disable_progress_bars()
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

        # Find the GGUF file matching the variant
        api = HfApi()
        files = api.list_repo_files(repo, repo_type = "model")
        gguf_files = [f for f in files if f.endswith(".gguf")]

        # Find all GGUF files matching the variant (may be split into shards)
        variant_lower = variant.lower().replace("-", "_")
        matching = sorted(
            f for f in gguf_files if variant_lower in f.lower().replace("-", "_")
        )

        if matching:
            logger.info(
                f"Pre-caching helper GGUF: {repo}/{matching[0]}"
                + (f" (+{len(matching) - 1} shards)" if len(matching) > 1 else "")
            )
            for target in matching:
                hf_hub_download(repo_id = repo, filename = target)
            logger.info(f"Helper GGUF cached: {len(matching)} file(s)")
        else:
            logger.warning(f"No GGUF matching variant '{variant}' in {repo}")
    except Exception as e:
        logger.warning(f"Failed to pre-cache helper GGUF: {e}")
    finally:
        try:
            enable_progress_bars()
        except Exception as e:
            pass


def _run_with_helper(prompt: str, max_tokens: int = 256) -> Optional[str]:
    """
    Load helper model, run one chat completion, unload.

    Returns the completion text, or None on any failure.
    """
    if os.environ.get("UNSLOTH_HELPER_MODEL_DISABLE", "").strip() in ("1", "true"):
        return None

    repo = os.environ.get("UNSLOTH_HELPER_MODEL_REPO", DEFAULT_HELPER_MODEL_REPO)
    variant = os.environ.get(
        "UNSLOTH_HELPER_MODEL_VARIANT", DEFAULT_HELPER_MODEL_VARIANT
    )

    backend = None
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend()
        logger.info(f"Loading helper model: {repo} ({variant})")

        ok = backend.load_model(
            hf_repo = repo,
            hf_variant = variant,
            model_identifier = f"helper:{repo}:{variant}",
            is_vision = False,
            n_ctx = 2048,
            n_gpu_layers = -1,
        )
        if not ok:
            logger.warning("Helper model failed to start")
            return None

        messages = [{"role": "user", "content": prompt}]
        logger.info(
            "Helper model request: enable_thinking=False (per-request override)"
        )
        cumulative = ""
        for chunk in backend.generate_chat_completion(
            messages = messages,
            temperature = 0.1,
            top_p = 0.9,
            top_k = 20,
            max_tokens = max_tokens,
            repetition_penalty = 1.0,
            enable_thinking = False,  # Always disable thinking for AI Assist
        ):
            if isinstance(chunk, dict):
                continue  # skip metadata events
            cumulative = chunk  # cumulative — last value is full text

        result = cumulative.strip()
        result = _strip_think_tags(result)
        logger.info(f"Helper model response ({len(result)} chars)")
        return result if result else None

    except Exception as e:
        logger.warning(f"Helper model failed: {e}")
        return None

    finally:
        if backend is not None:
            try:
                backend.unload_model()
                logger.info("Helper model unloaded")
            except Exception:
                pass


# ─── Public API ───────────────────────────────────────────────────────


def llm_generate_vlm_instruction(
    column_names: list[str],
    samples: list[dict],
    dataset_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Ask a helper LLM to generate a task-specific VLM instruction.

    Called when heuristic instruction generation returns low confidence
    or falls back to generic.

    Args:
        column_names: Column names in the dataset.
        samples: 3-5 sample rows with text values (images replaced by "<image>").
        dataset_name: Optional HF dataset identifier for context.

    Returns:
        {"instruction": str, "confidence": 0.85} or None.
    """
    # Format samples for the prompt
    formatted = ""
    for i, row in enumerate(samples[:5], 1):
        parts = []
        for col in column_names:
            val = str(row.get(col, ""))[:300]
            parts.append(f"  {col}: {val}")
        formatted += f"Sample {i}:\n" + "\n".join(parts) + "\n\n"

    prompt = (
        "You are a dataset analyst. Given a vision-language dataset, generate ONE "
        "instruction sentence that describes what the model should do with each image.\n\n"
        f"Dataset: {dataset_name or 'unknown'}\n"
        f"Columns: {column_names}\n\n"
        f"{formatted}"
        "Write ONE instruction sentence. Examples:\n"
        '- "Solve the math problem shown in the image and explain your reasoning."\n'
        '- "Transcribe all text visible in this image."\n'
        '- "Answer the question about this image."\n\n'
        "Respond with ONLY the instruction sentence, nothing else."
    )

    result = _run_with_helper(prompt, max_tokens = 100)
    if not result:
        return None

    # Clean up: strip quotes, ensure it's a single sentence
    instruction = result.strip().strip('"').strip("'").strip()
    # Reject obviously bad outputs (too short, too long, or multi-line)
    if len(instruction) < 10 or len(instruction) > 200 or "\n" in instruction:
        logger.warning(f"Helper model returned unusable instruction: {instruction!r}")
        return None

    logger.info(f"LLM-generated instruction: {instruction}")
    return {
        "instruction": instruction,
        "confidence": 0.85,
    }


def llm_classify_columns(
    column_names: list[str],
    samples: list[dict],
) -> Optional[dict[str, str]]:
    """
    Ask a helper LLM to classify dataset columns into roles.

    Called when heuristic column detection fails (returns None).

    Args:
        column_names: Column names in the dataset.
        samples: 3-5 sample rows with values truncated to 200 chars.

    Returns:
        Dict mapping column_name → role ("user"|"assistant"|"system"|"metadata"),
        or None on failure.
    """
    formatted = ""
    for i, row in enumerate(samples[:5], 1):
        parts = []
        for col in column_names:
            val = str(row.get(col, ""))[:200]
            parts.append(f"  {col}: {val}")
        formatted += f"Sample {i}:\n" + "\n".join(parts) + "\n\n"

    prompt = (
        "Classify each column in this dataset into one of these roles:\n"
        "- user: The input/question/prompt from the human\n"
        "- assistant: The expected output/answer/response from the AI\n"
        "- system: Context, persona, or task description\n"
        "- metadata: IDs, scores, labels, timestamps — not part of conversation\n\n"
        f"Columns: {column_names}\n\n"
        f"{formatted}"
        "Respond with ONLY a JSON object mapping column names to roles.\n"
        'Example: {"question": "user", "answer": "assistant", "id": "metadata"}'
    )

    result = _run_with_helper(prompt, max_tokens = 200)
    if not result:
        return None

    # Parse JSON from response (may have markdown fences)
    text = result.strip()
    if text.startswith("```"):
        # Strip markdown code fence
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    try:
        mapping = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        import re

        match = re.search(r"\{[^}]+\}", text)
        if match:
            try:
                mapping = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning(f"Could not parse helper model JSON: {text!r}")
                return None
        else:
            logger.warning(f"No JSON found in helper model response: {text!r}")
            return None

    if not isinstance(mapping, dict):
        return None

    # Validate: all values must be valid roles
    valid_roles = {"user", "assistant", "system", "metadata"}
    cleaned = {}
    for col, role in mapping.items():
        if (
            col in column_names
            and isinstance(role, str)
            and role.lower() in valid_roles
        ):
            cleaned[col] = role.lower()

    if not cleaned:
        return None

    # Must have at least user + assistant
    roles_present = set(cleaned.values())
    if "user" not in roles_present or "assistant" not in roles_present:
        logger.warning(f"Helper model mapping missing user/assistant: {cleaned}")
        return None

    logger.info(f"LLM-classified columns: {cleaned}")
    return cleaned


def llm_generate_dataset_warning(
    issues: list[str],
    dataset_name: Optional[str] = None,
    modality: str = "text",
    column_names: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Ask the helper LLM to turn technical dataset issues into a user-friendly warning.

    Works for all modalities (text, vision, audio).

    Args:
        issues: List of technical issue descriptions found during analysis.
        dataset_name: Optional HF dataset name.
        modality: "text", "vision", or "audio".
        column_names: Optional list of column names for context.

    Returns:
        A human-friendly warning string, or None on failure.
    """
    if not issues:
        return None

    issues_text = "\n".join(f"- {issue}" for issue in issues)
    cols_text = f"\nColumns: {column_names}" if column_names else ""

    prompt = (
        "You are a helpful assistant. A user is trying to fine-tune a model on a dataset.\n"
        "The following issues were found during dataset analysis:\n\n"
        f"{issues_text}\n\n"
        f"Dataset: {dataset_name or 'unknown'}\n"
        f"Modality: {modality}"
        f"{cols_text}\n\n"
        "Write a brief, friendly explanation of what's wrong and what the user can do about it.\n"
        "Keep it under 3 sentences. Be specific about the dataset."
    )

    result = _run_with_helper(prompt, max_tokens = 200)
    if not result:
        return None

    warning = result.strip()
    # Reject obviously bad outputs
    if len(warning) < 10 or len(warning) > 500:
        return None

    logger.info(f"LLM-generated warning: {warning}")
    return warning


# ─── Dataset Conversion Advisor ──────────────────────────────────────


def _parse_json_response(text: str) -> Optional[dict]:
    """Parse JSON from LLM response, handling markdown fences and noise."""
    if not text:
        return None

    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        end = -1 if lines[-1].strip().startswith("```") else len(lines)
        cleaned = "\n".join(lines[1:end]).strip()

    # Try direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Greedy match for outermost {...}
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _generate_with_backend(backend, messages: list[dict], max_tokens: int = 512) -> str:
    """Run one chat completion on an already-loaded backend. Returns raw text."""
    logger.info("Advisor request: enable_thinking=False (per-request override)")
    cumulative = ""
    for chunk in backend.generate_chat_completion(
        messages = messages,
        temperature = 0.1,
        top_p = 0.9,
        top_k = 20,
        max_tokens = max_tokens,
        repetition_penalty = 1.0,
        enable_thinking = False,  # Always disable thinking for AI Assist
    ):
        if isinstance(chunk, dict):
            continue  # skip metadata events
        cumulative = chunk
    result = cumulative.strip()
    result = _strip_think_tags(result)
    return result


def fetch_hf_dataset_card(
    dataset_name: str, hf_token: Optional[str] = None
) -> tuple[Optional[str], Optional[dict]]:
    """
    Fetch HF dataset card (README) and metadata.

    Returns:
        (readme_text, metadata_dict) or (None, None) on failure.
    """
    try:
        from huggingface_hub import DatasetCard

        card = DatasetCard.load(dataset_name, token = hf_token)
        readme = card.text or ""

        # Truncate at sentence boundary
        if len(readme) > README_MAX_CHARS:
            cut = readme[:README_MAX_CHARS].rfind(".")
            if cut > README_MAX_CHARS // 2:
                readme = readme[: cut + 1] + "\n[...truncated]"
            else:
                readme = readme[:README_MAX_CHARS] + "\n[...truncated]"

        # Extract metadata from YAML frontmatter
        metadata = {}
        if card.data:
            for key in (
                "task_categories",
                "task_ids",
                "language",
                "size_categories",
                "tags",
                "license",
                "pretty_name",
            ):
                val = getattr(card.data, key, None)
                if val is not None:
                    metadata[key] = val

        logger.info(
            f"Fetched dataset card: {len(readme)} chars, {len(metadata)} metadata fields"
        )
        return readme, metadata

    except Exception as e:
        logger.warning(f"Could not fetch dataset card for {dataset_name}: {e}")
        return None, None


def _run_multi_pass_advisor(
    columns: list[str],
    samples: list[dict],
    dataset_name: Optional[str] = None,
    dataset_card: Optional[str] = None,
    dataset_metadata: Optional[dict] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Multi-pass LLM analysis: classify → convert → validate.

    Keeps model loaded across all passes. Returns combined result dict or None.
    """
    if os.environ.get("UNSLOTH_HELPER_MODEL_DISABLE", "").strip() in ("1", "true"):
        return None

    repo = os.environ.get("UNSLOTH_HELPER_MODEL_REPO", DEFAULT_HELPER_MODEL_REPO)
    variant = os.environ.get(
        "UNSLOTH_HELPER_MODEL_VARIANT", DEFAULT_HELPER_MODEL_VARIANT
    )

    backend = None
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend()
        logger.info(f"Loading advisor model: {repo} ({variant})")
        t0 = time.monotonic()

        ok = backend.load_model(
            hf_repo = repo,
            hf_variant = variant,
            model_identifier = f"advisor:{repo}:{variant}",
            is_vision = False,
            n_ctx = 2048,
            n_gpu_layers = -1,
        )
        if not ok:
            logger.warning("Advisor model failed to start")
            return None

        logger.info(f"Advisor model loaded in {time.monotonic() - t0:.1f}s")
        # ── Format samples ──
        samples_text = ""
        for i, row in enumerate(samples[:5], 1):
            parts = [f"  {col}: {str(row.get(col, ''))[:200]}" for col in columns]
            samples_text += f"Row {i}:\n" + "\n".join(parts) + "\n"

        metadata_str = (
            json.dumps(dataset_metadata, indent = 2, default = str)[:500]
            if dataset_metadata
            else "N/A"
        )
        card_excerpt = (dataset_card or "")[:1200] or "N/A"

        # ── Target Model Hints ──
        target_hints = ""
        is_gemma_3n = False
        if model_name:
            try:
                from utils.models.model_config import load_model_config

                config = load_model_config(
                    model_name,
                    use_auth = True,
                    token = hf_token,
                    trust_remote_code = False,
                )
                archs = getattr(config, "architectures", [])
                if archs and "Gemma3nForConditionalGeneration" in archs:
                    is_gemma_3n = True
            except Exception:
                is_gemma_3n = "gemma-3n" in model_name.lower()

        if model_type == "audio" and not is_gemma_3n:
            target_hints = (
                "\n\nHINT: The user is training an AUDIO model. The dataset MUST contain "
                "a column with audio files/paths. Ensure one such column is selected "
                "as part of the input."
            )
        elif model_type == "embeddings":
            target_hints = (
                "\n\nHINT: The user is training an EMBEDDING model. These models typically "
                "do not use standard conversational input/output formats but instead use "
                "specific formats like:\n"
                "- Pairs of texts for Semantic Textual Similarity (STS)\n"
                "- Premise, hypothesis, and label for Natural Language Inference (NLI)\n"
                "- Queries and positive/negative documents for information retrieval\n"
                "Ensure the dataset format mapped reflects these specialized tasks."
            )

        # ── Pass 1: Classify ──
        logger.info("Pass 1: Classifying dataset...")
        t1 = time.monotonic()
        messages1 = [
            {
                "role": "system",
                "content": (
                    "You are a dataset analyst. Your job is to look at a HuggingFace dataset "
                    "and figure out what kind of data it contains and whether it is already in "
                    "a conversational format suitable for LLM fine-tuning. A dataset is "
                    '"conversational" if it already has columns like "messages", "conversations", '
                    'or multiturn "user"/"assistant" pairs. Some datasets are NOT conversational '
                    "— they are things like summarization, question answering, translation, "
                    "classification, etc. Those need conversion. You must respond with ONLY a "
                    "valid JSON object. Do not write any explanation before or after the JSON."
                    f"{target_hints}"
                ),
            },
            {
                "role": "user",
                "content": textwrap.dedent(f"""\
                    Look at this HuggingFace dataset and classify it.

                    DATASET CARD (excerpt):
                    {card_excerpt}

                    METADATA:
                    {metadata_str}

                    COLUMNS: {columns}

                    SAMPLE DATA (first 3 rows):
                    {samples_text}

                    Based on the above, respond with this exact JSON structure:
                    {{
                        "dataset_type": "<one of: summarization, question_answering, translation, classification, natural_language_inference, instruction_following, conversational, code_generation, other>",
                        "is_conversational": <true if the dataset already has message/conversation columns, false otherwise>,
                        "needs_conversion": <true if it needs to be converted into user/assistant turns, false if it is already conversational>,
                        "description": "<one sentence describing what this dataset contains>",
                        "task_description": "<one sentence describing the task: what input goes in and what output comes out>"
                    }}

                    Respond with ONLY the JSON object. No markdown, no explanation."""),
            },
        ]
        raw1 = _generate_with_backend(backend, messages1, max_tokens = 256)
        pass1 = _parse_json_response(raw1)
        logger.info(f"Pass 1 done ({time.monotonic() - t1:.1f}s): {pass1}")

        if not pass1:
            logger.warning(f"Advisor Pass 1 failed to produce JSON: {raw1[:200]}")
            return None

        # If dataset is already conversational, skip passes 2-3
        if pass1.get("is_conversational") and not pass1.get("needs_conversion"):
            return {
                "success": True,
                "dataset_type": pass1.get("dataset_type"),
                "is_conversational": True,
                "user_notification": (
                    "This dataset is already in conversational format. "
                    "No conversion needed — columns can be mapped directly."
                ),
            }

        # ── Pass 2: Map columns to roles ──
        logger.info("Pass 2: Mapping columns to roles...")

        t2 = time.monotonic()
        messages2 = [
            {
                "role": "system",
                "content": (
                    "You are a data preparation assistant. Your job is to assign each column "
                    "in a dataset to a conversation role for LLM fine-tuning. There are exactly "
                    "two roles:\n"
                    '- "user" = This column contains INPUT that the model will receive as a prompt.\n'
                    '- "assistant" = This column contains OUTPUT that the model should learn to generate.\n\n'
                    "CRITICAL RULES:\n"
                    '1. There MUST be at least one column assigned to "user" AND at least one '
                    'column assigned to "assistant". Never assign all columns to the same role.\n'
                    "2. The column that contains the TARGET or OUTPUT or ANSWER or LABEL must "
                    'ALWAYS be assigned to "assistant". This is the thing the model should learn '
                    "to produce.\n"
                    "3. The columns that contain the SOURCE or INPUT or CONTEXT or QUESTION must "
                    'be assigned to "user". This is what the model receives.\n'
                    '4. Metadata columns like "id", "index", "source", "url", "date" should be '
                    'set to "skip".\n\n'
                    "You must respond with ONLY a valid JSON object."
                    f"{target_hints}"
                ),
            },
            {
                "role": "user",
                "content": textwrap.dedent(f"""\
                    Here is a dataset that has been classified:

                    CLASSIFICATION:
                    {json.dumps(pass1, indent = 2)}

                    COLUMNS AVAILABLE: {columns}

                    SAMPLE DATA (first 3 rows):
                    {samples_text}

                    Your task: assign each column to either "user", "assistant", or "skip".

                    Here are worked examples to guide you:

                    Example 1 — Summarization dataset with columns ["document", "summary"]:
                      "document" is the input text → "user"
                      "summary" is the output the model should generate → "assistant"
                      Result: {{"document": "user", "summary": "assistant"}}

                    Example 2 — Question answering dataset with columns ["context", "question", "answer"]:
                      "context" is input → "user"
                      "question" is input → "user"
                      "answer" is what the model should generate → "assistant"
                      Result: {{"context": "user", "question": "user", "answer": "assistant"}}

                    Example 3 — Classification dataset with columns ["text", "label"]:
                      "text" is input → "user"
                      "label" is the output the model should predict → "assistant"
                      Result: {{"text": "user", "label": "assistant"}}

                    Example 4 — Translation dataset with columns ["en", "fr"]:
                      "en" is the source language (input) → "user"
                      "fr" is the target language (output) → "assistant"
                      Result: {{"en": "user", "fr": "assistant"}}

                    Now apply this logic to the actual dataset columns listed above.

                    Respond with this exact JSON structure:
                    {{
                        "column_roles": {{
                            "<column_name>": "<user|assistant|skip>"
                        }},
                        "label_mapping": <if any column contains integer labels (like 0, 1, 2), provide a mapping like {{"label": {{"0": "entailment", "1": "neutral", "2": "contradiction"}}}}, otherwise null>,
                        "notes": "<brief explanation of why you assigned roles this way>"
                    }}

                    REMEMBER: There must be at least one "user" column AND at least one "assistant" column. If all columns are "user", you made a mistake — the output/target column should be "assistant".

                    Respond with ONLY the JSON object."""),
            },
        ]
        raw2 = _generate_with_backend(backend, messages2, max_tokens = 512)
        pass2 = _parse_json_response(raw2)
        logger.info(f"Pass 2 done ({time.monotonic() - t2:.1f}s): {pass2}")

        if not pass2:
            logger.warning(f"Advisor Pass 2 failed to produce JSON: {raw2[:200]}")
            return None

        # ── Extract and validate column roles from Pass 2 ──
        column_roles = pass2.get("column_roles", {})
        label_map = pass2.get("label_mapping") or {}  # may be null

        # Validate: must have at least one user AND one assistant
        roles_present = set(column_roles.values())
        if "user" not in roles_present or "assistant" not in roles_present:
            logger.warning(
                f"Pass 2 sanity fail: missing user or assistant role: {column_roles}"
            )
            return None  # triggers fallback to simple classification

        # ── Pass 3: System prompt (non-conversational datasets only) ──
        sys_prompt = ""
        dtype = pass1.get("dataset_type", "unknown")
        is_conv = pass1.get("is_conversational", False)

        if not is_conv:
            logger.info("Pass 3: Generating system prompt...")
            t3 = time.monotonic()

            # Format label mapping info for the prompt
            label_info = ""
            if label_map:
                for col, mapping in label_map.items():
                    if isinstance(mapping, dict) and mapping:
                        pairs = ", ".join(f"{k} = {v}" for k, v in mapping.items())
                        label_info += f"\nLabel mapping for '{col}': {pairs}"

            # Describe the role assignments for context
            user_cols = [c for c, r in column_roles.items() if r == "user"]
            asst_cols = [c for c, r in column_roles.items() if r == "assistant"]
            task_desc = pass1.get("task_description") or pass1.get("description", "")

            messages3 = [
                {
                    "role": "user",
                    "content": textwrap.dedent(f"""\
                        I am building a fine-tuning dataset for an LLM. I need you to write a \
                        system prompt that will be included in every training example to tell \
                        the model what task it is performing.

                        Here is the task information:
                        - Dataset type: {dtype}
                        - Task description: {task_desc}
                        - The USER (input) columns are: {user_cols}
                        - The ASSISTANT (output) columns are: {asst_cols}
                        {label_info}

                        Write a system prompt that:
                        1. Explains what task the model is performing in plain language
                        2. Describes what input it will receive
                        3. Describes what output it should produce
                        4. Is 2-4 sentences long

                        Write ONLY the system prompt text. No quotes, no labels, no explanation around it."""),
                },
            ]
            raw3 = _generate_with_backend(backend, messages3, max_tokens = 256)
            logger.info(
                f"Pass 3 done ({time.monotonic() - t3:.1f}s): {raw3[:200] if raw3 else None}"
            )

            if raw3:
                # Pass 3 returns raw text, not JSON — clean it up
                cleaned = raw3.strip().strip('"').strip("'").strip()
                if len(cleaned) >= 20 and cleaned.lower() not in ("null", "none", ""):
                    sys_prompt = cleaned

        # Build suggested_mapping (column → role, for the frontend dropdowns)
        suggested_mapping = {}
        for col, role in column_roles.items():
            if col in columns and role in ("user", "assistant", "system"):
                suggested_mapping[col] = role

        # Build user notification from Pass 1 classification
        desc = pass1.get("task_description") or pass1.get("description", "")
        note_parts = [f"This is a {dtype} dataset (not conversational)."]
        if desc:
            note_parts.append(desc)
        note_parts.append(
            "Columns have been mapped to conversation roles. You can adjust the mapping if needed."
        )
        user_notification = " ".join(note_parts)

        total_time = time.monotonic() - t0
        logger.info(
            f"Advisor complete ({total_time:.1f}s): type={dtype}, mapping={suggested_mapping}, sys_prompt={bool(sys_prompt)}, label_map={bool(label_map)}"
        )

        return {
            "success": True,
            "suggested_mapping": suggested_mapping,
            "system_prompt": sys_prompt,
            "label_mapping": label_map if label_map else None,
            "dataset_type": dtype,
            "is_conversational": is_conv,
            "user_notification": user_notification,
        }

    except Exception as e:
        logger.warning(f"Advisor multi-pass failed: {e}")
        return None

    finally:
        if backend is not None:
            try:
                backend.unload_model()
                logger.info("Advisor model unloaded")
            except Exception:
                pass


def llm_conversion_advisor(
    column_names: list[str],
    samples: list[dict],
    dataset_name: Optional[str] = None,
    hf_token: Optional[str] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Full conversion advisor: fetch HF card → multi-pass LLM analysis.

    Falls back to simple llm_classify_columns() if the multi-pass advisor fails.

    Returns:
        Dict with keys: success, suggested_mapping, system_prompt, user_template,
        assistant_template, label_mapping, dataset_type, is_conversational,
        user_notification. Or None on complete failure.
    """
    # Fetch HF dataset card if this looks like a HF dataset (has a slash)
    dataset_card = None
    dataset_metadata = None
    if dataset_name and "/" in dataset_name:
        dataset_card, dataset_metadata = fetch_hf_dataset_card(dataset_name, hf_token)

    # Try multi-pass advisor
    result = _run_multi_pass_advisor(
        columns = column_names,
        samples = samples,
        dataset_name = dataset_name,
        dataset_card = dataset_card,
        dataset_metadata = dataset_metadata,
        model_name = model_name,
        model_type = model_type,
        hf_token = hf_token,
    )

    if result and result.get("success"):
        logger.info(f"Conversion advisor succeeded: type={result.get('dataset_type')}")
        return result

    # Fallback: simple column classification
    logger.info("Advisor failed, falling back to simple column classification")
    simple_mapping = llm_classify_columns(column_names, samples)
    if simple_mapping:
        return {
            "success": True,
            "suggested_mapping": {
                col: role
                for col, role in simple_mapping.items()
                if role in ("user", "assistant", "system")
            },
            "dataset_type": None,
            "is_conversational": None,
            "user_notification": None,
        }

    return None
