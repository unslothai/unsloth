# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

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
from itertools import islice
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_HELPER_MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_HELPER_MODEL_VARIANT = "Q8_0"


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
    variant = os.environ.get("UNSLOTH_HELPER_MODEL_VARIANT", DEFAULT_HELPER_MODEL_VARIANT)

    try:
        from huggingface_hub import HfApi, hf_hub_download

        # Find the GGUF file matching the variant
        api = HfApi()
        files = api.list_repo_files(repo, repo_type="model")
        gguf_files = [f for f in files if f.endswith(".gguf")]

        # Find all GGUF files matching the variant (may be split into shards)
        variant_lower = variant.lower().replace("-", "_")
        matching = sorted(
            f for f in gguf_files
            if variant_lower in f.lower().replace("-", "_")
        )

        if matching:
            logger.info(f"Pre-caching helper GGUF: {repo}/{matching[0]}"
                        + (f" (+{len(matching) - 1} shards)" if len(matching) > 1 else ""))
            for target in matching:
                hf_hub_download(repo_id=repo, filename=target)
            logger.info(f"Helper GGUF cached: {len(matching)} file(s)")
        else:
            logger.warning(f"No GGUF matching variant '{variant}' in {repo}")
    except Exception as e:
        logger.warning(f"Failed to pre-cache helper GGUF: {e}")


def _run_with_helper(prompt: str, max_tokens: int = 256) -> Optional[str]:
    """
    Load helper model, run one chat completion, unload.

    Returns the completion text, or None on any failure.
    """
    if os.environ.get("UNSLOTH_HELPER_MODEL_DISABLE", "").strip() in ("1", "true"):
        return None

    repo = os.environ.get("UNSLOTH_HELPER_MODEL_REPO", DEFAULT_HELPER_MODEL_REPO)
    variant = os.environ.get("UNSLOTH_HELPER_MODEL_VARIANT", DEFAULT_HELPER_MODEL_VARIANT)

    backend = None
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend()
        logger.info(f"Loading helper model: {repo} ({variant})")
        print(f"🤖 Loading helper model: {repo} ({variant})...")

        ok = backend.load_model(
            hf_repo=repo,
            hf_variant=variant,
            model_identifier=f"helper:{repo}:{variant}",
            is_vision=False,
            n_ctx=2048,
            n_gpu_layers=-1,
        )
        if not ok:
            logger.warning("Helper model failed to start")
            return None

        messages = [{"role": "user", "content": prompt}]
        cumulative = ""
        for text in backend.generate_chat_completion(
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            top_k=20,
            max_tokens=max_tokens,
            repetition_penalty=1.0,
        ):
            cumulative = text  # cumulative — last value is full text

        result = cumulative.strip()
        logger.info(f"Helper model response ({len(result)} chars)")
        return result if result else None

    except Exception as e:
        logger.warning(f"Helper model failed: {e}")
        return None

    finally:
        if backend is not None:
            try:
                backend.unload_model()
                print("🤖 Helper model unloaded")
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

    result = _run_with_helper(prompt, max_tokens=100)
    if not result:
        return None

    # Clean up: strip quotes, ensure it's a single sentence
    instruction = result.strip().strip('"').strip("'").strip()
    # Reject obviously bad outputs (too short, too long, or multi-line)
    if len(instruction) < 10 or len(instruction) > 200 or "\n" in instruction:
        logger.warning(f"Helper model returned unusable instruction: {instruction!r}")
        return None

    print(f"🤖 LLM-generated instruction: {instruction}")
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

    result = _run_with_helper(prompt, max_tokens=200)
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
        if col in column_names and isinstance(role, str) and role.lower() in valid_roles:
            cleaned[col] = role.lower()

    if not cleaned:
        return None

    # Must have at least user + assistant
    roles_present = set(cleaned.values())
    if "user" not in roles_present or "assistant" not in roles_present:
        logger.warning(f"Helper model mapping missing user/assistant: {cleaned}")
        return None

    print(f"🤖 LLM-classified columns: {cleaned}")
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

    result = _run_with_helper(prompt, max_tokens=200)
    if not result:
        return None

    warning = result.strip()
    # Reject obviously bad outputs
    if len(warning) < 10 or len(warning) > 500:
        return None

    print(f"🤖 LLM-generated warning: {warning}")
    return warning
