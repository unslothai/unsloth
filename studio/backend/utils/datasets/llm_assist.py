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
import re
import textwrap
import time
from itertools import islice
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_HELPER_MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_HELPER_MODEL_VARIANT = "Q8_0"

README_MAX_CHARS = 1500


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


def _generate_with_backend(
    backend, messages: list[dict], max_tokens: int = 512
) -> str:
    """Run one chat completion on an already-loaded backend. Returns raw text."""
    cumulative = ""
    for text in backend.generate_chat_completion(
        messages=messages,
        temperature=0.1,
        top_p=0.9,
        top_k=20,
        max_tokens=max_tokens,
        repetition_penalty=1.0,
    ):
        cumulative = text
    return cumulative.strip()


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

        card = DatasetCard.load(dataset_name, token=hf_token)
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
                "task_categories", "task_ids", "language",
                "size_categories", "tags", "license", "pretty_name",
            ):
                val = getattr(card.data, key, None)
                if val is not None:
                    metadata[key] = val

        logger.info(f"Fetched dataset card: {len(readme)} chars, {len(metadata)} metadata fields")
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
) -> Optional[dict[str, Any]]:
    """
    Multi-pass LLM analysis: classify → convert → validate.

    Keeps model loaded across all passes. Returns combined result dict or None.
    """
    if os.environ.get("UNSLOTH_HELPER_MODEL_DISABLE", "").strip() in ("1", "true"):
        return None

    repo = os.environ.get("UNSLOTH_HELPER_MODEL_REPO", DEFAULT_HELPER_MODEL_REPO)
    variant = os.environ.get("UNSLOTH_HELPER_MODEL_VARIANT", DEFAULT_HELPER_MODEL_VARIANT)

    backend = None
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend()
        print(f"🤖 Loading advisor model: {repo} ({variant})...")
        t0 = time.monotonic()

        ok = backend.load_model(
            hf_repo=repo,
            hf_variant=variant,
            model_identifier=f"advisor:{repo}:{variant}",
            is_vision=False,
            n_ctx=2048,
            n_gpu_layers=-1,
        )
        if not ok:
            logger.warning("Advisor model failed to start")
            return None

        print(f"🤖 Advisor model loaded in {time.monotonic() - t0:.1f}s")

        # ── Format samples ──
        samples_text = ""
        for i, row in enumerate(samples[:5], 1):
            parts = [f"  {col}: {str(row.get(col, ''))[:200]}" for col in columns]
            samples_text += f"Row {i}:\n" + "\n".join(parts) + "\n"

        metadata_str = (
            json.dumps(dataset_metadata, indent=2, default=str)[:500]
            if dataset_metadata else "N/A"
        )
        card_excerpt = (dataset_card or "")[:1200] or "N/A"

        # ── Pass 1: Classify ──
        print("🤖 Pass 1: Classifying dataset...", flush=True)
        t1 = time.monotonic()
        messages1 = [
            {
                "role": "system",
                "content": (
                    "You are a dataset analyst specializing in HuggingFace datasets for LLM fine-tuning. "
                    "You classify datasets and determine if they can be used directly for conversational "
                    "fine-tuning or if they need conversion. Respond with ONLY valid JSON, no explanation."
                ),
            },
            {
                "role": "user",
                "content": textwrap.dedent(f"""\
                    Analyze this HuggingFace dataset and classify it.

                    DATASET CARD (excerpt):
                    {card_excerpt}

                    METADATA:
                    {metadata_str}

                    COLUMNS: {columns}

                    SAMPLE DATA:
                    {samples_text}

                    Respond with a JSON object:
                    {{
                        "dataset_type": "<type like: nli, classification, summarization, qa, translation, etc.>",
                        "is_conversational": <true if already has user/assistant message structure, false otherwise>,
                        "needs_conversion": <true if columns need to be reorganized into conversation format>,
                        "description": "<1-2 sentence description of what this dataset is for>",
                        "task_description": "<what a model fine-tuned on this should do>"
                    }}"""),
            },
        ]
        raw1 = _generate_with_backend(backend, messages1, max_tokens=256)
        pass1 = _parse_json_response(raw1)
        print(f"🤖 Pass 1 done ({time.monotonic() - t1:.1f}s): {pass1}", flush=True)

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

        # ── Pass 2: Conversion strategy ──
        print("🤖 Pass 2: Generating conversion strategy...", flush=True)
        t2 = time.monotonic()
        messages2 = [
            {
                "role": "system",
                "content": (
                    "You are a dataset conversion specialist for LLM fine-tuning. "
                    "You design strategies to convert non-conversational datasets into "
                    "user/assistant conversation format. Respond with ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": textwrap.dedent(f"""\
                    This dataset was classified as:
                    {json.dumps(pass1, indent=2)}

                    COLUMNS: {columns}

                    SAMPLE DATA:
                    {samples_text}

                    Design a conversion strategy to turn this into conversation format for fine-tuning.
                    The strategy should create a system prompt, a user message template, and an assistant message template.

                    For the user template, use {{column_name}} placeholders for column values.
                    For the assistant template, use {{column_name}} placeholders.
                    If a column has integer values that represent categories, provide a label mapping.

                    Respond with a JSON object:
                    {{
                        "system_prompt": "<system prompt for the fine-tuned model>",
                        "user_template": "<template for user message using {{column}} placeholders>",
                        "assistant_template": "<template for assistant message using {{column}} placeholders>",
                        "column_roles": {{
                            "<column_name>": "<role: user|assistant|system|template_var>"
                        }},
                        "label_mapping": {{
                            "<column_name>": {{"0": "<string label>", "1": "<string label>"}}
                        }},
                        "notes": "<any important notes about this conversion>"
                    }}"""),
            },
        ]
        raw2 = _generate_with_backend(backend, messages2, max_tokens=512)
        pass2 = _parse_json_response(raw2)
        print(f"🤖 Pass 2 done ({time.monotonic() - t2:.1f}s): {pass2}", flush=True)

        if not pass2:
            logger.warning(f"Advisor Pass 2 failed to produce JSON: {raw2[:200]}")
            return None

        # ── Pass 3: Validate ──
        # Apply templates to samples for concrete examples
        sys_prompt = pass2.get("system_prompt", "")
        user_tpl = pass2.get("user_template", "")
        asst_tpl = pass2.get("assistant_template", "")
        label_map = pass2.get("label_mapping", {})

        examples_text = ""
        for i, row in enumerate(samples[:2], 1):
            row_vals = {}
            for col in columns:
                val = str(row.get(col, ""))
                # Apply label mapping
                if col in label_map and val in label_map[col]:
                    row_vals[col] = label_map[col][val]
                    row_vals[f"{col}_name"] = label_map[col][val]
                else:
                    row_vals[col] = val
                    row_vals[f"{col}_name"] = val

            try:
                user_msg = user_tpl.format(**row_vals)
            except (KeyError, IndexError, ValueError):
                user_msg = user_tpl
            try:
                asst_msg = asst_tpl.format(**row_vals)
            except (KeyError, IndexError, ValueError):
                asst_msg = asst_tpl

            examples_text += f"Example {i}:\n  System: {sys_prompt}\n  User: {user_msg}\n  Assistant: {asst_msg}\n\n"

        print("🤖 Pass 3: Validating conversion...", flush=True)
        t3 = time.monotonic()
        messages3 = [
            {
                "role": "system",
                "content": (
                    "You are a dataset quality reviewer for LLM fine-tuning. "
                    "Review converted training examples and suggest improvements. "
                    "Respond with ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": textwrap.dedent(f"""\
                    Review these converted training examples. The original dataset is:
                    {pass1.get('dataset_type', 'unknown')} — {pass1.get('description', '')}

                    CONVERTED EXAMPLES:
                    {examples_text}

                    Review the quality and generate a brief user-facing notification.

                    Respond with a JSON object:
                    {{
                        "quality_score": <1-10>,
                        "is_acceptable": <true/false>,
                        "revised_system_prompt": "<improved system prompt if needed, or null>",
                        "user_notification": "<friendly 2-3 sentence message for the training studio UI>"
                    }}"""),
            },
        ]
        raw3 = _generate_with_backend(backend, messages3, max_tokens=512)
        pass3 = _parse_json_response(raw3)
        print(f"🤖 Pass 3 done ({time.monotonic() - t3:.1f}s): {pass3}", flush=True)

        # ── Combine results ──
        final_sys_prompt = sys_prompt
        if pass3 and pass3.get("revised_system_prompt"):
            final_sys_prompt = pass3["revised_system_prompt"]

        # Build suggested_mapping (column → role, for the frontend dropdowns)
        suggested_mapping = {}
        column_roles = pass2.get("column_roles", {})
        for col, role in column_roles.items():
            if col in columns and role in ("user", "assistant", "system"):
                suggested_mapping[col] = role

        # Ensure at least user + assistant in mapping
        if "user" not in set(suggested_mapping.values()):
            # Try to infer from templates
            for col in columns:
                if f"{{{col}}}" in user_tpl and col not in suggested_mapping:
                    suggested_mapping[col] = "user"
                    break
        if "assistant" not in set(suggested_mapping.values()):
            for col in columns:
                if f"{{{col}}}" in asst_tpl and col not in suggested_mapping:
                    suggested_mapping[col] = "assistant"
                    break

        user_notification = None
        if pass3:
            user_notification = pass3.get("user_notification")

        return {
            "success": True,
            "suggested_mapping": suggested_mapping,
            "system_prompt": final_sys_prompt,
            "user_template": user_tpl,
            "assistant_template": asst_tpl,
            "label_mapping": label_map if label_map else None,
            "dataset_type": pass1.get("dataset_type"),
            "is_conversational": pass1.get("is_conversational", False),
            "user_notification": user_notification,
        }

    except Exception as e:
        logger.warning(f"Advisor multi-pass failed: {e}")
        return None

    finally:
        if backend is not None:
            try:
                backend.unload_model()
                print("🤖 Advisor model unloaded")
            except Exception:
                pass


def llm_conversion_advisor(
    column_names: list[str],
    samples: list[dict],
    dataset_name: Optional[str] = None,
    hf_token: Optional[str] = None,
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
        columns=column_names,
        samples=samples,
        dataset_name=dataset_name,
        dataset_card=dataset_card,
        dataset_metadata=dataset_metadata,
    )

    if result and result.get("success"):
        print(f"🤖 Conversion advisor succeeded: type={result.get('dataset_type')}")
        return result

    # Fallback: simple column classification
    logger.info("Advisor failed, falling back to simple column classification")
    simple_mapping = llm_classify_columns(column_names, samples)
    if simple_mapping:
        return {
            "success": True,
            "suggested_mapping": {
                col: role for col, role in simple_mapping.items()
                if role in ("user", "assistant", "system")
            },
            "dataset_type": None,
            "is_conversational": None,
            "user_notification": None,
        }

    return None
