# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from typing import Any, Optional

from loggers import get_logger

from hub.utils import download_registry
from hub.utils.hf_tokens import hf_token_arg

logger = get_logger(__name__)

DEFAULT_HELPER_MODEL_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_HELPER_MODEL_VARIANT = "UD-Q4_K_XL"
README_MAX_CHARS = 1500


def _helper_disabled() -> bool:
    return os.environ.get("UNSLOTH_HELPER_MODEL_DISABLE", "").strip().lower() in {
        "1",
        "true",
    }


def _strip_think_tags(text: str) -> str:
    if "<think>" not in text:
        return text
    stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags = re.DOTALL).strip()
    if stripped:
        return stripped
    matches = re.findall(r"<think>(.*?)</think>", text, flags = re.DOTALL)
    return matches[-1].strip() if matches else text


def _parse_json_response(text: str) -> Optional[dict[str, Any]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        end = -1 if lines and lines[-1].strip().startswith("```") else len(lines)
        cleaned = "\n".join(lines[1:end]).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _generate_with_backend(backend, messages: list[dict[str, str]], max_tokens: int) -> str:
    cumulative = ""
    for chunk in backend.generate_chat_completion(
        messages = messages,
        temperature = 0.1,
        top_p = 0.9,
        top_k = 20,
        max_tokens = max_tokens,
        repetition_penalty = 1.0,
        enable_thinking = False,
    ):
        if isinstance(chunk, dict):
            continue
        cumulative = chunk
    return _strip_think_tags(cumulative.strip())


def _fetch_hf_dataset_card(
    dataset_name: str, hf_token: Optional[str]
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    try:
        from huggingface_hub import DatasetCard

        card = DatasetCard.load(dataset_name, token = hf_token_arg(hf_token))
        readme = card.text or ""
        if len(readme) > README_MAX_CHARS:
            cut = readme[:README_MAX_CHARS].rfind(".")
            if cut > README_MAX_CHARS // 2:
                readme = readme[: cut + 1] + "\n[...truncated]"
            else:
                readme = readme[:README_MAX_CHARS] + "\n[...truncated]"
        metadata: dict[str, Any] = {}
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
                value = getattr(card.data, key, None)
                if value is not None:
                    metadata[key] = value
        return readme, metadata
    except Exception as exc:
        logger.warning(
            "Could not fetch dataset card for %s: %s",
            dataset_name,
            download_registry.scrub_secrets(str(exc), hf_token = hf_token),
        )
        return None, None


def _is_gemma_3n(model_name: Optional[str]) -> bool:
    normalized = (model_name or "").lower().replace("_", "-")
    return "gemma-3n" in normalized or "gemma3n" in normalized


def _sample_text(columns: list[str], samples: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for index, row in enumerate(samples[:5], 1):
        parts = [f"  {col}: {str(row.get(col, ''))[:200]}" for col in columns]
        rows.append(f"Row {index}:\n" + "\n".join(parts))
    return "\n".join(rows)


def _target_hints(model_name: Optional[str], model_type: Optional[str]) -> str:
    if model_type == "audio" and not _is_gemma_3n(model_name):
        return (
            "\n\nHINT: The user is training an AUDIO model. The dataset must contain "
            "a column with audio files or paths and one such column should be selected "
            "as part of the input."
        )
    if model_type == "embeddings":
        return (
            "\n\nHINT: The user is training an EMBEDDING model. Prefer dataset formats "
            "such as text pairs for STS, premise/hypothesis/label for NLI, or query "
            "and document columns for retrieval."
        )
    return ""


def _run_multi_pass_advisor(
    *,
    columns: list[str],
    samples: list[dict[str, Any]],
    dataset_name: Optional[str],
    dataset_card: Optional[str],
    dataset_metadata: Optional[dict[str, Any]],
    model_name: Optional[str],
    model_type: Optional[str],
) -> Optional[dict[str, Any]]:
    if _helper_disabled():
        return None

    repo = os.environ.get("UNSLOTH_HELPER_MODEL_REPO", DEFAULT_HELPER_MODEL_REPO)
    variant = os.environ.get("UNSLOTH_HELPER_MODEL_VARIANT", DEFAULT_HELPER_MODEL_VARIANT)
    backend = None
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend()
        started = time.monotonic()
        if not backend.load_model(
            hf_repo = repo,
            hf_variant = variant,
            model_identifier = f"hub-advisor:{repo}:{variant}",
            is_vision = False,
            n_ctx = 2048,
            n_gpu_layers = -1,
        ):
            return None
        logger.info("Hub advisor model loaded in %.1fs", time.monotonic() - started)

        samples_text = _sample_text(columns, samples)
        metadata_text = (
            json.dumps(dataset_metadata, indent = 2, default = str)[:500] if dataset_metadata else "N/A"
        )
        card_excerpt = (dataset_card or "")[:1200] or "N/A"
        hints = _target_hints(model_name, model_type)

        pass1_raw = _generate_with_backend(
            backend,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a dataset analyst. Classify the dataset and respond "
                        "with only a valid JSON object."
                        f"{hints}"
                    ),
                },
                {
                    "role": "user",
                    "content": textwrap.dedent(f"""\
                        Dataset: {dataset_name or "unknown"}

                        DATASET CARD:
                        {card_excerpt}

                        METADATA:
                        {metadata_text}

                        COLUMNS: {columns}

                        SAMPLE DATA:
                        {samples_text}

                        Return this JSON shape:
                        {{
                          "dataset_type": "<summarization|question_answering|translation|classification|natural_language_inference|instruction_following|conversational|code_generation|other>",
                          "is_conversational": <boolean>,
                          "needs_conversion": <boolean>,
                          "description": "<one sentence>",
                          "task_description": "<one sentence>"
                        }}"""),
                },
            ],
            256,
        )
        pass1 = _parse_json_response(pass1_raw)
        if not pass1:
            return None

        if pass1.get("is_conversational") and not pass1.get("needs_conversion"):
            return {
                "success": True,
                "dataset_type": pass1.get("dataset_type"),
                "is_conversational": True,
                "user_notification": (
                    "This dataset is already in conversational format. No conversion is needed."
                ),
            }

        pass2_raw = _generate_with_backend(
            backend,
            [
                {
                    "role": "system",
                    "content": (
                        "Assign each dataset column to user, assistant, or skip for "
                        "LLM fine-tuning. The target/output/answer/label column must be "
                        "assistant. Return only valid JSON."
                        f"{hints}"
                    ),
                },
                {
                    "role": "user",
                    "content": textwrap.dedent(f"""\
                        CLASSIFICATION:
                        {json.dumps(pass1, indent = 2)}

                        COLUMNS: {columns}

                        SAMPLE DATA:
                        {samples_text}

                        Return this JSON shape:
                        {{
                          "column_roles": {{"<column_name>": "<user|assistant|skip>"}},
                          "label_mapping": null,
                          "notes": "<short reason>"
                        }}"""),
                },
            ],
            512,
        )
        pass2 = _parse_json_response(pass2_raw)
        if not pass2:
            return None
        column_roles = pass2.get("column_roles")
        if not isinstance(column_roles, dict):
            return None
        roles_present = set(column_roles.values())
        if "user" not in roles_present or "assistant" not in roles_present:
            return None

        label_mapping = pass2.get("label_mapping") or None
        system_prompt = ""
        if not pass1.get("is_conversational"):
            user_cols = [col for col, role in column_roles.items() if role == "user"]
            assistant_cols = [col for col, role in column_roles.items() if role == "assistant"]
            prompt_raw = _generate_with_backend(
                backend,
                [
                    {
                        "role": "user",
                        "content": textwrap.dedent(f"""\
                            Write a concise system prompt for fine-tuning.

                            Dataset type: {pass1.get("dataset_type", "other")}
                            Task: {pass1.get("task_description") or pass1.get("description") or ""}
                            User input columns: {user_cols}
                            Assistant output columns: {assistant_cols}

                            Write only the system prompt text."""),
                    },
                ],
                256,
            )
            cleaned = prompt_raw.strip().strip('"').strip("'").strip()
            if 20 <= len(cleaned) <= 800 and cleaned.lower() not in {"null", "none"}:
                system_prompt = cleaned

        suggested_mapping = {
            col: role
            for col, role in column_roles.items()
            if col in columns and role in {"user", "assistant", "system"}
        }
        if (
            "user" not in suggested_mapping.values()
            or "assistant" not in suggested_mapping.values()
        ):
            return None

        dtype = str(pass1.get("dataset_type") or "other")
        notification_parts = [f"This is a {dtype} dataset."]
        description = pass1.get("task_description") or pass1.get("description")
        if description:
            notification_parts.append(str(description))
        notification_parts.append("Columns were mapped to conversation roles.")

        return {
            "success": True,
            "suggested_mapping": suggested_mapping,
            "system_prompt": system_prompt,
            "label_mapping": label_mapping if isinstance(label_mapping, dict) else None,
            "dataset_type": dtype,
            "is_conversational": bool(pass1.get("is_conversational")),
            "user_notification": " ".join(notification_parts),
        }
    except Exception as exc:
        logger.warning("Hub advisor failed: %s", exc)
        return None
    finally:
        if backend is not None:
            try:
                backend.unload_model()
            except Exception:
                pass


def _heuristic_mapping(columns: list[str]) -> Optional[dict[str, str]]:
    if not columns:
        return None
    lowered = {col: col.lower().replace("-", "_") for col in columns}
    metadata_terms = ("id", "uuid", "url", "source", "date", "time", "score", "index")
    assistant_terms = (
        "assistant",
        "answer",
        "response",
        "output",
        "completion",
        "target",
        "label",
        "summary",
        "translation",
    )
    user_terms = (
        "user",
        "human",
        "prompt",
        "instruction",
        "input",
        "question",
        "query",
        "context",
        "document",
        "article",
        "problem",
        "text",
    )
    mapping: dict[str, str] = {}
    for col, name in lowered.items():
        if any(term == name or name.endswith(f"_{term}") for term in metadata_terms):
            continue
        if any(term in name for term in assistant_terms):
            mapping[col] = "assistant"
        elif any(term in name for term in user_terms):
            mapping[col] = "user"

    if "assistant" not in mapping.values():
        candidates = [col for col in columns if col not in mapping]
        if candidates:
            mapping[candidates[-1]] = "assistant"
        elif columns:
            mapping[columns[-1]] = "assistant"
    if "user" not in mapping.values():
        for col in columns:
            if mapping.get(col) != "assistant":
                mapping[col] = "user"
                break
    if "user" not in mapping.values() or "assistant" not in mapping.values():
        return None
    return mapping


def llm_conversion_advisor(
    column_names: list[str],
    samples: list[dict[str, Any]],
    dataset_name: Optional[str] = None,
    hf_token: Optional[str] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    dataset_card = None
    dataset_metadata = None
    if dataset_name and "/" in dataset_name:
        dataset_card, dataset_metadata = _fetch_hf_dataset_card(dataset_name, hf_token)

    result = _run_multi_pass_advisor(
        columns = column_names,
        samples = samples,
        dataset_name = dataset_name,
        dataset_card = dataset_card,
        dataset_metadata = dataset_metadata,
        model_name = model_name,
        model_type = model_type,
    )
    if result and result.get("success"):
        return result

    mapping = _heuristic_mapping(column_names)
    if mapping:
        return {
            "success": True,
            "suggested_mapping": mapping,
            "dataset_type": None,
            "is_conversational": None,
            "warning": (
                "The helper model was unavailable, so Hub used column-name heuristics. "
                "Review the suggested mapping before training."
            ),
        }
    return None
