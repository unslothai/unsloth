from __future__ import annotations

import json
import logging
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OXC_VALIDATION_FN_MARKER = "unsloth_oxc_validator"

_OXC_LANG_TO_NODE_LANG = {
    "javascript": "js",
    "typescript": "ts",
    "jsx": "jsx",
    "tsx": "tsx",
}

_OXC_TOOL_DIR = Path(__file__).resolve().parent / "oxc-validator"
_OXC_RUNNER_PATH = _OXC_TOOL_DIR / "validate.mjs"


@dataclass(frozen=True)
class OxcLocalCallableValidatorSpec:
    name: str
    drop: bool
    target_columns: list[str]
    batch_size: int
    code_lang: str


def split_oxc_local_callable_validators(
    recipe_core: dict[str, Any],
) -> tuple[dict[str, Any], list[OxcLocalCallableValidatorSpec]]:
    columns = recipe_core.get("columns")
    if not isinstance(columns, list):
        return recipe_core, []

    sanitized = deepcopy(recipe_core)
    sanitized_columns = sanitized.get("columns")
    if not isinstance(sanitized_columns, list):
        return sanitized, []

    llm_code_lang_by_name = _extract_llm_code_lang_by_name(sanitized_columns)
    kept_columns: list[Any] = []
    oxc_specs: list[OxcLocalCallableValidatorSpec] = []

    for column in sanitized_columns:
        if not isinstance(column, dict):
            kept_columns.append(column)
            continue

        maybe_spec = _parse_oxc_spec(
            column=column,
            llm_code_lang_by_name=llm_code_lang_by_name,
        )
        if maybe_spec is None:
            kept_columns.append(column)
            continue
        oxc_specs.append(maybe_spec)

    sanitized["columns"] = kept_columns
    return sanitized, oxc_specs


def register_oxc_local_callable_validators(
    *,
    builder,
    specs: list[OxcLocalCallableValidatorSpec],
) -> None:
    if not specs:
        return

    from data_designer.config.column_configs import ValidationColumnConfig
    from data_designer.config.validator_params import (
        LocalCallableValidatorParams,
        ValidatorType,
    )

    for spec in specs:
        validation_function = _build_oxc_validation_function(spec.code_lang)
        builder.add_column(
            ValidationColumnConfig(
                name=spec.name,
                drop=spec.drop,
                target_columns=spec.target_columns,
                validator_type=ValidatorType.LOCAL_CALLABLE,
                validator_params=LocalCallableValidatorParams(
                    validation_function=validation_function,
                ),
                batch_size=spec.batch_size,
            )
        )


def _parse_oxc_spec(
    *,
    column: dict[str, Any],
    llm_code_lang_by_name: dict[str, str],
) -> OxcLocalCallableValidatorSpec | None:
    if str(column.get("column_type") or "").strip() != "validation":
        return None
    if str(column.get("validator_type") or "").strip() != "local_callable":
        return None

    params = column.get("validator_params")
    if not isinstance(params, dict):
        return None

    fn_raw = params.get("validation_function")
    fn_name = fn_raw.strip() if isinstance(fn_raw, str) else ""
    if not fn_name.startswith(OXC_VALIDATION_FN_MARKER):
        return None

    name = str(column.get("name") or "").strip()
    if not name:
        return None

    target_columns_raw = column.get("target_columns")
    target_columns = (
        [value.strip() for value in target_columns_raw if isinstance(value, str) and value.strip()]
        if isinstance(target_columns_raw, list)
        else []
    )
    if not target_columns:
        return None

    code_lang = _resolve_oxc_lang(
        fn_name=fn_name,
        target_columns=target_columns,
        llm_code_lang_by_name=llm_code_lang_by_name,
    )
    batch_size = _parse_batch_size(column.get("batch_size"))
    drop = bool(column.get("drop") is True)

    return OxcLocalCallableValidatorSpec(
        name=name,
        drop=drop,
        target_columns=target_columns,
        batch_size=batch_size,
        code_lang=code_lang,
    )


def _parse_batch_size(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10
    return parsed if parsed >= 1 else 10


def _extract_llm_code_lang_by_name(columns: list[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for column in columns:
        if not isinstance(column, dict):
            continue
        if str(column.get("column_type") or "").strip() != "llm-code":
            continue
        name = str(column.get("name") or "").strip()
        code_lang = str(column.get("code_lang") or "").strip()
        if name and code_lang:
            out[name] = code_lang
    return out


def _resolve_oxc_lang(
    *,
    fn_name: str,
    target_columns: list[str],
    llm_code_lang_by_name: dict[str, str],
) -> str:
    _, _, marker_lang = fn_name.partition(":")
    marker_lang = marker_lang.strip()
    if marker_lang in _OXC_LANG_TO_NODE_LANG:
        return marker_lang

    first_target = target_columns[0]
    target_lang = llm_code_lang_by_name.get(first_target, "").strip()
    if target_lang in _OXC_LANG_TO_NODE_LANG:
        return target_lang
    return "javascript"


@lru_cache(maxsize=8)
def _build_oxc_validation_function(lang: str):
    node_lang = _OXC_LANG_TO_NODE_LANG.get(lang, "js")

    def _validator(df):
        import pandas as pd  # imported lazily for local callable runtime

        row_count = int(len(df.index))
        if row_count == 0:
            return pd.DataFrame({"is_valid": []})

        code_column = str(df.columns[0]) if len(df.columns) > 0 else ""
        code_values = (
            ["" for _ in range(row_count)]
            if not code_column
            else ["" if value is None else str(value) for value in df[code_column].tolist()]
        )

        results = _run_oxc_batch(node_lang=node_lang, code_values=code_values)
        if len(results) != row_count:
            results = _fallback_results(
                row_count,
                "OXC validator returned mismatched result size.",
            )
        return pd.DataFrame(results)

    _validator.__name__ = f"{OXC_VALIDATION_FN_MARKER}_{node_lang}"
    return _validator


def _run_oxc_batch(*, node_lang: str, code_values: list[str]) -> list[dict[str, Any]]:
    if not _OXC_RUNNER_PATH.exists():
        return _fallback_results(
            len(code_values),
            f"OXC runner missing at {_OXC_RUNNER_PATH}",
        )

    payload = {
        "lang": node_lang,
        "codes": code_values,
    }
    try:
        proc = subprocess.run(
            ["node", str(_OXC_RUNNER_PATH)],
            cwd=str(_OXC_TOOL_DIR),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
    except (OSError, ValueError) as exc:
        logger.warning("OXC subprocess launch failed: %s", exc)
        return _fallback_results(len(code_values), f"OXC launch failed: {exc}")

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "unknown error").strip()
        if len(message) > 300:
            message = f"{message[:300]}..."
        return _fallback_results(len(code_values), f"OXC failed: {message}")

    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _fallback_results(len(code_values), "OXC output parse failed.")

    if not isinstance(raw, list):
        return _fallback_results(len(code_values), "OXC output must be an array.")

    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            out.append(
                {
                    "is_valid": False,
                    "error_count": 1,
                    "error_message": "Invalid OXC result entry.",
                    "severity": None,
                    "labels": [],
                    "codeframe": None,
                }
            )
            continue
        is_valid_raw = item.get("is_valid")
        error_count_raw = item.get("error_count")
        message_raw = item.get("error_message")
        severity_raw = item.get("severity")
        labels_raw = item.get("labels")
        codeframe_raw = item.get("codeframe")
        out.append(
            {
                "is_valid": bool(is_valid_raw) if isinstance(is_valid_raw, bool) else False,
                "error_count": int(error_count_raw) if isinstance(error_count_raw, int) else 0,
                "error_message": str(message_raw or ""),
                "severity": str(severity_raw) if isinstance(severity_raw, str) else None,
                "labels": labels_raw if isinstance(labels_raw, list) else [],
                "codeframe": str(codeframe_raw) if isinstance(codeframe_raw, str) else None,
            }
        )
    return out


def _fallback_results(row_count: int, message: str) -> list[dict[str, Any]]:
    return [
        {
            "is_valid": False,
            "error_count": 1,
            "error_message": message,
            "severity": None,
            "labels": [],
            "codeframe": None,
        }
        for _ in range(row_count)
    ]
