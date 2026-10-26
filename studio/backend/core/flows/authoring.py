# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Schema-first LLM authoring: ask for JSON under a schema, check it, feed the error back, else fall back.

The schema goes to llama-server as ``response_format: json_schema`` (``ctx.llm_json`` retries as plain JSON when the
server refuses it), so names the game knows (props, items, enemies) are enums the model cannot leave. The answer is
checked against the schema here too, since the plain-JSON path is unconstrained, then by ``validate``; a refused
answer is asked again with the reason, at most ``repair_max`` times, and then ``fallback`` decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from core.flows.registry import FlowNodeError

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
}


class AuthoringError(ValueError):
    """A model answer that is valid JSON but not usable; the message is shown to the model on the repair turn."""


@dataclass
class Authored:
    value: Any
    attempts: int
    errors: list[str] = field(default_factory = list)
    fallback: bool = False


def string_enum(names: Iterable[Any], *, allow_empty: bool = False) -> dict:
    """A string limited to ``names``; a plain string when there are none (an empty enum breaks the grammar)."""
    out = list(dict.fromkeys(str(n) for n in names if isinstance(n, str) and n))
    if allow_empty and out:
        out.insert(0, "")
    return {"type": "string", "enum": out} if out else {"type": "string"}


def strict_schema(schema: dict) -> dict:
    """Every object closed (additionalProperties false) with every key required: llama.cpp grammars handle optional
    and open structures worst, so absent values are sent as "", 0 or [] sentinels instead."""
    out = dict(schema)
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: strict_schema(v) for k, v in out["properties"].items()}
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if isinstance(out.get("items"), dict):
        out["items"] = strict_schema(out["items"])
    return out


def schema_depth(schema: dict) -> int:
    """Object nesting levels: a flat object is 1, an array of flat objects inside it is 2."""
    subs = [*(schema.get("properties") or {}).values()]
    if isinstance(schema.get("items"), dict):
        subs.append(schema["items"])
    inner = max((schema_depth(s) for s in subs), default = 0)
    return inner + (1 if schema.get("type") == "object" else 0)


def schema_errors(
    value: Any,
    schema: dict,
    path: str = "the answer",
) -> list[str]:
    """The ways ``value`` breaks ``schema``, for the subset llama.cpp grammars support."""
    errs: list[str] = []
    want = schema.get("type")
    if want:
        py = _TYPES.get(want)
        if py is not None and (
            not isinstance(value, py) or (isinstance(value, bool) and want in ("integer", "number"))
        ):
            return [f"{path} should be {want}"]
    if "enum" in schema and value not in schema["enum"]:
        shown = ", ".join(repr(v) for v in schema["enum"][:12])
        errs.append(f"{path} is {value!r}, not one of {shown}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path} is above {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errs.append(f"{_key(path, key)} is missing")
        for key, v in value.items():
            if key in props:
                errs += schema_errors(v, props[key], _key(path, key))
            elif schema.get("additionalProperties") is False:
                errs.append(f"{_key(path, key)} is not allowed")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path} needs at least {schema['minItems']} entries")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path} has more than {schema['maxItems']} entries")
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                errs += schema_errors(
                    v,
                    schema["items"],
                    f"{path[:-1]}[{i}]`" if path.endswith("`") else f"{path}[{i}]",
                )
    return errs


def _key(path: str, key: str) -> str:
    return f"`{key}`" if path == "the answer" else f"{path[:-1]}.{key}`"


def repair_note(error: str) -> str:
    return f"Your last answer was refused: {error}. Answer again and fix that.\n"


async def constrained_call(
    ctx: Any,
    schema: dict,
    prompt: str,
    validate: Callable[[Any], Any],
    repair_max: int = 2,
    fallback: Optional[Callable[[], Any]] = None,
    *,
    system: str = "",
    model: Optional[dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
    seed: Optional[int] = None,
    defaults: Optional[dict] = None,
) -> Authored:
    """Ask, check and repair up to ``repair_max`` times, then call ``fallback`` (or raise a retryable error).

    ``validate`` returns the cleaned value or raises ``AuthoringError`` / ``ValueError`` with a reason the model can act
    on. Only an unparseable answer is treated as a refusal among model errors; anything else (no model, cancelled)
    propagates, since a fallback cannot fix the setup. ``defaults`` fills keys an object answer left out."""
    errors: list[str] = []
    attempts = 0
    for _ in range(max(0, int(repair_max)) + 1):
        attempts += 1
        try:
            raw = await ctx.llm_json(
                model = model if model is not None else ctx.params.get("model"),
                system = system,
                prompt = prompt + (repair_note(errors[-1]) if errors else ""),
                schema = schema,
                temperature = temperature,
                max_tokens = max_tokens,
                seed = seed,
            )
        except FlowNodeError as exc:
            if "JSON" not in str(exc):
                raise
            errors.append("it was not valid JSON")
            continue
        if defaults and isinstance(raw, dict):
            raw = {**defaults, **raw}
        bad = schema_errors(raw, schema)
        if bad:
            errors.append("; ".join(bad[:4]))
            continue
        try:
            return Authored(validate(raw), attempts, errors)
        except ValueError as exc:
            errors.append(str(exc) or type(exc).__name__)
    if fallback is None:
        raise FlowNodeError(
            errors[-1] if errors else "The model gave no usable answer", retryable = True
        )
    try:
        value = fallback()
    except ValueError as exc:
        raise FlowNodeError(str(exc), retryable = True) from exc
    ctx.log(
        f"Model answer refused {attempts} times ({errors[-1]}); used the built-in fallback",
        "warning",
    )
    return Authored(value, attempts, errors, fallback = True)
