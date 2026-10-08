# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GRPO reward library: REWARD.md folders on disk, and the evaluator for rule rewards.

A reward lives in ``rewards/<name>/REWARD.md``: YAML frontmatter (name, kind, description) followed
by a YAML body describing the rule. Rule rewards are pure data scored here, so an imported one runs
no user code. ``kind: python`` is reserved for sandboxed code rewards and is refused for now.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import threading
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import yaml

try:  # regex takes a per-call timeout; a user pattern must not stall a GRPO step or the server
    import regex as _rx
except ImportError:  # pragma: no cover
    _rx = None

MAX_REWARD_MD_BYTES = 64 * 1024
MAX_REWARDS_PER_ROOT = 500
MAX_PATTERN_CHARS = 2_000
RULE_TYPES = ("regex", "exact_match", "numeric", "json_schema", "length")
NORMALIZERS = ("strip", "lower", "remove_commas", "collapse_spaces")

_BUNDLED_ROOT = Path(__file__).with_name("bundled_rewards")
_MANAGED_DIR = "rewards"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_LOCK = threading.RLock()


class RewardError(ValueError):
    pass


class RewardNotFoundError(RewardError):
    pass


class RewardExistsError(RewardError):
    pass


def normalize_reward_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", str(name or "")).strip().lower()
    if not _NAME_RE.fullmatch(name):
        raise RewardError(
            "Reward names use lowercase letters, digits and hyphens (max 64 characters)."
        )
    return name


def _user_root() -> Path:
    from utils.paths import workspace_root
    return workspace_root() / _MANAGED_DIR


def _split_markdown(raw: str) -> tuple[dict, dict]:
    if len(raw.encode("utf-8")) > MAX_REWARD_MD_BYTES:
        raise RewardError("REWARD.md exceeds the 64 KB limit.")
    if not raw.startswith("---"):
        raise RewardError("REWARD.md must start with YAML frontmatter.")
    parts = raw.split("\n---", 1)
    if len(parts) != 2:
        raise RewardError("REWARD.md YAML frontmatter is not closed.")
    head, body = parts[0][3:], parts[1]
    try:
        meta = yaml.safe_load(head) or {}
        # A Python body is code, not YAML: refuse it before trying to parse it.
        if isinstance(meta, dict) and meta.get("kind") == "python":
            raise RewardError("Python rewards are not supported yet; only rule rewards can run.")
        rule = yaml.safe_load(body.lstrip("-\n")) or {}
    except yaml.YAMLError as exc:
        raise RewardError("REWARD.md contains invalid YAML.") from exc
    if not isinstance(meta, dict) or not isinstance(rule, dict):
        raise RewardError("REWARD.md frontmatter and body must both be mappings.")
    return meta, rule


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RewardError(f"{field} must be a finite number.")
    return float(value)


def _validate_extract(extract: Any) -> Optional[dict]:
    if extract is None:
        return None
    if not isinstance(extract, dict):
        raise RewardError("extract must be a mapping.")
    if "between" in extract:
        pair = extract["between"]
        if not (
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(p, str) and p for p in pair)
        ):
            raise RewardError("extract.between takes two non-empty strings.")
        return {"between": pair}
    if "regex" in extract:
        _compile(extract["regex"])
        return {"regex": extract["regex"]}
    raise RewardError("extract needs 'between' or 'regex'.")


# Seconds one user-pattern match may take before it scores as a miss (needs the regex module).
MATCH_TIMEOUT_S = 1.0


def _match(
    pattern: str,
    text: str,
    fullmatch: bool = False,
):
    if _rx is None:
        compiled = re.compile(pattern, re.DOTALL | re.MULTILINE)
        return compiled.fullmatch(text) if fullmatch else compiled.search(text)
    compiled = _rx.compile(pattern, _rx.DOTALL | _rx.MULTILINE)
    try:
        if fullmatch:
            return compiled.fullmatch(text, timeout = MATCH_TIMEOUT_S)
        return compiled.search(text, timeout = MATCH_TIMEOUT_S)
    except TimeoutError:
        return None


def _compile(pattern: Any) -> re.Pattern:
    if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_PATTERN_CHARS:
        raise RewardError(f"pattern must be a non-empty string under {MAX_PATTERN_CHARS} chars.")
    try:
        return re.compile(pattern, re.DOTALL | re.MULTILINE)
    except re.error as exc:
        raise RewardError(f"Invalid regex: {exc}") from exc


def _validate_score(score: Any, keys: Iterable[str]) -> dict:
    if not isinstance(score, dict):
        raise RewardError("score must be a mapping.")
    return {k: _number(score.get(k, 0.0), f"score.{k}") for k in keys}


def validate_rule(rule: dict) -> dict:
    """Return a cleaned copy of a rule body, or raise RewardError."""
    kind = rule.get("type")
    if kind not in RULE_TYPES:
        raise RewardError(f"type must be one of {', '.join(RULE_TYPES)}.")
    out: dict[str, Any] = {"type": kind}
    if kind == "regex":
        _compile(rule.get("pattern"))
        out["pattern"] = rule["pattern"]
        mode = rule.get("mode", "fullmatch")
        if mode not in ("fullmatch", "search"):
            raise RewardError("regex mode must be 'fullmatch' or 'search'.")
        out["mode"] = mode
        out["score"] = _validate_score(rule.get("score", {}), ("match", "miss"))
    elif kind in ("exact_match", "numeric"):
        out["extract"] = _validate_extract(rule.get("extract"))
        column = rule.get("compare_to", "answer")
        if not isinstance(column, str) or not column:
            raise RewardError("compare_to must name a dataset column.")
        out["compare_to"] = column
        # GSM8K-style references ("...reasoning #### 72") need their answer pulled out too.
        out["reference_extract"] = _validate_extract(rule.get("reference_extract"))
        out["missing"] = _number(rule.get("missing", 0.0), "missing")
        if kind == "exact_match":
            norms = rule.get("normalize", ["strip"])
            if not isinstance(norms, list) or any(n not in NORMALIZERS for n in norms):
                raise RewardError(f"normalize takes a list of {', '.join(NORMALIZERS)}.")
            out["normalize"] = norms
            out["score"] = _validate_score(rule.get("score", {}), ("match", "miss"))
        else:
            bands = rule.get("bands")
            if not isinstance(bands, list) or not bands:
                raise RewardError("numeric needs a non-empty bands list.")
            if not all(isinstance(b, dict) for b in bands):
                raise RewardError("Each numeric band needs 'within' and 'score'.")
            out["bands"] = sorted(
                (
                    {
                        "within": _number(b.get("within"), "bands.within"),
                        "score": _number(b.get("score"), "bands.score"),
                    }
                    for b in bands
                ),
                key = lambda b: b["within"],
            )
            out["else"] = _number(rule.get("else", 0.0), "else")
    elif kind == "json_schema":
        out["extract"] = _validate_extract(rule.get("extract"))
        schema = rule.get("schema", {})
        if not isinstance(schema, dict):
            raise RewardError("schema must be a mapping.")
        schema_type = schema.get("type", "object")
        if not isinstance(schema_type, str) or schema_type not in _JSON_TYPES:
            raise RewardError(f"schema.type must be one of {', '.join(_JSON_TYPES)}.")
        # Strings only: str() of a nested YAML alias expands it exponentially.
        required = schema.get("required", []) or []
        if (
            not isinstance(required, list)
            or len(required) > 64
            or not all(isinstance(k, str) and len(k) <= 256 for k in required)
        ):
            raise RewardError("schema.required takes up to 64 key names.")
        out["schema"] = {"type": schema_type, "required": list(required)}
        out["score"] = _validate_score(rule.get("score", {}), ("match", "miss"))
    elif kind == "length":
        out["max_chars"] = int(_number(rule.get("max_chars"), "max_chars"))
        out["score"] = _validate_score(rule.get("score", {}), ("over", "under"))
    return out


def parse_reward_markdown(raw: str, folder_name: Optional[str] = None) -> dict:
    meta, rule = _split_markdown(raw)
    name = normalize_reward_name(meta.get("name") or folder_name or "")
    if folder_name and name != folder_name:
        raise RewardError(f"Frontmatter name '{name}' does not match folder '{folder_name}'.")
    kind = meta.get("kind", "rule")
    if kind == "python":
        raise RewardError("Python rewards are not supported yet; only rule rewards can run.")
    if kind != "rule":
        raise RewardError("kind must be 'rule'.")
    description = meta.get("description", "")
    if not isinstance(description, str):
        raise RewardError("description must be a string.")
    return {
        "name": name,
        "kind": kind,
        "description": description.strip(),
        "rule": validate_rule(rule),
    }


def render_reward_markdown(spec: dict) -> str:
    head = yaml.safe_dump(
        {
            "name": spec["name"],
            "kind": spec.get("kind", "rule"),
            "description": spec.get("description", ""),
        },
        sort_keys = False,
        allow_unicode = True,
    )
    body = yaml.safe_dump(spec["rule"], sort_keys = False, allow_unicode = True)
    return f"---\n{head}---\n{body}"


def _roots() -> tuple[tuple[str, Path], ...]:
    return (("user", _user_root()), ("bundled", _BUNDLED_ROOT))


def _read_root(source: str, root: Path) -> list[dict]:
    records: list[dict] = []
    if not root.is_dir():
        return records
    for child in sorted(root.iterdir())[:MAX_REWARDS_PER_ROOT]:
        manifest = child / "REWARD.md"
        if (
            child.is_symlink()
            or not child.is_dir()
            or not manifest.is_file()
            or manifest.is_symlink()
        ):
            continue
        record = {"name": child.name, "source": source, "valid": True, "error": None}
        try:
            record.update(parse_reward_markdown(manifest.read_text("utf-8"), child.name))
        except (RewardError, OSError, UnicodeDecodeError) as exc:
            record.update(
                {"valid": False, "error": str(exc), "kind": "rule", "description": "", "rule": None}
            )
        records.append(record)
    return records


def list_rewards() -> list[dict]:
    """User rewards shadow bundled ones of the same name."""
    with _LOCK:
        seen: set[str] = set()
        out: list[dict] = []
        for source, root in _roots():
            for record in _read_root(source, root):
                record["shadowed"] = record["name"] in seen
                seen.add(record["name"])
                out.append(record)
        return out


def get_reward(name: str) -> dict:
    name = normalize_reward_name(name)
    for record in list_rewards():
        if record["name"] == name and not record["shadowed"]:
            if not record["valid"]:
                raise RewardError(f"Reward '{name}' is invalid: {record['error']}")
            return record
    raise RewardNotFoundError(f"Reward '{name}' was not found.")


def import_reward(raw: str, *, overwrite: bool = False) -> dict:
    spec = parse_reward_markdown(raw)
    target = _user_root() / spec["name"]
    with _LOCK:
        if target.exists() and not overwrite:
            raise RewardExistsError(f"A reward named '{spec['name']}' already exists.")
        target.mkdir(parents = True, exist_ok = True)
        # Re-render from the parsed spec so only validated fields reach disk.
        (target / "REWARD.md").write_text(render_reward_markdown(spec), "utf-8")
    return {**spec, "source": "user", "valid": True, "error": None, "shadowed": False}


def delete_reward(name: str) -> None:
    name = normalize_reward_name(name)
    target = _user_root() / name
    with _LOCK:
        if not (target / "REWARD.md").is_file() or target.is_symlink():
            raise RewardNotFoundError(
                f"No user reward named '{name}'; bundled rewards cannot be deleted."
            )
        shutil.rmtree(target)


def export_reward(name: str) -> str:
    return render_reward_markdown(get_reward(name))


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(
            m.get("content", "")
            if isinstance(m, dict) and isinstance(m.get("content"), str)
            else ""
            for m in completion
        )
    if isinstance(completion, dict):
        content = completion.get("content")
        return content if isinstance(content, str) else ""
    return ""


def _extract(text: str, extract: Optional[dict]) -> Optional[str]:
    if extract is None:
        return text
    if "between" in extract:
        start, end = extract["between"]
        i = text.rfind(start)
        if i < 0:
            return None
        j = text.find(end, i + len(start))
        return text[i + len(start) : j] if j >= 0 else None
    match = _match(extract["regex"], text)
    if not match:
        return None
    return match.group(1) if match.groups() else match.group(0)


def _normalize(value: str, norms: list[str]) -> str:
    for norm in norms:
        if norm == "strip":
            value = value.strip()
        elif norm == "lower":
            value = value.lower()
        elif norm == "remove_commas":
            value = value.replace(",", "")
        elif norm == "collapse_spaces":
            value = " ".join(value.split())
    return value


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


_JSON_TYPES = {"object": dict, "array": list, "string": str, "number": (int, float)}


def _json_matches(text: str, schema: dict) -> bool:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    try:
        value = json.loads(fenced.group(1) if fenced else text)
    except (ValueError, TypeError):
        return False
    expected = _JSON_TYPES[schema["type"]]
    if not isinstance(value, expected) or (schema["type"] == "number" and isinstance(value, bool)):
        return False
    return (
        all(key in value for key in schema["required"])
        if isinstance(value, dict)
        else not schema["required"]
    )


def score_rule(
    rule: dict,
    text: str,
    reference: Any = None,
) -> float:
    kind = rule["type"]
    if kind == "regex":
        if rule["mode"] == "fullmatch":
            hit = _match(rule["pattern"], text.strip(), fullmatch = True)
        else:
            hit = _match(rule["pattern"], text)
        return rule["score"]["match" if hit else "miss"]
    if kind == "length":
        return rule["score"]["over" if len(text) > rule["max_chars"] else "under"]
    if kind == "json_schema":
        part = _extract(text, rule["extract"])
        return rule["score"][
            "match" if part is not None and _json_matches(part, rule["schema"]) else "miss"
        ]
    part = _extract(text, rule["extract"])
    if part is None or reference is None:
        return rule["missing"]
    reference = str(reference)
    if rule.get("reference_extract"):
        # No match means the reference is already the bare answer.
        reference = _extract(reference, rule["reference_extract"]) or reference
    if kind == "exact_match":
        same = _normalize(part, rule["normalize"]) == _normalize(str(reference), rule["normalize"])
        return rule["score"]["match" if same else "miss"]
    guess, truth = _to_float(part), _to_float(reference)
    if guess is None or truth is None:
        return rule["missing"]
    error = abs(guess - truth) if truth == 0 else abs(guess / truth - 1.0)
    for band in rule["bands"]:
        if error <= band["within"] + 1e-12:
            return band["score"]
    return rule["else"]


def make_reward_func(spec: dict) -> Callable[..., list[float]]:
    """A TRL reward function; its __name__ becomes the rewards/<name>/mean log key."""
    rule = spec["rule"]
    column = rule.get("compare_to")

    def reward(
        prompts = None,
        completions = None,
        **kwargs,
    ) -> list[float]:
        references = kwargs.get(column) if column else None
        out = []
        for i, completion in enumerate(completions or []):
            ref = (
                references[i]
                if isinstance(references, (list, tuple)) and i < len(references)
                else None
            )
            out.append(float(score_rule(rule, completion_text(completion), ref)))
        return out

    reward.__name__ = spec["name"].replace("-", "_")
    return reward


def preview_scores(
    specs: list[dict],
    text: str,
    reference: Any = None,
) -> list[dict]:
    return [{"name": s["name"], "score": score_rule(s["rule"], text, reference)} for s in specs]
