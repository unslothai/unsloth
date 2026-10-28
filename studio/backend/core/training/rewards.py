# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GRPO reward library: REWARD.md folders on disk, and the evaluator for rule rewards.

A reward lives in ``rewards/<name>/REWARD.md``: YAML frontmatter (name, kind, description) followed
by a YAML body describing the rule. Rule rewards are pure data scored here, so an imported one runs
no user code. ``kind: python`` rewards are a Python function in the body, run by
``python_rewards`` in a sandboxed worker, never in this process.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import threading
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import yaml

try:
    import re._constants as _sre
    import re._parser as _sre_parse
except ImportError:  # pragma: no cover  (Python < 3.11)
    import sre_constants as _sre
    import sre_parse as _sre_parse

try:  # regex takes a per-call timeout; a user pattern must not stall a GRPO step or the server
    import regex as _rx
except ImportError:  # pragma: no cover
    _rx = None

MAX_REWARD_MD_BYTES = 64 * 1024
MAX_REWARDS_PER_ROOT = 500
MAX_PATTERN_CHARS = 2_000
# regex compiles {n} repeats by unrolling them, nested counts multiplying.
MAX_PATTERN_EXPANSION = 10_000
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
    # account_path refuses a library swapped for a link out of the account's workspace.
    from utils.paths.storage_roots import account_path
    try:
        return account_path(_MANAGED_DIR)
    except ValueError as exc:
        raise RewardError("The reward library is a link outside this account's workspace.") from exc


def _split_markdown(raw: str) -> tuple[dict, dict]:
    if len(raw.encode("utf-8")) > MAX_REWARD_MD_BYTES:
        raise RewardError("REWARD.md exceeds the 64 KB limit.")
    raw = raw.replace("\r\n", "\n")
    if not raw.startswith("---"):
        raise RewardError("REWARD.md must start with YAML frontmatter.")
    parts = raw.split("\n---", 1)
    if len(parts) != 2:
        raise RewardError("REWARD.md YAML frontmatter is not closed.")
    head, body = parts[0][3:], parts[1]
    try:
        meta = yaml.load(head, Loader = _NoAliasLoader) or {}
        # A Python body is code, not YAML.
        if isinstance(meta, dict) and meta.get("kind") == "python":
            return meta, {"code": body.lstrip("-").strip("\n")}
        rule = yaml.load(body.lstrip("-\n"), Loader = _NoAliasLoader) or {}
    except yaml.YAMLError as exc:
        raise RewardError("REWARD.md contains invalid YAML.") from exc
    if not isinstance(meta, dict) or not isinstance(rule, dict):
        raise RewardError("REWARD.md frontmatter and body must both be mappings.")
    return meta, rule


class _NoAliasLoader(yaml.SafeLoader):
    """Aliases (and the merge keys built on them) let a few hundred bytes expand without bound."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.events.AliasEvent):
            raise RewardError("REWARD.md cannot use YAML aliases (*name) or merge keys.")
        return super().compose_node(parent, index)


def _number(value: Any, field: str) -> float:
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        try:
            if math.isfinite(value):
                return float(value)
        except OverflowError:  # an int past the float range
            pass
    raise RewardError(f"{field} must be a finite number.")


def _validate_extract(extract: Any) -> Optional[dict]:
    if extract is None:
        return None
    if not isinstance(extract, dict):
        raise RewardError("extract must be a mapping.")
    # Two steps (or a misspelled one) would score a different text than the rule says.
    if len(extract) != 1 or not ("between" in extract or "regex" in extract):
        raise RewardError("extract takes exactly one of 'between' or 'regex'.")
    if "between" in extract:
        pair = extract["between"]
        if not (
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(p, str) and p for p in pair)
        ):
            raise RewardError("extract.between takes two non-empty strings.")
        return {"between": pair}
    _compile(extract["regex"])
    return {"regex": extract["regex"]}


# Names GRPOTrainer passes to reward functions itself (or never passes); a column cannot use them.
# Names TRL passes to rewards itself, plus dataset schema keys it reinterprets (chosen next to a
# conversational prompt raises "Invalid keys in the example").
_TRL_REWARD_ARGS = frozenset(
    {"prompt", "prompts", "completion", "completions", "completion_ids", "trainer_state"}
    | {"chosen", "rejected", "messages", "label", "image", "images"}
)

# Seconds one user-pattern match may take before it scores as a miss (needs the regex module).
MATCH_TIMEOUT_S = 1.0


def _expanded_size(parsed) -> int:
    total = 0
    for op, av in parsed:
        count = 1
        if op in (_sre.MAX_REPEAT, _sre.MIN_REPEAT, getattr(_sre, "POSSESSIVE_REPEAT", None)):
            low, high, av = av
            count = max(low if high == _sre.MAXREPEAT else high, 1)
        stack, inner = [av], 0
        while stack:
            item = stack.pop()
            if isinstance(item, _sre_parse.SubPattern):
                inner += _expanded_size(item)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        total += count * max(inner, 1)
        if total > MAX_PATTERN_EXPANSION:
            break
    return total


def _check_expansion(pattern: str) -> None:
    try:
        size = _expanded_size(_sre_parse.parse(pattern, re.DOTALL | re.MULTILINE))
    except re.error as exc:
        raise RewardError(f"Invalid regex: {exc}") from exc
    if size > MAX_PATTERN_EXPANSION:
        raise RewardError("Regex repeat counts are too large; use * or + for long runs.")


def _match(
    pattern: str,
    text: str,
    fullmatch: bool = False,
):
    if _rx is None:
        raise RewardError(_NO_REGEX_MODULE)
    _check_expansion(pattern)
    compiled = _rx.compile(pattern, _rx.DOTALL | _rx.MULTILINE)
    try:
        if fullmatch:
            return compiled.fullmatch(text, timeout = MATCH_TIMEOUT_S)
        return compiled.search(text, timeout = MATCH_TIMEOUT_S)
    except TimeoutError:
        return None


_NO_REGEX_MODULE = "Regex rules need the 'regex' package, which bounds each match; it is missing from this install."


def _compile(pattern: Any) -> re.Pattern:
    if _rx is None:
        raise RewardError(_NO_REGEX_MODULE)
    if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_PATTERN_CHARS:
        raise RewardError(f"pattern must be a non-empty string under {MAX_PATTERN_CHARS} chars.")
    _check_expansion(pattern)
    try:
        return re.compile(pattern, re.DOTALL | re.MULTILINE)
    except re.error as exc:
        raise RewardError(f"Invalid regex: {exc}") from exc


def _validate_score(score: Any, keys: Iterable[str]) -> dict:
    if not isinstance(score, dict):
        raise RewardError("score must be a mapping.")
    # A misspelled key would silently score 0.0 for the one it meant.
    unknown = [str(k) for k in score if k not in keys]
    if unknown:
        raise RewardError(f"score.{unknown[0]} is not a score key; use {', '.join(keys)}.")
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
        if rule.get("extract") is not None:
            # Kept only when given, so rules saved without one keep their shape.
            out["extract"] = _validate_extract(rule["extract"])
        out["score"] = _validate_score(rule.get("score", {}), ("match", "miss"))
    elif kind in ("exact_match", "numeric"):
        out["extract"] = _validate_extract(rule.get("extract"))
        column = rule.get("compare_to", "answer")
        if not isinstance(column, str) or not column:
            raise RewardError("compare_to must name a dataset column.")
        if column in _TRL_REWARD_ARGS:
            raise RewardError(
                f"compare_to cannot be '{column}': TRL reserves that name. Map the column as "
                "the answer role and compare to 'answer' instead."
            )
        # GRPO prompt rendering drops tools and writes the prefill column before rewards run.
        if column in ("tools", PREFILL_COLUMN, PROMPT_TURNS_COLUMN, TOOLS_COLUMN):
            raise RewardError(f"compare_to cannot be '{column}': Studio uses that column itself.")
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
            # The error a band is compared with is never negative, so the band would never match.
            if out["bands"][0]["within"] < 0:
                raise RewardError("bands.within can't be negative.")
            out["else"] = _number(rule.get("else", 0.0), "else")
    elif kind == "json_schema":
        out["extract"] = _validate_extract(rule.get("extract"))
        schema = rule.get("schema", {})
        if not isinstance(schema, dict):
            raise RewardError("schema must be a mapping.")
        # Only type and required are enforced; silently dropping others would reward wrong output.
        unsupported = sorted(str(k) for k in schema if k not in ("type", "required"))
        if unsupported:
            raise RewardError(
                f"schema supports only type and required (got {', '.join(unsupported)})."
            )
        schema_type = schema.get("type", "object")
        if not isinstance(schema_type, str) or schema_type not in _JSON_TYPES:
            raise RewardError(f"schema.type must be one of {', '.join(_JSON_TYPES)}.")
        # Strings only: str() of a nested YAML alias expands it exponentially.
        required = schema.get("required", [])
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
        if out["max_chars"] < 0:
            raise RewardError("max_chars can't be negative.")
        out["score"] = _validate_score(rule.get("score", {}), ("over", "under"))
    # Dropped silently, it would score a different text than the rule says.
    for key in ("extract", "reference_extract"):
        if rule.get(key) is not None and key not in out:
            raise RewardError(f"{kind} rules take no {key}.")
    # A misspelled field (normalise) would fall back to its default instead.
    unknown = [str(k) for k in rule if k not in out and k not in ("extract", "reference_extract")]
    if unknown:
        raise RewardError(f"{kind} rules take no {unknown[0]}.")
    return out


def parse_reward_markdown(raw: str, folder_name: Optional[str] = None) -> dict:
    meta, rule = _split_markdown(raw)
    # str() on a YAML alias tree expands it exponentially, so check the type first.
    if not isinstance(meta.get("name") or "", str):
        raise RewardError("name must be a string.")
    name = normalize_reward_name(meta.get("name") or folder_name or "")
    if folder_name and name != folder_name:
        raise RewardError(f"Frontmatter name '{name}' does not match folder '{folder_name}'.")
    kind = meta.get("kind", "rule")
    if kind not in ("rule", "python"):
        raise RewardError("kind must be 'rule' or 'python'.")
    description = meta.get("description", "")
    if not isinstance(description, str):
        raise RewardError("description must be a string.")
    if kind == "python":
        entry = meta.get("entry", "reward")
        return {
            "name": name,
            "kind": kind,
            "description": description.strip(),
            "entry": entry,
            "code": validate_python(rule["code"], entry),
            "rule": None,
        }
    return {
        "name": name,
        "kind": kind,
        "description": description.strip(),
        "rule": validate_rule(rule),
    }


_FENCE_RE = re.compile(r"^```(?:python|py)?[ \t]*\n(.*?)\n```[ \t]*$", re.DOTALL)


def validate_python(code: Any, entry: Any) -> str:
    """Syntax and shape only; the code itself never runs in this process."""
    if not isinstance(entry, str) or not entry.isidentifier():
        raise RewardError("entry must be the name of a Python function.")
    if not isinstance(code, str):
        raise RewardError("A Python reward needs code in the body.")
    fenced = _FENCE_RE.match(code.strip())
    code = (fenced.group(1) if fenced else code).strip("\n")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise RewardError(f"Python syntax error on line {exc.lineno}: {exc.msg}") from exc
    defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry
    ]
    if not defs:
        raise RewardError(f"The code does not define a top-level function named '{entry}'.")
    if isinstance(defs[-1], ast.AsyncFunctionDef):
        raise RewardError(f"'{entry}' is async; rewards are called directly, so use a plain def.")
    return code + "\n"


def render_reward_markdown(spec: dict) -> str:
    kind = spec.get("kind", "rule")
    meta = {"name": spec["name"], "kind": kind, "description": spec.get("description", "")}
    if kind == "python":
        meta["entry"] = spec["entry"]
    head = yaml.safe_dump(meta, sort_keys = False, allow_unicode = True)
    if kind == "python":
        return f"---\n{head}---\n```python\n{spec['code']}```\n"
    body = yaml.safe_dump(spec["rule"], sort_keys = False, allow_unicode = True)
    return f"---\n{head}---\n{body}"


def _roots() -> tuple[tuple[str, Path], ...]:
    try:
        return (("user", _user_root()), ("bundled", _BUNDLED_ROOT))
    except RewardError:
        return (("bundled", _BUNDLED_ROOT),)


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
            record.update(parse_reward_markdown(manifest.read_text(encoding = "utf-8"), child.name))
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
        # Same rule as the listing: never write through a link planted in the library.
        if target.is_symlink() or (target / "REWARD.md").is_symlink():
            raise RewardError(f"'{spec['name']}' is a link; remove it before importing.")
        if (target / "REWARD.md").exists() and not overwrite:
            raise RewardExistsError(f"A reward named '{spec['name']}' already exists.")
        root = target.parent
        if (
            not target.exists()
            and root.is_dir()
            and sum(1 for _ in root.iterdir()) >= MAX_REWARDS_PER_ROOT
        ):
            # The listing reads only the first MAX_REWARDS_PER_ROOT entries, so more would vanish.
            raise RewardError(f"The reward library is full ({MAX_REWARDS_PER_ROOT} rewards).")
        # Re-render from the parsed spec so only validated fields reach disk; defaults it fills in
        # can push a file near the limit over it.
        text = render_reward_markdown(spec)
        if len(text.encode("utf-8")) > MAX_REWARD_MD_BYTES:
            raise RewardError(
                f"REWARD.md is larger than {MAX_REWARD_MD_BYTES // 1024} KB once saved."
            )
        created = not target.exists()
        target.mkdir(parents = True, exist_ok = True)
        # Replace atomically so a failed write never leaves a truncated reward (or an empty folder).
        tmp = target / f".REWARD.md.{os.getpid()}.tmp"
        try:
            tmp.write_text(text, encoding = "utf-8")
            os.replace(tmp, target / "REWARD.md")
        except BaseException:
            tmp.unlink(missing_ok = True)
            if created:
                shutil.rmtree(target, ignore_errors = True)
            raise
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


def _reject_constant(name: str):
    # NaN / Infinity are not JSON (RFC 8259 section 6); json.loads accepts them by default.
    raise ValueError(f"{name} is not valid JSON")


_JSON_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")


def _json_matches(text: str, schema: dict) -> bool:
    # No \s* before the body: with .*? it backtracks quadratically on an unclosed fence, and
    # json.loads skips the whitespace anyway.
    fenced = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    try:
        value = json.loads(fenced.group(1) if fenced else text, parse_constant = _reject_constant)
    except (ValueError, TypeError, RecursionError):
        return False
    kind = schema["type"]
    # bool is an int subclass, so it is excluded from number and integer by hand.
    if kind == "null":
        ok = value is None
    elif kind in ("number", "integer"):
        ok = isinstance(value, int) and not isinstance(value, bool)
        # 1e999 parses to inf without passing through parse_constant.
        ok = ok or (isinstance(value, float) and math.isfinite(value))
        # An int may be past float range, so only a float is asked whether it is whole.
        ok = ok and (kind == "number" or isinstance(value, int) or value.is_integer())
    else:
        ok = isinstance(
            value, {"object": dict, "array": list, "string": str, "boolean": bool}[kind]
        )
    if not ok:
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
        if rule.get("extract"):
            text = _extract(text, rule["extract"])
            if text is None:
                return rule["score"]["miss"]
        if rule["mode"] == "fullmatch":
            # Stripped as a fallback only: a pattern may require boundary whitespace itself.
            hit = _match(rule["pattern"], text, fullmatch = True) or (
                text != text.strip() and _match(rule["pattern"], text.strip(), fullmatch = True)
            )
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
        extracted = _extract(reference, rule["reference_extract"])
        reference = reference if extracted is None else extracted
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


# A rendered prompt's assistant prefill, under a name no dataset column (reward reference) takes.
PREFILL_COLUMN = "_studio_prefill"
# The structured turns of a prompt rendered to text, for Python rewards (make_python_reward_funcs).
PROMPT_TURNS_COLUMN = "_studio_prompt_turns"
# The row's tool schemas, which GRPO rendering drops, for Python rewards.
TOOLS_COLUMN = "_studio_tools"


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
        # A pre-rendered prompt's assistant prefill, which TRL would have restored itself.
        prefills = kwargs.get(PREFILL_COLUMN)
        out = []
        for i, completion in enumerate(completions or []):
            ref = (
                references[i]
                if isinstance(references, (list, tuple)) and i < len(references)
                else None
            )
            text = completion_text(completion)
            if isinstance(prefills, (list, tuple)) and i < len(prefills) and prefills[i]:
                text = prefills[i] + text
            out.append(float(score_rule(rule, text, ref)))
        return out

    reward.__name__ = spec["name"].replace("-", "_")
    return reward


def preview_scores(
    specs: list[dict],
    text: str,
    reference: Any = None,
    row: Optional[dict] = None,
    prompt: Any = None,
    owner: Optional[str] = None,
    python_isolation: str = "required",
    system_prompt: Optional[str] = None,
) -> tuple[list[dict], Optional[dict]]:
    """Scores in ``specs`` order, plus the isolation the Python ones ran under (None if none did).
    A Python reward that fails carries its own ``error``; the rest still score."""
    python = [s for s in specs if s.get("kind") == "python"]
    python_scores: dict = {}
    python_errors: dict = {}
    isolation = None
    if python:
        from core.training.python_rewards import PythonRewardError, preview_python_scores
        if row is None and reference is not None:
            row = {"answer": reference}
        try:
            python_scores, isolation, python_errors = preview_python_scores(
                python,
                text,
                row or {},
                prompt,
                owner = owner,
                mode = python_isolation,
                system_prompt = system_prompt,
            )
        except PythonRewardError as exc:
            # No worker (no sandbox, a busy preview): every Python reward shares the reason.
            python_errors = {s["name"]: str(exc) for s in python}
    out = []
    for s in specs:
        error = None
        if s.get("kind") == "python":
            score = python_scores.get(s["name"])
            error = python_errors.get(s["name"])
        else:
            # Training reads the rule's compare_to column (make_reward_func), so the preview does too.
            column = s["rule"].get("compare_to")
            ref = row[column] if row and column in row else reference
            score = score_rule(s["rule"], text, ref)
        out.append({"name": s["name"], "score": score, "error": error})
    return out, isolation
