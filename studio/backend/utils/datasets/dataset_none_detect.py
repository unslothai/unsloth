"""
dataset_none_detect.py

Detect None/empty content turns in conversation datasets.
Reports findings without modifying data.

Usage:
    from .dataset_none_detect import scan_dataset, print_report
    stats = scan_dataset(dataset)               # auto-detect + scan
    stats = scan_dataset(dataset, fmt="chatml")  # explicit format
    print_report(stats, stats["format"])

Dependencies: only `datasets` (already in studio/unsloth) + stdlib.

Supported formats (via FORMAT_REGISTRY -- auto-scales)
-------------------------------------------------------
Format     Columns / turn keys                What is checked
---------  ---------------------------------  ----------------------------------
alpaca     instruction, output, [input]       instruction + output must be set
chatml     messages / conversations / texts   role + content per turn
           -> role, content                   (covers ALL model chat templates:
                                              llama-3, mistral, gemma, phi-4,
                                              qwen-2.5, qwen-3, gemma-3, gemma-3n,
                                              zephyr, vicuna, starling, yi-chat,
                                              lfm-2, qwen3-thinking, phi-3, etc.)
sharegpt   conversations -> from, value       value per turn
gptoss     messages -> role, content           content per turn; unsloth alias for
           (alias: gpt-oss)                   openai gpt-oss; detected via the
                                              presence of a developer role turn

New model-specific chat templates auto-match the chatml entry because they
all use role/content turn keys (the OpenAI messages standard). Only a genuinely
new structural pattern (different column names or turn keys) requires adding
an entry to FORMAT_REGISTRY.
"""

from collections import Counter, defaultdict
from datasets import Dataset

# ---------------------------------------------------------------------------
# Conversation column probing (shared by detection + scanning)
# ---------------------------------------------------------------------------

# Candidate column names for conversational datasets, checked in priority order.
CONVERSATION_COLUMNS = ("messages", "conversations", "texts")

# Minimum turn key sets that identify a column as conversational (not e.g. messages=[{"id":1}]).
_CHAT_KEY_SETS = (frozenset({"role", "content"}), frozenset({"from", "value"}))


def _probe_conversation(dataset: Dataset, candidates = None):
    """
    Probe a dataset for its conversation column and turn structure.

    candidates - iterable of column names to try, in priority order.
                 Defaults to CONVERSATION_COLUMNS when None.

    Returns a dict with:
        column    - name of the conversation column found
        turn_keys - set of keys present in the first turn dict
        roles     - set of all role values seen across the first few samples

    Returns None if no conversation column is found.
    """
    if candidates is None:
        candidates = CONVERSATION_COLUMNS
    columns = set(dataset.column_names)
    # Keep the first all-corrupt candidate as a fallback in case no healthy
    # column is found.  This avoids returning all_corrupt prematurely and
    # skipping a valid second candidate (e.g. messages is garbage but
    # conversations is fine).
    all_corrupt_fallback = None
    for col in candidates:
        if col not in columns:
            continue
        # Scan up to 100 rows — row 0 alone may be empty or malformed.
        first = None
        for i in range(min(len(dataset), 100)):
            sample = dataset[i][col]
            if not isinstance(sample, list) or len(sample) == 0:
                continue
            # Skip non-dict leading turns (e.g. [None, {"role": ...}]).
            first_turn = next((t for t in sample if isinstance(t, dict)), None)
            if first_turn is not None:
                first = first_turn
                break
        if first is None:
            # No usable dict turn in the first 100 rows — record as fallback
            # and continue probing the remaining candidates.  A later column
            # may be healthy and should take priority.
            if all_corrupt_fallback is None:
                # P2 fix: only treat the column as a corrupt conversation column
                # when its values are list or None.  Plain scalars (strings,
                # ints, etc.) indicate a non-conversation column that happens
                # to share a candidate name — they should not match chatml.
                has_list_or_none = any(
                    isinstance(dataset[i][col], list) or dataset[i][col] is None
                    for i in range(min(len(dataset), 100))
                )
                all_corrupt_fallback = {
                    "column": col,
                    "turn_keys": set(),
                    "roles": set(),
                    "all_corrupt": True,
                    "has_list_or_none": has_list_or_none,
                }
            continue

        # Use the same 100-row window to gather keys/roles.
        turn_keys = set()
        roles = set()
        for i in range(min(len(dataset), 100)):
            conv = dataset[i][col]
            if isinstance(conv, list):
                for t in conv:
                    if isinstance(t, dict):
                        turn_keys.update(t.keys())
                        r = t.get("role") or t.get("from")
                        if r:
                            roles.add(str(r))
        # Reject columns that don't match a known chat schema.
        if not any(keys <= turn_keys for keys in _CHAT_KEY_SETS):
            continue
        return {"column": col, "turn_keys": turn_keys, "roles": roles}
    # No healthy column found; return the all_corrupt fallback if any.
    return all_corrupt_fallback


# None-detection helpers
# ---------------------------------------------------------------------------


def is_none_or_empty(value) -> bool:
    """True if value is None, empty string, or whitespace-only."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _classify_empty(value) -> str:
    """Return a human-readable label for why this value is considered empty."""
    if value is None:
        return "None"
    if isinstance(value, str):
        if len(value) == 0:
            return "empty_string"
        if not value.strip():
            return "whitespace_only"
    return "valid"  # should not reach here if is_none_or_empty was True


# ---------------------------------------------------------------------------
# Alpaca detection
# ---------------------------------------------------------------------------


def find_none_alpaca(dataset: Dataset) -> dict:
    """
    Scan alpaca dataset for None/empty instruction or output fields.
    Returns stats dict with a detailed 'findings' list.
    """
    stats = {
        "total_rows": len(dataset),
        "none_instruction": 0,
        "none_output": 0,
        "bad_row_indices": [],
        "findings": [],  # [{row, field, value_type, raw_value}, ...]
    }

    for i, row in enumerate(dataset):
        bad = False
        for field in ("instruction", "output"):
            val = row.get(field)
            if is_none_or_empty(val):
                stats[f"none_{field}"] = stats.get(f"none_{field}", 0) + 1
                bad = True
                stats["findings"].append(
                    {
                        "row_index": i,
                        "field": field,
                        "value_type": _classify_empty(val),
                        "raw_value": repr(val),
                    }
                )
        if bad:
            stats["bad_row_indices"].append(i)

    return stats


# ---------------------------------------------------------------------------
# ChatML / conversational detection
# ---------------------------------------------------------------------------


def find_none_chatml(dataset: Dataset, col: str = None) -> dict:
    """
    Scan chatml/sharegpt/gptoss dataset for turns with None/empty content.
    Auto-detects the conversation column if col=None.

    Returns a stats dict that includes a complete 'findings' list — one entry
    per bad turn with row_index, turn_index, role, value_type, and raw_value.
    """
    if col is None:
        # Reuse _probe_conversation so the all_corrupt path (fully malformed
        # columns) is handled correctly without duplicating probe logic here.
        _cinfo = _probe_conversation(dataset)
        if _cinfo is not None:
            col = _cinfo["column"]

    if col is None or col not in dataset.column_names:
        raise ValueError(
            f"No conversation column found. "
            f"Expected one of {CONVERSATION_COLUMNS}, got columns: {dataset.column_names}"
        )

    stats = {
        "total_rows": len(dataset),
        "column": col,
        "rows_with_none_turns": 0,
        "total_none_turns": 0,
        "none_by_role": {},  # role -> count of None turns
        "none_by_type": {},  # "None" | "empty_string" | "whitespace_only" -> count
        "rows_all_none": 0,  # rows where every turn is bad
        "bad_row_indices": [],  # every row index that has at least one bad turn
        "findings": [],  # detailed per-turn list
    }

    for i, row in enumerate(dataset):
        conversation = row[col]
        if not isinstance(conversation, list):
            # P1 fix: record non-list conversation rows as bad instead of
            # silently skipping them.  These rows are unusable for training
            # and would produce a false-clean report if ignored.
            vtype = "None" if conversation is None else "invalid_type"
            stats["bad_row_indices"].append(i)
            stats["rows_with_none_turns"] += 1
            stats["total_none_turns"] += 1
            stats["none_by_role"]["unknown"] = (
                stats["none_by_role"].get("unknown", 0) + 1
            )
            stats["none_by_type"][vtype] = stats["none_by_type"].get(vtype, 0) + 1
            stats["findings"].append(
                {
                    "row_index": i,
                    "turn_index": 0,
                    "role": "unknown",
                    "value_type": vtype,
                    "raw_value": repr(conversation),
                }
            )
            continue

        row_findings = []
        for turn_idx, turn in enumerate(conversation):
            # Non-dict turn — record it rather than crash or silently skip.
            if not isinstance(turn, dict):
                row_findings.append(
                    {
                        "row_index": i,
                        "turn_index": turn_idx,
                        "role": "unknown",
                        "value_type": "None" if turn is None else "invalid_type",
                        "raw_value": repr(turn),
                    }
                )
                stats["none_by_role"]["unknown"] = (
                    stats["none_by_role"].get("unknown", 0) + 1
                )
                vtype = "None" if turn is None else "invalid_type"
                stats["none_by_type"][vtype] = stats["none_by_type"].get(vtype, 0) + 1
                continue
            # Stringify role to handle unhashable/non-string types.
            role = turn.get("role") or turn.get("from") or "unknown"
            if not isinstance(role, str):
                role = str(role)
            content = turn.get("content") if "content" in turn else turn.get("value")
            if is_none_or_empty(content):
                vtype = _classify_empty(content)
                row_findings.append(
                    {
                        "row_index": i,
                        "turn_index": turn_idx,
                        "role": role,
                        "value_type": vtype,
                        "raw_value": repr(content),
                    }
                )
                stats["none_by_role"][role] = stats["none_by_role"].get(role, 0) + 1
                stats["none_by_type"][vtype] = stats["none_by_type"].get(vtype, 0) + 1

        if row_findings:
            stats["rows_with_none_turns"] += 1
            stats["total_none_turns"] += len(row_findings)
            stats["bad_row_indices"].append(i)
            stats["findings"].extend(row_findings)

            if len(row_findings) == len(conversation):
                stats["rows_all_none"] += 1

    return stats


# ---------------------------------------------------------------------------
# Convenience wrappers per format (all delegate to the same scan logic)
# ---------------------------------------------------------------------------


def find_none_sharegpt(dataset: Dataset, col: str = None) -> dict:
    """ShareGPT uses 'from'/'value' keys — same scan logic handles both."""
    if col is None:
        # ShareGPT data lives in 'conversations'; restrict probe to that column
        # so a corrupt conversations column is always scanned instead of being
        # silently replaced by a healthy messages column (P1 fix).
        conv_info = _probe_conversation(dataset, candidates = ("conversations",))
        if conv_info is None:
            raise ValueError(
                f"No valid conversation column found in {dataset.column_names}. "
                "Expected a 'conversations' column with 'from'/'value' or 'role'/'content' turn keys."
            )
        col = conv_info["column"]
    return find_none_chatml(dataset, col = col)


def find_none_gptoss(dataset: Dataset, col: str = None) -> dict:
    """gptoss: role/content plus optional thinking/tool_calls. Only content checked."""
    if col is None:
        # gptoss data always lives in 'messages'; probing 'conversations' as a
        # fallback can silently scan the wrong column and return false-clean
        # results when messages is corrupt but conversations exists and is valid.
        conv_info = _probe_conversation(dataset, candidates = ("messages",))
        if conv_info is None:
            raise ValueError(
                f"No valid conversation column found in {dataset.column_names}. "
                "Expected a 'messages' column with 'role'/'content' turn keys."
            )
        col = conv_info["column"]
    return find_none_chatml(dataset, col = col)


# ---------------------------------------------------------------------------
# Format registry -- add new formats here; detect_format auto-scales.
# ---------------------------------------------------------------------------
#
# Each entry defines one dataset structure the None detector can scan.
#
#   name    - label returned by detect_format() and accepted by --format
#   match   - callable(dataset, conv_info) -> bool
#             conv_info is the dict from _probe_conversation(), or None
#   scan    - reference to the scanner function (find_none_*)
#
# Order matters: first match wins.  Put specific formats before their
# generalisations (e.g. gptoss before chatml, since gptoss IS chatml
# with a 'developer' role).
#
# To add a new format:
#   1. Write a find_none_<name>() function (or reuse find_none_chatml).
#   2. Append a dict to FORMAT_REGISTRY.
#   That's it -- detect_format(), the CLI --format choices, and
#   scan_dataset() all pick it up automatically.
#
# All model-specific chat templates in unsloth/chat_templates.py
# (llama-3, mistral, gemma, phi-4, qwen-2.5, qwen-3, gemma-3,
# gemma-3n, phi-3, zephyr, vicuna, starling, yi-chat, lfm-2, etc.)
# use role/content turn keys, so they match the 'chatml' entry
# without needing their own registry entries.

FORMAT_REGISTRY = [
    {
        "name": "alpaca",
        # Only match alpaca when no conversation column is detected.  A dataset
        # with both instruction/output columns AND a messages/conversations
        # column should be scanned as conversational so None turns are caught.
        # Without this guard, alpaca wins first and conversational turns are
        # never inspected, producing a silent false negative (P1 fix).
        "match": lambda ds, conv: (
            {"instruction", "output"}.issubset(ds.column_names) and conv is None
        ),
        "scan": find_none_alpaca,
    },
    {
        "name": "gptoss",
        "match": lambda ds, conv: (
            conv is not None
            and {"role", "content"} <= conv["turn_keys"]
            and "developer" in conv["roles"]
        ),
        "scan": find_none_gptoss,
    },
    {
        "name": "sharegpt",
        "match": lambda ds, conv: (
            conv is not None and {"from", "value"} <= conv["turn_keys"]
        ),
        "scan": find_none_sharegpt,
    },
    {
        "name": "chatml",
        "match": lambda ds, conv: (
            conv is not None
            and (
                {"role", "content"} <= conv["turn_keys"]
                # all_corrupt path: probe found the column but every row is
                # malformed.  Require has_list_or_none so plain-string or
                # other scalar columns are not misclassified as chatml (P2 fix).
                or (conv.get("all_corrupt") and conv.get("has_list_or_none"))
            )
        ),
        "scan": find_none_chatml,
    },
]

# Derived list of known format names (used by CLI --format choices).
FORMAT_NAMES = [entry["name"] for entry in FORMAT_REGISTRY]


def detect_format(dataset: Dataset) -> str:
    """
    Auto-detect dataset format by probing columns and turn structure.

    Returns one of the format names in FORMAT_REGISTRY, or 'unknown'.
    Walks the registry in order; first match wins.
    """
    conv_info = _probe_conversation(dataset)
    for entry in FORMAT_REGISTRY:
        if entry["match"](dataset, conv_info):
            return entry["name"]
    return "unknown"


def get_scanner(fmt: str):
    """Return the scanner function for a format name, or None if unknown."""
    for entry in FORMAT_REGISTRY:
        if entry["name"] == fmt:
            return entry["scan"]
    return None


def scan_dataset(dataset: Dataset, fmt: str = "auto") -> dict:
    """
    One-liner: detect format (if 'auto') and scan for None/empty content.

    Returns the stats dict with an added 'format' key.
    Raises ValueError if the format is unknown or unsupported.
    """
    was_auto = fmt == "auto"
    # Always probe so detection and column selection share one scan pass.
    conv_info = _probe_conversation(dataset)
    if was_auto:
        fmt = "unknown"
        for entry in FORMAT_REGISTRY:
            if entry["match"](dataset, conv_info):
                fmt = entry["name"]
                break
    scanner = get_scanner(fmt)
    if scanner is None:
        raise ValueError(f"Unknown or unsupported format: '{fmt}'")
    # Column forwarding rules:
    # - auto-detect: always pass the probed column (probe already chose best)
    # - explicit format: let each scanner's own per-format probe handle column
    #   selection.  Forwarding the generic probe's column for explicit formats
    #   breaks semantics when the generic probe (ordered messages→conversations)
    #   picks a different column than the format expects, e.g. scan_dataset(
    #   fmt='sharegpt') should always scan 'conversations', not 'messages'
    #   even when both columns are all-corrupt (P1 fix).
    use_probed_col = (
        conv_info is not None
        and fmt != "alpaca"
        and was_auto
    )
    if use_probed_col:
        stats = scanner(dataset, col = conv_info["column"])
    else:
        stats = scanner(dataset)
    stats["format"] = fmt
    return stats


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def _print_summary_header(stats: dict, fmt: str) -> bool:
    """Print the top-level stats block (shared by all report modes). Returns True if findings exist."""
    total = stats["total_rows"]
    findings = stats.get("findings", [])

    print(f"\n{'=' * 64}")
    print(f"  None / Empty Detection Report")
    print(f"{'=' * 64}")
    print(f"  Format:       {fmt}")
    print(f"  Total rows:   {total}")

    if not findings:
        print(f"  Result:       CLEAN -- no None or empty values found")
        print(f"{'=' * 64}")
        return False

    if fmt == "alpaca":
        bad_rows = len(stats.get("bad_row_indices", []))
        print(f"  Rows with Nones: {bad_rows} / {total}")
        print(f"  None instruction: {stats.get('none_instruction', 0)}")
        print(f"  None output:      {stats.get('none_output', 0)}")
    else:
        col = stats.get("column", "?")
        print(f"  Column:       {col}")
        print(f"  Rows with bad turns: {stats['rows_with_none_turns']} / {total}")
        print(f"  Total bad turns:     {len(findings)}")
        print(f"  By type:      {stats.get('none_by_type', {})}")
        print(f"  By role:      {stats.get('none_by_role', {})}")
        rows_all = stats.get("rows_all_none", 0)
        if rows_all:
            print(f"  Rows ALL bad: {rows_all} (every turn is None/empty)")

    # Rows with no Nones
    all_indices = set(range(total))
    bad_indices = set(stats.get("bad_row_indices", []))
    clean_indices = sorted(all_indices - bad_indices)
    clean_count = len(clean_indices)
    if 0 < clean_count <= 20:
        print(f"  Rows with no Nones: {clean_count} / {total}  {clean_indices}")
    else:
        print(f"  Rows with no Nones: {clean_count} / {total}")

    print(f"{'=' * 64}")
    return True


def print_report(stats: dict, fmt: str, summary_only: bool = False):
    """Print a human-readable summary, optionally with full findings list."""
    has_findings = _print_summary_header(stats, fmt)
    if not has_findings or summary_only:
        return

    findings = stats.get("findings", [])
    print(f"\n  {'-' * 60}")
    print(f"  Findings ({len(findings)} total):")
    print(f"  {'-' * 60}")

    for f in findings:
        if fmt == "alpaca":
            print(
                f"  row {f['row_index']:>5d}  "
                f"field={f['field']:<12s}  "
                f"type={f['value_type']:<16s}  "
                f"raw={f['raw_value']}"
            )
        else:
            print(
                f"  row {f['row_index']:>5d}  "
                f"turn {f['turn_index']}  "
                f"role={str(f['role']):<12s}  "
                f"type={f['value_type']:<16s}  "
                f"raw={f['raw_value']}"
            )

    print(f"{'=' * 64}")


def show_row(dataset: Dataset, row_indices: list[int], fmt: str, col: str = None):
    """Print the full contents of specific rows for inspection.

    Used by test_codex_fixes.py to verify row rendering behaviour.
    Not part of the production API.
    """
    if col is None:
        for candidate in ("messages", "conversations", "texts"):
            if candidate in dataset.column_names:
                col = candidate
                break

    for ri in row_indices:
        if ri < 0 or ri >= len(dataset):
            print(f"\n  [ERROR] Row {ri} out of range (0-{len(dataset)-1})")
            continue

        row = dataset[ri]
        print(f"\n{'=' * 64}")
        print(f"  Row {ri}")
        print(f"{'=' * 64}")

        # Print non-conversation columns
        for key in dataset.column_names:
            if key == col:
                continue
            val = row[key]
            if isinstance(val, str) and len(val) > 120:
                val = val[:120] + "..."
            print(f"  {key}: {val}")

        if fmt == "alpaca":
            for field in ("instruction", "input", "output"):
                val = row.get(field)
                status = "  [NONE]" if is_none_or_empty(val) else ""
                if val and len(str(val)) > 200:
                    val = str(val)[:200] + "..."
                print(f"  {field}: {val}{status}")
        elif col:
            conversation = row[col]
            if isinstance(conversation, list):
                none_count = sum(
                    1
                    for t in conversation
                    if not isinstance(t, dict)
                    or is_none_or_empty(
                        t.get("content") if "content" in t else t.get("value")
                    )
                )
                print(f"  {col}: {len(conversation)} turns ({none_count} None)")
                print(f"  {'-' * 60}")
                for i, turn in enumerate(conversation):
                    # Non-dict turn — can't extract role or content normally.
                    if not isinstance(turn, dict):
                        label = "None" if turn is None else "invalid_type"
                        print(f"  [{i:>3d}] {'unknown':<12s} [{label}]  << NONE")
                        continue
                    role = str(turn.get("role") or turn.get("from", "?"))
                    content = (
                        turn.get("content") if "content" in turn else turn.get("value")
                    )
                    if is_none_or_empty(content):
                        status = "  << NONE"
                    else:
                        status = ""
                    if content is None:
                        preview = "None"
                    else:
                        preview_str = str(content)  # cast: content may not be a string
                        if len(preview_str) > 150:
                            preview = preview_str[:150].replace("\n", "\\n") + "..."
                        else:
                            preview = preview_str.replace("\n", "\\n")
                    print(f"  [{i:>3d}] {role:<12s} {preview}{status}")

        print(f"{'=' * 64}")
