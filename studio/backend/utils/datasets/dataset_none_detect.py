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
            # Only treat the column as a corrupt conversation column when
            # we actually saw evidence of turn-shaped data: a list that
            # contains at least one dict or None element, or the cell
            # itself is None.  This keeps plain scalar columns (strings,
            # ints, etc.) and plain list-of-strings columns (e.g. a
            # `texts` column of raw text) from being misclassified as
            # chatml with a bogus all_corrupt match.
            # Upgrade the fallback when a later candidate looks more
            # plausible than the currently stored one, so probe order does
            # not silently discard a better match.
            if (
                all_corrupt_fallback is None
                or not all_corrupt_fallback.get("has_plausible_turns")
            ):
                has_plausible_turns = False
                for i in range(min(len(dataset), 100)):
                    cell = dataset[i][col]
                    if cell is None:
                        has_plausible_turns = True
                        break
                    if isinstance(cell, dict):
                        # Struct-typed column (row is a single dict instead
                        # of a list of dicts, e.g. a user typo of
                        # messages=[{"role":...}] instead of
                        # [[{"role":...}]]).  The scanner's non-list branch
                        # handles this by flagging each row as invalid_type,
                        # so route it to chatml instead of falling through
                        # to `unknown`.
                        has_plausible_turns = True
                        break
                    if isinstance(cell, list):
                        # Empty list is a valid (but empty) conversation
                        # shape; a non-empty list is only plausible if it
                        # contains dict/None turns, so plain list-of-string
                        # columns (e.g. raw-text `texts`) don't match.
                        if len(cell) == 0 or any(
                            t is None or isinstance(t, dict) for t in cell
                        ):
                            has_plausible_turns = True
                            break
                all_corrupt_fallback = {
                    "column": col,
                    "turn_keys": set(),
                    "roles": set(),
                    "all_corrupt": True,
                    "has_plausible_turns": has_plausible_turns,
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
    """True if value is None, empty string, whitespace-only, or an empty/whitespace-only VLM content block list."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list):
        # OpenAI/VLM multimodal shape: content = [{"type":"text","text":"..."}, {"type":"image",...}].
        # Flag empty lists outright.  For non-empty lists, only flag when
        # text blocks exist *and* every text block is empty/whitespace *and*
        # there is no non-text block (image / audio / tool-call) that
        # provides real content.  An image-only turn with no text is not
        # corrupt — the image itself is the content.
        if len(value) == 0:
            return True
        non_text_blocks = [
            item
            for item in value
            if isinstance(item, dict) and item.get("type") != "text"
        ]
        if non_text_blocks:
            return False
        text_values = [
            item.get("text")
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_values and all(
            t is None or (isinstance(t, str) and not t.strip())
            for t in text_values
        ):
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
    if isinstance(value, list):
        # Mirrors the VLM/OpenAI content-block handling in is_none_or_empty.
        if len(value) == 0:
            return "empty_list"
        return "empty_vlm_content"
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
            # Record non-list conversation rows as bad instead of silently
            # skipping them.  These rows are unusable for training and would
            # produce a false-clean report if ignored.
            vtype = "None" if conversation is None else "invalid_type"
            stats["bad_row_indices"].append(i)
            stats["rows_with_none_turns"] += 1
            stats["total_none_turns"] += 1
            stats["rows_all_none"] += 1
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

        if len(conversation) == 0:
            # Zero-turn conversations are unusable for training; flag them so
            # they don't scan as clean.
            stats["bad_row_indices"].append(i)
            stats["rows_with_none_turns"] += 1
            stats["total_none_turns"] += 1
            stats["rows_all_none"] += 1
            stats["none_by_role"]["unknown"] = (
                stats["none_by_role"].get("unknown", 0) + 1
            )
            stats["none_by_type"]["empty_conversation"] = (
                stats["none_by_type"].get("empty_conversation", 0) + 1
            )
            stats["findings"].append(
                {
                    "row_index": i,
                    "turn_index": 0,
                    "role": "unknown",
                    "value_type": "empty_conversation",
                    "raw_value": "[]",
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
            # Resolve role without collapsing falsy values (role=0, "", False)
            # into "unknown" — explicit None check preserves the real key.
            r = turn.get("role")
            if r is None:
                r = turn.get("from")
            if r is None:
                role = "unknown"
            elif isinstance(r, str):
                role = r
            else:
                role = str(r)
            content = turn.get("content") if "content" in turn else turn.get("value")
            # Valid OpenAI tool-calling assistant turns carry empty content
            # plus a populated tool_calls array — these are not bad data.
            if is_none_or_empty(content) and not turn.get("tool_calls"):
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
        # gptoss data canonically lives in 'messages'.  Preserve the original
        # P1 invariant that a corrupt messages column alongside a clean
        # conversations column is never silently replaced by the cleaner
        # column: if messages exists at all, always target it.  Fall back to
        # 'conversations' only when the messages column is absent entirely
        # (some third-party exports rename the gptoss column to
        # 'conversations').  This keeps explicit fmt='gptoss' on a
        # conversations-only dataset working without masking real errors on
        # a corrupt canonical column.
        if "messages" in dataset.column_names:
            conv_info = _probe_conversation(dataset, candidates = ("messages",))
        else:
            conv_info = _probe_conversation(dataset, candidates = ("conversations",))
        if conv_info is None:
            raise ValueError(
                f"No valid conversation column found in {dataset.column_names}. "
                "Expected a 'messages' or 'conversations' column with 'role'/'content' turn keys."
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
        # Match alpaca when instruction/output are present and either:
        #   - no conversation column exists (conv is None), or
        #   - the only conversation column found is fully corrupt
        #     (e.g. a stray `messages=None` metadata column on an
        #     otherwise valid alpaca dataset).
        # A dataset with both instruction/output AND a *healthy* messages /
        # conversations column still prefers the conversational scanners
        # below so None turns in the chat column are caught.
        "match": lambda ds, conv: (
            {"instruction", "output"}.issubset(ds.column_names)
            and (conv is None or conv.get("all_corrupt"))
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
                # malformed.  Require has_plausible_turns so plain-string,
                # list-of-string, and other scalar columns are not
                # misclassified as chatml.
                or (conv.get("all_corrupt") and conv.get("has_plausible_turns"))
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
    # Guard against accidentally passing a DatasetDict (e.g. load_dataset
    # without split=...) — otherwise column_names returns a mapping of split
    # names and the probe silently produces a confusing "unknown format"
    # error.  IterableDatasetDict (streaming load without split) is *not* a
    # subclass of DatasetDict, so both types must be checked.  Import
    # locally so importing this module never requires the DatasetDict
    # symbols from `datasets`.
    _dict_types = []
    try:
        from datasets import DatasetDict as _DatasetDict
        _dict_types.append(_DatasetDict)
    except ImportError:
        pass
    try:
        from datasets import IterableDatasetDict as _IterableDatasetDict
        _dict_types.append(_IterableDatasetDict)
    except ImportError:
        pass
    if _dict_types and isinstance(dataset, tuple(_dict_types)):
        raise ValueError(
            "scan_dataset requires a single Dataset split, not a DatasetDict. "
            f"Available splits: {list(dataset.keys())}. "
            "Pass dataset[<split>] or use load_dataset(..., split='train')."
        )
    was_auto = fmt == "auto"
    # Zero-row datasets have nothing to scan and would otherwise fall
    # through the probe to an "unknown format" error.  Return a trivially
    # clean stats dict so callers don't have to special-case empty inputs.
    if was_auto and len(dataset) == 0:
        return {
            "format": "unknown",
            "total_rows": 0,
            "findings": [],
            "bad_row_indices": [],
        }
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
    use_probed_col = conv_info is not None and fmt != "alpaca" and was_auto
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

    # Rows with no Nones — compute the count directly instead of allocating
    # a full set of row indices, which OOMs on large (10M+ row) datasets.
    bad_indices = set(stats.get("bad_row_indices", []))
    clean_count = total - len(bad_indices)
    if 0 < clean_count <= 20:
        clean_indices = [i for i in range(total) if i not in bad_indices]
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

        # Print non-conversation columns.  For alpaca, skip the fields that
        # the alpaca-specific block below prints with status markers to
        # avoid rendering them twice.
        _ALPACA_FIELDS = {"instruction", "input", "output"}
        for key in dataset.column_names:
            if key == col:
                continue
            if fmt == "alpaca" and key in _ALPACA_FIELDS:
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
                def _is_bad_turn(t):
                    if not isinstance(t, dict):
                        return True
                    c = t.get("content") if "content" in t else t.get("value")
                    # Valid tool-calling assistant turns have empty content
                    # but a populated tool_calls array; scan_dataset ignores
                    # them, so the display should too.
                    if is_none_or_empty(c) and not t.get("tool_calls"):
                        return True
                    return False
                none_count = sum(1 for t in conversation if _is_bad_turn(t))
                print(f"  {col}: {len(conversation)} turns ({none_count} None)")
                print(f"  {'-' * 60}")
                for i, turn in enumerate(conversation):
                    # Non-dict turn — can't extract role or content normally.
                    if not isinstance(turn, dict):
                        label = "None" if turn is None else "invalid_type"
                        print(f"  [{i:>3d}] {'unknown':<12s} [{label}]  << NONE")
                        continue
                    r = turn.get("role")
                    if r is None:
                        r = turn.get("from")
                    role = "?" if r is None else str(r)
                    content = (
                        turn.get("content") if "content" in turn else turn.get("value")
                    )
                    if is_none_or_empty(content) and not turn.get("tool_calls"):
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        prog = "dataset_none_detect",
        description = "Scan a HuggingFace dataset for None/empty content turns.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
examples:
  python dataset_none_detect.py org/my-dataset
  python dataset_none_detect.py org/my-dataset --split train
  python dataset_none_detect.py org/my-dataset --format sharegpt
  python dataset_none_detect.py org/my-dataset --summary-only
  python dataset_none_detect.py org/my-dataset --token hf_...
        """,
    )
    parser.add_argument(
        "dataset", help = "HuggingFace dataset repo id (e.g. org/my-dataset)"
    )
    parser.add_argument(
        "--split", default = "train", help = "Dataset split to load (default: train)"
    )
    parser.add_argument(
        "--format",
        default = "auto",
        choices = ["auto"] + FORMAT_NAMES,
        help = "Force a specific format instead of auto-detecting (default: auto)",
    )
    parser.add_argument(
        "--summary-only",
        action = "store_true",
        help = "Print summary header only — skip the per-turn findings list",
    )
    parser.add_argument(
        "--token",
        default = os.environ.get("HF_TOKEN"),
        help = (
            "HuggingFace API token for private datasets (default: $HF_TOKEN). "
            "Prefer setting $HF_TOKEN; passing --token on the command line "
            "exposes it in process listings."
        ),
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: 'datasets' package not found. Install with: pip install datasets",
            file = sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {args.dataset!r} (split={args.split!r})...")
    try:
        ds = load_dataset(args.dataset, split = args.split, token = args.token)
    except Exception as exc:
        # Some `datasets` / `requests` versions include the Authorization
        # header in exception messages.  Redact the token before printing.
        msg = str(exc)
        if args.token:
            msg = msg.replace(args.token, "hf_***REDACTED***")
        print(f"Error loading dataset: {msg}", file = sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(ds)} rows, columns: {ds.column_names}")

    try:
        stats = scan_dataset(ds, fmt = args.format)
    except ValueError as exc:
        print(f"Error: {exc}", file = sys.stderr)
        sys.exit(1)

    print_report(stats, stats["format"], summary_only = args.summary_only)
