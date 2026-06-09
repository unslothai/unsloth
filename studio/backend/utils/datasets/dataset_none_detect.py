"""
Detect None/empty content turns in conversation datasets. Reports findings
without modifying data.

Usage:
    from .dataset_none_detect import scan_dataset, print_report
    stats = scan_dataset(dataset)               # auto-detect + scan
    stats = scan_dataset(dataset, fmt="chatml")  # explicit format
    print_report(stats, stats["format"])

Dependencies: only `datasets` (already in studio/unsloth) + stdlib.

Supported formats (via FORMAT_REGISTRY):
    alpaca     instruction/output            instruction + output must be set
    chatml     messages/conversations/texts  role + content per turn
    sharegpt   conversations                 from/value per turn
    gptoss     messages (alias: gpt-oss)     role/content; has a developer turn

Any role/content chat template matches chatml, so new templates need no change;
add a FORMAT_REGISTRY entry only for a genuinely new column/turn shape.
"""

from datasets import Dataset

# ---------------------------------------------------------------------------
# Conversation column probing (shared by detection + scanning)
# ---------------------------------------------------------------------------

# Candidate column names for conversational datasets, checked in priority order.
CONVERSATION_COLUMNS = ("messages", "conversations", "texts")

# Minimum turn key sets identifying a column as conversational (not e.g. messages=[{"id":1}]).
_CHAT_KEY_SETS = (frozenset({"role", "content"}), frozenset({"from", "value"}))


def _probe_conversation(dataset: Dataset, candidates = None):
    """
    Probe a dataset for its conversation column and turn structure.

    candidates - column names to try, in priority order.
                 Defaults to CONVERSATION_COLUMNS when None.

    Returns a dict with:
        column    - conversation column found
        turn_keys - keys present in the first turn dict
        roles     - all role values seen across the first few samples

    Returns None if no conversation column is found.
    """
    if candidates is None:
        candidates = CONVERSATION_COLUMNS
    columns = set(dataset.column_names)
    # Remember the first all-corrupt candidate, but keep probing: a later column
    # may be healthy and win (e.g. bad messages, good conversations).
    all_corrupt_fallback = None
    for col in candidates:
        if col not in columns:
            continue
        # Scan up to 100 rows - row 0 alone may be empty/malformed.
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
            # No usable dict turn in 100 rows. Record an all_corrupt fallback,
            # plausible only with turn-shaped data (None cell or list of dict/None
            # turns); a later plausible candidate upgrades a non-plausible one.
            if all_corrupt_fallback is None or not all_corrupt_fallback.get("has_plausible_turns"):
                has_plausible_turns = False
                for i in range(min(len(dataset), 100)):
                    cell = dataset[i][col]
                    if cell is None:
                        has_plausible_turns = True
                        break
                    # A struct-typed cell (single dict, not a list) is metadata,
                    # not chat: leave it for "unknown format", matching
                    # format_detection.py.
                    if isinstance(cell, list):
                        # Plausible only if the list holds a dict/None turn;
                        # empty lists and list-of-strings are not chat data.
                        if any(t is None or isinstance(t, dict) for t in cell):
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
        # Column lacks a full chat key pair. If it has a conversational key
        # (role/from/content/value) it is a corrupt-but-real chat column, so
        # save a plausible fallback for find_none_chatml to flag. Pure metadata
        # (e.g. [{"id":1}]) is not plausible, so a later real-but-corrupt column
        # (e.g. conversations=None) can still win.
        _CONV_KEYS = {"role", "from", "content", "value"}
        if not any(keys <= turn_keys for keys in _CHAT_KEY_SETS):
            schema_less_plausible = bool(turn_keys & _CONV_KEYS)
            if all_corrupt_fallback is None or not all_corrupt_fallback.get("has_plausible_turns"):
                all_corrupt_fallback = {
                    "column": col,
                    "turn_keys": turn_keys,
                    "roles": roles,
                    "all_corrupt": True,
                    "has_plausible_turns": schema_less_plausible,
                }
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
    if isinstance(value, str):
        # Treat zero-width/BOM chars (U+FEFF/200B/200C/200D/2060) as empty;
        # they render invisibly. Two-pass strip (ws, invisibles, ws) catches
        # mixed cases like "\u200b \u200b".
        stripped = value.strip().strip("\ufeff\u200b\u200c\u200d\u2060").strip()
        if not stripped:
            return True
    if isinstance(value, list):
        # VLM content blocks, e.g. [{"type":"text",...}, {"type":"image",...}].
        # Empty list -> empty. A non-text block (image/audio/tool) is real
        # content; only flag when every text block is blank and no such block
        # exists (an image-only turn is valid).
        if len(value) == 0:
            return True
        # No dict blocks (e.g. [None], ['  ']) -> malformed/empty.
        dict_blocks = [item for item in value if isinstance(item, dict)]
        if not dict_blocks:
            return True
        non_text_blocks = [item for item in dict_blocks if item.get("type") != "text"]
        if non_text_blocks:
            return False
        text_values = [item.get("text") for item in dict_blocks if item.get("type") == "text"]
        if text_values and all(
            t is None
            or (
                isinstance(t, str) and not t.strip().strip("\ufeff\u200b\u200c\u200d\u2060").strip()
            )
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
        # Only-whitespace or only-invisible (BOM/zero-width) strings render empty.
        if not value.strip().strip("\ufeff\u200b\u200c\u200d\u2060").strip():
            return "whitespace_only"
    if isinstance(value, list):
        # Mirrors the VLM/OpenAI content-block handling in is_none_or_empty.
        if len(value) == 0:
            return "empty_list"
        return "empty_vlm_content"
    return "valid"  # unreachable if is_none_or_empty was True


# ---------------------------------------------------------------------------
# Alpaca detection
# ---------------------------------------------------------------------------


def find_none_alpaca(dataset: Dataset) -> dict:
    """
    Scan alpaca dataset for None/empty instruction or output fields.
    Returns a stats dict with a detailed 'findings' list.
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

    Returns a stats dict with a complete 'findings' list - one entry per bad
    turn with row_index, turn_index, role, value_type, and raw_value.
    """
    if col is None:
        # Reuse _probe_conversation so the all_corrupt path is handled here too.
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
            # Non-list conversation: unusable for training, flag as bad.
            vtype = "None" if conversation is None else "invalid_type"
            stats["bad_row_indices"].append(i)
            stats["rows_with_none_turns"] += 1
            stats["total_none_turns"] += 1
            stats["rows_all_none"] += 1
            stats["none_by_role"]["unknown"] = stats["none_by_role"].get("unknown", 0) + 1
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
            # Zero-turn conversation: flag so it doesn't scan as clean.
            stats["bad_row_indices"].append(i)
            stats["rows_with_none_turns"] += 1
            stats["total_none_turns"] += 1
            stats["rows_all_none"] += 1
            stats["none_by_role"]["unknown"] = stats["none_by_role"].get("unknown", 0) + 1
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
            # Non-dict turn - record it rather than crash or silently skip.
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
                stats["none_by_role"]["unknown"] = stats["none_by_role"].get("unknown", 0) + 1
                vtype = "None" if turn is None else "invalid_type"
                stats["none_by_type"][vtype] = stats["none_by_type"].get(vtype, 0) + 1
                continue
            # Explicit None check so falsy roles (0, "", False) are kept, not
            # collapsed to "unknown".
            r = turn.get("role")
            if r is None:
                r = turn.get("from")
            if r is None:
                role = "unknown"
            elif isinstance(r, str):
                role = r
            else:
                role = str(r)
            # Pick the content key: from+value -> value (ShareGPT, even if role
            # is set); role -> content (or value); from only -> value (None when
            # missing, so it is flagged); neither -> content then value.
            if "from" in turn and "value" in turn:
                content = turn.get("value")
            elif "role" in turn:
                content = turn.get("content") if "content" in turn else turn.get("value")
            elif "from" in turn:
                content = turn.get("value")
            else:
                content = turn.get("content") if "content" in turn else turn.get("value")
            # Assistant tool-call turns carry empty content + tool_calls and are
            # valid; the exemption is assistant-only.
            if is_none_or_empty(content) and not (role == "assistant" and turn.get("tool_calls")):
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
    """ShareGPT uses 'from'/'value' keys - same scan logic handles both."""
    if col is None:
        # ShareGPT lives in 'conversations'; probe only that column so a corrupt
        # one is still scanned, not replaced by healthy 'messages' (P1 fix).
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
        # gptoss lives in 'messages': target it whenever present (even if
        # corrupt); fall back to 'conversations' only if 'messages' is absent.
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
# Format registry - first match wins; detect_format() auto-scales.
# Each entry: name (label/--format value), match(dataset, conv_info) -> bool,
# scan (find_none_* function). Put specific formats before general ones
# (gptoss before chatml, since gptoss is chatml with a 'developer' role).
# To add a format: write find_none_<name>() (or reuse find_none_chatml) and
# append an entry; detect_format(), --format, and scan_dataset() pick it up.
# ---------------------------------------------------------------------------

FORMAT_REGISTRY = [
    {
        "name": "alpaca",
        # instruction/output present and no usable chat column: either none
        # exists, or the only one is fully corrupt (e.g. a stray all-None or
        # metadata `messages` column). A healthy chat column falls through to
        # the conversational scanners below.
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
        "match": lambda ds, conv: (conv is not None and {"from", "value"} <= conv["turn_keys"]),
        "scan": find_none_sharegpt,
    },
    {
        "name": "chatml",
        "match": lambda ds, conv: (
            conv is not None
            and (
                {"role", "content"} <= conv["turn_keys"]
                # all_corrupt: column found but every row malformed; require
                # has_plausible_turns so scalar/string columns aren't chatml.
                or (conv.get("all_corrupt") and conv.get("has_plausible_turns"))
            )
        ),
        "scan": find_none_chatml,
    },
]

# Derived list of known format names (used by CLI --format choices).
FORMAT_NAMES = [entry["name"] for entry in FORMAT_REGISTRY]

# Documented aliases accepted by both the Python API and the CLI.
FORMAT_ALIASES = {"gpt-oss": "gptoss"}


def detect_format(dataset: Dataset) -> str:
    """
    Auto-detect dataset format by probing columns and turn structure.

    Returns a format name from FORMAT_REGISTRY, or 'unknown'.
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
    # Reject a DatasetDict / IterableDatasetDict (load_dataset without split):
    # its column_names is a split map and would yield a confusing "unknown
    # format". Check both (IterableDatasetDict is not a DatasetDict subclass);
    # import locally so this module never hard-requires them.
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
    # Streaming IterableDataset has no len()/column_names; give a clear error
    # instead of a confusing downstream TypeError.
    try:
        from datasets import IterableDataset as _IterableDataset
        if isinstance(dataset, _IterableDataset):
            raise ValueError(
                "scan_dataset requires a materialized Dataset, not an IterableDataset. "
                "Load without streaming=True, or materialize a slice first: "
                "Dataset.from_list(list(dataset.take(N)))."
            )
    except ImportError:
        pass
    fmt = FORMAT_ALIASES.get(fmt, fmt)
    was_auto = fmt == "auto"
    # Zero-row dataset: return a trivially clean stats dict.
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
        # No format matched: return clean stats (format="unknown") instead of
        # raising, so callers can branch on stats["format"].
        if fmt == "unknown":
            return {
                "format": "unknown",
                "total_rows": len(dataset),
                "findings": [],
                "bad_row_indices": [],
            }
    scanner = get_scanner(fmt)
    if scanner is None:
        raise ValueError(f"Unknown or unsupported format: '{fmt}'")
    # Column forwarding: on auto-detect pass the probed column (the best
    # choice). On an explicit format let that scanner pick its own column, so
    # e.g. fmt='sharegpt' always scans 'conversations', not 'messages' (P1 fix);
    # gptoss has its own messages-first rule. alpaca never takes a column.
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
        if fmt == "unknown":
            print(f"  Result:       NOT SCANNED -- format could not be detected")
        else:
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

    # Rows with no Nones - compute the count directly rather than allocating a
    # full set of row indices, which OOMs on large (10M+ row) datasets.
    bad_indices = set(stats.get("bad_row_indices", []))
    clean_count = total - len(bad_indices)
    if 0 < clean_count <= 20:
        clean_indices = [i for i in range(total) if i not in bad_indices]
        print(f"  Rows with no Nones: {clean_count} / {total}  {clean_indices}")
    else:
        print(f"  Rows with no Nones: {clean_count} / {total}")

    print(f"{'=' * 64}")
    return True


def print_report(
    stats: dict,
    fmt: str,
    summary_only: bool = False,
):
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


def show_row(
    dataset: Dataset,
    row_indices: list[int],
    fmt: str,
    col: str = None,
):
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

        # Print non-conversation columns. For alpaca, skip fields the alpaca
        # block below prints with status markers (avoid double render).
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
                    # Mirror scanner logic: from+value wins, then role, then from alone.
                    if "from" in t and "value" in t:
                        c = t.get("value")
                    elif "role" in t:
                        c = t.get("content") if "content" in t else t.get("value")
                    elif "from" in t:
                        c = t.get("value")
                    else:
                        c = t.get("content") if "content" in t else t.get("value")
                    # Mirror scanner: tool_calls exemption is assistant-only;
                    # other roles with empty content + tool_calls are still bad.
                    r = t.get("role") if t.get("role") is not None else t.get("from")
                    if is_none_or_empty(c) and not (str(r) == "assistant" and t.get("tool_calls")):
                        return True
                    return False

                none_count = sum(1 for t in conversation if _is_bad_turn(t))
                print(f"  {col}: {len(conversation)} turns ({none_count} None)")
                print(f"  {'-' * 60}")
                for i, turn in enumerate(conversation):
                    # Non-dict turn - can't extract role/content normally.
                    if not isinstance(turn, dict):
                        label = "None" if turn is None else "invalid_type"
                        print(f"  [{i:>3d}] {'unknown':<12s} [{label}]  << NONE")
                        continue
                    r = turn.get("role")
                    if r is None:
                        r = turn.get("from")
                    role = "?" if r is None else str(r)
                    # Mirror scanner logic: from+value wins, then role, then from alone.
                    if "from" in turn and "value" in turn:
                        content = turn.get("value")
                    elif "role" in turn:
                        content = turn.get("content") if "content" in turn else turn.get("value")
                    elif "from" in turn:
                        content = turn.get("value")
                    else:
                        content = turn.get("content") if "content" in turn else turn.get("value")
                    if is_none_or_empty(content) and not (
                        role == "assistant" and turn.get("tool_calls")
                    ):
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
    parser.add_argument("dataset", help = "HuggingFace dataset repo id (e.g. org/my-dataset)")
    parser.add_argument("--split", default = "train", help = "Dataset split to load (default: train)")
    parser.add_argument(
        "--format",
        default = "auto",
        choices = ["auto"] + FORMAT_NAMES + list(FORMAT_ALIASES),
        help = "Force a specific format instead of auto-detecting (default: auto). "
        "Documented aliases (e.g. 'gpt-oss' for 'gptoss') are also accepted.",
    )
    parser.add_argument(
        "--summary-only",
        action = "store_true",
        help = "Print summary header only - skip the per-turn findings list",
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
        # header in exception messages. Redact the token before printing.
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
