"""Run dataset_none_detect.py against synthetic + two HF datasets; log to tests/logs/none_detect_results.log."""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path

# Import dataset_none_detect directly, bypassing utils/datasets/__init__.py (heavy deps).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "studio" / "backend" / "utils" / "datasets"))

from dataset_none_detect import (
    find_none_chatml,
    print_report,
    scan_dataset,
)

LOG_DIR = REPO_ROOT / "tests" / "logs"
LOG_DIR.mkdir(parents = True, exist_ok = True)
LOG_PATH = LOG_DIR / "none_detect_results.log"

# Helpers


class Tee:
    """Write to both stdout and a file simultaneously."""

    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def section(title: str):
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def run_scan(
    dataset,
    label: str,
    fmt: str = "auto",
) -> dict | None:
    print(f"\n--- Scanning: {label} (fmt={fmt}) ---")
    try:
        stats = scan_dataset(dataset, fmt = fmt)
        print_report(stats, stats["format"])
        return stats
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        traceback.print_exc()
        return None


def assert_bad_rows(stats: dict, expected_min: int, label: str):
    bad = len(stats.get("bad_row_indices", []))
    status = "PASS" if bad >= expected_min else "FAIL"
    print(f"  [{status}] {label}: found {bad} bad rows (expected >= {expected_min})")
    return status == "PASS"


def assert_exact_recall(stats: dict, expected_bad: set, label: str):
    """Every injected bad row index must appear in bad_row_indices."""
    actual_bad = set(stats.get("bad_row_indices", []))
    missed = expected_bad - actual_bad
    all_caught = len(missed) == 0
    status = "PASS" if all_caught else "FAIL"
    caught_count = len(expected_bad) - len(missed)
    print(
        f"  [{status}] {label}: exact recall — "
        f"{caught_count}/{len(expected_bad)} injected bad rows caught",
        end = "",
    )
    if missed:
        print(f"  (missed rows: {sorted(missed)})")
    else:
        print()
    return all_caught


# 1. Synthetic datasets

# Minimal mock for hand-crafted rows pyarrow can't represent (e.g. messages=None / "not a list").


class _MockDataset:
    """Behaves like an HF Dataset for iteration, len(), and index access."""

    def __init__(self, rows: list, columns: list):
        self.column_names = columns
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, idx):
        """Support dataset[i] and dataset[i][col] patterns used by _probe_conversation."""
        return self._rows[idx]


def test_p1_fix():
    """find_none_chatml records rows where messages is None or non-list."""
    section("P1 Fix Verification — non-list conversation column values")

    sys.path.insert(0, str(REPO_ROOT / "tests" / "utils"))
    from generate_dataset_with_none import make_chatml_p1_rows

    p1_rows = make_chatml_p1_rows()
    mock_ds = _MockDataset(p1_rows, ["messages"])

    print(f"  Rows under test: {p1_rows}")
    stats = find_none_chatml(mock_ds, col = "messages")
    print_report(stats, "chatml")

    expected_bad = set(range(len(p1_rows)))
    actual_bad = set(stats.get("bad_row_indices", []))
    all_caught = expected_bad.issubset(actual_bad)
    print(
        f"  [{'PASS' if all_caught else 'FAIL'}] P1 fix: all {len(expected_bad)} non-list rows caught"
    )

    for row in stats.get("findings", []):
        vtype = row.get("value_type", "?")
        raw = row.get("raw_value", "?")
        print(f"    row {row['row_index']}: value_type={vtype!r}  raw={raw}")

    return stats


def test_probe_p1_fix():
    """scan_dataset(fmt='auto') on an all-corrupt messages column returns findings, not ValueError."""
    section("P1 Fix Verification — probe skip on all-corrupt column (auto-detect path)")

    # All rows have messages=None, so the probe finds no dict turn.
    all_corrupt_rows = [{"messages": None}] * 5
    mock_ds = _MockDataset(all_corrupt_rows, ["messages"])

    print(f"  Rows under test: {len(all_corrupt_rows)} rows all with messages=None")
    try:
        stats = scan_dataset(mock_ds, fmt = "auto")
        print_report(stats, stats.get("format", "?"))
        bad = len(stats.get("bad_row_indices", []))
        status = "PASS" if bad == len(all_corrupt_rows) else "FAIL"
        print(
            f"  [{status}] Probe P1 fix: {bad}/{len(all_corrupt_rows)} all-corrupt rows caught via auto-detect"
        )
        return stats
    except ValueError as exc:
        print(
            f"  [FAIL] scan_dataset raised ValueError (probe P1 bug NOT fixed): {exc}"
        )
        return None


def test_probe_string_corrupt():
    """P2 fix: a plain-string 'messages' column must NOT be classified as chatml (raises ValueError)."""
    section("P2 Fix Verification — plain-string messages not classified as chatml")

    string_rows = [{"messages": "this is a string, not a list"}] * 5
    mock_ds = _MockDataset(string_rows, ["messages"])

    print(f"  Rows under test: {len(string_rows)} rows all with messages='string'")
    try:
        stats = scan_dataset(mock_ds, fmt = "auto")
        fmt = stats.get("format", "?")
        not_chatml = fmt != "chatml"
        status = "PASS" if not_chatml else "FAIL"
        print(
            f"  [{status}] String-corrupt probe: detected fmt={fmt!r} "
            f"(expected: anything except 'chatml')"
        )
        return stats
    except ValueError as exc:
        # ValueError (unknown format) is the correct outcome for a non-conversation column.
        print(
            f"  [PASS] String-corrupt probe: scan_dataset raised ValueError (not chatml, as expected): {exc}"
        )
        return None


def test_explicit_fmt_corrupt():
    """scan_dataset(fmt='chatml') on an all-corrupt column returns findings, not ValueError."""
    section("P1 Fix Verification — explicit fmt='chatml' on all-corrupt column")

    all_corrupt_rows = [{"messages": None}] * 4 + [{"messages": "not a list"}] * 3
    mock_ds = _MockDataset(all_corrupt_rows, ["messages"])

    print(f"  Rows under test: {len(all_corrupt_rows)} rows (4×None, 3×string)")
    try:
        stats = scan_dataset(mock_ds, fmt = "chatml")
        print_report(stats, stats.get("format", "?"))
        bad = len(stats.get("bad_row_indices", []))
        status = "PASS" if bad == len(all_corrupt_rows) else "FAIL"
        print(
            f"  [{status}] Explicit-fmt P1 fix: {bad}/{len(all_corrupt_rows)} rows caught with fmt='chatml'"
        )
        return stats
    except ValueError as exc:
        print(
            f"  [FAIL] scan_dataset raised ValueError (explicit-fmt P1 NOT fixed): {exc}"
        )
        return None


def test_p2_probe_skips_corrupt_prefers_valid():
    """P2 fix: probe continues past a corrupt 'messages' column to a valid 'conversations' column."""
    section(
        "P2 Fix Verification — probe continues past corrupt first column to valid second"
    )

    # messages column is all-None; conversations is a valid ShareGPT column.
    rows = [
        {
            "messages": None,
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi!"},
            ],
        }
    ] * 5 + [
        {
            "messages": None,
            "conversations": [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": "OK"},
            ],
        }
    ] * 2
    mock_ds = _MockDataset(rows, ["messages", "conversations"])

    print(
        f"  Rows: {len(rows)} — messages=None, conversations=valid ShareGPT (2 bad value='')"
    )
    try:
        stats = scan_dataset(mock_ds, fmt = "auto")
        fmt = stats.get("format", "?")
        col = stats.get("column", "?")
        bad = len(stats.get("bad_row_indices", []))
        # Must detect 'conversations' (sharegpt), not 'messages'.
        correct_col = col == "conversations"
        correct_fmt = fmt == "sharegpt"
        correct_bad = bad == 2
        status = "PASS" if (correct_col and correct_fmt and correct_bad) else "FAIL"
        print(
            f"  [{status}] Probe P2 fix: fmt={fmt!r} col={col!r} bad_rows={bad} "
            f"(expected fmt='sharegpt' col='conversations' bad=2)"
        )
        print_report(stats, fmt)
        return stats
    except ValueError as exc:
        print(f"  [FAIL] scan_dataset raised ValueError: {exc}")
        return None


def test_p2_explicit_fmt_col_priority():
    """P2 fix: explicit fmt='sharegpt' lets find_none_sharegpt pick its own column (conversations)."""
    section(
        "P2 Fix Verification — explicit fmt='sharegpt' respects per-scanner column priority"
    )

    # messages has valid role/content turns (chatml-ish); conversations has bad sharegpt turns.
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "conversations": [
                {"from": "human", "value": None},
                {"from": "gpt", "value": "OK"},
            ],
        }
    ] * 5
    mock_ds = _MockDataset(rows, ["messages", "conversations"])

    print(
        f"  Rows: {len(rows)} — messages=clean chatml, conversations=bad sharegpt (value=None)"
    )
    stats = scan_dataset(mock_ds, fmt = "sharegpt")
    col = stats.get("column", "?")
    bad = len(stats.get("bad_row_indices", []))
    # fmt='sharegpt' scans 'conversations' -> 5 bad rows.
    correct_col = col == "conversations"
    correct_bad = bad == 5
    status = "PASS" if (correct_col and correct_bad) else "FAIL"
    print(
        f"  [{status}] Explicit-fmt P2 fix: col={col!r} bad_rows={bad} "
        f"(expected col='conversations' bad=5)"
    )
    print_report(stats, "sharegpt")
    return stats


def test_p2_gptoss_col_priority():
    """P2 fix: fmt='gptoss' scans 'messages' only, not a clean 'conversations' fallback."""
    section("P2 Fix Verification — fmt='gptoss' scans messages only, not conversations")

    # messages is all-None (corrupt); conversations is clean sharegpt.
    rows = [
        {
            "messages": None,
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi!"},
            ],
        }
    ] * 5
    mock_ds = _MockDataset(rows, ["messages", "conversations"])

    print(
        f"  Rows: {len(rows)} — messages=None (corrupt), conversations=clean sharegpt"
    )
    try:
        stats = scan_dataset(mock_ds, fmt = "gptoss")
        col = stats.get("column", "?")
        bad = len(stats.get("bad_row_indices", []))
        correct_col = col == "messages"
        correct_bad = bad == 5
        status = "PASS" if (correct_col and correct_bad) else "FAIL"
        print(
            f"  [{status}] gptoss P2 fix: col={col!r} bad_rows={bad} "
            f"(expected col='messages' bad=5)"
        )
        print_report(stats, "gptoss")
        return stats
    except ValueError as exc:
        print(f"  [FAIL] scan_dataset raised ValueError: {exc}")
        return None


def test_new_p1_explicit_sharegpt_both_all_corrupt():
    """NEW P1 (commit eb7fea3b7e): fmt='sharegpt' with both columns all-corrupt scans 'conversations', not 'messages'."""
    section(
        "NEW P1 — explicit fmt='sharegpt' scans 'conversations' even when both columns all-corrupt"
    )

    # Both columns are all-corrupt: every row has None.
    rows = [{"messages": None, "conversations": None}] * 5
    mock_ds = _MockDataset(rows, ["messages", "conversations"])

    print(f"  Rows: {len(rows)} — messages=None, conversations=None (both all-corrupt)")
    try:
        stats = scan_dataset(mock_ds, fmt = "sharegpt")
        col = stats.get("column", "?")
        bad = len(stats.get("bad_row_indices", []))
        # Must scan 'conversations', not 'messages'.
        correct_col = col == "conversations"
        correct_bad = bad == 5
        status = "PASS" if (correct_col and correct_bad) else "FAIL"
        print(
            f"  [{status}] New-P1 explicit sharegpt: col={col!r} bad_rows={bad} "
            f"(expected col='conversations' bad=5)"
        )
        print_report(stats, "sharegpt")
        return stats
    except ValueError as exc:
        print(f"  [FAIL] scan_dataset raised ValueError: {exc}")
        return None


def test_new_p2_plain_string_messages_not_chatml():
    """NEW P2 (commit eb7fea3b7e): plain-string 'messages' must NOT be auto-classified as chatml."""
    section("NEW P2 — plain-string 'messages' column must NOT be classified as chatml")

    # messages is a plain text column, not a conversation column.
    rows = [{"messages": "hello world"}] * 5
    mock_ds = _MockDataset(rows, ["messages"])

    print(
        f"  Rows: {len(rows)} — messages='hello world' (plain strings, not conversation)"
    )
    try:
        stats = scan_dataset(mock_ds, fmt = "auto")
        fmt = stats.get("format", "?")
        not_chatml = fmt != "chatml"
        status = "PASS" if not_chatml else "FAIL"
        print(
            f"  [{status}] New-P2 plain-string messages: detected fmt={fmt!r} "
            f"(expected: anything except 'chatml')"
        )
        return stats
    except ValueError as exc:
        # ValueError is also acceptable: not a valid conversation format.
        print(
            f"  [PASS] New-P2 plain-string messages: scan_dataset raised ValueError (not chatml): {exc}"
        )
        return {
            "format": "unknown",
            "total_rows": 5,
            "bad_row_indices": [],
            "findings": [],
        }


def test_synthetic():
    section("1. Synthetic Datasets (generated in-memory)")

    sys.path.insert(0, str(REPO_ROOT / "tests" / "utils"))
    from generate_dataset_with_none import (
        make_alpaca_dataset,
        make_chatml_dataset,
        make_sharegpt_dataset,
    )

    results = {}

    # ChatML — 10 clean rows (0-9), 8 bad rows (10-17)
    ds_chatml = make_chatml_dataset()
    stats = run_scan(ds_chatml, "Synthetic ChatML (messages/role/content)")
    assert_bad_rows(stats, 8, "ChatML bad rows")
    assert_exact_recall(stats, set(range(10, 18)), "ChatML exact recall")
    results["chatml"] = stats

    # ShareGPT — 5 clean rows (0-4), 5 bad rows (5-9)
    ds_sgpt = make_sharegpt_dataset()
    stats = run_scan(ds_sgpt, "Synthetic ShareGPT (conversations/from/value)")
    assert_bad_rows(stats, 3, "ShareGPT bad rows")
    assert_exact_recall(stats, set(range(5, 10)), "ShareGPT exact recall")
    results["sharegpt"] = stats

    # Alpaca — 5 clean rows (0-4), 5 bad rows (5-9)
    ds_alpaca = make_alpaca_dataset()
    stats = run_scan(ds_alpaca, "Synthetic Alpaca (instruction/output)")
    assert_bad_rows(stats, 4, "Alpaca bad rows")
    assert_exact_recall(stats, set(range(5, 10)), "Alpaca exact recall")
    results["alpaca"] = stats

    return results


# 2. HuggingFace: peteromallet/dataclaw-peteromallet


def _brute_force_bad_rows(ds, fmt: str) -> set:
    """Pure-Python ground-truth scanner (no shared code with dataset_none_detect) for independent proof.

    Flags a row bad if any field/turn is None, empty, or whitespace-only; returns bad row indices.
    """

    def _blank(val) -> bool:
        if val is None:
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        return False

    bad: set = set()
    for i, row in enumerate(ds):
        if fmt in ("chatml", "gptoss"):
            msgs = row.get("messages")
            if msgs is None or not isinstance(msgs, list):
                bad.add(i)
                continue
            for turn in msgs:
                if turn is None or (
                    isinstance(turn, dict) and _blank(turn.get("content"))
                ):
                    bad.add(i)
                    break
        elif fmt == "sharegpt":
            convs = row.get("conversations")
            if convs is None or not isinstance(convs, list):
                bad.add(i)
                continue
            for turn in convs:
                if turn is None or (
                    isinstance(turn, dict) and _blank(turn.get("value"))
                ):
                    bad.add(i)
                    break
        elif fmt == "alpaca":
            if _blank(row.get("instruction")) or _blank(row.get("output")):
                bad.add(i)
    return bad


def _assert_hf_no_misses(ds, stats: dict, label: str) -> bool:
    """Independent check: scan_dataset() must find every bad row brute-force finds (no misses)."""
    fmt = stats.get("format", "unknown")
    module_bad = set(stats.get("bad_row_indices", []))

    print(f"  Running brute-force independent scan (fmt={fmt!r}, {len(ds)} rows)...")
    brute_bad = _brute_force_bad_rows(ds, fmt)

    missed = brute_bad - module_bad  # brute-force found, module missed
    extra = module_bad - brute_bad  # module flagged, brute-force didn't

    no_misses = len(missed) == 0
    snippet = ""
    if missed:
        sample = sorted(missed)[:10]
        snippet = f"  (first missed rows: {sample}{'...' if len(missed) > 10 else ''})"
    status = "PASS" if no_misses else "FAIL"
    print(
        f"  [{status}] {label} no-miss check — "
        f"brute-force: {len(brute_bad)} bad rows  |  "
        f"module: {len(module_bad)} bad rows  |  "
        f"missed: {len(missed)}{snippet}"
    )
    if extra:
        # Module may legitimately flag more rows (extra structural checks); informational only.
        print(
            f"  [INFO] {label} — module flagged {len(extra)} rows not in brute-force "
            f"(may reflect additional structural checks, not false positives)"
        )
    return no_misses


def test_dataclaw():
    section("2. HuggingFace — peteromallet/dataclaw-peteromallet")
    try:
        from datasets import load_dataset

        print("  Loading dataset (streaming first 500 rows for speed)...")
        ds = load_dataset(
            "peteromallet/dataclaw-peteromallet",
            split = "train",
            streaming = False,
        )
        print(f"  Loaded {len(ds)} rows, columns: {ds.column_names}")
        stats = run_scan(ds, "dataclaw-peteromallet")
        if stats:
            _assert_hf_no_misses(ds, stats, "dataclaw-peteromallet")
        return stats
    except Exception as exc:
        print(f"  [ERROR] Could not load dataclaw dataset: {exc}")
        traceback.print_exc()
        return None


# 3. HuggingFace: peteromallet/my-personal-codex-data


def test_codex_data():
    section("3. HuggingFace — peteromallet/my-personal-codex-data")
    try:
        # load_dataset fails here (ujson chokes on the large JSONL batch); download + parse raw instead.
        from huggingface_hub import hf_hub_download
        from datasets import Dataset

        print("  Downloading conversations.jsonl via huggingface_hub...")
        path = hf_hub_download(
            "peteromallet/my-personal-codex-data",
            "conversations.jsonl",
            repo_type = "dataset",
        )
        rows = []
        with open(path, encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        ds = Dataset.from_list(rows)
        print(f"  Loaded {len(ds)} rows, columns: {ds.column_names}")
        stats = run_scan(ds, "my-personal-codex-data")
        if stats:
            _assert_hf_no_misses(ds, stats, "my-personal-codex-data")
        return stats
    except Exception as exc:
        print(f"  [ERROR] Could not load codex dataset: {exc}")
        traceback.print_exc()
        return None


# Main


def main():
    started = datetime.now().isoformat()

    with open(LOG_PATH, "w", encoding = "utf-8") as log_file:
        sys.stdout = Tee(log_file)

        print(f"dataset_none_detect.py — Test Run")
        print(f"Started: {started}")
        print(f"Python:  {sys.version}")
        print(f"Log:     {LOG_PATH}")

        all_results = {}

        all_results["p1_fix"] = test_p1_fix()
        all_results["probe_p1_fix"] = test_probe_p1_fix()
        all_results["probe_string_corrupt"] = test_probe_string_corrupt()
        all_results["explicit_fmt_corrupt"] = test_explicit_fmt_corrupt()
        all_results["p2_probe_valid_fallback"] = (
            test_p2_probe_skips_corrupt_prefers_valid()
        )
        all_results["p2_explicit_col_priority"] = test_p2_explicit_fmt_col_priority()
        all_results["p2_gptoss_col_priority"] = test_p2_gptoss_col_priority()
        all_results["new_p1_sharegpt_all_corrupt"] = (
            test_new_p1_explicit_sharegpt_both_all_corrupt()
        )
        all_results["new_p2_plain_string_not_chatml"] = (
            test_new_p2_plain_string_messages_not_chatml()
        )
        all_results["synthetic"] = test_synthetic()
        all_results["dataclaw"] = test_dataclaw()
        all_results["codex_data"] = test_codex_data()

        # Summary table
        section("SUMMARY")
        rows = [
            ("Dataset", "Format", "Total rows", "Bad rows", "Bad turns"),
        ]

        def _row(label, stats):
            if stats is None:
                return (label, "ERROR", "-", "-", "-")
            fmt = stats.get("format", "?")
            total = stats.get("total_rows", "?")
            bad = len(stats.get("bad_row_indices", []))
            turns = stats.get("total_none_turns") or len(stats.get("findings", []))
            return (label, fmt, str(total), str(bad), str(turns))

        for key, label in [
            ("chatml", "Synthetic chatml"),
            ("sharegpt", "Synthetic sharegpt"),
            ("alpaca", "Synthetic alpaca"),
        ]:
            s = all_results.get("synthetic") or {}
            rows.append(_row(label, s.get(key) if isinstance(s, dict) else None))

        rows.append(_row("dataclaw-peteromallet", all_results.get("dataclaw")))
        rows.append(_row("my-personal-codex-data", all_results.get("codex_data")))

        col_widths = [max(len(r[i]) for r in rows) for i in range(5)]
        fmt_str = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
        header = rows[0]
        print(fmt_str.format(*header))
        print("  " + "-" * (sum(col_widths) + 10))
        for row in rows[1:]:
            print(fmt_str.format(*row))

        # Write machine-readable JSON summary alongside the log
        json_path = LOG_DIR / "none_detect_results.json"
        summary = {}
        for key, val in all_results.items():
            if val is None:
                summary[key] = None
            elif isinstance(val, dict):
                # Flatten synthetic sub-keys
                if key == "synthetic":
                    for subkey, subval in val.items():
                        summary[f"synthetic_{subkey}"] = (
                            {
                                "format": subval.get("format"),
                                "total_rows": subval.get("total_rows"),
                                "bad_row_count": len(subval.get("bad_row_indices", [])),
                                "bad_turn_count": subval.get(
                                    "total_none_turns", len(subval.get("findings", []))
                                ),
                            }
                            if subval
                            else None
                        )
                else:
                    summary[key] = {
                        "format": val.get("format"),
                        "total_rows": val.get("total_rows"),
                        "bad_row_count": len(val.get("bad_row_indices", [])),
                        "bad_turn_count": val.get(
                            "total_none_turns", len(val.get("findings", []))
                        ),
                    }

        json_path.write_text(json.dumps(summary, indent = 2), encoding = "utf-8")

        finished = datetime.now().isoformat()
        print(f"\nFinished: {finished}")
        print(f"Log:      {LOG_PATH}")
        print(f"JSON:     {json_path}")

        sys.stdout = sys.stdout.stdout  # restore


if __name__ == "__main__":
    main()
