"""
run_none_detect_tests.py

Runs dataset_none_detect.py against:
  1. Synthetic datasets (chatml, sharegpt, alpaca) with deliberately injected Nones
  2. peteromallet/dataclaw-peteromallet (HuggingFace)
  3. peteromallet/my-personal-codex-data (HuggingFace)

Writes a detailed log to tests/logs/none_detect_results.log

Usage (from repo root, with venv active):
    python tests/run_none_detect_tests.py
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path

# Allow running from repo root without install
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "studio" / "backend"))

from utils.datasets.dataset_none_detect import (
    find_none_chatml,
    print_report,
    scan_dataset,
)

LOG_DIR = REPO_ROOT / "tests" / "logs"
LOG_DIR.mkdir(parents = True, exist_ok = True)
LOG_PATH = LOG_DIR / "none_detect_results.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def run_scan(dataset, label: str, fmt: str = "auto") -> dict | None:
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


# ---------------------------------------------------------------------------
# 1. Synthetic datasets
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Minimal mock so find_none_chatml can be called with hand-crafted rows that
# pyarrow can't represent (e.g. messages=None, messages="not a list").
# ---------------------------------------------------------------------------


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
        """Support dataset[i] (int) and dataset[i][col] patterns used by _probe_conversation."""
        return self._rows[idx]


def test_p1_fix():
    """Verify that find_none_chatml records rows where messages is None or non-list."""
    section("P1 Fix Verification — non-list conversation column values")

    sys.path.insert(0, str(REPO_ROOT / "tests"))
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
    """Verify that scan_dataset(fmt='auto') on a fully-corrupt messages column
    routes through _probe_conversation's all_corrupt=True path and returns
    findings rather than raising ValueError."""
    section("P1 Fix Verification — probe skip on all-corrupt column (auto-detect path)")

    # All 5 rows have messages=None — no dict turn will ever be found in the probe.
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
    """Verify all-corrupt probe path also handles non-list values like strings
    (previously has_list_values guard would skip those columns)."""
    section("P1 Fix Verification — all-corrupt probe with non-list string values")

    string_rows = [{"messages": "this is a string, not a list"}] * 5
    mock_ds = _MockDataset(string_rows, ["messages"])

    print(f"  Rows under test: {len(string_rows)} rows all with messages='string'")
    try:
        stats = scan_dataset(mock_ds, fmt="auto")
        print_report(stats, stats.get("format", "?"))
        bad = len(stats.get("bad_row_indices", []))
        status = "PASS" if bad == len(string_rows) else "FAIL"
        print(f"  [{status}] String-corrupt probe fix: {bad}/{len(string_rows)} rows caught via auto-detect")
        return stats
    except ValueError as exc:
        print(f"  [FAIL] scan_dataset raised ValueError (string-corrupt probe NOT fixed): {exc}")
        return None


def test_explicit_fmt_corrupt():
    """Verify that scan_dataset(fmt='chatml') on an all-corrupt column returns
    findings instead of raising ValueError (explicit format + all_corrupt path)."""
    section("P1 Fix Verification — explicit fmt='chatml' on all-corrupt column")

    all_corrupt_rows = [{"messages": None}] * 4 + [{"messages": "not a list"}] * 3
    mock_ds = _MockDataset(all_corrupt_rows, ["messages"])

    print(f"  Rows under test: {len(all_corrupt_rows)} rows (4×None, 3×string)")
    try:
        stats = scan_dataset(mock_ds, fmt="chatml")
        print_report(stats, stats.get("format", "?"))
        bad = len(stats.get("bad_row_indices", []))
        status = "PASS" if bad == len(all_corrupt_rows) else "FAIL"
        print(f"  [{status}] Explicit-fmt P1 fix: {bad}/{len(all_corrupt_rows)} rows caught with fmt='chatml'")
        return stats
    except ValueError as exc:
        print(f"  [FAIL] scan_dataset raised ValueError (explicit-fmt P1 NOT fixed): {exc}")
        return None


def test_synthetic():
    section("1. Synthetic Datasets (generated in-memory)")

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from generate_dataset_with_none import (
        make_alpaca_dataset,
        make_chatml_dataset,
        make_sharegpt_dataset,
    )

    results = {}

    # ChatML
    ds_chatml = make_chatml_dataset()
    stats = run_scan(ds_chatml, "Synthetic ChatML (messages/role/content)")
    assert_bad_rows(stats, 8, "ChatML bad rows")
    results["chatml"] = stats

    # ShareGPT
    ds_sgpt = make_sharegpt_dataset()
    stats = run_scan(ds_sgpt, "Synthetic ShareGPT (conversations/from/value)")
    assert_bad_rows(stats, 3, "ShareGPT bad rows")
    results["sharegpt"] = stats

    # Alpaca
    ds_alpaca = make_alpaca_dataset()
    stats = run_scan(ds_alpaca, "Synthetic Alpaca (instruction/output)")
    assert_bad_rows(stats, 4, "Alpaca bad rows")
    results["alpaca"] = stats

    return results


# ---------------------------------------------------------------------------
# 2. HuggingFace: peteromallet/dataclaw-peteromallet
# ---------------------------------------------------------------------------


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
        return stats
    except Exception as exc:
        print(f"  [ERROR] Could not load dataclaw dataset: {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 3. HuggingFace: peteromallet/my-personal-codex-data
# ---------------------------------------------------------------------------


def test_codex_data():
    section("3. HuggingFace — peteromallet/my-personal-codex-data")
    try:
        from datasets import load_dataset

        print("  Loading dataset...")
        ds = load_dataset(
            "peteromallet/my-personal-codex-data",
            split = "train",
        )
        print(f"  Loaded {len(ds)} rows, columns: {ds.column_names}")
        stats = run_scan(ds, "my-personal-codex-data")
        return stats
    except Exception as exc:
        print(f"  [ERROR] Could not load codex dataset: {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        all_results["synthetic"] = test_synthetic()
        all_results["dataclaw"] = test_dataclaw()
        all_results["codex_data"] = test_codex_data()

        # ---------------------------------------------------------------------------
        # Summary table
        # ---------------------------------------------------------------------------
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
