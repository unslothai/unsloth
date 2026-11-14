# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-file CSV uploads whose headers differ still read every cell as written."""

import sys
from pathlib import Path

import pytest
from datasets import load_dataset
from datasets.exceptions import DatasetGenerationCastError

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.datasets.cells import csv_as_text_kwargs  # noqa: E402


def test_csv_files_differing_headers_keep_cells_as_written(tmp_path):
    wide = tmp_path / "wide.csv"
    wide.write_text("instruction,input,output\nZip code?,Holtsville,00501\nPrice?,N/A,72\n")
    narrow = tmp_path / "narrow.csv"
    narrow.write_text("instruction,output\nState?,NA\n")

    files = [str(wide), str(narrow)]
    dataset = load_dataset("csv", data_files = files, split = "train", **csv_as_text_kwargs(files))
    assert len(dataset) == 3
    rows = {row["instruction"]: row for row in dataset}
    assert rows["Zip code?"]["input"] == "Holtsville"
    assert rows["Zip code?"]["output"] == "00501"
    assert rows["Price?"]["input"] == "N/A"
    assert rows["Price?"]["output"] == "72"
    assert rows["State?"]["input"] is None
    assert rows["State?"]["output"] == "NA"

    # A later file adding columns still hits the builder's same-columns rule:
    # a loud error, never silently mistyped cells.
    files = [str(narrow), str(wide)]
    with pytest.raises(DatasetGenerationCastError):
        load_dataset("csv", data_files = files, split = "train", **csv_as_text_kwargs(files))


def test_csv_files_differing_headers_keep_numeric_columns_typed(tmp_path):
    wide = tmp_path / "wide.csv"
    wide.write_text("anchor,positive,score,note\nhi,hello,0.9,N/A\n")
    narrow = tmp_path / "narrow.csv"
    narrow.write_text("anchor,positive,score\ncat,kitten,0.75\n")

    files = [str(wide), str(narrow)]
    dataset = load_dataset("csv", data_files = files, split = "train", **csv_as_text_kwargs(files))
    # Embedding training reads score as a number; only text columns are kept as written.
    assert dataset.features["score"].dtype == "float64"
    assert dataset["score"] == [0.9, 0.75]
    assert dataset["note"] == ["N/A", None]
