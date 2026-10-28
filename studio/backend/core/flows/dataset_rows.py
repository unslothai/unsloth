# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rows of a flow `dataset` value for the viewers, read the way the Train page's dataset check reads them."""

from __future__ import annotations

import re
from typing import Any, Optional

# The viewers page through the first rows only; a dataset is read from its start each time.
VIEW_ROWS = 1000
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class DatasetRowsError(ValueError):
    pass


def dataset_source(value: Any) -> dict:
    """The dataset name (Hub id or local path) plus split and subset a `dataset` value points at."""
    if not isinstance(value, dict) or value.get("kind") != "dataset":
        raise DatasetRowsError("Not a dataset")
    source, ref = value.get("source"), str(value.get("ref") or "").strip()
    if not ref:
        raise DatasetRowsError("This dataset has no reference")
    if source == "hf":
        if not _REPO.match(ref):
            raise DatasetRowsError("Name a Hugging Face dataset like owner/name")
        return {
            "dataset_name": ref,
            "subset": value.get("subset") or None,
            "train_split": value.get("split") or "train",
        }
    if source in ("local", "recipe"):
        name = ref
        if ref.startswith("ref:"):
            from hub.utils.host_paths import resolve_host_path_reference
            name = resolve_host_path_reference(ref) or ""
            if not name:
                raise DatasetRowsError(
                    "This dataset is no longer available; run the step that made it again"
                )
        from utils.paths import resolve_dataset_path

        # The account-scoped path, so the managed-account gate never reads a relative ref from the cwd.
        return {"dataset_name": str(resolve_dataset_path(name)), "subset": None, "train_split": "train"}
    raise DatasetRowsError(f"Unsloth Studio cannot show a {source!r} dataset")


def dataset_rows(value: Any, offset: int, limit: int, hf_token: Optional[str]) -> dict:
    from core.training.account_jobs import account_hf_token, account_path
    from hub.schemas.datasets import CheckFormatRequest
    from hub.services.datasets.formatting import _serialize_preview_rows, load_preview_slice

    end = min(offset + limit, VIEW_ROWS)
    request = CheckFormatRequest(**dataset_source(value))
    account_path(request.dataset_name, reference = True)
    rows, total = load_preview_slice(request, account_hf_token(hf_token), max(end, 1))
    page = _serialize_preview_rows(rows)[offset:end] if end > offset else []
    return {"rows": page, "total": total, "offset": offset, "view_rows": VIEW_ROWS}
