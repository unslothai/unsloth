# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG callers stop opening an account's storage once retirement starts."""

from storage import rag_db as _db
from core.training.account_jobs import account_is_retired


def _require_live_account():
    if account_is_retired():
        raise RuntimeError("Account is retired")


def get_connection():
    _require_live_account()
    return _db.get_connection()


def get_metadata_connection():
    _require_live_account()
    return _db.get_metadata_connection()


def __getattr__(name):
    return getattr(_db, name)
