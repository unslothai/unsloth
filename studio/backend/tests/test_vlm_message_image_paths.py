# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""String images in messages rows go to the vision collator, which opens them itself: a managed account may
name only files it may read there, as in flat image columns and video parts. The owner is unchanged."""

import io

import pytest

from auth import policy
from utils import hf_cache_settings
from utils.account_context import AccountContext, run_as
from utils.paths import storage_roots

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def guarded(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    cache = tmp_path / "hub-cache"
    cache.mkdir()
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(cache))
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])


def _png(path):
    from PIL import Image

    path.parent.mkdir(parents = True, exist_ok = True)
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format = "PNG")
    path.write_bytes(buf.getvalue())
    return str(path)


def _rows(image):
    return [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "what?"}, image]},
                {"role": "assistant", "content": [{"type": "text", "text": "red"}]},
            ]
        }
    ]


def _format(rows, account):
    from utils.datasets.dataset_utils import format_and_template_dataset
    return run_as(
        account,
        format_and_template_dataset,
        rows,
        model_name = "tiny-vlm",
        tokenizer = None,
        is_vlm = True,
        format_type = "auto",
    )


@pytest.mark.parametrize(
    "part",
    [
        lambda p: {"type": "image", "image": p},
        lambda p: {"type": "image", "image": "file://" + p},
        lambda p: {"type": "image_url", "image_url": {"url": p}},
    ],
)
def test_a_managed_account_cannot_name_a_host_image_in_a_messages_row(tmp_path, part):
    secret = _png(tmp_path / "owner-private" / "scan.png")

    result = _format(_rows(part(secret)), ALICE)

    assert result["success"] is False
    assert any("not readable by this account" in error for error in result["errors"])


def test_a_managed_accounts_own_image_still_trains(tmp_path):
    own = run_as(ALICE, lambda: _png(storage_roots.dataset_uploads_root() / "mine.png"))

    result = _format(_rows({"type": "image", "image": own}), ALICE)

    assert result["success"] is True, result["errors"]


def test_the_owner_still_names_any_image(tmp_path):
    from utils.account_context import OWNER

    secret = _png(tmp_path / "owner-private" / "scan.png")

    result = _format(_rows({"type": "image", "image": secret}), OWNER)

    assert result["success"] is True, result["errors"]
