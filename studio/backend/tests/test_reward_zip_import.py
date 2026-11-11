# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ZIP import of REWARD.md files: bounded, link and traversal free, and all-or-nothing."""

import io
import stat
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.training import rewards
from routes.rewards import router


def _md(name, body = "type: length\nmax_chars: 10"):
    return f"---\nname: {name}\nkind: rule\n---\n{body}"


def _zip(*entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            if isinstance(entry, zipfile.ZipInfo):
                archive.writestr(entry, "target")
            else:
                archive.writestr(*entry)
    return buffer.getvalue()


@pytest.fixture
def user_root(tmp_path, monkeypatch):
    root = tmp_path / "rewards"
    monkeypatch.setattr(rewards, "_user_root", lambda: root)
    return root


@pytest.fixture
def client(user_root):
    app = FastAPI()
    app.include_router(router, prefix = "/api/rewards")
    app.dependency_overrides[get_current_subject] = lambda: "test"
    return TestClient(app)


def test_zip_round_trip_imports_every_reward(client, user_root):
    data = _zip(
        ("a/REWARD.md", _md("a")),
        ("b/REWARD.md", _md("b")),
        ("__MACOSX/a/REWARD.md", "junk"),
        ("notes.txt", "ignored"),
    )
    response = client.post("/api/rewards/import-zip", content = data)
    assert response.status_code == 201, response.text
    assert [r["name"] for r in response.json()] == ["a", "b"]
    assert sorted(p.name for p in user_root.iterdir()) == ["a", "b"]


def test_one_invalid_definition_writes_nothing(client, user_root):
    data = _zip(("a/REWARD.md", _md("a")), ("b/REWARD.md", _md("b", "type: nope")))
    response = client.post("/api/rewards/import-zip", content = data)
    assert response.status_code == 400
    assert not user_root.exists() or not any(user_root.iterdir())


def test_a_failed_write_restores_the_rewards_written_before_it(user_root):
    rewards.import_reward(_md("a", "type: length\nmax_chars: 1"))
    # A plain file where "c" goes passes every check, then fails its write.
    (user_root / "c").write_text("not a folder")
    data = _zip(("a/REWARD.md", _md("a")), ("b/REWARD.md", _md("b")), ("c/REWARD.md", _md("c")))
    with pytest.raises(OSError):
        rewards.import_reward_zip(data, overwrite = True)
    assert rewards.get_reward("a")["rule"]["max_chars"] == 1
    assert not (user_root / "b").exists()


def test_existing_names_need_explicit_replace(client, user_root):
    rewards.import_reward(_md("a", "type: length\nmax_chars: 1"))
    data = _zip(("a/REWARD.md", _md("a")), ("b/REWARD.md", _md("b")))
    refused = client.post("/api/rewards/import-zip", content = data)
    assert refused.status_code == 409
    assert "a" in refused.json()["detail"]
    assert not (user_root / "b").exists()
    replaced = client.post("/api/rewards/import-zip?overwrite=true", content = data)
    assert replaced.status_code == 201
    assert rewards.get_reward("a")["rule"]["max_chars"] == 10


@pytest.mark.parametrize("name", ["../a/REWARD.md", "/abs/REWARD.md", "C:/x/REWARD.md"])
def test_paths_outside_the_archive_are_refused(user_root, name):
    with pytest.raises(rewards.RewardError, match = "outside"):
        rewards.import_reward_zip(_zip((name, _md("a"))))


def test_links_are_refused(user_root):
    link = zipfile.ZipInfo("a/REWARD.md")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(rewards.RewardError, match = "link"):
        rewards.import_reward_zip(_zip(link))


def test_sizes_are_bounded_while_inflating(user_root, monkeypatch):
    big = _md("a", "type: length\nmax_chars: 10\n" + "#" * rewards.MAX_REWARD_MD_BYTES)
    with pytest.raises(rewards.RewardError, match = "64 KB"):
        rewards.import_reward_zip(_zip(("a/REWARD.md", big)))
    monkeypatch.setattr(rewards, "MAX_REWARD_ZIP_BYTES", 64)
    with pytest.raises(rewards.RewardError, match = "larger"):
        rewards.import_reward_zip(_zip(("a/REWARD.md", _md("a"))))


def test_duplicates_aliases_and_junk_are_refused(user_root):
    with pytest.raises(rewards.RewardError, match = "twice"):
        rewards.import_reward_zip(_zip(("a/REWARD.md", _md("a")), ("x/REWARD.md", _md("a"))))
    alias = "---\nname: a\nkind: rule\n---\ntype: length\nmax_chars: &n 10\nx: *n"
    with pytest.raises(rewards.RewardError, match = "aliases"):
        rewards.import_reward_zip(_zip(("a/REWARD.md", alias)))
    with pytest.raises(rewards.RewardError, match = "no REWARD.md"):
        rewards.import_reward_zip(_zip(("readme.txt", "x")))
    with pytest.raises(rewards.RewardError, match = "not a valid ZIP"):
        rewards.import_reward_zip(b"not a zip")


def test_an_oversized_upload_is_refused_before_parsing(client, monkeypatch):
    import routes.rewards as route

    monkeypatch.setattr(route, "MAX_REWARD_ZIP_BYTES", 8)
    response = client.post("/api/rewards/import-zip", content = b"x" * 64)
    assert response.status_code == 413
