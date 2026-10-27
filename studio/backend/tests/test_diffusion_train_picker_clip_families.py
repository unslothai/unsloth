# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The picker narrows families to what its dataset list can feed; the API is not narrowed."""

from __future__ import annotations

import inspect
import io
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from auth.authentication import get_current_subject
from models.training import DiffusionDatasetSummary
from routes.training import router as training_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(training_router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


@pytest.fixture
def ds_root(monkeypatch, tmp_path):
    import utils.paths as up

    root = tmp_path / "assets" / "datasets"
    root.mkdir(parents = True)
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(up, "datasets_root", lambda: root)
    monkeypatch.setattr(up, "outputs_root", lambda: out)
    return root


def _stills_dataset(root: Path, name: str = "cat-photos") -> Path:
    folder = root / name
    folder.mkdir()
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format = "PNG")
    (folder / "a.png").write_bytes(buf.getvalue())
    (folder / "a.txt").write_text("a cat", encoding = "utf-8")
    return folder


def _clip_families() -> set[str]:
    from core.training.diffusion_train_common import CLIP_TRAINED_FAMILIES
    return set(CLIP_TRAINED_FAMILIES)


def _trainable_here() -> set[str]:
    """The families this install can actually train. A diffusers without H3's transformer drops
    it from ``family_train_infos()`` for an unrelated reason, and these tests are about the
    picker gate, not about that probe."""
    from core.training.diffusion_train_common import family_train_infos
    return {i["name"] for i in family_train_infos()}


def _report_clips(
    monkeypatch,
    count: int,
    images: int | None = None,
) -> None:
    """Clip count comes from DiffusionDatasetSummary.clip_count when present, else from a subclass."""
    import routes.training as tr

    has_field = "clip_count" in getattr(DiffusionDatasetSummary, "model_fields", {})

    class _WithClips(DiffusionDatasetSummary):
        clip_count: int = 0

    model = DiffusionDatasetSummary if has_field else _WithClips
    real = tr._diffusion_dataset_summary

    def fake(folder: Path):
        base = real(folder)
        return model(
            name = base.name,
            path = base.path,
            image_count = base.image_count if images is None else images,
            caption_count = base.caption_count,
            clip_count = count,
        )

    monkeypatch.setattr(tr, "_diffusion_dataset_summary", fake)


def test_a_clip_family_is_not_offered_while_every_listed_dataset_is_stills(client, ds_root):
    """The bug: H3 was advertised in the Train tab while the only datasets that tab can list are
    folders of images. Every pick was a dead end -- Start rejected the dataset the picker had
    just offered for it."""
    clip_families = _clip_families() & _trainable_here()
    if not clip_families:
        pytest.skip("this install trains no clip family")
    _stills_dataset(ds_root)

    body = client.get("/api/train/diffusion/info").json()
    assert [d["name"] for d in body["datasets"]] == ["cat-photos"]
    offered = {f["name"] for f in body["families"]}
    assert not (
        offered & clip_families
    ), "a family that trains only on clips must not be offered while no clip dataset is listable"
    assert offered == _trainable_here() - clip_families
    assert "sdxl" in offered


def test_a_clip_family_is_offered_as_soon_as_a_clip_dataset_is_listable(
    client, ds_root, monkeypatch
):
    """The other half, and the reason the gate reads a count rather than naming a family: nothing
    below has to be edited for H3 to light up. The dataset layer reporting a clip is the whole
    trigger."""
    clip_families = _clip_families() & _trainable_here()
    if not clip_families:
        pytest.skip("this install trains no clip family")
    _stills_dataset(ds_root, "video-clips")
    # images = 0: leftover stills would be the mixed case, which Start refuses.
    _report_clips(monkeypatch, 3, images = 0)

    body = client.get("/api/train/diffusion/info").json()
    assert [d["name"] for d in body["datasets"]] == ["video-clips"]
    offered = {f["name"] for f in body["families"]}
    assert clip_families <= offered
    assert offered == _trainable_here()


def test_a_zero_clip_report_is_not_a_clip_dataset(client, ds_root, monkeypatch):
    """The layer reporting the field but counting none is the image-only case, not the clip one.
    Asserted separately because it is the state every existing image dataset will be in once the
    field exists."""
    clip_families = _clip_families() & _trainable_here()
    if not clip_families:
        pytest.skip("this install trains no clip family")
    _stills_dataset(ds_root)
    _report_clips(monkeypatch, 0)

    offered = {f["name"] for f in client.get("/api/train/diffusion/info").json()["families"]}
    assert not (offered & clip_families)


def test_the_gate_withholds_exactly_the_clip_trained_families():
    """Named from ``CLIP_TRAINED_FAMILIES`` -- the set that already says which families read
    clips instead of stills -- so the gate cannot drift from the discovery split it mirrors, and
    a family that trains from stills is never withheld for being video."""
    import routes.training as tr
    from core.training.diffusion_train_common import CLIP_TRAINED_FAMILIES, family_train_infos

    every = {i["name"] for i in family_train_infos()}
    withheld = every - {i["name"] for i in tr._ui_trainable_families([])}
    assert withheld == CLIP_TRAINED_FAMILIES & every
    # LTX-2 is video but trains from stills, so it stays offered.
    if "ltx-2" in every:
        assert "ltx-2" not in withheld

    # The gate must derive from the dataset report, not a hardcoded family name.
    src = (
        inspect.getsource(tr._ui_trainable_families)
        + inspect.getsource(tr._listed_dataset_clip_count)
        + inspect.getsource(tr._listed_dataset_trains_clips)
    )
    assert "minimax" not in src.lower()


def test_a_mixed_folder_does_not_advertise_a_clip_family():
    """A mixed folder must not advertise a clip family, since Start refuses stills for clip families."""
    import routes.training as tr
    from core.training.diffusion_train_common import CLIP_TRAINED_FAMILIES, family_train_infos

    every = {i["name"] for i in family_train_infos()}
    clip_families = CLIP_TRAINED_FAMILIES & every
    assert clip_families, "nothing to assert about if no family trains from clips"

    def summary(name, images, clips_):
        return DiffusionDatasetSummary(
            name = name,
            path = f"/ds/{name}",
            image_count = images,
            clip_count = clips_,
            caption_count = clips_ + images,
        )

    mixed_only = {i["name"] for i in tr._ui_trainable_families([summary("mixed", 4, 6)])}
    assert not (mixed_only & clip_families)

    # The converse, so this cannot pass by withholding the family unconditionally.
    with_clean = {
        i["name"]
        for i in tr._ui_trainable_families([summary("mixed", 4, 6), summary("clips", 0, 6)])
    }
    assert clip_families <= with_clean


def test_a_summary_with_an_unreadable_clip_count_is_treated_as_no_clips():
    """The count is read off whatever the layer hands over, so a junk value must not raise out of
    the info route. No clips is the safe reading: it withholds an option rather than offering one
    that cannot run."""
    import routes.training as tr

    class _Junk:
        clip_count = "lots"

    class _None:
        clip_count = None

    assert tr._listed_dataset_clip_count(_Junk()) == 0
    assert tr._listed_dataset_clip_count(_None()) == 0
    assert tr._listed_dataset_clip_count(object()) == 0


def test_the_api_still_carries_the_family_the_picker_withholds(client, ds_root):
    """The review item asked for the routing to be KEPT: only the advertisement changes. So on a
    stills-only host, where the picker withholds H3, the trainable set the start route resolves
    against still has it."""
    import routes.training as tr
    from core.training.diffusion_train_common import (
        TRAINABLE_VIDEO_FAMILIES,
        family_train_infos,
    )

    clip_families = _clip_families()
    assert clip_families <= TRAINABLE_VIDEO_FAMILIES
    trainable = clip_families & _trainable_here()
    if not trainable:
        pytest.skip("this install trains no clip family")
    _stills_dataset(ds_root)

    offered = {f["name"] for f in client.get("/api/train/diffusion/info").json()["families"]}
    assert not (offered & trainable)
    assert trainable <= {i["name"] for i in family_train_infos()}

    start = inspect.getsource(tr.start_diffusion_training)
    assert "_ui_trainable_families" not in start
    assert "_listed_dataset_clip_count" not in start


def test_the_start_preflight_still_takes_a_clip_dataset_for_the_withheld_family(tmp_path):
    """The Start preflight still accepts clip folders for families the picker withholds."""
    from core.training.diffusion_train_common import discover_training_pairs
    for family in sorted(_clip_families()):
        (tmp_path / f"{family}.mp4").write_bytes(b"\x00" * 16)
        (tmp_path / f"{family}.txt").write_text("a rabbit in a meadow", encoding = "utf-8")
        pairs = discover_training_pairs(family, tmp_path, verify_images = True)
        assert [Path(p).name for p, _ in pairs] == [f"{family}.mp4"]
        assert pairs[0][1] == "a rabbit in a meadow"
        for p in tmp_path.iterdir():
            p.unlink()
