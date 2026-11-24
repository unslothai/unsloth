# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A dataset Image cell holding PostScript must not reach the host's Ghostscript."""

import subprocess

import pytest


def test_an_eps_image_cell_never_starts_ghostscript(monkeypatch):
    datasets = pytest.importorskip("datasets")
    from PIL import EpsImagePlugin

    from datasets.features import image as hf_image
    from utils.datasets import remote_images

    remote_images.install_guarded_image_decode(hf_image)
    started = []

    def fake_check_call(cmd, *args, **kwargs):
        started.append(cmd)
        return 0

    def fake_check_output(cmd, *args, **kwargs):
        started.append(cmd)
        raise OSError("not run")

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 4 4\n{ } loop\n%%EOF\n"

    with pytest.raises(Exception):
        datasets.Image().decode_example({"bytes": eps, "path": None}).load()

    assert started == []
    assert EpsImagePlugin.has_ghostscript() is False
