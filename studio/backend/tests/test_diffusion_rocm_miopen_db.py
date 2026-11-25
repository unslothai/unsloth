# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shipped ROCm MIOpen db for diffusion VAEs: exact-stack match, opt-out, release restores (no GPU)."""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_rocm_miopen_db as md  # noqa: E402

STACK = {
    "arch": "gfx1151",
    "torch": "2.11.0+rocm7.14.0",
    "hip": "7.14.60850-fake",
    "miopen": "3005002",
    "platform": "win32",
}
DB_NAME = "gfx1151_20.HIP.3_5_2_cd957402"


def _ship(
    root: Path,
    name: str = "gfx1151_win_test",
    stack: dict = STACK,
    files: tuple = ("ufdb", "udb"),
) -> Path:
    d = root / name
    d.mkdir(parents = True)
    names = []
    for kind in files:
        f = d / f"{DB_NAME}.{kind}.txt"
        f.write_text(f"{kind}-record=Solver:1.0\n", encoding = "utf-8")
        names.append(f.name)
    (d / md.MANIFEST).write_text(
        json.dumps({"stack": stack, "files": names, "measured": "test"}), encoding = "utf-8"
    )
    return d


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(md, "is_rocm_torch", lambda _t: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(md, "_arch", lambda device = None: "gfx1151")
    monkeypatch.setattr(md, "_device_count", lambda _t: 1)
    monkeypatch.setattr(md, "current_stack", lambda arch: dict(STACK, arch = arch))
    monkeypatch.setattr(md, "_DB_DIR", tmp_path / "shipped")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR", str(tmp_path / "cache"))
    for k in [k for k in __import__("os").environ if k.startswith("MIOPEN_")] + [md.MIOPEN_DB_ENV]:
        monkeypatch.delenv(k, raising = False)
    md._STATE["owners"].clear()
    md._STATE.update(active = False, path = None, snapshot = md._UNSET, db = None)
    yield tmp_path
    md._STATE["owners"].clear()
    md._STATE.update(active = False, path = None, snapshot = md._UNSET, db = None)


def _pipe():
    return types.SimpleNamespace(vae = object())


def test_matching_db_is_seeded_and_release_restores(env, monkeypatch):
    import os

    _ship(env / "shipped")
    pipe = _pipe()
    assert md.install_for_pipe(pipe)
    dest = Path(os.environ[md.USER_DB_ENV])
    assert dest == env / "cache" / "miopen_db" / "gfx1151_win_test"
    assert (dest / f"{DB_NAME}.ufdb.txt").read_text(encoding = "utf-8") == "ufdb-record=Solver:1.0\n"
    assert md.is_active()
    md.release(pipe)
    assert md.USER_DB_ENV not in os.environ
    assert not md.is_active()


@pytest.mark.parametrize("key", md.STACK_KEYS)
def test_any_stack_difference_leaves_miopen_alone(env, key):
    import os

    _ship(env / "shipped", stack = dict(STACK, **{key: "other"}))
    assert not md.install_for_pipe(_pipe())
    assert md.USER_DB_ENV not in os.environ


@pytest.mark.parametrize("key", md.STACK_KEYS)
def test_a_manifest_missing_a_key_never_matches(env, key):
    stack = {k: v for k, v in STACK.items() if k != key}
    _ship(env / "shipped", stack = stack)
    assert md.shipped_db(STACK) is None


def test_a_manifest_naming_a_missing_file_never_matches(env):
    d = _ship(env / "shipped")
    (d / f"{DB_NAME}.udb.txt").unlink()
    assert md.shipped_db(STACK) is None


@pytest.mark.parametrize(
    "var,value",
    [
        (md.USER_DB_ENV, "/home/u/my_db"),
        ("MIOPEN_FIND_MODE", "1"),
        ("MIOPEN_FIND_ENFORCE", "4"),
        (md.MIOPEN_DB_ENV, "0"),
    ],
)
def test_user_settings_and_opt_out_decline(env, monkeypatch, var, value):
    import os

    _ship(env / "shipped")
    monkeypatch.setenv(var, value)
    assert not md.install_for_pipe(_pipe())
    assert os.environ.get(md.USER_DB_ENV) == (value if var == md.USER_DB_ENV else None)


def test_opting_out_after_a_load_restores_on_the_next_load(env, monkeypatch):
    import os

    _ship(env / "shipped")
    assert md.install_for_pipe(_pipe())
    monkeypatch.setenv(md.MIOPEN_DB_ENV, "0")
    assert not md.install_for_pipe(_pipe())
    assert md.USER_DB_ENV not in os.environ and not md.is_active()


@pytest.mark.parametrize("case", ["cuda", "arch", "mixed"])
def test_non_matching_hosts_decline(env, monkeypatch, case):
    import os

    _ship(env / "shipped")
    if case == "cuda":
        monkeypatch.setattr(md, "is_rocm_torch", lambda _t: False)
    elif case == "arch":
        monkeypatch.setattr(md, "_arch", lambda device = None: "gfx1100")
    else:
        monkeypatch.setattr(md, "_device_count", lambda _t: 2)
        monkeypatch.setattr(
            md, "_arch", lambda device = None: "gfx1100" if device == 1 else "gfx1151"
        )
    assert not md.install_for_pipe(_pipe())
    assert md.USER_DB_ENV not in os.environ


def test_the_last_release_restores(env):
    import os

    _ship(env / "shipped")
    a, b = _pipe(), _pipe()
    assert md.install_for_pipe(a) and md.install_for_pipe(b)
    md.release(a)
    assert os.environ.get(md.USER_DB_ENV) and md.is_active()
    md.release(b)
    assert md.USER_DB_ENV not in os.environ


def test_seeded_dir_keeps_miopen_records_until_the_shipped_db_changes(env):
    import os

    d = _ship(env / "shipped")
    p = _pipe()
    assert md.install_for_pipe(p)
    dest = Path(os.environ[md.USER_DB_ENV]) / f"{DB_NAME}.ufdb.txt"
    dest.write_text(dest.read_text(encoding = "utf-8") + "appended=X:2.0\n", encoding = "utf-8")
    md.release(p)
    p = _pipe()
    assert md.install_for_pipe(p)
    assert "appended" in dest.read_text(encoding = "utf-8")
    md.release(p)
    (d / f"{DB_NAME}.ufdb.txt").write_text("ufdb-record=Solver:0.5\n", encoding = "utf-8")
    assert md.install_for_pipe(_pipe())
    assert dest.read_text(encoding = "utf-8") == "ufdb-record=Solver:0.5\n"


def test_a_crlf_checkout_is_seeded_with_lf(env):
    import os

    d = _ship(env / "shipped")
    f = d / f"{DB_NAME}.ufdb.txt"
    f.write_bytes(b"a=X:1.0\r\nb=Y:2.0\r\n")
    assert md.install_for_pipe(_pipe())
    assert (Path(os.environ[md.USER_DB_ENV]) / f.name).read_bytes() == b"a=X:1.0\nb=Y:2.0\n"


def test_a_vanished_pipe_releases_its_claim(env):
    import gc
    import os

    _ship(env / "shipped")

    class Pipe:
        pass

    p = Pipe()
    assert md.install_for_pipe(p)
    del p
    gc.collect()
    assert md.USER_DB_ENV not in os.environ


def test_speed_layer_installs_on_rocm_tiers_and_removes_on_off(monkeypatch):
    from torch import nn

    from core.inference import diffusion_rocm_tunableop as rt
    from core.inference import diffusion_speed as ds

    seen = []
    monkeypatch.setattr(rt, "install_for_pipe", lambda pipe, device = None, logger = None: None)
    monkeypatch.setattr(rt, "uninstall", lambda pipe = None: None)
    monkeypatch.setattr(
        md, "install_for_pipe", lambda pipe, device = None, logger = None: seen.append("install")
    )
    monkeypatch.setattr(md, "uninstall", lambda pipe = None: seen.append("uninstall"))
    pipe = types.SimpleNamespace(transformer = nn.Linear(2, 2), vae = None)
    rocm = types.SimpleNamespace(backend = "rocm", device = "cuda", dtype = torch.bfloat16)
    ds.apply_speed_optims(pipe, rocm, is_gguf = False, family = None, speed_mode = "off")
    assert seen == ["uninstall"]
    cuda = types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = torch.bfloat16)
    ds.apply_speed_optims(pipe, cuda, is_gguf = False, family = None, speed_mode = "off")
    assert seen == ["uninstall"]


def test_unload_releases_the_miopen_db_claim(env):
    import os

    from core.inference import diffusion_speed as ds

    _ship(env / "shipped")
    pipe = _pipe()
    assert md.install_for_pipe(pipe)
    ds.release_rocm_tunableop(pipe)
    assert md.USER_DB_ENV not in os.environ


_MIOPEN_FILE = re.compile(r"^(gfx[0-9a-f]+)_\d+\.HIP\.\d+_\d+_\d+_[0-9a-f]+\.u(f)?db\.txt$")


def test_shipped_dbs_are_well_formed():
    dirs = sorted(p for p in md._DB_DIR.iterdir() if p.is_dir())
    assert dirs, "no shipped MIOpen db"
    for d in dirs:
        m = md._manifest(d)
        stack = m.get("stack") or {}
        assert all(stack.get(k) for k in md.STACK_KEYS), d.name
        assert stack["arch"] in md.TUNED_ARCHES and d.name.startswith(stack["arch"] + "_")
        files = md._db_files(d, m)
        assert files, d.name
        kinds = set()
        for f in files:
            mt = _MIOPEN_FILE.match(f.name)
            assert mt and mt.group(1) == stack["arch"], f.name
            kinds.add(f.name.rsplit(".", 2)[-2])
            for line in f.read_text(encoding = "utf-8").splitlines():
                assert "=" in line, (f.name, line)
        assert "ufdb" in kinds, d.name


def test_shipped_dbs_are_package_data():
    root = Path(md.__file__).resolve().parents[4]
    text = (root / "pyproject.toml").read_text(encoding = "utf-8")
    assert '"backend/core/inference/rocm_miopen_db/*/*"' in text
