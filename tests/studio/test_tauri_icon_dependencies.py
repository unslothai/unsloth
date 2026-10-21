# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prevent downgrading the Windows executable-resource and ICO-selection fixes."""

import hashlib
import json
from pathlib import Path
import re

import pytest


TAURI = Path(__file__).resolve().parents[2] / "studio/src-tauri"


@pytest.mark.parametrize(
    "crate,minimum",
    [("tauri", (2, 12, 0)), ("tauri-codegen", (2, 7, 0))],
)
def test_windows_icons_use_resource_aware_tauri(crate, minimum) -> None:
    # tauri #15241 (ICO entry selection), #15274 (exe icon resource).
    entry = re.search(
        rf'\[\[package\]\]\s+name = "{crate}"\s+version = "([^"]+)"',
        (TAURI / "Cargo.lock").read_text(encoding = "utf-8"),
    )
    assert entry is not None, f"{crate} is absent from Cargo.lock"
    assert (
        tuple(map(int, entry.group(1).split("."))) >= minimum
    ), f"{crate} {entry.group(1)} predates the Windows icon fixes"


def test_supplied_icons_are_not_regenerated() -> None:
    # SHA-256 of the supplied design export; artwork must stay byte-for-byte.
    expected = {
        "32x32.png": "908b71cd54669a88ad6559c9325aa28e7517fea14f71b07a42bd597318758952",
        "128x128.png": "9274f7e007422b1de4066167c960f548a9bf0651ed7b421cb4a918bfcc26c503",
        "128x128@2x.png": "e3eed82d6c2f0802c588e0e419a32fb1d558aa8bb508caa5fc8e8151c532e6b1",
        "icon.png": "9cc5a39c109d588fa9d1f57562cf8f49926f611fe490d9863787be3a0011a61b",
        "icon.ico": "2cb4995d79005aeb1a62eb66e20e8d27817772a865fb363343375a40ad08fbfe",
        "linux/16x16.png": "b61159c19a7ec8c5cc0f5d51ba72f968569ad1778f0988a8723829f843ddf5ce",
        "linux/22x22.png": "93097997606ccc085aa0081046a868bb046e20ade901f8f4e538ad191d2ab72a",
        "linux/24x24.png": "785dbb08ab468db67dc9958f2edf8550553857bc388a2091a2b40a31397029ea",
        "linux/32x32.png": "908b71cd54669a88ad6559c9325aa28e7517fea14f71b07a42bd597318758952",
        "linux/48x48.png": "439ee16198b9e28741787e348604a97b21a3edf39f333c51888fae4bec3af7c9",
        "linux/64x64.png": "4319bb0f45b1a5e73aa72b7922478ce9d8221bf92e4cfa593844c3578677d68c",
        "linux/96x96.png": "bb2ef84b4aa6ff574102c39d1e580d529fc26b731fffa6541356ac67e031b83d",
        "linux/128x128.png": "9274f7e007422b1de4066167c960f548a9bf0651ed7b421cb4a918bfcc26c503",
        "linux/256x256.png": "e3eed82d6c2f0802c588e0e419a32fb1d558aa8bb508caa5fc8e8151c532e6b1",
        "linux/512x512.png": "ab9caf2566f3bb98c38cb69595566a4ae50b59639ed258be5b9e68b337bd1f92",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((TAURI / "icons" / name).read_bytes()).hexdigest() == digest, name


def test_macos_and_color_trays_keep_original_artwork() -> None:
    expected = {
        "macos/32x32.png": "791d080e9e7d2f3bf2164bdbeea7ba416b7f9e6c4c06e892ade72b06563fd0fc",
        "macos/128x128.png": "216dcc4b8b6113bbf93b9483d039188965c01b398020d184dd89fd75cb1b1eed",
        "tray-icon-color.png": "791d080e9e7d2f3bf2164bdbeea7ba416b7f9e6c4c06e892ade72b06563fd0fc",
        "icon.icns": "6d4887812a19e536981f4c1e58b4248d73889b2f176284d3b533e6fefa1de919",
        "tray-icon@2x.png": "7fd387bb458e46b97e177071f3d200790637caab21a2b0ed8f10ab8877b2844e",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((TAURI / "icons" / name).read_bytes()).hexdigest() == digest, name
    macos = json.loads((TAURI / "tauri.macos.conf.json").read_text(encoding = "utf-8"))
    assert macos["bundle"]["icon"] == [
        "icons/macos/32x32.png",
        "icons/macos/128x128.png",
        "icons/icon.icns",
    ]
    linux = json.loads((TAURI / "tauri.linux.conf.json").read_text(encoding = "utf-8"))
    assert linux["bundle"]["icon"][0] == "icons/linux/512x512.png"
    assert len(linux["bundle"]["icon"]) == 10
    source = (TAURI / "src/main.rs").read_text(encoding = "utf-8")
    assert 'let tray_icon = tauri::include_image!("./icons/tray-icon-color.png");' in source
    assert "let tray_icon = app.default_window_icon()" not in source
