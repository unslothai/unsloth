# SPDX-License-Identifier: AGPL-3.0-only
"""The About tab must show a runtime row on an Intel XPU host.

`hardware.py` has always emitted `versions["xpu"]`, but the frontend only ever read `cuda`
and `rocm`. On an Arc host both of those are null, so the runtime row vanished entirely
while the GPU name and VRAM rows still rendered -- a host that looks half-detected. That
was unreachable on Windows until the installer learned to select XPU wheels, which is what
makes it worth a guard now.
"""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HARDWARE = REPO / "studio/backend/utils/hardware/hardware.py"
HOOK = REPO / "studio/frontend/src/hooks/use-hardware-info.ts"
ABOUT = REPO / "studio/frontend/src/features/settings/tabs/about-tab.tsx"
LOCALES = REPO / "studio/frontend/src/i18n/locales"


def test_backend_still_emits_an_xpu_version():
    # If this ever stops being true the frontend field below is dead weight, so fail loudly
    # rather than leave a field nothing populates.
    src = HARDWARE.read_text(encoding = "utf-8")
    assert 'versions["xpu"]' in src


def test_hardware_info_declares_and_maps_xpu():
    src = HOOK.read_text(encoding = "utf-8")
    assert "xpu: string | null;" in src, "HardwareInfo must declare xpu"
    assert "xpu: data?.versions?.xpu ?? null," in src, "xpu must be mapped from the API response"
    # The default has to carry the key too, otherwise the first render is `undefined`
    # rather than null and the row flickers.
    assert src.count("xpu: null,") >= 1


def test_about_tab_renders_the_xpu_runtime_row():
    src = ABOUT.read_text(encoding = "utf-8")
    assert 'labelKey: "settings.about.xpu"' in src, "About tab must offer an xpu runtime label"
    assert "hw.xpu" in src
    # The section itself must open for an xpu-only host, not just for cuda/rocm.
    assert "hw.gpus.length > 0 || runtime" in src


def test_the_xpu_label_resolves_in_every_locale():
    # en is the fallback source, so it is the only file that MUST carry the key --
    # check-parity.ts states overlays may be partial and missing keys fall back to English.
    # Requiring it everywhere would break the next locale anyone adds for no gain: the
    # label is a proper noun, so the fallback is byte-identical to a translation.
    assert 'xpu: "XPU",' in (LOCALES / "en.ts").read_text(encoding = "utf-8")
    # What the overlays must not do is disagree with en, which is what would actually
    # render wrong. check-parity.ts rejects a key absent from en; this catches the value.
    wrong = [
        p.name
        for p in sorted(LOCALES.glob("*.ts"))
        for line in p.read_text(encoding = "utf-8").splitlines()
        if line.strip().startswith("xpu:") and line.strip() != 'xpu: "XPU",'
    ]
    assert not wrong, f"locales whose xpu label diverges from en: {wrong}"
