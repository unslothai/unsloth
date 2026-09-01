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
    # If this stops being true the frontend field below is dead weight;
    src = HARDWARE.read_text(encoding = "utf-8")
    assert 'versions["xpu"]' in src


def test_hardware_info_declares_and_maps_xpu():
    src = HOOK.read_text(encoding = "utf-8")
    assert "xpu: string | null;" in src, "HardwareInfo must declare xpu"
    assert "xpu: data?.versions?.xpu ?? null," in src, "xpu must be mapped from the API response"
    # The default must carry the key too, or the first render is `undefined` and the row flickers.
    assert src.count("xpu: null,") >= 1


def test_about_tab_renders_the_xpu_runtime_row():
    src = ABOUT.read_text(encoding = "utf-8")
    assert 'labelKey: "settings.about.xpu"' in src, "About tab must offer an xpu runtime label"
    assert "hw.xpu" in src
    # The section itself must open for an xpu-only host, not just for cuda/rocm.
    assert "hw.gpus.length > 0 || runtimes.length > 0" in src


def test_about_tab_shows_every_runtime_not_just_the_first():
    # hardware.py sets versions["cuda"] and versions["xpu"] independently, so a dual build in forced-XPU mode reports
    src = ABOUT.read_text(encoding = "utf-8")
    assert "acceleratorRuntimes" in src, "the runtime picker must return all matches"
    assert src.count("rows.push(") == 3, "cuda, rocm and xpu must each be pushed"
    assert "runtimes.map(" in src, "every reported runtime must be rendered"
    # An early return is what caused the bug;
    picker = src.split("function acceleratorRuntimes")[1].split("\n}")[0]
    assert picker.count("return") == 1, "the picker must not return early on the first hit"


def test_the_xpu_label_resolves_in_every_locale():
    # en is the fallback source, so it is the only file that MUST carry the key (check-parity.ts allows partial
    assert 'xpu: "XPU",' in (LOCALES / "en.ts").read_text(encoding = "utf-8")
    # Overlays must not DISAGREE with en. check-parity.ts rejects a key absent from en; this
    wrong = [
        p.name
        for p in sorted(LOCALES.glob("*.ts"))
        for line in p.read_text(encoding = "utf-8").splitlines()
        if line.strip().startswith("xpu:") and line.strip() != 'xpu: "XPU",'
    ]
    assert not wrong, f"locales whose xpu label diverges from en: {wrong}"
