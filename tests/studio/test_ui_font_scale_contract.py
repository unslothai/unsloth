# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UI font size scaling contracts (Settings > Appearance).

The preference must scale typography through the --ui-font-scale tokens,
never by mutating the root font size, so rem-based layout stays put. These
contracts also act as the guard against reintroducing raw pixel typography
that would silently ignore the preference.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "studio/frontend/src"
INDEX_CSS = (SRC / "index.css").read_text(encoding = "utf-8")
STORE = (SRC / "features/settings/stores/appearance-custom-store.ts").read_text(encoding = "utf-8")
SELECT = (SRC / "components/ui/select.tsx").read_text(encoding = "utf-8")
UTILS = (SRC / "lib/utils.ts").read_text(encoding = "utf-8")

# Raw numeric fontSize props are only allowed where a scaled stylesheet rule
# (.recharts-text) overrides the presentation attribute at render time.
FONTSIZE_PROP_ALLOWED_DIRS = (
    "features/studio/sections/charts",
    "features/studio/sections/training-section.tsx",
)

# Non-visible typography that intentionally stays fixed.
FONTSIZE_STYLE_ALLOWLIST = {
    # Offscreen textarea; 12pt+ suppresses the iOS focus zoom. Never rendered.
    "lib/copy-to-clipboard.ts",
}


def _frontend_sources():
    for path in sorted(SRC.rglob("*")):
        if path.suffix in {".ts", ".tsx", ".css"}:
            yield path


def test_preference_writes_a_scale_not_the_root_font_size():
    assert 'setVar("--ui-font-scale"' in STORE
    assert 'el.setAttribute("data-ui-font-size"' in STORE
    # Older builds set an inline root font-size; the applier must clear it.
    assert 'style.removeProperty("font-size")' in STORE
    assert "style.fontSize" not in STORE


def test_named_text_tokens_scale():
    for token, rem in (
        ("--text-xs", "0.75rem"),
        ("--text-sm", "0.875rem"),
        ("--text-base", "1rem"),
        ("--text-lg", "1.125rem"),
    ):
        assert f"{token}: calc({rem} * var(--ui-font-scale, 1));" in INDEX_CSS


def test_numeric_leading_scales_with_the_preference():
    for n, rem in ((3, "0.75rem"), (5, "1.25rem"), (6, "1.5rem")):
        assert f"--leading-{n}: calc({rem} * var(--ui-font-scale, 1));" in INDEX_CSS


def test_ui_token_families_exist():
    assert "--text-ui-11: calc(0.6875rem * var(--ui-font-scale, 1));" in INDEX_CSS
    assert "--text-ui-10p5: calc(0.65625rem * var(--ui-font-scale, 1));" in INDEX_CSS
    assert "--leading-ui-17: calc(1.0625rem * var(--ui-font-scale, 1));" in INDEX_CSS


def test_explicit_code_font_size_is_never_multiplied():
    match = re.search(r"html\[data-code-font-size\][^{]*\{([^}]*)\}", INDEX_CSS)
    assert match is not None
    body = match.group(1)
    assert "var(--custom-code-font-size)" in body
    assert "--ui-font-scale" not in body


def test_radix_select_viewport_owns_the_scroll_state():
    viewport = SELECT[SELECT.index("SelectPrimitive.Viewport") :]
    assert "overflow-y-auto" in viewport.split("</SelectPrimitive.Viewport>")[0]
    # The rounded surface itself must not scroll (WebKit squares its corners).
    content_cls = re.search(
        r"SelectPrimitive\.Content[\s\S]*?className=\{cn\(\s*\"([^\"]+)\"", SELECT
    )
    assert content_cls is not None
    assert "overflow-hidden" in content_cls.group(1)
    assert "overflow-y-auto" not in content_cls.group(1)


def test_cn_knows_the_ui_typography_tokens():
    """Stock tailwind-merge classifies text-ui-* as a text color and deletes
    it whenever a real color class follows in the same cn() call, so the
    element falls back to the unscaled inherited font size."""
    assert "extendTailwindMerge" in UTILS
    assert '"font-size": [{ text: [isUiToken] }]' in UTILS
    assert "leading: [{ leading: [isUiToken] }]" in UTILS
    assert "/^ui-\\d+(p5)?$/.test(value)" in UTILS


def test_icons_follow_the_ui_font_scale():
    """Glyphs beside scaled labels track the preference: the shared
    --icon-size token, the scoped menu/toast/chat svg overrides, and the
    sonner toast text that is otherwise pinned at 13px."""
    assert "--icon-size: calc(18px * var(--ui-font-scale, 1));" in INDEX_CSS
    assert "& svg.size-4 { width: calc(1rem * var(--ui-font-scale, 1));" in INDEX_CSS
    assert "font-size: calc(13px * var(--ui-font-scale, 1)) !important;" in INDEX_CSS
    for scope in (
        "[data-slot='dropdown-menu-content']",
        "[data-slot='select-content']",
        "[data-sonner-toast]",
        ".aui-root",
    ):
        assert scope in INDEX_CSS


def test_no_raw_pixel_text_utilities():
    offenders = []
    for path in _frontend_sources():
        text = path.read_text(encoding = "utf-8")
        for m in re.finditer(r"(?<![\w-])(?:text|leading)-\[[0-9.]+px\]", text):
            offenders.append(f"{path.relative_to(SRC)}: {m.group(0)}")
    assert offenders == [], (
        "Raw px text utilities ignore the UI font size preference; use the "
        f"text-ui-* / leading-ui-* tokens in index.css instead: {offenders[:10]}"
    )


def test_css_font_sizes_reference_the_scale():
    offenders = []
    for path in _frontend_sources():
        if path.suffix != ".css":
            continue
        text = path.read_text(encoding = "utf-8")
        for m in re.finditer(r"(font-size|line-height):[^;{}]*;", text):
            decl = m.group(0)
            if re.search(r"[0-9.]+(px|rem)", decl) is None:
                continue  # unitless ratios and vars scale naturally
            if "--ui-font-scale" in decl:
                continue
            if "1px" in decl:
                continue  # library layout tricks (KaTeX-style), not text
            offenders.append(f"{path.relative_to(SRC)}: {decl.strip()[:80]}")
    assert offenders == [], (
        "CSS typography must multiply by var(--ui-font-scale, 1) or be "
        f"allowlisted here with a reason: {offenders[:10]}"
    )


def test_inline_font_size_styles_reference_the_scale():
    offenders = []
    for path in _frontend_sources():
        rel = str(path.relative_to(SRC))
        if rel in FONTSIZE_STYLE_ALLOWLIST:
            continue
        text = path.read_text(encoding = "utf-8")
        for m in re.finditer(r"fontSize:\s*([\"'][^\"']+[\"']|[0-9.]+)", text):
            value = m.group(1)
            if "--ui-font-scale" in value:
                continue
            if value.replace(".", "").isdigit() and any(
                rel.startswith(d) for d in FONTSIZE_PROP_ALLOWED_DIRS
            ):
                continue  # covered by the .recharts-text override
            offenders.append(f"{rel}: fontSize {value}")
        for m in re.finditer(r"fontSize=\{?([0-9.]+)\}?", text):
            if not any(rel.startswith(d) for d in FONTSIZE_PROP_ALLOWED_DIRS):
                offenders.append(f"{rel}: fontSize={m.group(1)}")
    assert offenders == [], (
        "Inline font sizes must scale with var(--ui-font-scale, 1) or be "
        f"documented in the allowlist: {offenders[:10]}"
    )
