# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Every appearance setting the app has, read from its source, is either measured by the long-chat
gate (a mode in _area_matrix.modes) or listed here with the reason it cannot slow an area down.

Glass shipped as a new look that was itself several times slower than the default; the gate only
caught it because MATERIALS happened to be measured. This module closes the gap for every other
appearance setting: a new key in DEFAULT_CUSTOMIZATION or DEFAULT_ADJUSTMENTS, a new value of an
enum setting, a new palette outside the shared palette rule, or a new <html> attribute or class that
switches on costly CSS (backdrop-filter, filter, animation, box-shadow, ...) fails
test_appearance_coverage.py until it is measured or allowlisted with a reason. No browser needed.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "studio" / "frontend" / "src"
MODEL_TS = SRC / "features" / "chat-background" / "model.ts"
CUSTOM_TS = SRC / "features" / "settings" / "lib" / "appearance-customization.ts"
THEMES_TS = SRC / "features" / "settings" / "lib" / "color-themes.ts"

# Measured by the `+custom` modes: the costliest value of each setting.
CUSTOM_MEASURED = {
    "contrast": 100,
    "composerOutline": 100,
    "elementOutline": 100,
    "pointerCursors": True,
    "fontSmoothing": False,
    "codeLineNumbers": True,
    "codeWordWrap": True,
    "codeLigatures": False,
    "chatWidth": "full",
    "uiFontSize": 20,
    "codeFontSize": 20,
    # Forced motion, the default until Reduce motion turned on by default.
    "reduceMotion": "off",
}
# Root attributes / classes the `+custom` modes must show, so a renamed key cannot measure nothing.
CUSTOM_EXPECT = [
    "data-contrast-adjust",
    "data-element-outline",
    "data-composer-outline",
    "data-code-wrap",
    "data-code-line-numbers",
    "data-code-no-ligatures",
    "data-ui-font-size",
    "data-code-font-size",
    "pointer-cursors",
    "no-font-smoothing",
    "force-motion",
]
CUSTOM_ALLOW = {
    "colors": "custom colour tokens (CSS variables only, no effect properties)",
    "uiFont": "font-family variable only",
    "headingFont": "font-family variable only",
    "chatFont": "font-family variable only",
    "codeFont": "font-family variable only",
    "importedFonts": "@font-face list for the font settings above",
    "sentAttachments": "layout of sent attachment rows only",
    "sidebarMenu": "which sidebar menu items show",
    "sidebarNav": "which sidebar items are pinned",
    "sidebarNavAuto": "which sidebar items auto-show",
}
# Enum settings: every value must be the default, measured, or listed here.
ENUM_ALLOW = {
    ("chatWidth", "wide"): "between standard and the measured full width",
    ("reduceMotion", "on"): "removes motion (the default)",
    ("reduceMotion", "system"): "follows the OS; no more than forced motion",
    ("sentAttachments", "list"): "attachment row layout",
    ("sentAttachments", "chips"): "attachment row layout",
}

# Measured by solid+palette: the only palette with its own effect rules (composer glow).
PALETTE_MEASURED = "neon-cyberpunk"

# Measured: the `+wallpaper` modes seed a stored chat wallpaper with these adjustments.
WALLPAPER_MEASURED = {"scope": "all", "conversation": 100, "blur": 20}
WALLPAPER_EXPECT = ["data-chat-wallpaper-ready", "data-chat-wallpaper-workspace-ready"]
WALLPAPER_ALLOW = {
    "effect": "pixel effects are rendered once into the image off the main thread (render.ts worker)",
    "visibility": "opacity of the static image",
    "fade": "static gradient over the image",
    "outline": "outline strength of the static image frame",
    "sizing": "background-size of the static image",
    "zoom": "scale of the static image",
    "position": "background-position of the static image",
    "glassLightVisibility": "opacity of the static image on Glass",
    "glassDarkVisibility": "opacity of the static image on Glass",
    "glassFade": "static gradient over the image on Glass",
}

# <html> attributes / classes that switch on costly CSS: what measures each, or why it needs nothing.
# A value naming a mode must be one of _area_matrix.app_modes() (checked by the test).
ROOT_FLAGS = {
    "data-glass": "liquid-3d",
    "data-glass=liquid-3d": "liquid-3d",
    "data-glass=macos": "macos",
    "data-glass-clear": "transparent",
    "data-glass-menus": "liquid-3d",
    "data-glass-plain": "liquid-3d",
    "data-glass-layers": "allow: WebKit only (layers for Glass surfaces); the Chromium gate cannot run it",
    "data-glass-cpu": "allow: WebKitGTK only (cheaper Glass); the Chromium gate cannot run it",
    "data-glass-motion": "allow: set only while scrolling under software compositing; scroll areas measure it",
    "data-chat-wallpaper-ready": "solid+wallpaper",
    "data-chat-wallpaper-workspace-ready": "liquid-3d+wallpaper",
    "data-chat-welcome": "allow: the empty new-chat screen, not a long chat",
    "data-browser-overlay-open": "allow: set while the browser pane is open; browser_pane measures it",
    "data-input-modality": "allow: keyboard vs pointer focus rings, set by every area",
    "data-input-modality=pointer": "allow: pointer focus rings, set by every clicked area",
    "data-palette": "solid+palette",
    "data-palette=neon-cyberpunk": "solid+palette",
    "data-contrast-adjust": "solid+custom",
    "data-element-outline": "solid+custom",
    "data-composer-outline": "solid+custom",
    "data-code-wrap": "solid+custom",
    "data-interface-zooming": "allow: set only during a zoom gesture",
    "dark": "solid+dark",
    "force-motion": "default",
    "force-reduced-motion": "allow: removes motion",
    "pointer-cursors": "solid+custom",
    "no-font-smoothing": "solid+custom",
    "theme-switching": "allow: set for one frame while the theme changes",
    "tauri": "allow: desktop app shell, not reachable from the browser gate",
    "render-linux": "allow: Linux desktop renderer, not reachable from the browser gate",
    "client-windows": "allow: Windows desktop shell, not reachable from the browser gate",
}

# Persisted Glass settings (useMaterialStore partialize in material.ts): what measures each.
MATERIAL_KEYS = {
    "material": "every MATERIALS entry is its own mode",
    "style": "follows material for every Glass mode",
    "menus": "liquid-3d+menus-off",
    "strength": "liquid-3d+strength-max",
    "lens": "every LENS_MODES entry is its own mode",
    "scope": "allow: 'all' (measured) shows Glass everywhere; other scopes show it on fewer pages",
}
MATERIAL_TS = SRC / "features" / "chat-background" / "material.ts"

COSTLY = re.compile(
    r"(?<![-\w])(?:-webkit-)?(?:backdrop-filter|filter|animation(?:-name)?|will-change|mix-blend-mode"
    r"|mask(?:-image)?|box-shadow)\s*:"
)


def object_keys(source: str, name: str) -> list[str]:
    """Top-level keys of `export const NAME ... = { ... }` in a TS module."""
    m = re.search(rf"export const {name}\b[^=]*=\s*\{{", source)
    if not m:
        raise ValueError(f"{name} not found")
    i, depth, start, keys = m.end(), 1, m.end(), []
    while depth:
        c = source[i]
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        i += 1
    body = source[start : i - 1]
    depth, line = 0, ""
    for c in body:
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        if depth == 0 and c == "\n":
            line = ""
            continue
        line += c
        km = re.match(r"\s*(\w+)\s*:", line)
        if depth == 0 and km and line.rstrip().endswith(":"):
            keys.append(km.group(1))
    return list(dict.fromkeys(keys))


def string_registry(source: str, name: str) -> list[str]:
    m = re.search(rf"export const {name}\s*=\s*\[(.*?)\]\s*as const", source, re.S)
    if not m:
        raise ValueError(f"{name} registry not found")
    return re.findall(r"""["']([^"']+)["']""", m.group(1))


def union_values(source: str, type_name: str) -> list[str]:
    m = re.search(rf"export type {type_name}\s*=([^;]+);", source)
    if not m:
        raise ValueError(f"type {type_name} not found")
    return re.findall(r'"([^"]+)"', m.group(1))


def enum_settings(source: str) -> dict[str, list[str]]:
    """Customization keys typed by an exported string-literal union: {key: values}."""
    body = re.search(r"export type AppearanceCustomization\s*=\s*\{(.*?)\n\};", source, re.S)
    if not body:
        raise ValueError("AppearanceCustomization type not found")
    out = {}
    for key, typ in re.findall(r"^\s*(\w+)\s*:\s*(\w+Setting)\s*;", body.group(1), re.M):
        out[key] = union_values(source, typ)
    return out


def palettes() -> tuple[list[str], set[str]]:
    """Every palette id, and the ids listed in the shared palette rule of index.css."""
    themes = THEMES_TS.read_text(encoding = "utf-8")
    ids = []
    for name in ("UNSLOTH_THEME_IDS", "FLAVOR_THEME_IDS"):
        m = re.search(rf"(?:export )?const {name}\s*=\s*\[(.*?)\]", themes, re.S)
        if m:
            ids += re.findall(r'"([^"]+)"', m.group(1))
    css = (SRC / "index.css").read_text(encoding = "utf-8")
    shared = re.search(
        r":root:where\(((?:\[data-palette=\"[^\"]+\"\],?\s*)+)\)\[data-palette\]\s*\{", css
    )
    listed = set(re.findall(r'data-palette="([^"]+)"', shared.group(1))) if shared else set()
    return [i for i in ids if i != "standard"], listed


def _split_top(selector: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for c in selector:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    return out + [cur]


def _root_compound(part: str) -> str | None:
    """The first compound of a selector when it targets <html> / :root, else None."""
    part = part.strip()
    if not re.match(r"(html|:root)(?![\w-])", part):
        return None
    depth, out = 0, ""
    for c in part:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if depth == 0 and (c.isspace() or c in ">+~"):
            break
        out += c
    return out


def _flags(compound: str) -> set[str]:
    """Attribute names (`name=value` when the selector pins an exact value) and classes."""
    out = set(re.findall(r"\.([\w-]+)", compound))
    for name, op, value in re.findall(
        r"\[([\w-]+)(?:\s*([~|^$*]?=)\s*\"?([^\]\"]*)\"?)?\]", compound
    ):
        out.add(f"{name}={value}" if op == "=" else name)
    return out


def root_flags(css: str) -> dict[str, list[str]]:
    """{flag: [selector, ...]} for <html> attributes / classes on rules that set costly properties,
    nested rules included (a flag on an enclosing rule applies to its children)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags = re.S)
    out: dict[str, list[str]] = {}

    def walk(i: int, inherited: set[str]) -> int:
        text = ""
        while i < len(css):
            c = css[i]
            if c == "{":
                selector = text.split(";")[-1].strip()
                flags = set(inherited)
                if not selector.startswith("@"):
                    for part in _split_top(selector):
                        compound = _root_compound(part.replace("&", ""))
                        if compound:
                            flags |= _flags(compound)
                j, body = i + 1, ""
                depth = 1
                # direct declarations of this block (nested blocks are walked separately)
                k = j
                while k < len(css) and depth:
                    if css[k] == "{":
                        depth += 1
                    elif css[k] == "}":
                        depth -= 1
                    elif depth == 1:
                        body += css[k]
                    k += 1
                if COSTLY.search(body):
                    for f in flags:
                        out.setdefault(f, []).append(selector[:120])
                walk(j, flags)
                i, text = k, ""
                continue
            if c == "}":
                return i + 1
            text += c
            i += 1
        return i

    walk(0, set())
    return out


def css_root_flags() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted(SRC.rglob("*.css")):
        for flag, sels in root_flags(f.read_text(encoding = "utf-8")).items():
            out.setdefault(flag, []).extend(f"{f.relative_to(SRC)}: {s}" for s in sels)
    return out


def uncovered(mode_names: set[str]) -> list[str]:
    """Problems, one sentence each telling the author what to add; empty when covered."""
    problems = []
    custom = CUSTOM_TS.read_text(encoding = "utf-8")
    for key in object_keys(custom, "DEFAULT_CUSTOMIZATION"):
        if key not in CUSTOM_MEASURED and key not in CUSTOM_ALLOW:
            problems.append(
                f"appearance setting {key!r} (DEFAULT_CUSTOMIZATION) is not measured by the long-chat gate: "
                f"add its costliest value to CUSTOM_MEASURED (and its root flag to CUSTOM_EXPECT), or list it in "
                f"CUSTOM_ALLOW with why it cannot slow an area down (tests/studio/_appearance_coverage.py)"
            )
    defaults = dict(
        re.findall(
            r'^\s*(\w+):\s*"([^"]+)"', custom.split("DEFAULT_CUSTOMIZATION", 1)[1][:3000], re.M
        )
    )
    for key, values in enum_settings(custom).items():
        for v in values:
            if v == defaults.get(key) or CUSTOM_MEASURED.get(key) == v or (key, v) in ENUM_ALLOW:
                continue
            problems.append(
                f"appearance setting {key}={v!r} is not measured: set it in CUSTOM_MEASURED or list "
                f"({key!r}, {v!r}) in ENUM_ALLOW with a reason (tests/studio/_appearance_coverage.py)"
            )
    material = MATERIAL_TS.read_text(encoding = "utf-8")
    part = re.search(r"partialize:\s*\(s\)\s*=>\s*\(\{(.*?)\}\)", material, re.S)
    for key in re.findall(r"(\w+)\s*:", part.group(1) if part else ""):
        target = MATERIAL_KEYS.get(key)
        if target is None:
            problems.append(
                f"persisted Glass setting {key!r} (useMaterialStore) is not measured: add a mode for its costliest "
                f"value in _area_matrix.modes and map it in MATERIAL_KEYS, or map it to 'allow: <reason>'"
            )
        elif "+" in target and target not in mode_names:
            problems.append(
                f"MATERIAL_KEYS maps {key!r} to mode {target!r}, which the gate does not run"
            )
    model = MODEL_TS.read_text(encoding = "utf-8")
    for key in object_keys(model, "DEFAULT_ADJUSTMENTS"):
        if key not in WALLPAPER_MEASURED and key not in WALLPAPER_ALLOW:
            problems.append(
                f"chat wallpaper setting {key!r} (DEFAULT_ADJUSTMENTS) is not measured: add its costliest value "
                f"to WALLPAPER_MEASURED or list it in WALLPAPER_ALLOW with a reason (tests/studio/_appearance_coverage.py)"
            )
    ids, shared = palettes()
    for p in ids:
        if p not in shared:
            problems.append(
                f"palette {p!r} is not in the shared palette rule of index.css that solid+palette measures: "
                f"add it there, or measure it as its own mode"
            )
    for flag, where in sorted(css_root_flags().items()):
        target = ROOT_FLAGS.get(flag)
        if target is None:
            problems.append(
                f"<html> flag {flag!r} switches on costly CSS ({where[0]}) but no gate mode measures it: map it in "
                f"ROOT_FLAGS to the mode that sets it (add one in _area_matrix.modes if none does), or to "
                f"'allow: <reason>' (tests/studio/_appearance_coverage.py)"
            )
        elif not target.startswith("allow:") and target not in mode_names:
            problems.append(
                f"ROOT_FLAGS maps {flag!r} to mode {target!r}, which the gate does not run"
            )
    return problems
