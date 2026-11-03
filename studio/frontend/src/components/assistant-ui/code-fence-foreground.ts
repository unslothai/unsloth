// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HighlightResult } from "@streamdown/code";

type TokenLine = HighlightResult["tokens"][number];

const COLOR = /^#(?:[\da-f]{3,4}|[\da-f]{6}|[\da-f]{8})$/i;
const EXTRA_STYLES = new Set([
  "background-color",
  "font-style",
  "font-weight",
  "text-decoration",
  "--shiki-dark-bg",
  "--shiki-dark-font-style",
  "--shiki-dark-font-weight",
  "--shiki-dark-text-decoration",
]);

export function fenceLineForeground(line: TokenLine): {
  light: string;
  dark: string;
  inherited: boolean[];
} | null {
  const candidates: (string | null)[] = [];
  let chosen = "";
  for (const token of line) {
    const style = token.htmlStyle ?? {};
    const light = style.color ?? token.color;
    const dark = style["--shiki-dark"];
    if (
      !light ||
      !dark ||
      !COLOR.test(light) ||
      !COLOR.test(dark) ||
      (token.htmlAttrs && Object.keys(token.htmlAttrs).length > 0)
    )
      return null;
    const extra = Object.keys(style).filter(
      (key) => key !== "color" && key !== "--shiki-dark",
    );
    if (extra.some((key) => !EXTRA_STYLES.has(key))) return null;
    if (token.bgColor || token.fontStyle || extra.length > 0) {
      candidates.push(null);
      continue;
    }
    const key = `${light.toLowerCase()}|${dark.toLowerCase()}`;
    candidates.push(key);
    if (!chosen) chosen = key;
  }
  if (!chosen) return null;
  const [light, dark] = chosen.split("|");
  return { light, dark, inherited: candidates.map((key) => key === chosen) };
}
