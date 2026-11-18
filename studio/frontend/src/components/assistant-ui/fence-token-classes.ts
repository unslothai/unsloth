// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThemeRegistrationAny } from "shiki";

// WebKit shares no computed style across inline-styled elements (3x style work), so token colours are classes written once.
// Must be installed before any chat renders: a sheet edited later restyles the whole document on WebKit.

const COLOR = /^#(?:[\da-f]{3,4}|[\da-f]{6}|[\da-f]{8})$/i;
const COLOR_PROPERTIES = ["--sdm-c", "--shiki-dark"] as const;
const KEYWORD_PROPERTIES: Readonly<Record<string, readonly string[]>> = {
  "font-style": ["italic", "normal", "inherit"],
  "font-weight": ["bold", "normal", "inherit"],
  "text-decoration": ["underline", "line-through", "none", "inherit"],
};

const themeColors = (theme: ThemeRegistrationAny): string[] => {
  const found = new Set<string>();
  const add = (value: unknown): void => {
    if (typeof value === "string" && COLOR.test(value)) found.add(value);
  };
  add(theme.fg);
  add(theme.colors?.["editor.foreground"]);
  for (const rule of theme.tokenColors ?? theme.settings ?? []) add(rule.settings?.foreground);
  return [...found];
};

export function buildTokenClasses(
  themes: readonly ThemeRegistrationAny[],
): { names: Map<string, string>; css: string } {
  const names = new Map<string, string>();
  const rules: string[] = [];
  const add = (property: string, value: string): void => {
    const key = `${property}:${value}`;
    if (names.has(key)) return;
    const name = `sdm-t${names.size.toString(36)}`;
    names.set(key, name);
    rules.push(`.${name}{${property}:${value}}`);
  };
  for (const color of themes.flatMap(themeColors)) {
    for (const spelling of new Set([color, color.toLowerCase(), color.toUpperCase()])) {
      for (const property of COLOR_PROPERTIES) add(property, spelling);
    }
  }
  for (const [property, values] of Object.entries(KEYWORD_PROPERTIES)) {
    for (const prefix of ["", "--shiki-light-", "--shiki-dark-"]) {
      for (const value of values) add(prefix ? `${prefix}${property}` : property, value);
    }
  }
  return { names, css: rules.join("\n") };
}

let installed: Map<string, string> | null = null;

export function installTokenClasses(themes: readonly ThemeRegistrationAny[]): void {
  if (installed !== null || typeof document === "undefined" || !document.head) return;
  const { names, css } = buildTokenClasses(themes);
  const element = document.createElement("style");
  element.setAttribute("data-fence-token-classes", "");
  element.textContent = css;
  document.head.appendChild(element);
  installed = names;
}

export function tokenClassesFor(style: Readonly<Record<string, string>>): string | null {
  if (installed === null) return null;
  let classes = "";
  for (const [property, value] of Object.entries(style)) {
    const name = installed.get(`${property}:${value}`);
    if (name === undefined) return null;
    classes += ` ${name}`;
  }
  return classes;
}
