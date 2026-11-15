// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Plugin } from "vite";

/**
 * Tailwind v4 writes `group-hover:x` as `.x:is(:where(.group):hover *)`. WebKit cannot key that invalidation
 * on `.group` inside `:where()`, so when :hover, :focus-within or :active flips on any element (each ancestor
 * of the pointer or focus, up to <html>) it re-styles that element's whole subtree: about 150 ms per focus
 * move into a menu or dialog on a 150-turn chat. `.group:hover *` is keyed on the class. The selector matches
 * the same elements; specificity goes up by one class, as in Tailwind v3, for all three states alike.
 */
const GROUP_STATE = /:where\((\.(?:[\w-]|\\.)+)\):(hover|focus-within|active)(?=\s*\*)/g;

export function keyedGroupStates(css: string): string {
  return css.replace(GROUP_STATE, "$1:$2");
}

export function groupStateKeys(): Plugin {
  return {
    name: "group-state-keys",
    transform: {
      filter: { id: /\.css(?:$|\?)/, code: ":where(" },
      handler(code) {
        const next = keyedGroupStates(code);
        return next === code ? null : { code: next, map: null };
      },
    },
  };
}
