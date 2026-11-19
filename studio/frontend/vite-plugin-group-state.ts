// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Plugin } from "vite";

/**
 * Tailwind v4 writes group-hover as :where(.group):hover; rewrite to .group:hover (same elements,
 * specificity up one class as in Tailwind v3, all three states); WebKit restyles subtrees on the
 * :where form.
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
