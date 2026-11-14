// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import postcss from "postcss";
import type { Plugin } from "vite";

const KATEX_CSS = /[\\/]node_modules[\\/]katex[\\/]dist[\\/]katex(?:\.min)?\.css(?:$|\?)/;
const WOFF2_SOURCE = /url\(\s*(?:"[^"]*"|'[^']*'|[^)"']*)\s*\)\s*format\(\s*["']?woff2["']?\s*\)/gi;

export function keepWoff2Sources(css: string): { css: string; rules: number } {
  const root = postcss.parse(css);
  let rules = 0;
  root.walkAtRules("font-face", (rule) => {
    rule.walkDecls("src", (decl) => {
      const woff2 = decl.value.match(WOFF2_SOURCE);
      if (!woff2 || woff2.length !== 1) {
        throw new Error(`@font-face src "${decl.value}" has ${woff2?.length ?? 0} woff2 sources, expected 1`);
      }
      decl.value = woff2[0];
      rules += 1;
    });
  });
  return { css: root.toString(), rules };
}

export function katexWoff2Only(): Plugin {
  return {
    name: "katex-woff2-only",
    // Rewrite before vite:css emits font assets.
    enforce: "pre",
    transform: {
      filter: { id: KATEX_CSS, code: "@font-face" },
      handler(code, id) {
        if (/[?&]raw\b/.test(id)) {
          return null;
        }
        const { css, rules } = keepWoff2Sources(code);
        if (rules === 0) throw new Error(`no KaTeX @font-face src found in ${id}`);
        return { code: css, map: null };
      },
    },
  };
}
