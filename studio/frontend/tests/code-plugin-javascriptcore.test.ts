// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type {
  HighlightOptions,
  HighlightResult,
  ThemeInput,
} from "@streamdown/code";
import { createHighlighter } from "shiki";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

const WEBKIT =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";
Object.defineProperty(globalThis, "navigator", {
  value: { userAgent: WEBKIT },
  configurable: true,
});
const { createCodePlugin, TOKENIZE_LIMITS } = await import(
  "../src/components/assistant-ui/code-plugin.ts"
);

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];

const SAMPLES: Record<string, string> = {
  html: `<!doctype html>
<html lang="en">
<head>
  <style>
    canvas { display: block; margin: 0 auto; background: #70c5ce; }
    .score::after { content: "pts"; color: rgb(255 255 255 / 0.8); }
  </style>
</head>
<body>
  <canvas id="game" width="400" height="600"></canvas>
  <script>
    const canvas = document.getElementById("game");
    const ctx = canvas.getContext("2d");
    let bird = { x: 80, y: 300, velocity: 0 };
    const pipes = [];
    function tick(now) {
      bird.velocity = Math.min(bird.velocity + 0.5, 12);
      bird.y += bird.velocity;
      if (pipes.length === 0 || pipes.at(-1).x < 200) pipes.push({ x: 400, gap: 120 + Math.random() * 200 });
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      requestAnimationFrame(tick);
    }
    document.addEventListener("keydown", (event) => { if (event.code === "Space") bird.velocity = -8; });
    requestAnimationFrame(tick);
  </script>
</body>
</html>
`,
  typescript: `type Pipe = { x: number; gap: number };
export class Game<T extends { score: number }> {
  #pipes: Pipe[] = [];
  constructor(private readonly state: T) {}
  async load(url: string): Promise<Map<string, number>> {
    const response = await fetch(\`\${url}?v=\${Date.now()}\`);
    return new Map(Object.entries(await response.json()) as [string, number][]);
  }
  get best(): number { return this.#pipes.reduce((max, { gap }) => (gap > max ? gap : max), 0); }
}
const re = /(?<=\\$)\\d+(?:\\.\\d{2})?/gu;
`,
  python: `import re
from dataclasses import dataclass, field

PRICE = re.compile(r"(?<=\\$)\\d+(?:\\.\\d{2})?")

@dataclass
class Bird:
    x: float = 80.0
    y: float = 300.0
    velocity: float = field(default=0.0)

    def flap(self, *, strength: float = 8.0) -> None:
        """Jump, but never faster than the cap."""
        self.velocity = max(-strength, self.velocity - strength)

print(f"{Bird()!r} {PRICE.findall('$12.50 and $3')}")
`,
  css: `@layer base {
  :root { --sky: oklch(78% 0.08 220); }
  .pipe:is(.top, .bottom):not([hidden]) > .cap::before {
    inset: 0 auto auto 0;
    transform: translate3d(calc(var(--x, 0) * 1px), 0, 0) rotate(-2deg);
  }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
}
`,
  json: `{"bird": {"x": 80, "y": 300, "velocity": -8.5}, "pipes": [{"x": 400, "gap": 160}], "scores": [12, 7, 33], "name": "flappy \\"bird\\"", "ok": true, "none": null}
`,
  shellscript: `#!/usr/bin/env bash
set -euo pipefail
for file in "\${@:-src/*.ts}"; do
  if [[ "$file" =~ \\.test\\.ts$ ]]; then echo "skip \${file##*/}"; continue; fi
  sed -E 's/(foo|bar)+/baz/g' "$file" | tee "\${file%.ts}.out" >/dev/null
done
cat <<EOF
done: $(date +%s)
EOF
`,
};

async function reference(code: string, language: string) {
  const highlighter = await createHighlighter({
    themes: THEMES,
    langs: [language],
    engine: createJavaScriptRegexEngine({ forgiving: true }),
  });
  return highlighter.codeToTokens(code, {
    lang: language as never,
    themes: { light: "github-light", dark: "github-dark" },
    ...TOKENIZE_LIMITS,
  });
}

function highlightOnce(
  plugin: ReturnType<typeof createCodePlugin>,
  options: HighlightOptions,
): Promise<HighlightResult> {
  return new Promise((resolve) => {
    const immediate = plugin.highlightExact(options, resolve);
    if (immediate) resolve(immediate);
  });
}

for (const [language, code] of Object.entries(SAMPLES)) {
  test(`under JavaScriptCore's patterns, ${language} tokenizes exactly as with the engine's own`, async () => {
    const plugin = createCodePlugin({ themes: THEMES });
    const result = await highlightOnce(plugin, {
      code,
      language: language as HighlightOptions["language"],
      themes: THEMES,
    });
    const full = await reference(code, language);
    assert.deepEqual(result.tokens, full.tokens);
  });
}
