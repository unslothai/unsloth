// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type GrammarSample = { language: string; lines: string[] };

export type PrewarmHighlight = (
  code: string,
  language: string,
  late: () => void,
) => unknown;

export type PrewarmDeps = {
  idle: (callback: () => void, timeout: number) => () => void;
  wait: (callback: () => void, ms: number) => () => void;
  quiet: () => boolean;
  skip: (language: string) => boolean;
};

export const PREWARM_IDLE_TIMEOUT_MS = 2_000;
export const PREWARM_RETRY_MS = 1_000;
export const PREWARM_GUARD_MS = 5_000;

export const GRAMMAR_SAMPLES: readonly GrammarSample[] = [
  {
    language: "python",
    lines: [
      "# unsloth grammar warm",
      "import os",
      "from typing import Optional",
      "",
      "class Trainer:",
      '    """Docstring."""',
      "    def __init__(self, lr: float = 1e-3) -> None:",
      "        self.lr = lr",
      "",
      "    def step(self, batch: list[int]) -> Optional[float]:",
      "        total = sum(x * 2 for x in batch if x > 0)",
      "        return total / len(batch) if batch else None",
      "",
      'if __name__ == "__main__":',
      '    print(f"loss={Trainer().step([1, 2, 3]):.3f}")',
    ],
  },
  {
    language: "html",
    lines: [
      "<!-- unsloth grammar warm -->",
      "<!doctype html>",
      '<html lang="en">',
      "  <head>",
      "    <style>",
      "      :root { --gap: 8px; }",
      "      .card { display: flex; gap: var(--gap); color: #333; }",
      "      @media (max-width: 600px) { .card { flex-direction: column; } }",
      "    </style>",
      "  </head>",
      "  <body>",
      '    <div class="card" data-note="a > b">text &amp; more</div>',
      "    <script>",
      "      const items = [1, 2, 3];",
      "      const total = items.reduce((sum, n) => sum + n, 0);",
      "      function tick(dt) { if (dt > 1) { return `total ${total}`; } }",
      "      class Game { constructor() { this.state = { t: 0 }; } }",
      '      document.querySelector(".card").addEventListener("click", () => console.log(/re+/g.test("ree")));',
      "    </script>",
      "  </body>",
      "</html>",
    ],
  },
  {
    language: "javascript",
    lines: [
      "// unsloth grammar warm",
      'import { readFile } from "node:fs/promises";',
      "const items = [1, 2, 3];",
      "const total = items.reduce((sum, n) => sum + n, 0);",
      "function tick(dt) { if (dt > 1) { return `total ${total}`; } }",
      'class Game { constructor() { this.state = { t: 0 }; } async load() { return await readFile("x"); } }',
      "export default Game;",
    ],
  },
  {
    language: "typescript",
    lines: [
      "// unsloth grammar warm",
      'import type { Foo } from "./foo";',
      "interface Props { items: string[]; onPick?: (item: string) => void }",
      "export function pick<T extends object>(xs: T[], i: number): T | undefined { return xs[i]; }",
      "const total: number = [1, 2, 3].reduce((sum, n) => sum + n, 0);",
      'enum Mode { A = "a", B = "b" }',
      "export class Store<T> { private items = new Map<string, T>(); get(k: string) { return this.items.get(k) ?? null; } }",
    ],
  },
  {
    language: "shellscript",
    lines: [
      "# unsloth grammar warm",
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      'for f in "$@"; do',
      '  if [[ -f "$f" ]]; then echo "ok $f"; fi',
      "done",
      "cat <<END",
      "$HOME here",
      "END",
      'pip install -U unsloth && python -m unsloth.cli --model "$MODEL"',
    ],
  },
  {
    language: "json",
    lines: [
      '{"unsloth grammar warm": 0,',
      ' "name": "x", "version": "1.0.0", "list": [1, 2.5, true, null], "nested": {"a": "b"}}',
    ],
  },
  {
    language: "css",
    lines: [
      "/* unsloth grammar warm */",
      ":root { --gap: 8px; }",
      ".card { display: flex; gap: var(--gap); color: #333; }",
      "@media (max-width: 600px) { .card { flex-direction: column; } }",
      '.card:hover > .title::before { content: "x"; }',
    ],
  },
];

export type GrammarPrewarm = { start: () => void; cancel: () => void };

export function createGrammarPrewarm(
  highlight: PrewarmHighlight,
  deps: PrewarmDeps,
  samples: readonly GrammarSample[] = GRAMMAR_SAMPLES,
): GrammarPrewarm {
  let sample = 0;
  let line = 0;
  let pending: (() => void) | null = null;
  let guard: (() => void) | null = null;
  let epoch = 0;
  let started = false;
  let done = false;
  const step = (): void => {
    pending = null;
    while (sample < samples.length && deps.skip(samples[sample].language)) {
      sample += 1;
      line = 0;
    }
    if (sample >= samples.length) {
      done = true;
      return;
    }
    if (!deps.quiet()) {
      pending = deps.wait(step, PREWARM_RETRY_MS);
      return;
    }
    const { language, lines } = samples[sample];
    line += 1;
    const code = `${lines.slice(0, line).join("\n")}\n`;
    const mine = epoch;
    let advanced = false;
    const advance = (): void => {
      if (advanced || mine !== epoch) return;
      advanced = true;
      guard?.();
      guard = null;
      if (line >= lines.length) {
        sample += 1;
        line = 0;
      }
      pending = deps.idle(step, PREWARM_IDLE_TIMEOUT_MS);
    };
    guard = deps.wait(advance, PREWARM_GUARD_MS);
    if (highlight(code, language, advance)) advance();
  };
  return {
    start: () => {
      if (started || done) return;
      started = true;
      pending = deps.idle(step, PREWARM_IDLE_TIMEOUT_MS);
    },
    cancel: () => {
      epoch += 1;
      guard?.();
      guard = null;
      pending?.();
      pending = null;
      started = false;
    },
  };
}
