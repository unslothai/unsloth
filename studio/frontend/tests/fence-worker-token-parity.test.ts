// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { HighlightOptions, HighlightResult, ThemeInput } from "@streamdown/code";

import { createCodePlugin } from "../src/components/assistant-ui/code-plugin.ts";
import { unslothDarkTheme, unslothLightTheme } from "../src/components/assistant-ui/code-themes.ts";
import {
  reasoningHighlightReply,
  transferableResult,
} from "../src/components/assistant-ui/reasoning-highlight.ts";

// The chat renderer's themes (markdown-text.tsx) and the worker's (reasoning-highlight.worker.ts).
const THEMES: [ThemeInput, ThemeInput] = [unslothLightTheme, unslothDarkTheme];

const FIXTURES: Record<string, string> = {
  python: 'from dataclasses import dataclass\n\n@dataclass\nclass Row:\n    weight: float = 0.5  # default\n\ndef score(rows: list[Row]) -> float:\n    """Sum the weights."""\n    return sum(r.weight for r in rows if r.weight > 0)\n',
  typescript: "export interface Row { readonly weight: number }\nexport const score = (rows: readonly Row[]): number =>\n  rows.reduce((t, r) => t + r.weight * 2, 0); // total\n",
  tsx: "export function Badge({ label }: { label: string }) {\n  return <span className=\"badge\">{label.toUpperCase()}</span>;\n}\n",
  javascript: "const rows = [{ w: 1 }, { w: 2 }];\nconsole.log(rows.map((r) => r.w * 2).join(`, `));\n",
  rust: "fn score(rows: &[Row]) -> f64 {\n    rows.iter().map(|r| r.weight * 1.5).sum::<f64>() // step\n}\n",
  go: "func Score(rows []Row) float64 {\n\ttotal := 0.0\n\tfor _, r := range rows { total += r.Weight }\n\treturn total\n}\n",
  cpp: "#include <vector>\ndouble score(const std::vector<Row>& rows) {\n  double t = 0; for (auto& r : rows) t += r.weight; return t;\n}\n",
  c: "#include <stdio.h>\nint main(void) { printf(\"%d\\n\", 42); return 0; }\n",
  java: "public final class Score {\n  public static double of(List<Row> rows) { return rows.stream().mapToDouble(r -> r.weight).sum(); }\n}\n",
  json: '{"id": "row-1", "weight": 0.5, "tags": ["a", "b"], "ok": true, "next": null}\n',
  yaml: "steps:\n  - name: build\n    run: npm ci && npm run build # ci\n    env: { CI: true }\n",
  bash: 'for f in ./data/*.csv; do\n  echo "file: $f"; wc -l "$f" | awk \'{print $1}\'\ndone\n',
  sql: "SELECT id, SUM(weight) AS total FROM rows WHERE label <> 'skip' GROUP BY id ORDER BY total DESC;\n",
  css: ".row:hover { color: hsl(210, 70%, 55%); margin: 4px 0; }\n@media (max-width: 600px) { .row { display: none } }\n",
  html: '<!doctype html>\n<div class="row" data-step="1"><a href="/rows/1">open</a></div>\n<script>const x = 1 < 2;</script>\n',
  markdown: "# Title\n\nSome *emphasis* and `code`, and a [link](https://example.com).\n",
  diff: "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old = 1\n+new = 2\n",
  text: "plain text stays plain\n",
};

const exact = (plugin: ReturnType<typeof createCodePlugin>, options: HighlightOptions): Promise<HighlightResult> =>
  new Promise((resolve) => {
    const now = plugin.highlightExact(options, resolve);
    if (now) resolve(now);
  });

test("a fence coloured by the worker renders exactly the tokens the main thread would have", async () => {
  const main = createCodePlugin({ themes: THEMES });
  const worker = createCodePlugin({ themes: THEMES });
  for (const [language, code] of Object.entries(FIXTURES)) {
    const options = { code, language: language as never, themes: THEMES };
    const local = await exact(main, options);
    const reply = reasoningHighlightReply(
      { client: 1, revision: 1, source: code, language, lines: [], full: true },
      await exact(worker, options),
    );
    const received = structuredClone(reply).result as HighlightResult;
    assert.deepEqual(received.tokens, local.tokens, `${language}: same tokens`);
    for (const key of ["bg", "fg", "themeName", "rootStyle"] as const) {
      assert.deepEqual(received[key], local[key], `${language}: same ${key}`);
    }
    assert.equal("grammarState" in received, false, `${language}: no grammar state crosses over`);
  }
});

test("a worker result shares one style object per distinct style, and survives the clone that way", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const code = Array.from({ length: 200 }, (_, i) => `def step_${i}(rows):\n    return sum(r.w for r in rows) * ${i}  # step`).join("\n");
  const result = await exact(plugin, { code, language: "python" as never, themes: THEMES });
  const compact = structuredClone(transferableResult(result));
  const tokens = compact.tokens.flat();
  const styles = new Set(tokens.map((token) => token.htmlStyle));
  assert.ok(tokens.length > 1000);
  assert.ok(styles.size < 20, `${styles.size} style objects for ${tokens.length} tokens`);
  assert.deepEqual(compact.tokens, result.tokens);
  assert.equal(result.grammarState !== undefined, true, "the cached result keeps its own grammar state");
});

test("a failed or null worker result still replies, and a partial request carries no result", () => {
  const request = { client: 2, revision: 1, source: "x", language: "python", lines: [0], full: true };
  assert.deepEqual(reasoningHighlightReply(request, null), { client: 2, revision: 1, lines: [], result: null });
  assert.equal("result" in reasoningHighlightReply({ ...request, full: false }, null), false);
});
