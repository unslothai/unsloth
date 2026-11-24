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

import {
  createCodePlugin,
  MIN_INCREMENTAL_CHARS,
  TOKENIZE_LIMITS,
} from "../src/components/assistant-ui/code-plugin.ts";

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];

// Longer than REFRESH_MS: inside that window the throttled path skips grammar state.
const settle = (): Promise<unknown> =>
  new Promise((resolve) => setTimeout(resolve, 260));

const highlightOnce = (
  plugin: ReturnType<typeof createCodePlugin>,
  options: HighlightOptions,
): Promise<HighlightResult> =>
  new Promise((resolve) => {
    const immediate = plugin.highlight(options, resolve);
    if (immediate) resolve(immediate);
  });

const referenceHighlighters = new Map<
  string,
  ReturnType<typeof createHighlighter>
>();

async function reference(code: string, language: HighlightOptions["language"]) {
  let loading = referenceHighlighters.get(language);
  if (!loading) {
    loading = createHighlighter({
      themes: THEMES,
      langs: [language],
      engine: createJavaScriptRegexEngine({ forgiving: true }),
    });
    referenceHighlighters.set(language, loading);
  }
  const highlighter = await loading;
  return highlighter.codeToTokens(code, {
    lang: language,
    themes: { light: "github-light", dark: "github-dark" },
    // Use the plugin's tokenizer limits so shiki's wall-clock bail cannot skew either side.
    ...TOKENIZE_LIMITS,
  });
}

async function assertMatchesWholeDocument(
  source: string,
  language: HighlightOptions["language"],
  // Annotated because TS cannot infer a binding another default in the same pattern reads.
  {
    step = 1,
    throttledStep = step,
    cuts = [],
  }: { step?: number; throttledStep?: number; cuts?: number[] } = {},
) {
  const lengths = new Set<number>(cuts.filter((n) => n > 0 && n <= source.length));
  for (let length = 1; length <= source.length; ) {
    lengths.add(length);
    length += length >= MIN_INCREMENTAL_CHARS ? throttledStep : step;
  }
  lengths.add(source.length);

  const plugin = createCodePlugin({ themes: THEMES });
  let previous = 0;
  for (const length of [...lengths].sort((a, b) => a - b)) {
    if (previous >= MIN_INCREMENTAL_CHARS) {
      await settle();
    }
    previous = length;
    const code = source.slice(0, length);
    const streamed = await highlightOnce(plugin, {
      code,
      language,
      themes: THEMES,
    });
    const full = await reference(code, language);
    assert.deepEqual(
      streamed.tokens,
      full.tokens,
      `${language} diverged at ${length} of ${source.length} characters, ` +
        `after ${JSON.stringify(code.slice(-24))}`,
    );
  }
}

const cutsAround = (source: string, marker: string): number[] => {
  const out: number[] = [];
  for (let i = source.indexOf(marker); i >= 0; i = source.indexOf(marker, i + 1)) {
    out.push(i, i + 1, i + marker.length, i + marker.length + 1);
  }
  return out;
};

const HTML = `<!doctype html>
<html lang="en">
  <head>
    <style>
      .card { color: #333; /* a comment
         spanning lines */ }
    </style>
  </head>
  <body>
    <div class="card" data-note="a > b">text</div>
    <script>
      const total = items.reduce((sum, item) => sum + item.n, 0);
      /* block comment
         still open */
      console.log(\`total \${total}\`);
    </script>
  </body>
</html>
`;

test("an HTML fence with embedded script and style matches whole-document tokenization", async () => {
  await assertMatchesWholeDocument(HTML, "html", {
    step: 7,
    cuts: [
      ...cutsAround(HTML, "<style>"),
      ...cutsAround(HTML, "</style>"),
      ...cutsAround(HTML, "<script>"),
      ...cutsAround(HTML, "</script>"),
      ...cutsAround(HTML, "/*"),
      ...cutsAround(HTML, "*/"),
      ...cutsAround(HTML, "${"),
    ],
  });
});

const TSX = `type Props = { items: string[] };

export function List({ items }: Props) {
  return (
    <ul className="list">
      {items.map((item) => (
        <li key={item} title={\`row \${item}\`}>
          {/* a JSX comment, which is not a JS comment */}
          {item.length > 2 ? <strong>{item}</strong> : item}
        </li>
      ))}
    </ul>
  );
}
`;

test("a TSX fence with JSX children matches whole-document tokenization", async () => {
  await assertMatchesWholeDocument(TSX, "tsx", {
    step: 5,
    cuts: [
      ...cutsAround(TSX, "<ul"),
      ...cutsAround(TSX, "{items"),
      ...cutsAround(TSX, "{/*"),
      ...cutsAround(TSX, "*/}"),
      ...cutsAround(TSX, "${"),
      ...cutsAround(TSX, "</ul>"),
    ],
  });
});

const MARKDOWN = `# Title

Some prose with \`inline code\` and a [link](https://example.com).

<!-- an HTML comment
that stays open across
several lines -->

\`\`\`python
def f(x):
    """docstring
    across lines"""
    return x
\`\`\`

More prose after the fence.
`;

test("a markdown fence with a multi-line comment matches whole-document tokenization", async () => {
  await assertMatchesWholeDocument(MARKDOWN, "markdown", {
    step: 5,
    cuts: [
      ...cutsAround(MARKDOWN, "<!--"),
      ...cutsAround(MARKDOWN, "-->"),
      ...cutsAround(MARKDOWN, "```python"),
      ...cutsAround(MARKDOWN, '"""'),
      ...cutsAround(MARKDOWN, "```\n\nMore"),
    ],
  });
});

const NOTES = Array.from(
  { length: 17 },
  (_, index) => `Paragraph ${index + 1} of the notes, long enough that the
document clears the incremental threshold before the next fence opens.`,
).join("\n\n");

const MARKDOWN_NESTED = `# Release notes

The block below is markdown, so a fence in its body opens a second one.

\`\`\`python
# Collect the rows before rendering them.
def render(rows):
    """Return the rows as text.

    * Not a list item, just a docstring line.
    """
    return "\\n".join(rows)
\`\`\`

Prose after the first nested fence, with **bold**, \`inline code\` and a
[link](https://example.com), none of which is markdown at all unless that
fence really closed.

${NOTES}

\`\`\`bash
# Restart the worker after editing the config.
set -euo pipefail
./scripts/worker.sh --config config.yaml
\`\`\`

# A heading the document only has once the second fence has closed too

Trailing prose with **bold** and \`inline code\`.
`;

test("nested markdown fences match whole-document tokenization", async () => {
  await assertMatchesWholeDocument(MARKDOWN_NESTED, "markdown", {
    step: 23,
    throttledStep: 150,
    cuts: [
      ...cutsAround(MARKDOWN_NESTED, "```"),
      ...cutsAround(MARKDOWN_NESTED, "```python"),
      ...cutsAround(MARKDOWN_NESTED, "```bash"),
      ...cutsAround(MARKDOWN_NESTED, '"""'),
    ],
  });
});

const SHELL = `#!/usr/bin/env bash
set -euo pipefail

cat <<'END_SQL'
$HOME is not expanded here
SELECT '\${value}' FROM t;
END_SQL

cat <<EOF
$HOME is expanded here
EOF

echo done
`;

test("a shell heredoc keeps its scope across updates", async () => {
  await assertMatchesWholeDocument(SHELL, "shellscript", {
    step: 4,
    cuts: [
      ...cutsAround(SHELL, "<<'END_SQL'"),
      ...cutsAround(SHELL, "END_SQL"),
      ...cutsAround(SHELL, "<<EOF"),
      ...cutsAround(SHELL, "EOF"),
    ],
  });
});

const TEMPLATE = `const name = "row";
const value = \`outer
\${render({
  inner: \`nested \${name} deep\`,
  note: "a } brace in a string",
})}
tail\`;
const escaped = \`not \\\${an} interpolation\`;
const done = true;
`;

test("nested template literals match whole-document tokenization", async () => {
  await assertMatchesWholeDocument(TEMPLATE, "typescript", {
    step: 3,
    cuts: [
      ...cutsAround(TEMPLATE, "`outer"),
      ...cutsAround(TEMPLATE, "${render"),
      ...cutsAround(TEMPLATE, "`nested"),
      ...cutsAround(TEMPLATE, "})}"),
      ...cutsAround(TEMPLATE, "tail`"),
      ...cutsAround(TEMPLATE, "\\${an}"),
    ],
  });
});
