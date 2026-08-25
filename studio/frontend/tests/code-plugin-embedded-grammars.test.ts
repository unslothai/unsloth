// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Incremental tokenization has to survive a grammar switching mid-document.
 *
 * `code-plugin.ts` commits completed lines with the shiki `GrammarState` that
 * follows them and resumes from it on the next refresh. Every fixture in
 * `code-plugin-incremental.test.ts` is python, json or typescript, where one
 * grammar covers the whole fence. The interesting case is the one none of them
 * reach: HTML pushing into javascript and css, TSX alternating between JS and
 * JSX, markdown opening a nested fence, a heredoc whose terminator is chosen at
 * runtime. There the saved state is a stack several grammars deep, and a
 * resume that loses a level produces plausible-looking tokens with the wrong
 * scopes rather than an obvious break.
 *
 * The oracle is the same one the sibling file uses: whole-document
 * `codeToTokens` at every prefix. Deliberately self-contained, so these can be
 * read and moved independently of the sibling file's helpers.
 */

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
} from "../src/components/assistant-ui/code-plugin.ts";

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];

// REFRESH_MS in code-plugin.ts, plus a margin. A fence past
// MIN_INCREMENTAL_CHARS that is updated twice inside that window is answered
// from the throttled approximation, which renders the uncommitted tail plain
// and never reads the grammar state, so a comparison there cannot fail however
// badly the resume is broken. Measured, not assumed: drop the wait and the
// first comparison past the threshold takes that path on correct code too.
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

/** What shiki returns for the whole string in one call. */
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
  });
}

/**
 * Stream `source` one prefix at a time and require every settled result to
 * equal whole-document tokenization of that same prefix.
 *
 * `cuts` are extra prefix lengths to test on top of the `step` walk, for the
 * boundaries that matter here: the character before and after a delimiter that
 * pushes or pops a grammar. A step walk alone can stride straight over them.
 *
 * Once the fence is past MIN_INCREMENTAL_CHARS every update has to wait out the
 * refresh interval to be tokenized at all, so `throttledStep` walks that
 * stretch on its own coarser stride. Fixtures below the threshold never reach
 * either and pay nothing.
 */
async function assertMatchesWholeDocument(
  source: string,
  language: HighlightOptions["language"],
  // Annotated rather than inferred: `throttledStep` defaults to `step`, and TS
  // cannot infer a binding that another default in the same pattern reads.
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

/** Every index just before and just after each occurrence of `marker`. */
const cutsAround = (source: string, marker: string): number[] => {
  const out: number[] = [];
  for (let i = source.indexOf(marker); i >= 0; i = source.indexOf(marker, i + 1)) {
    out.push(i, i + 1, i + marker.length, i + marker.length + 1);
  }
  return out;
};

// ── HTML: the grammar pushes into javascript and css and back ──────────

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

// ── TSX: JSX children and an expression container ──────────────────────

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

// ── Markdown containing a fence: the grammar nests into itself ─────────

// The nested fence is here because it is what users actually paste, but it is
// NOT what makes this discriminate: shiki's markdown grammar leaves a fenced
// body uncoloured, so its tail tokenizes the same with or without a resumed
// state. The multi-line HTML comment is the part that carries state across
// lines, verified by tokenizing the tail both ways at every line boundary.
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

// ── Markdown whose nested fences open and close again ──────────────────

// Prose that carries no state of its own, here to push the second nested fence
// past MIN_INCREMENTAL_CHARS.
const NOTES = Array.from(
  { length: 17 },
  (_, index) => `Paragraph ${index + 1} of the notes, long enough that the
document clears the incremental threshold before the next fence opens.`,
).join("\n\n");

// MARKDOWN above pushes one shallow level and, as its own comment says, leaves
// its nested fence's body doing no work. This fixture is the opposite: the body
// and the pop back out of it are the whole point. A `#` line inside a fence is
// body text and the same line outside one is a heading, so a resume that stays
// a level too deep, or comes back a level too shallow, colours them the other
// way round, and the prose after each fence goes with them.
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
    // Every comparison past the threshold waits out a refresh interval, so that
    // stretch is walked coarsely: 15 of the 119 comparisons are there, and they
    // are what makes this test take about four seconds.
    throttledStep: 150,
    cuts: [
      // Both sides of every delimiter run, opening and closing alike.
      ...cutsAround(MARKDOWN_NESTED, "```"),
      // And after the info string, which is where the nested language is named.
      ...cutsAround(MARKDOWN_NESTED, "```python"),
      ...cutsAround(MARKDOWN_NESTED, "```bash"),
      ...cutsAround(MARKDOWN_NESTED, '"""'),
    ],
  });
});

// ── Shell heredoc: the terminator is chosen by the document ────────────

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

// ── Nested template literals: interpolation inside interpolation ───────

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
