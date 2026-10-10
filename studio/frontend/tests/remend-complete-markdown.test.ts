// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { type Dirent, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import remend from "remend";
import { Streamdown } from "streamdown";
import {
  LITERAL_LINK_REMEND,
  hasIncompleteLinkRepair,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";

/**
 * remend runs on every settled body, so a version bump is a rendering change.
 * Only bare Streamdown is covered: no plugins, preprocessing, or DOM.
 */

// remend 1.3.0 misreads `\(`/`\[` math subscripts as open emphasis and appends a stray `_`.
const COMPLETE_LATEX_DOCUMENTS = [
  String.raw`where \( \delta_{r} = 1 \) holds.`,
  "\\[ \\delta_{r} = 1 \\]\n",
  String.raw`where $ \delta_{r} = 1 $ holds.`,
  String.raw`where \( \delta_{r} = \beta_{k} \) holds.`,
];

test("a complete document with LaTeX subscripts comes back untouched", () => {
  for (const complete of COMPLETE_LATEX_DOCUMENTS) {
    assert.equal(
      remend(complete, {}),
      complete,
      `remend rewrote a complete document: ${JSON.stringify(complete)}`,
    );
  }
});

test("ordinary complete markdown is returned unchanged", () => {
  for (const complete of [
    "see [the docs](https://example.com) for more",
    "this is **bold** text",
    "call `foo()` now",
    "```js\nconst a = 1;\n```\n",
    "the value $x + y$ holds",
    "an array literal like [1, 2, 3] in prose",
    "the pattern [^a-z] matches",
    "a snake_case identifier in prose",
    "```js\nconst r = /[^a-z]/;\n```\n",
  ]) {
    assert.equal(
      remend(complete, {}),
      complete,
      `remend rewrote ${JSON.stringify(complete)}`,
    );
  }
});

test("a truncated stream is still repaired", () => {
  // Ensures the repair still runs; dropping it would otherwise pass the tests above.
  const repairs: [string, string][] = [
    ["see [the docs](https://exa", "]("],
    ["this is **bol", "**"],
    ["call `foo(", "`"],
  ];
  for (const [truncated, marker] of repairs) {
    const repaired = remend(truncated, {});
    assert.notEqual(
      repaired,
      truncated,
      `remend left a truncated ${marker} unrepaired: ${JSON.stringify(truncated)}`,
    );
    assert.ok(
      repaired.length >= truncated.length - marker.length,
      `the repair of ${JSON.stringify(truncated)} lost the document`,
    );
  }
});

const MARKDOWN_TEXT = new URL(
  "../src/components/assistant-ui/markdown-text.tsx",
  import.meta.url,
);

/** Evaluates markdown-text.tsx's real expression at incrementalRender === null. */
function settledParseIncompleteMarkdown(): boolean {
  const source = readFileSync(MARKDOWN_TEXT, "utf8");
  const opened = source.indexOf("<Streamdown");
  assert.notEqual(
    opened,
    -1,
    "markdown-text.tsx no longer renders a <Streamdown>, so the settled render path this file " +
      "describes cannot be located",
  );
  const expression = /parseIncompleteMarkdown=\{([^}]*)\}/.exec(
    source.slice(opened),
  )?.[1];
  assert.ok(
    expression,
    "markdown-text.tsx no longer passes parseIncompleteMarkdown to Streamdown; the repair these " +
      "documents depend on is no longer requested where they claim it is",
  );
  const value: unknown = new Function(
    "incrementalRender",
    `return (${expression});`,
  )(null);
  assert.equal(
    typeof value,
    "boolean",
    `parseIncompleteMarkdown={${expression}} did not evaluate to a boolean at incrementalRender === null`,
  );
  return value as boolean;
}

function renderSettled(markdown: string): string {
  return renderToStaticMarkup(
    createElement(
      Streamdown,
      {
        mode: "streaming",
        parseIncompleteMarkdown: settledParseIncompleteMarkdown(),
        remend: hasIncompleteLinkRepair(markdown)
          ? LITERAL_LINK_REMEND
          : undefined,
      },
      markdown,
    ),
  );
}

test("a settled unfinished link remains text without a blocked placeholder", () => {
  const html = renderSettled("See [example](https://exa");
  assert.match(html, /See \[example\]\(/);
  assert.match(html, /https:\/\/exa/);
  assert.doesNotMatch(html, /\[blocked\]|streamdown:incomplete-link/);
  assert.match(renderSettled("See [example](javascript:alert)"), /\[blocked\]/);
  const list = renderSettled("- >= 16 GB\n\nSee [foo");
  assert.match(list, /<li[^>]*>&gt;= 16 GB<\/li>/);
  assert.doesNotMatch(list, /blockquote|\[blocked\]/);
});

test("the settled render path runs the repair, not just the package", () => {
  // The tests above call remend directly; only rendering a truncated body shows the UI wiring.
  // Cancelled or errored responses settle truncated, so they reach the settled path.
  for (const [truncated, repaired] of [
    ["this is **bol", 'data-streamdown="strong"'],
    ["call `foo(", 'data-streamdown="inline-code"'],
  ] as const) {
    assert.ok(
      renderSettled(truncated).includes(repaired),
      `a settled <Streamdown> rendered ${JSON.stringify(truncated)} without repairing it: no ` +
        `${repaired} in the output. The repair is off on the path Unsloth actually renders, ` +
        "whatever the direct calls to remend above report.",
    );
  }
});

// A scan, not a `<[^>]*>` replace, which CodeQL flags; React escapes `<` in text.
const renderedText = (markup: string): string => {
  let text = "";
  let inTag = false;
  for (const ch of markup) {
    if (inTag) {
      inTag = ch !== ">";
    } else if (ch === "<") {
      inTag = true;
    } else {
      text += ch;
    }
  }
  return text;
};

test("a complete document survives the settled render path unchanged", () => {
  const underscores = (text: string): number => (text.match(/_/g) ?? []).length;
  for (const complete of COMPLETE_LATEX_DOCUMENTS) {
    const text = renderedText(renderSettled(complete));
    assert.equal(
      underscores(text),
      underscores(complete),
      `the settled render of ${JSON.stringify(complete)} changed how many underscores reach the ` +
        `reader: ${JSON.stringify(text)}`,
    );
  }
});

const FRONTEND_ROOT = fileURLToPath(new URL("..", import.meta.url));

// A package's private copy lives at <package>/node_modules; scoped dirs hold packages.
function collectRemendCopies(nodeModules: string, found: string[]): string[] {
  let entries: Dirent[];
  try {
    entries = readdirSync(nodeModules, { withFileTypes: true });
  } catch {
    return found;
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const full = path.join(nodeModules, entry.name);
    if (entry.name === "remend") {
      found.push(full);
    } else if (entry.name.startsWith("@")) {
      collectRemendCopies(full, found);
    } else {
      collectRemendCopies(path.join(full, "node_modules"), found);
    }
  }
  return found;
}

test("every remend in the tree is the pinned one, including Streamdown's", async () => {
  // streamdown pins remend exactly, so without overrides.remend npm nests an older copy.
  // Assert the version where Streamdown resolves it, not just where this file does.
  const copies = collectRemendCopies(
    path.join(FRONTEND_ROOT, "node_modules"),
    [],
  );
  assert.ok(
    copies.length > 0,
    "no remend under node_modules: run `npm ci` first",
  );

  const manifest = JSON.parse(
    readFileSync(path.join(FRONTEND_ROOT, "package.json"), "utf8"),
  ) as { dependencies: Record<string, string> };
  const pinned = manifest.dependencies.remend;

  for (const copy of copies) {
    const copyManifest = JSON.parse(
      readFileSync(path.join(copy, "package.json"), "utf8"),
    ) as { version: string; module?: string; main?: string };
    assert.equal(
      copyManifest.version,
      pinned,
      `${path.relative(FRONTEND_ROOT, copy)} is remend ${copyManifest.version}, not the pinned ${pinned}`,
    );
    const entry = copyManifest.module ?? copyManifest.main;
    assert.ok(entry, `remend at ${copy} has no module entry point`);
    const loaded = (await import(
      pathToFileURL(path.join(copy, entry)).href
    )) as {
      default: (text: string, options?: object) => string;
    };
    for (const complete of COMPLETE_LATEX_DOCUMENTS) {
      // `undefined`, not `{}`: Streamdown forwards its remend prop and Unsloth passes none.
      assert.equal(
        loaded.default(complete, undefined),
        complete,
        `${path.relative(FRONTEND_ROOT, copy)} rewrote a complete document: ${JSON.stringify(complete)}`,
      );
    }
  }
});

/* No timing assertion: a doubling-factor check had too little margin and flaked under load. */
