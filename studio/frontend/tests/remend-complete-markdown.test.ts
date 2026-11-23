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
  FINISHED_TEXT_REMEND,
  LITERAL_LINK_REMEND,
  hasIncompleteLinkRepair,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";

/**
 * WHY THIS FILE EXISTS. `remend` is the incomplete-markdown repair Streamdown runs over a message
 * before parsing it, and Unsloth runs it on every settled body (marker completions only on a cut
 * one), so a version bump is a rendering change.
 *
 * These are RUN, not scraped. Every assertion below is either a call into the resolved package or
 * a render through Streamdown, so the file fails if the dependency is downgraded, if the override
 * that forces one copy of it is dropped, or if the settled branch stops asking for the repair.
 * That last one is checked by EVALUATING the `parseIncompleteMarkdown` expression that
 * `markdown-text.tsx` writes and handing the result to a real
 * `<Streamdown>` -- not by asserting on the package in isolation and hoping the wiring agrees.
 *
 * WHAT THIS FILE DOES NOT COVER, so that the paragraph above is not read as wider than it is: the
 * render below is a bare `<Streamdown mode="streaming">` to static markup. It carries none of the
 * other props `markdown-text.tsx` passes (`STREAMDOWN_PLUGINS`, `STREAMDOWN_COMPONENTS`,
 * `parseMarkdownIntoBlocksFn`, `BlockComponent`), none of the `preprocessLaTeX` /
 * `stabilizeStreamingMarkdown` pipeline that runs upstream of it, and no DOM. A regression that
 * lives in one of those is not caught here.
 *
 * The first test fails on remend 1.3.0, which is the point of the bump.
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

/** The settled `parseIncompleteMarkdown`, evaluated from the source rather than restated. */
function settledParseIncompleteMarkdown(cut: boolean): boolean {
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
    "cut",
    `return (${expression});`,
  )(null, cut);
  assert.equal(
    typeof value,
    "boolean",
    `parseIncompleteMarkdown={${expression}} did not evaluate to a boolean for a settled part`,
  );
  return value as boolean;
}

function renderSettled(markdown: string, cut = true): string {
  return renderToStaticMarkup(
    createElement(
      Streamdown,
      {
        mode: "streaming",
        parseIncompleteMarkdown: settledParseIncompleteMarkdown(cut),
        remend: !cut
          ? FINISHED_TEXT_REMEND
          : hasIncompleteLinkRepair(markdown)
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
  // The tests above call `remend` directly, so only a render shows the UI still repairs cut bodies.
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
    const text = renderedText(renderSettled(complete, false));
    assert.equal(
      underscores(text),
      underscores(complete),
      `the settled render of ${JSON.stringify(complete)} changed how many underscores reach the ` +
        `reader: ${JSON.stringify(text)}`,
    );
  }
});

test("a finished reply keeps its lone markers as written", () => {
  for (const reply of [
    "Fields marked with an asterisk (*) are required.",
    "Add *.pyc to your .gitignore file.",
    "Pass options with **kwargs to forward them.",
    "Press the ` key to open the console.",
    "Delete files matching *.tmp first.\n\nThen restart the server.",
    "Unpack it with ***args in Python 2 style.",
    "Use ***args to unpack them.\n",
    "It costs ~~ 5 dollars.",
    "Load it as ~~fast mode in the config.",
  ]) {
    const html = renderSettled(reply, false);
    assert.doesNotMatch(
      html,
      /data-streamdown="(emphasis|strong|inline-code|strikethrough)"|<em>|<strong>|<code|<del/,
      `a finished ${JSON.stringify(reply)} was restyled: ${html}`,
    );
    assert.equal(
      renderedText(html).replace(/\s+/g, ""),
      reply.replace(/\s+/g, ""),
      `a finished ${JSON.stringify(reply)} did not reach the reader as written`,
    );
  }
});

test("a cut reply still hides a dangling image", () => {
  for (const cut of [
    "see ![alt](https://example.com/very/long/partial",
    "see ![alt](data:image/png;base64,iVBORw0KGgo",
  ]) {
    const text = renderedText(renderSettled(cut));
    assert.doesNotMatch(
      text,
      /!\[|example\.com|base64/,
      `a cut ${JSON.stringify(cut)} showed its partial image: ${text}`,
    );
  }
});

test("a finished reply keeps remend's literal escapes", () => {
  const list = renderSettled("- >= 16 GB of RAM", false);
  assert.match(list, /<li[^>]*>&gt;= 16 GB of RAM<\/li>/);
  assert.doesNotMatch(list, /blockquote/);
  const ranges = renderSettled(
    "It takes 5~10 minutes and uses 20~30 GB.",
    false,
  );
  assert.doesNotMatch(ranges, /<del|data-streamdown="strikethrough"/);
  assert.equal(
    renderedText(ranges),
    "It takes 5~10 minutes and uses 20~30 GB.",
  );
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

/* No timing assertion: a 15% margin would fail on a busy runner; the claim belongs in the PR. */
