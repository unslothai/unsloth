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

/**
 * WHY THIS FILE EXISTS. `remend` is the incomplete-markdown repair Streamdown runs over a message
 * before parsing it, and Unsloth runs it on every SETTLED body: `markdown-text.tsx` passes
 * `parseIncompleteMarkdown={!incrementalRender}` and `incrementalRender` is null exactly when the
 * message is not streaming. So this dependency decides what a finished message LOOKS like, not
 * only what a half-written one looks like, and a version bump is a rendering change.
 *
 * These are RUN, not scraped. Every assertion below is either a call into the resolved package or
 * a render through Streamdown, so the file fails if the dependency is downgraded, if the override
 * that forces one copy of it is dropped, or if the settled branch stops asking for the repair.
 * That last one is checked by EVALUATING the `parseIncompleteMarkdown` expression that
 * `markdown-text.tsx` writes, at `incrementalRender === null`, and handing the result to a real
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

// remend 1.3.0 does not recognise `\( ... \)` or `\[ ... \]` as math, so it counts the `_` of a
// subscript as an unmatched emphasis marker and "completes" it by appending another one. The
// document is COMPLETE, so there is nothing to complete, and Unsloth renders the extra character
// literally at the end of the message. Shared with the copy sweep below, which runs the same four
// documents through every remend on disk rather than through the one this file imports.
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
  // The other half of the bump: the repair must still DO something. Without this, dropping the
  // dependency entirely, or passing `parseIncompleteMarkdown={false}`, would pass the tests above
  // and silently take the streaming repair with it.
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

/**
 * The `parseIncompleteMarkdown` a SETTLED body is rendered with, taken from the source and
 * evaluated rather than restated. `incrementalRender` is null exactly when the message is not
 * streaming, so evaluating the real expression in that state is the same question the component
 * answers on the settled path.
 */
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
      },
      markdown,
    ),
  );
}

test("the settled render path runs the repair, not just the package", () => {
  // THE TESTS ABOVE CALL `remend` THEMSELVES, so all of them pass while the UI has the repair
  // switched off: mutating `parseIncompleteMarkdown={!incrementalRender}` to `{false}` in
  // `markdown-text.tsx` left this file green until this test existed. The only way to see the
  // difference is to render, and the only documents where the repair is VISIBLE are truncated
  // ones -- a complete document is by construction unchanged either way.
  //
  // A truncated body does reach the settled path: a cancelled or errored response settles
  // mid-construct and is then rendered with `incrementalRender === null` forever.
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

// The text a reader would see, given markup produced by renderToStaticMarkup. Deliberately a scan
// rather than a `<[^>]*>` replace: that pattern is bypassable in general, and CodeQL fails the
// build over it at high severity, correctly, because nothing in the type system says the input is
// trusted. The scan is sound for this input for a reason worth stating: React escapes `<` in text
// to `&lt;`, so every raw `<` in the output really does open a tag.
//
// Measured, so nobody has to guess: today this changes no count, because Streamdown's attributes
// happen to contain no underscores, and the assertion below is red with or without it. It is here
// so that stays true if an attribute ever gains one, not because it is currently load-bearing.
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
  // The rendering half of the bump, asserted where the reader sees it. Under remend 1.3.0 the
  // subscript `_` is "completed" with a second one, so the paragraph gains a character that is
  // not in the source; count them rather than matching a fixed string, since the renderer resolves
  // the `\(` escapes.
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

// Every `node_modules/**/remend` in the tree, in npm's own layout: a package's private copy lives
// at `<package>/node_modules/remend`, and a scoped directory holds packages rather than packages
// of its own.
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
  // THE COPY THIS FILE IMPORTS IS NOT NECESSARILY THE ONE THE UI RUNS. Unsloth renders settled
  // bodies through `<Streamdown parseIncompleteMarkdown>`, and streamdown@2.5.0 depends on
  // `"remend": "1.3.0"` EXACTLY, so its own `import remend from "remend"` resolves against
  // `node_modules/streamdown/node_modules` first. Bumping the top-level pin to 1.3.1 therefore
  // makes npm nest a second, older copy under streamdown unless `overrides.remend` forces one
  // version on the whole tree; verified with `npm install --package-lock-only` after deleting the
  // override, which writes `node_modules/streamdown/node_modules/remend -> 1.3.0` into the lock.
  //
  // In that state the tests above still pass, because they import the hoisted 1.3.1, while the
  // rendered message goes back to `where \( \delta_{r} = 1 \) holds._`. So the version is asserted
  // where Streamdown would find it, not only where this file finds it.
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
      // `undefined`, not `{}`: Streamdown forwards its optional `remend` prop straight through, and
      // Unsloth does not pass one.
      assert.equal(
        loaded.default(complete, undefined),
        complete,
        `${path.relative(FRONTEND_ROOT, copy)} rewrote a complete document: ${JSON.stringify(complete)}`,
      );
    }
  }
});

/*
 * NO TIMING ASSERTION HERE, deliberately.
 *
 * The bump is a performance change as well as a rendering one: 1.3.0 answers "is this offset
 * inside a code block" by scanning from the start of the document every time it is asked, once per
 * candidate marker, so the repair grows faster than the length of a body. Timed on the frozen
 * studiobench corpus with no browser and no profiler attached, ten bodies cost 697.6 ms under
 * 1.3.0 and 111.0 ms under 1.3.1, and the largest body alone reads 2.95 / 10.93 / 158.40 / 539.92
 * ms at 13,347 / 26,694 / 53,388 / 106,776 characters against 3.30 / 11.21 / 29.39 / 69.42.
 *
 * A doubling-factor assertion over that shape was written, measured, and DELETED. On this host it
 * separates the two versions by 3.10 against 2.61, which is a 15 percent margin on a quantity that
 * only moves one way under load: it would fail on a busy runner for a reason that has nothing to
 * do with the property, and a test whose failure mode is "the machine was busy" gets re-run rather
 * than read. The performance claim belongs in the pull request beside the rungs it was measured
 * at, not in a unit test that cannot hold it.
 */
