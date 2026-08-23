// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// DOM parity check for the reasoning pagination flag.
//
// Arm A: the REAL PartitionedMarkdownText from markdown-text.tsx, rendered with
//        paginateReasoning = REASONING_PAGINATION_ENABLED (the shipped flag).
// Arm B: the same component with paginateReasoning = true (discriminating control).
// Arm C: a pagination-free PartitionedMarkdownText, built from the SAME module
//        internals with every pagination expression removed.
//
// A === C byte for byte is the claim. A !== B on a long trace proves the harness
// can see pagination at all.

import assert from "node:assert/strict";
import { createElement as h } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

// The tree under test. Defaults to this file's own directory rather than any
// hardcoded path: a harness that silently verifies somebody else's worktree
// passes regardless of what the branch it is quoted against contains.
import { execFileSync } from "node:child_process";
import path from "node:path";

const ROOT = path.resolve(
  process.argv[2] ??
    path.join(path.dirname(new URL(import.meta.url).pathname), ".."),
);
const TARGET = `${ROOT}/src/components/assistant-ui/markdown-text.tsx`;
const SCHEDULE = `${ROOT}/src/components/assistant-ui/streaming-render-schedule.ts`;
const FLAGS = `${ROOT}/src/components/assistant-ui/thread-feature-flags.ts`;

// Top-level declarations only: nested ones are indented, and the file's own
// `export const MarkdownText` already starts with `export`.
const exposeInternals = {
  name: "expose-internals",
  enforce: "pre",
  transform(code, id) {
    if (id.split("?")[0] !== TARGET) return null;
    return { code: code.replace(/^(const|function) /gm, "export $1 "), map: null };
  },
};

const head = execFileSync("git", ["-C", ROOT, "log", "--oneline", "-1"], {
  encoding: "utf8",
}).trim();
const dirty = execFileSync("git", ["-C", ROOT, "status", "--short", "."], {
  encoding: "utf8",
}).trim();
console.log(`tree under test: ${ROOT}`);
console.log(`  HEAD: ${head}`);
console.log(`  uncommitted here: ${dirty ? dirty.split("\n").length + " path(s)" : "none"}\n`);

const server = await createServer({
  root: ROOT,
  configFile: `${ROOT}/vite.config.ts`,
  appType: "custom",
  logLevel: "error",
  server: { middlewareMode: true, hmr: false, watch: null },
  plugins: [exposeInternals],
});

const M = await server.ssrLoadModule(TARGET);
const S = await server.ssrLoadModule(SCHEDULE);
const F = await server.ssrLoadModule(FLAGS);
const { Streamdown } = await server.ssrLoadModule("streamdown");
const { safeMarkdownUrl } = await server.ssrLoadModule(`${ROOT}/src/lib/safe-markdown-url.ts`);

// Every render plan the cache hands out, per arm. The plan carries the block ids
// that become React keys inside PartitionedStreamdownShell, so comparing the
// recorded plans covers keys, which HTML does not show.
let recording = null;
const realUpdate = S.IncrementalMarkdownCache.prototype.update;
S.IncrementalMarkdownCache.prototype.update = function update(...args) {
  const result = realUpdate.apply(this, args);
  if (recording) recording.push({ args, result });
  return result;
};

// Arm C. Same tree PartitionedMarkdownText renders, with the pagination
// expressions gone: no page selection, no page-boundary state, no Show more /
// Show less controls, no canonical-source provider, and the untouched markdown
// and provenance handed straight to the cache.
function NoPaginationMarkdownText(props) {
  const {
    codeHighlighting,
    isStreaming,
    markdown,
    messageId,
    persistedTrailingLfOrdinals,
    statusType,
  } = props;
  const cache = new S.IncrementalMarkdownCache(persistedTrailingLfOrdinals);
  const incrementalRender = cache.update(
    markdown,
    isStreaming,
    persistedTrailingLfOrdinals,
  );
  return h(
    "div",
    { "data-status": statusType, className: "min-w-0 max-w-full" },
    h(
      M.MarkdownCodeHighlightingContext.Provider,
      { value: codeHighlighting },
      h(
        M.StreamingMarkdownPlanContext.Provider,
        { value: incrementalRender },
        h(
          Streamdown,
          {
            key: messageId,
            mode: "streaming",
            parseIncompleteMarkdown: false,
            parseMarkdownIntoBlocksFn: M.parseProviderShellBlock,
            isAnimating: isStreaming,
            animated: M.STREAMDOWN_IMMEDIATE_UPDATES,
            plugins:
              codeHighlighting === "syntax"
                ? M.STREAMDOWN_SYNTAX_PLUGINS
                : M.STREAMDOWN_PLAIN_CODE_PLUGINS,
            components: M.STREAMDOWN_COMPONENTS,
            urlTransform: safeMarkdownUrl,
            controls: M.STREAMDOWN_CONTROLS,
            shikiTheme: M.STREAMDOWN_SHIKI_THEME,
            BlockComponent: M.PartitionedStreamdownShell,
          },
          incrementalRender.shellMarkdown,
        ),
      ),
    ),
  );
}

const render = (Component, props) => {
  recording = [];
  const html = renderToStaticMarkup(h(Component, props));
  const plans = recording;
  recording = null;
  return { html, plans };
};

const long = (unit, times) =>
  Array.from({ length: times }, (_, i) => unit.replace("{i}", String(i))).join("\n");

const corpus = {
  "empty": "",
  "short paragraph": "Let me think about **this** and then [check](https://example.com).",
  "short list + table":
    "> quoted\n\n- one\n- two\n\n| a | b |\n| - | - |\n| 1 | 2 |",
  "short fence": "Before\n\n```ts\nconst answer = 42;\n```\n\nAfter",
  "incomplete fence": "Before\n\n```ts\nconst answer = 42;",
  "long list (46K)": long("- step {i}: reasoning about the problem at hand", 1_000),
  "long prose (60K)": long(
    "Paragraph {i}. " + "The model considers the constraint carefully. ".repeat(4),
    600,
  ),
  "long mixed (70K)": long(
    "## Section {i}\n\nText for {i} with `inline` code and $x_{i}$ math.\n\n" +
      "```py\nvalue = {i}\nprint(value)\n```\n\n- bullet {i}\n- bullet again\n\n" +
      "| col | val |\n| - | - |\n| {i} | y |",
    120,
  ),
  "long fence (40K)": "```py\n" + long("value_{i} = compute({i})", 1_500) + "\n```",
  "long unbroken line (30K)": "x".repeat(30_000),
};

let failures = 0;
let armsCompared = 0;
let discriminating = 0;

// `pageUsesSourceSlice` swaps persisted provenance for the empty list, so the
// off arm has to be compared with a non-empty list too.
const provenances = { "": [], " +prov": [0, 1, 2] };

for (const [name, markdown] of Object.entries(corpus)) {
  for (const isStreaming of [false, true]) {
   for (const [suffix, persistedTrailingLfOrdinals] of Object.entries(provenances)) {
    const base = {
      codeHighlighting: "plain",
      isStreaming,
      markdown,
      messageId: "msg-1",
      persistedTrailingLfOrdinals,
      statusType: isStreaming ? "running" : "complete",
    };
    const label = `${name}${suffix} / streaming=${isStreaming}`;

    const armA = render(M.PartitionedMarkdownText, {
      ...base,
      paginateReasoning: F.REASONING_PAGINATION_ENABLED,
    });
    const armB = render(M.PartitionedMarkdownText, {
      ...base,
      paginateReasoning: true,
    });
    const armC = render(NoPaginationMarkdownText, base);

    armsCompared += 1;
    try {
      assert.equal(armA.html.length, armC.html.length, "html length");
      assert.equal(armA.html, armC.html, "html bytes");
      assert.deepEqual(armA.plans, armC.plans, "render plans (block ids -> keys)");
      assert.equal(armA.plans.length, armC.plans.length, "plan count");
    } catch (error) {
      failures += 1;
      console.log(`FAIL  ${label}: ${error.message}`);
      const i = [...armA.html].findIndex((c, k) => c !== armC.html[k]);
      if (i >= 0) {
        console.log(`  first divergence at byte ${i}`);
        console.log(`  off : ${JSON.stringify(armA.html.slice(Math.max(0, i - 60), i + 120))}`);
        console.log(`  none: ${JSON.stringify(armC.html.slice(Math.max(0, i - 60), i + 120))}`);
      }
      continue;
    }

    const differs = armA.html !== armB.html;
    if (differs) discriminating += 1;
    console.log(
      `ok    ${label.padEnd(38)} ${String(armA.html.length).padStart(8)} bytes` +
        `  A==C  A!=B:${differs ? "yes" : "no "}` +
        `  showEarlier:${armB.html.includes("reasoning-show-earlier") ? "B" : "-"}` +
        `  offHasControls:${armA.html.includes("reasoning-show-") ? "YES(BUG)" : "no"}`,
    );
   }
  }
}

console.log(
  `\n${armsCompared - failures}/${armsCompared} cases byte-identical with the flag off; ` +
    `${discriminating}/${armsCompared} cases where pagination on is visibly different.`,
);
await server.close();
process.exit(failures === 0 && discriminating > 0 ? 0 : 1);
