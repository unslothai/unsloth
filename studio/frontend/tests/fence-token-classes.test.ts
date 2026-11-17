// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { unslothDarkTheme, unslothLightTheme } from "../src/components/assistant-ui/code-themes.ts";
import {
  buildTokenClasses,
  installTokenClasses,
  tokenClassesFor,
} from "../src/components/assistant-ui/fence-token-classes.ts";
import { readSrc } from "./helpers/kit.ts";

const THEMES = [unslothLightTheme, unslothDarkTheme];

// Every declaration the bundled themes put on token spans in a long rendered chat (both colour cases).
const SEEN = [
  "--sdm-c:#383A42", "--sdm-c:#383a42", "--sdm-c:#E45649", "--sdm-c:#e45649", "--sdm-c:#A626A4",
  "--sdm-c:#a626a4", "--sdm-c:#986801", "--sdm-c:#4078F2", "--sdm-c:#50A14F", "--sdm-c:#C18401",
  "--sdm-c:#0184BC", "--sdm-c:#A0A1A7", "--shiki-dark:#ABB2BF", "--shiki-dark:#abb2bf",
  "--shiki-dark:#E06C75", "--shiki-dark:#e06c75", "--shiki-dark:#C678DD", "--shiki-dark:#c678dd",
  "--shiki-dark:#E5C07B", "--shiki-dark:#D19A66", "--shiki-dark:#d19a66", "--shiki-dark:#61AFEF",
  "--shiki-dark:#56B6C2", "--shiki-dark:#98C379", "--shiki-dark:#7F848E",
  "--shiki-dark-font-style:italic", "--shiki-light-font-style:inherit", "--shiki-light-font-style:italic",
];

test("the bundled themes yield a class for every declaration their tokens carry, each setting exactly it", () => {
  const { names, css } = buildTokenClasses(THEMES);
  for (const key of SEEN) assert.ok(names.has(key), key);
  const rules = css.split("\n");
  assert.equal(rules.length, names.size);
  for (const [key, name] of names) {
    const colon = key.indexOf(":");
    assert.ok(rules.includes(`.${name}{${key.slice(0, colon)}:${key.slice(colon + 1)}}`), key);
  }
  assert.ok(names.size < 400, "a few hundred rules at most, written once");
});

test("without a document nothing is installed and spans keep inline colours; once installed, known sets map to classes", () => {
  assert.equal(tokenClassesFor({ "--sdm-c": "#383A42" }), null);
  const appended: { textContent: string }[] = [];
  const element = { textContent: "", setAttribute: () => {} };
  (globalThis as { document?: unknown }).document = {
    head: { appendChild: (node: { textContent: string }) => appended.push(node) },
    createElement: () => element,
  };
  try {
    installTokenClasses(THEMES);
    installTokenClasses(THEMES);
    assert.equal(appended.length, 1, "the sheet is written once and never edited");
    const classes = tokenClassesFor({ "--sdm-c": "#383A42", "--shiki-dark": "#ABB2BF" });
    assert.match(classes ?? "", /^ sdm-t\w+ sdm-t\w+$/);
    assert.equal(tokenClassesFor({ "--sdm-c": "#123456" }), null, "a colour outside the themes stays inline");
    assert.equal(tokenClassesFor({ "--sdm-c": "#383A42", "--sdm-tbg": "#383A42" }), null);
  } finally {
    delete (globalThis as { document?: unknown }).document;
  }
});

test("every token span takes its colours through the shared classes, installed at module load from the bundled themes", () => {
  const defer = readSrc("components/assistant-ui/code-fence-defer.tsx");
  assert.match(defer, /installTokenClasses\(\[unslothLightTheme, unslothDarkTheme\]\);/);
  assert.equal(defer.includes("style={style}"), false);
  assert.equal((defer.match(/\{\.\.\.styledSpan\(/g) ?? []).length, 2);
  assert.equal((defer.match(/styledSpan\(`\$\{(?:lineClass|LINE_CLASS)\} \$\{TOKEN_CLASS\}`/g) ?? []).length, 2);
  const markdown = readSrc("components/assistant-ui/markdown-text.tsx");
  assert.match(markdown, /const STREAMDOWN_SHIKI_THEME = \[\s*unslothLightTheme,\s*unslothDarkTheme,\s*\]/,
    "the classes come from the themes the highlighter uses");
});
