// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { applyThreadWidthSteps } from "../src/components/assistant-ui/thread-width-steps.ts";
import { readSrc } from "./helpers/kit.ts";

function fakeRoot() {
  const attributes = new Set<string>();
  return {
    attributes,
    toggleAttribute(name: string, force: boolean) {
      if (force) attributes.add(name);
      else attributes.delete(name);
      return force;
    },
  };
}

test("the steps match the container queries they replace, bounds included", () => {
  const root = fakeRoot();
  const at = (width: number) => {
    applyThreadWidthSteps(root as unknown as Element, width, 16);
    return [...root.attributes].sort();
  };
  assert.deepEqual(at(800), ["data-thread-wide"]);
  assert.deepEqual(at(576), ["data-thread-wide"], "min-width: 36rem holds at exactly 576 px");
  assert.deepEqual(at(575.5), []);
  assert.deepEqual(at(504), ["data-thread-narrow"], "max-width: 31.5rem holds at exactly 504 px");
  assert.deepEqual(at(504.5), []);
  assert.deepEqual(at(300), ["data-thread-narrow", "data-thread-tiny"], "max-width: 18.75rem holds at exactly 300 px");
  assert.deepEqual(at(301), ["data-thread-narrow"]);
  applyThreadWidthSteps(root as unknown as Element, 600, 20);
  assert.deepEqual([...root.attributes], ["data-thread-narrow"], "rem follows the root font size, as in a container query: 600 px is under 31.5 x 20 px and not over 36 x 20 px");
});

test("the thread root is no longer a size container, so a width change restyles only what the steps select", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const open = thread.indexOf("<ThreadPrimitive.Root");
  const root = thread.slice(open, thread.indexOf(">", thread.indexOf("className=", open)));
  assert.match(root, /ref=\{watchWidthSteps\}/);
  assert.equal(root.includes("@container"), false);
  assert.ok(root.includes("[contain:style]"), "the style containment the size container carried scopes KaTeX's equation counter; without it each numbered equation counts on from the ones in earlier messages");
  assert.match(thread, /const watchWidthSteps = useCallback\(\s*\(root: HTMLDivElement \| null\) =>\s*root \? observeThreadWidthSteps\(root\) : undefined,\s*\[\],\s*\);/);
  const css = readSrc("index.css");
  assert.equal(/@container \((max|min)-width: (31\.5|18\.75)rem\)/.test(css), false);
  assert.equal(css.includes("@container (min-width: 36rem)"), false);
  for (const selector of [
    ":where(.aui-thread-root[data-thread-narrow]) .unsloth-composer-dock-inner {",
    ":where(.aui-thread-root[data-thread-narrow]) .aui-thread-welcome-message {",
    ":where(.aui-thread-root[data-thread-narrow]) .unsloth-welcome-title {",
    ":where(.aui-thread-root[data-thread-tiny]) .unsloth-welcome-title {",
    '.aui-thread-root:where([data-thread-wide]) [data-streamdown="code-block-header"] {',
  ]) {
    assert.ok(css.includes(selector), `${selector} keeps the selector's own specificity`);
  }
  assert.ok(css.includes("@container (max-width: 36rem)"), "the thinking pill still queries its composer surface");
  assert.match(css, /\.aui-thread-root:where\(\[data-thread-wide\]\) \[data-streamdown="code-block-header"\] \{\s*font-size: calc\(0\.75rem \* var\(--ui-font-scale, 1\)\);\s*\}\s*@media print \{\s*\.aui-thread-root \[data-streamdown="code-block-header"\] \{\s*font-size: calc\(0\.75rem \* var\(--ui-font-scale, 1\)\);\s*\}\s*\}/, "a print is laid out at the page's width, which no observer reports, so it takes the wide header a page always was under the container query");
});
