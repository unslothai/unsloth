// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The desktop sidebar collapses to nothing, so it never shows the web build's
// icon rail, but it used to wear that rail's styling on the way there: a frame
// with the labels hidden, the buttons centred and the panel white but the width
// not yet gone drew a bare column of icons. That is the reported collapse ghost.

async function source(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

test("a sidebar that collapses to zero never enters icon-rail mode", async () => {
  const sidebar = await source("../src/components/ui/sidebar.tsx");
  const attribute = sidebar.match(/data-collapsible=\{([\s\S]*?)\n\s*\}/);
  assert.ok(attribute, "data-collapsible is no longer written as an expression");
  const expression = attribute[1];
  assert.match(
    expression,
    /collapseToZero/,
    "the value has to depend on collapseToZero",
  );
  assert.match(expression, /"zero"/, "and resolve to a value no rule matches");
});

test("no styling keys off the zero-width collapse value", async () => {
  // The point of "zero" is that it is inert; anything matching it puts the rail
  // styling back into a state that is never the destination.
  for (const path of [
    "../src/components/ui/sidebar.tsx",
    "../src/components/app-sidebar.tsx",
    "../src/index.css",
  ]) {
    const text = await source(path);
    assert.ok(
      !text.includes("collapsible=zero"),
      `${path} styles the zero-width collapse state`,
    );
    assert.ok(
      !text.includes('data-collapsible="zero"]'),
      `${path} styles the zero-width collapse state`,
    );
  }
});

test("the icon rail styling still exists for the web build", async () => {
  // The web sidebar really does collapse to the rail, so its rules must stay.
  const css = await source("../src/index.css");
  assert.match(css, /\[data-collapsible="icon"\] \[data-sidebar="sidebar"\]/);
  const sidebar = await source("../src/components/ui/sidebar.tsx");
  assert.match(sidebar, /group-data-\[collapsible=icon\]:/);
});
