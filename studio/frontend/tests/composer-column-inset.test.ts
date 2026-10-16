// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  composerColumnInset,
  composerInsetFromColumn,
} from "../src/components/assistant-ui/composer-column-inset.ts";
import { readSrc } from "./helpers/kit.ts";

const DOCK_INNER_RULE =
  /\.unsloth-composer-dock-inner \{\s*padding-left: var\(--composer-dock-inset-left/;
const FOOTER_COLUMN_CALL =
  /composerInsetFromColumn\(footer\.getBoundingClientRect\(\)/;
const NARROW_DOCK_RULE = /data-thread-narrow\]\) \.unsloth-composer-dock-inner/;

const base = {
  left: 0,
  right: 700,
  offsetWidth: 700,
  clientWidth: 678,
  borderLeft: 0,
  borderRight: 0,
  paddingLeft: 20,
  paddingRight: 20,
  bothEdges: true,
  rtl: false,
};

test("a both-edges gutter splits the scrollbar across the sides", () => {
  assert.deepEqual(composerColumnInset(base, { left: 0, right: 700 }), {
    left: 31,
    right: 31,
  });
});

test("overlay scrollbars reserve nothing, so only the padding is inset", () => {
  assert.deepEqual(
    composerColumnInset({ ...base, clientWidth: 700 }, { left: 0, right: 700 }),
    { left: 20, right: 20 },
  );
});

test("the dock's own offset is subtracted", () => {
  assert.deepEqual(composerColumnInset(base, { left: 10, right: 690 }), {
    left: 21,
    right: 21,
  });
});

test("a one-sided gutter with a border, as the Windows low-scale viewport has", () => {
  assert.deepEqual(
    composerColumnInset(
      { ...base, clientWidth: 680, borderLeft: 10, bothEdges: false },
      { left: 0, right: 700 },
    ),
    { left: 30, right: 30 },
  );
  assert.deepEqual(
    composerColumnInset(
      {
        ...base,
        clientWidth: 680,
        borderLeft: 10,
        bothEdges: false,
        rtl: true,
      },
      { left: 0, right: 700 },
    ),
    { left: 40, right: 20 },
  );
});

test("a measured column box wins over the computed one", () => {
  // WebKit: both-edges reserves 26px on the right while clientWidth drops 13px.
  assert.deepEqual(
    composerInsetFromColumn(
      { left: 300, right: 1394 },
      { left: 290, right: 1430 },
    ),
    { left: 10, right: 36 },
  );
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, FOOTER_COLUMN_CALL);
});

test("the dock pads from the measured inset, not a fixed gutter", () => {
  const css = readSrc("index.css");
  assert.match(css, DOCK_INNER_RULE);
  assert.ok(!NARROW_DOCK_RULE.test(css));
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.ok(!thread.includes('"unsloth-composer-dock-inner relative px-5'));
});
