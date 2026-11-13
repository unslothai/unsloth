// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type PillCompact,
  measurePillCompact,
} from "../src/hooks/use-composer-pill-fit.ts";

const LINE_HEIGHT = 36;
const CONTROLS_WIDTH = 84;

function stubRow(widths: Record<string, number>, available: number) {
  let attribute: string | null = null;
  const width = () => widths[attribute ?? "none"];
  const rowWraps = () => width() > available;
  const controlsWrap = () =>
    !rowWraps() && width() + CONTROLS_WIDTH > available;
  const controls = {
    getBoundingClientRect: () => ({
      width: CONTROLS_WIDTH,
      top: controlsWrap() ? LINE_HEIGHT : 0,
    }),
    nextElementSibling: null,
  };
  return {
    get attribute() {
      return attribute;
    },
    setAttribute: (_name: string, value: string) => {
      attribute = value;
    },
    removeAttribute: () => {
      attribute = null;
    },
    children: [{ getBoundingClientRect: () => ({ height: LINE_HEIGHT }) }],
    getBoundingClientRect: () => ({
      height: rowWraps() ? LINE_HEIGHT * 2 : LINE_HEIGHT,
      bottom: rowWraps() ? LINE_HEIGHT * 2 : LINE_HEIGHT,
    }),
    nextElementSibling: {
      getBoundingClientRect: () => ({ width: 0, top: 0 }),
      nextElementSibling: controls,
    },
  };
}

function measure(
  widths: Record<string, number>,
  available: number,
  forceCompact = false,
): { result: PillCompact; attribute: string | null } {
  const row = stubRow(widths, available);
  const result = measurePillCompact(
    row as unknown as HTMLElement,
    forceCompact,
  );
  return { result, attribute: row.attribute };
}

const WIDTHS = { none: 470, first: 330, true: 190 };

test("keeps every label when the row already fits", () => {
  assert.equal(measure(WIDTHS, 640).result, undefined);
});

test("collapses the leading permission pill rather than wrap the controls", () => {
  assert.equal(measure(WIDTHS, 500).result, "first");
});

test("collapses the whole row when the first pill is not enough", () => {
  assert.equal(measure(WIDTHS, 360).result, "true");
});

test("stays collapsed when even icons overflow", () => {
  assert.equal(measure(WIDTHS, 120).result, "true");
});

test("skips measuring when the count or mobile rule already forces compact", () => {
  assert.equal(measure(WIDTHS, 1200, true).result, "true");
});

test("leaves the row on the stage it returns, so the render agrees", () => {
  assert.equal(measure(WIDTHS, 500).attribute, "first");
  assert.equal(measure(WIDTHS, 640).attribute, null);
});
