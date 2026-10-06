// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");

function shadowOf(selector: string): string {
  const start = CSS.indexOf(`${selector} {`);
  assert.ok(start >= 0, `${selector} is styled`);
  const rule = CSS.slice(start, CSS.indexOf("}", start));
  const shadow = /box-shadow:\s*([^;]+);/.exec(rule)?.[1];
  assert.ok(shadow, `${selector} sets a shadow`);
  return shadow.replace(/\s*!important$/, "").trim();
}

test("toasts cast the composer's shadow in light and dark", () => {
  assert.equal(
    shadowOf("[data-sonner-toast][data-styled='true']"),
    shadowOf("\t.chat-composer-surface"),
  );
  assert.equal(
    shadowOf(".dark [data-sonner-toast][data-styled='true']"),
    shadowOf("\t.dark .chat-composer-surface"),
  );
});
