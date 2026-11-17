// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// When the last used model is on the device but fails to load, the send reports that failure
// instead of loading a different model. The guard is lifted from source, like the budget tests.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const source = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

function braced(from: number, what: string): string {
  const open = source.indexOf("{", from);
  assert.ok(open >= 0, `no block found for ${what}`);
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === "{") {
      depth += 1;
    } else if (source[i] === "}") {
      depth -= 1;
      if (depth === 0) {
        return source.slice(open + 1, i);
      }
    }
  }
  throw new Error(`unbalanced block for ${what}`);
}

const cascadeStart = source.indexOf("async function autoLoadSmallestModel(");
assert.ok(cascadeStart >= 0, "autoLoadSmallestModel is no longer defined");
const loopStart = source.indexOf(
  "for (const source of sources) {",
  cascadeStart,
);
assert.ok(loopStart >= 0, "the cascade's source loop is no longer there");
const loopBody = braced(loopStart, "the cascade's source loop");

// The loop's `if (...) break;` guards that read the remembered model's outcome.
const breakGuards = [...loopBody.matchAll(/if \(([^;{}]+)\) break;/g)]
  .map((match) => match[1])
  .filter((condition) => condition.includes("isRemembered"));
assert.ok(
  breakGuards.length > 0,
  "no guard stops the sweep after the last used model",
);

function stopsAfter(isRemembered: boolean, failed: boolean): boolean {
  const loadFailure = {
    current: failed
      ? {
          label: "unsloth/gemma-4-26B-A4B-it-GGUF",
          detail: "x",
          blamesModel: true,
        }
      : null,
  };
  return breakGuards.some((condition) =>
    (
      new Function(
        "isRemembered",
        "loadFailure",
        `return Boolean(${condition});`,
      ) as (a: boolean, b: typeof loadFailure) => boolean
    )(isRemembered, loadFailure),
  );
}

test("a failed last used model ends the sweep instead of loading the next model", () => {
  assert.ok(stopsAfter(true, true));
});

test("the last used model that did not fail does not end the sweep", () => {
  assert.ok(!stopsAfter(true, false));
});

test("without a remembered model a failure still moves on to the next one", () => {
  assert.ok(!stopsAfter(false, true));
});
