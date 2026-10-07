// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reads can be in flight across an eject and land out of order; a ticket drops stale ones.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PAGES = [
  ["images", "features/images/images-page.tsx", "getDiffusionStatus", "unloadDiffusionModel"],
  ["video", "features/video/video-page.tsx", "getVideoStatus", "unloadVideoModel"],
] as const;

function callbackBody(source: string, name: string): string {
  const declaration = `const ${name} = useCallback`;
  const at = source.indexOf(declaration);
  assert.ok(at >= 0, `${name} is not declared as a useCallback`);
  const start = source.indexOf("(", at + declaration.length);
  let depth = 0;
  for (let i = start; i < source.length; i += 1) {
    if (source[i] === "(") depth += 1;
    else if (source[i] === ")") {
      depth -= 1;
      if (depth === 0) return source.slice(start + 1, i);
    }
  }
  assert.fail(`${name}'s callback never closes`);
}

for (const [name, path, read, unload] of PAGES) {
  test(`the ${name} page lets only the newest status read write`, () => {
    const page = readSrc(path);
    assert.match(page, /const statusTicket = useRef\(0\);/);
    // Checks the guard semantically so equivalent spellings pass.
    const body = callbackBody(page, "setStatusIfNewest");
    const write = body.indexOf("setStatus(");
    assert.notEqual(write, -1, "setStatusIfNewest no longer writes the status");
    const held = /if\s*\(\s*ticket\s*===\s*statusTicket\.current\s*\)[\s{]*setStatus\(/.exec(body);
    // The stale branch's return must be bare, not return setStatus(next).
    const early = /if\s*\(\s*ticket\s*!==\s*statusTicket\.current\s*\)[\s{]*return\s*(?:[;}]|\r?\n)/.exec(
      body,
    );
    const guard = held ?? early;
    assert.ok(guard, "a superseded read must not write");
    assert.ok(
      guard.index < write,
      "the ticket guard must come before the status write, not after it",
    );
    assert.equal(
      (body.match(/setStatus\(/g) ?? []).length,
      1,
      "setStatusIfNewest must write the status exactly once, under the ticket guard",
    );
    // The write must be reachable past the stale branch's block.
    if (early && !held && /\{/.test(early[0])) {
      const open = body.indexOf("{", early.index);
      let depth = 0;
      let close = -1;
      for (let i = open; i < body.length; i += 1) {
        if (body[i] === "{") depth += 1;
        else if (body[i] === "}") {
          depth -= 1;
          if (depth === 0) {
            close = i;
            break;
          }
        }
      }
      assert.notEqual(close, -1, "the stale branch never closes");
      assert.ok(write > close, "the status write is stranded inside the stale branch");
    }
    assert.doesNotMatch(
      page,
      new RegExp(`setStatus\\(await ${read}\\(\\)\\)`),
      "the bare read must not write directly",
    );
    assert.doesNotMatch(
      page,
      new RegExp(`setStatus\\(await ${unload}\\(\\)\\)`),
      "nor the unload",
    );
    assert.equal(
      (page.match(/setStatusIfNewest\(/g) ?? []).length,
      3,
      "the refresh, the load-progress read and the unload all go through it",
    );
  });

  test(`the ${name} page claims its ticket before awaiting, not after`, () => {
    const page = readSrc(path);
    // Claiming the ticket after the await would give every read the newest ticket.
    assert.match(page, /const ticket = \+\+statusTicket\.current;\s*\n\s*try \{/);
  });
}
