// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images and Video pages hold their own status and re-read it on tab
// activation and on their own actions, never on a timer. So two reads can be in
// flight across an eject: an activation read that saw the pipeline loaded, and
// the post-eject read that saw it gone. Responses have no order, and the older
// one landing last left the page offering to generate against a freed runtime,
// with no poll coming to correct it.
//
// Asserted by reading the source: both pages pull in the whole media runtime,
// which the node suite cannot mount.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PAGES = [
  ["images", "features/images/images-page.tsx", "getDiffusionStatus", "unloadDiffusionModel"],
  ["video", "features/video/video-page.tsx", "getVideoStatus", "unloadVideoModel"],
] as const;

/** The argument list of `const NAME = useCallback(...)`, parentheses balanced. */
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
    // Read the GUARD, not the one line that spelled it. #10788 rewrote this as an early
    // return, which admits exactly the same reads, and the exact-text form went red over a
    // refactor that changed nothing. Both spellings are checked against the callback's own
    // body, so a guard that lives somewhere else in the file cannot stand in for it.
    const body = callbackBody(page, "setStatusIfNewest");
    const write = body.indexOf("setStatus(");
    assert.notEqual(write, -1, "setStatusIfNewest no longer writes the status");
    const guard =
      /if\s*\(\s*ticket\s*===\s*statusTicket\.current\s*\)[\s{]*setStatus\(/.exec(body) ??
      // The stale branch's return must be BARE. `return setStatus(next);` also reads as an
      // early return and also precedes the normal write, while writing the superseded
      // status out of the return expression itself.
      /if\s*\(\s*ticket\s*!==\s*statusTicket\.current\s*\)[\s{]*return\s*(?:[;}]|\r?\n)/.exec(
        body,
      );
    assert.ok(guard, "a superseded read must not write");
    // Ordering, not just presence. Either spelling can be present while the write happens
    // FIRST, and `setStatus(next); if (ticket !== statusTicket.current) return;` has already
    // published the superseded status by the time it returns, which is the whole bug.
    assert.ok(
      guard.index < write,
      "the ticket guard must come before the status write, not after it",
    );
    // Every writer goes through it, so none can be the one that slips past.
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
    // Claiming after the await would hand every read the newest ticket and
    // defeat the whole thing.
    assert.match(page, /const ticket = \+\+statusTicket\.current;\s*\n\s*try \{/);
  });
}
