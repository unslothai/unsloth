// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// The whole point of this change is that the panel names the cause. A kind the
// renderer does not handle falls through to the generic wording, and because the
// catalog blanks `message` whenever a failure exists, the classified text is not
// shown anywhere else either. So every kind that can reach the panel must have a
// case, and this is asserted against the type rather than a list, so adding a
// kind without a branch fails here instead of silently degrading the panel.

// A superseded request is never rendered: it is not a failure the user caused.
const NOT_RENDERED = new Set(["aborted"]);

function read(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

test("every renderable Hub failure kind has a branch in the panel", async () => {
  const network = await read("../src/features/hub/lib/network.ts");
  const decl = /export type HubFailureKind =([\s\S]*?);/.exec(network);
  assert.ok(decl, "could not find HubFailureKind in network.ts");
  const kinds = [...decl[1].matchAll(/"([a-z-]+)"/g)].map((m) => m[1]);
  assert.ok(kinds.length >= 5, `parsed too few kinds: ${kinds.join(", ")}`);

  const states = await read("../src/features/hub/catalog/catalog-states.tsx");
  const start = states.indexOf("function describeFailure");
  assert.notEqual(start, -1, "could not find describeFailure");
  const body = states.slice(start, states.indexOf("\nexport function", start));

  const missing = kinds.filter(
    (kind) => !NOT_RENDERED.has(kind) && !body.includes(`case "${kind}":`),
  );
  assert.deepEqual(
    missing,
    [],
    `these kinds fall through to the generic panel: ${missing.join(", ")}`,
  );
});

test("the http branch reports the status the server actually returned", async () => {
  // Reachable on a mirror, where discovery is proxy-first and a non-2xx from the
  // backend is the only signal there is.
  const states = await read("../src/features/hub/catalog/catalog-states.tsx");
  const start = states.indexOf('case "http":');
  assert.notEqual(start, -1);
  const branch = states.slice(start, start + 400);
  assert.match(branch, /failure\.status/, "the status is the diagnosis here");
});
