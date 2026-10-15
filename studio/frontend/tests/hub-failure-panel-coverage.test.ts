// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

// Asserted against the type, so adding a failure kind without a panel branch fails here.

const NOT_RENDERED = new Set(["aborted"]);

test("every renderable Hub failure kind has a branch in the panel", async () => {
  const network = await readText("../src/features/hub/lib/network.ts");
  const decl = /export type HubFailureKind =([\s\S]*?);/.exec(network);
  assert.ok(decl, "could not find HubFailureKind in network.ts");
  const kinds = [...decl[1].matchAll(/"([a-z-]+)"/g)].map((m) => m[1]);
  assert.ok(kinds.length >= 5, `parsed too few kinds: ${kinds.join(", ")}`);

  const states = await readText(
    "../src/features/hub/catalog/catalog-states.tsx",
  );
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
