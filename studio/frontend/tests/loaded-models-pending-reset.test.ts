// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An optimistic "Loading" row is only ever retired by the terminal lifecycle
// event, and nothing listens for that while the indicator is disabled. So a
// load in flight when the pref goes off leaves a pending entry that survives
// into the next enable, and `withPendingLoads` will keep rendering it: it
// yields only to a polled row for the same runtime, and a load that failed or
// was since unloaded has none. The result is a loading row no refresh can
// remove.
//
// The behaviour half is asserted against `withPendingLoads` directly; the
// wiring half by reading the source, since the node suite has no DOM to mount
// the hook into.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  type LoadedModelEntry,
  type LoadedModelSource,
  withPendingLoads,
} from "../src/features/loaded-models/loaded-models-sources.ts";

const SOURCE = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/loaded-models/use-loaded-models.ts",
      import.meta.url,
    ),
  ),
  "utf8",
);

test("a stale pending entry outlives every poll, so it must not survive a disable", () => {
  const pending = new Map<LoadedModelSource, string | null>([
    ["image", "unsloth/flux"],
  ]);
  // What the poll reports after the load failed: this runtime holds nothing.
  const polled: LoadedModelEntry[] = [
    {
      id: "chat:qwen",
      kind: "text",
      source: "chat",
      name: "qwen",
      detail: "GGUF",
    },
  ];

  const rows = withPendingLoads(polled, pending);
  const image = rows.filter((row) => row.source === "image");
  assert.equal(image.length, 1);
  assert.equal(image[0].loading, true);

  // Only an empty pending map clears it; no status refresh can.
  assert.deepEqual(withPendingLoads(polled, new Map()), polled);
});

test("the hook drops pending loads while the indicator is disabled", () => {
  const guard = SOURCE.slice(
    SOURCE.indexOf("if (wasEnabled !== enabled)"),
    SOURCE.indexOf("// The load call announces itself"),
  );
  assert.ok(guard.length > 0, "expected the enable-transition guard");
  assert.match(
    guard,
    /if \(!enabled && pending\.size > 0\) setPending\(new Map\(\)\)/,
    "the disable transition must empty the pending map",
  );
});

test("the clear runs once per transition, not on every render", () => {
  // Adjusting state during render only terminates if it is guarded by a change
  // in the value it tracks; an unguarded setPending would re-render forever.
  assert.match(
    SOURCE,
    /const \[wasEnabled, setWasEnabled\] = useState\(enabled\);\s*if \(wasEnabled !== enabled\) \{\s*setWasEnabled\(enabled\);/,
    "the adjustment must be gated on the previous enabled value",
  );
});

test("pending rows are cleared on disable, not on enable", () => {
  // Clearing on the way back in would race the subscription: a load started
  // from another tab could be announced and then wiped.
  const guard = SOURCE.slice(
    SOURCE.indexOf("if (wasEnabled !== enabled)"),
    SOURCE.indexOf("// The load call announces itself"),
  );
  assert.doesNotMatch(guard, /if \(enabled\) setPending/);
  assert.match(guard, /!enabled &&/);
});
