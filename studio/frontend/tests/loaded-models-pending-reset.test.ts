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

test("the hook drops pending loads once recording is turned off", () => {
  const guard = SOURCE.slice(
    SOURCE.indexOf("if (wasTracking !== track)"),
    SOURCE.indexOf("// The load call announces itself"),
  );
  assert.ok(guard.length > 0, "expected the enable-transition guard");
  assert.match(
    guard,
    /if \(!track && pending\.size > 0\) setPending\(new Map\(\)\)/,
    "turning recording off must empty the pending map",
  );
});

test("the clear runs once per transition, not on every render", () => {
  // Adjusting state during render only terminates if it is guarded by a change
  // in the value it tracks; an unguarded setPending would re-render forever.
  assert.match(
    SOURCE,
    /const \[wasTracking, setWasTracking\] = useState\(track\);\s*if \(wasTracking !== track\) \{\s*setWasTracking\(track\);/,
    "the adjustment must be gated on the previous tracking value",
  );
});

test("pending rows are cleared when recording stops, not when it starts", () => {
  // Clearing on the way back in would race the subscription: a load started
  // from another tab could be announced and then wiped.
  const guard = SOURCE.slice(
    SOURCE.indexOf("if (wasTracking !== track)"),
    SOURCE.indexOf("// The load call announces itself"),
  );
  assert.doesNotMatch(guard, /if \(track\) setPending/);
  assert.match(guard, /!track &&/);
});

// A replacement load is the case the source-only suppression got wrong. The
// backend keeps the old pipeline in _state and frees it only at the commit,
// after the whole download, so /images/status reports the OLD model for
// minutes while the announcement names the new one.
test("a replacement load shows alongside the model it is replacing", () => {
  const resident: LoadedModelEntry[] = [
    {
      id: "image:unsloth/flux-old",
      kind: "image",
      source: "image",
      name: "unsloth/flux-old",
      detail: "FLUX · BF16 · cuda",
    },
  ];
  const pending = new Map<LoadedModelSource, string | null>([
    ["image", "unsloth/flux-new"],
  ]);

  const rows = withPendingLoads(resident, pending);
  assert.equal(rows.length, 2);
  assert.equal(rows[1].name, "unsloth/flux-new");
  assert.equal(rows[1].loading, true);
});

test("the resident row wins once the replacement has committed", () => {
  const committed: LoadedModelEntry[] = [
    {
      id: "image:unsloth/flux-new",
      kind: "image",
      source: "image",
      name: "unsloth/flux-new",
      detail: "FLUX · BF16 · cuda",
    },
  ];
  const pending = new Map<LoadedModelSource, string | null>([
    ["image", "unsloth/flux-new"],
  ]);
  assert.deepEqual(withPendingLoads(committed, pending), committed);
});

test("a status loading row still suppresses the announcement", () => {
  // Chat and dictation report their own loading rows, and the backend may spell
  // the name differently, so those must not double up.
  const loadingRow: LoadedModelEntry[] = [
    {
      id: "chat:models/qwen3-0.6b.gguf",
      kind: "text",
      source: "chat",
      name: "models/qwen3-0.6b.gguf",
      detail: "Loading",
      loading: true,
    },
  ];
  const pending = new Map<LoadedModelSource, string | null>([
    ["chat", "unsloth/Qwen3-0.6B-GGUF"],
  ]);
  assert.deepEqual(withPendingLoads(loadingRow, pending), loadingRow);
});

test("an unnamed announcement defers to any row for its runtime", () => {
  const resident: LoadedModelEntry[] = [
    {
      id: "video:unsloth/wan",
      kind: "video",
      source: "video",
      name: "unsloth/wan",
      detail: "WAN",
    },
  ];
  const pending = new Map<LoadedModelSource, string | null>([["video", null]]);
  assert.deepEqual(withPendingLoads(resident, pending), resident);
});

// Closing the card means "not now", so the next load reopens it. That only
// works if the announcement is still being RECORDED while the card is closed:
// the indicator clears the dismissal from its own subscription, but the rows
// come from this hook's, and gating that one on `enabled` lost the very event
// that was meant to bring the card back. Chat and dictation would have limped
// on, since the poll synthesises their loading rows; images and video have no
// such fallback, so the card stayed hidden for the whole load.
test("recording is gated on the preference, not on whether the card shows", () => {
  const SOURCE = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/loaded-models/use-loaded-models.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.match(
    SOURCE,
    /track: boolean = enabled/,
    "the hook must take a recording flag distinct from the showing flag",
  );
  const subscribe = SOURCE.slice(
    SOURCE.indexOf("// The load call announces itself"),
    SOURCE.indexOf("}, [track, refresh]);") + 22,
  );
  assert.match(subscribe, /if \(!track\) return;/);
  assert.doesNotMatch(subscribe, /if \(!enabled\) return;/);

  const INDICATOR = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/loaded-models/loaded-models-indicator.tsx",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  // showIndicator is the Settings toggle alone: dismissal and route gating must
  // not stop the recording, or the card cannot reopen for the load.
  assert.match(
    INDICATOR,
    /useLoadedModels\(\s*enabled,\s*showIndicator,\s*\)/,
    "recording follows the preference, not what is on screen",
  );
});
