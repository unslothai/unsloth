// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pending rows are only retired by a lifecycle event, which is not heard while disabled.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type LoadedModelEntry,
  type LoadedModelSource,
  withPendingLoads,
} from "../src/features/loaded-models/loaded-models-sources.ts";

import { readSrc } from "./helpers/kit.ts";

const SOURCE = readSrc("features/loaded-models/use-loaded-models.ts");

test("a stale pending entry outlives every poll, so it must not survive a disable", () => {
  const pending = new Map<LoadedModelSource, string | null>([
    ["image", "unsloth/flux"],
  ]);
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
  // Render-time setState must be guarded by a change, or it re-renders forever.
  assert.match(
    SOURCE,
    /const \[wasTracking, setWasTracking\] = useState\(track\);\s*if \(wasTracking !== track\) \{\s*setWasTracking\(track\);/,
    "the adjustment must be gated on the previous tracking value",
  );
});

test("pending rows are cleared when recording stops, not when it starts", () => {
  // Clearing on re-enable would race a load announced from another tab.
  const guard = SOURCE.slice(
    SOURCE.indexOf("if (wasTracking !== track)"),
    SOURCE.indexOf("// The load call announces itself"),
  );
  assert.doesNotMatch(guard, /if \(track\) setPending/);
  assert.match(guard, /!track &&/);
});

// /images/status reports the old model until the replacement commits.
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
  // Chat and dictation report their own loading rows.
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

// Recording must continue while the card is closed, or images/video loads cannot reopen it.
test("recording is gated on the preference, not on whether the card shows", () => {
  const SOURCE = readSrc("features/loaded-models/use-loaded-models.ts");
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

  const INDICATOR = readSrc(
    "features/loaded-models/loaded-models-indicator.tsx",
  );
  // Reachability carries the auth gate; dismissal must not stop recording.
  assert.match(
    INDICATOR,
    /useLoadedModels\(\s*enabled,\s*showIndicator && reachable,\s*\)/,
    "recording survives a dismissal but not a route with no session",
  );
});
