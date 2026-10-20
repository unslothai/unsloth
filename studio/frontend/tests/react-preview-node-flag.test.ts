// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// The artifacts modules import their siblings without an extension, as Vite allows.
register(
  `data:text/javascript,${encodeURIComponent(
    "export function resolve(s, c, next) { return s.startsWith('.') && c.parentURL?.includes('/features/chat/artifacts/') && !/\\.\\w+$/.test(s) ? next(s + '.ts', c) : next(s, c); }",
  )}`,
);
const { noteNodeAvailability } = await import("../src/features/chat/artifacts/react-preview/node-availability.ts");
const { useChatArtifactsStore } = await import("../src/features/chat/artifacts/store.ts");

const flag = () => useChatArtifactsStore.getState().reactPreviewUnavailable;

test("the Needs Node.js note follows the last compile's answer", () => {
  assert.equal(flag(), false);
  noteNodeAvailability({ status: "unavailable", reason: "node_missing" });
  assert.equal(flag(), true);
  // A timeout or failed request says nothing about Node.
  noteNodeAvailability({ status: "unavailable", reason: "timeout" });
  noteNodeAvailability({ status: "unavailable", reason: "failed" });
  assert.equal(flag(), true);
  // Once a compile gets an answer from Node, the note goes away.
  noteNodeAvailability({ status: "compile-error" });
  assert.equal(flag(), false);
  noteNodeAvailability({ status: "unavailable", reason: "transform_missing" });
  assert.equal(flag(), true);
  noteNodeAvailability({ status: "ready" });
  assert.equal(flag(), false);
});
