// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { fireWindowEvent } = installLocalStorageFake();

const { bumpAuthSessionEpoch, setAuthFetchHandler } =
  await import("./helpers/store-stubs/auth.ts");
const { useRlWorkspaceStore } =
  await import("../src/features/training/stores/rl-workspace-store.ts");

test("signing out clears the reward library and the dataset sample", () => {
  useRlWorkspaceStore.setState({
    previewRow: { prompt: "account A's row" },
    library: [{ name: "a-private-reward" } as never],
    libraryError: "x",
  });
  const delivered = fireWindowEvent(
    "unsloth:auth-session-cleared",
    new Event("unsloth:auth-session-cleared"),
  );
  assert.ok(delivered >= 1);
  const s = useRlWorkspaceStore.getState();
  assert.equal(s.previewRow, null);
  assert.deepEqual(s.library, []);
  assert.equal(s.libraryError, null);
});

test("a reward list that arrives after a sign-out is dropped", async () => {
  useRlWorkspaceStore.setState({ library: [], libraryError: null });
  let release: () => void = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  setAuthFetchHandler(async () => {
    await held;
    return Response.json([{ name: "previous-account-reward" }]);
  });
  const pending = useRlWorkspaceStore.getState().refreshLibrary();
  bumpAuthSessionEpoch();
  release();
  await pending;
  setAuthFetchHandler(null);
  assert.deepEqual(useRlWorkspaceStore.getState().library, []);
});
