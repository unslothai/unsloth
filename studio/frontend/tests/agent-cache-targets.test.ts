// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { agentCacheLoadIds } from "../src/features/settings/lib/agent-cache-targets.ts";

test("Agents preserves an active snapshot target when the repo ref cannot load it", () => {
  assert.deepEqual(
    agentCacheLoadIds([
      {
        repo_id: "Org/Model",
        load_id: "/active/snapshots/verified",
        active_cache: true,
      },
      { repo_id: "Org/Model", load_id: "/old/rev", active_cache: false },
    ]),
    { "org/model": "/active/snapshots/verified" },
  );
});

test("Agents chooses a complete copy before an active partial download", () => {
  const active = {
    repo_id: "Org/Model",
    load_id: "Org/Model",
    active_cache: true,
    partial: true,
  };
  const complete = {
    repo_id: "Org/Model",
    load_id: "/old/complete",
    active_cache: false,
    partial: false,
  };
  for (const copies of [
    [active, complete],
    [complete, active],
  ]) {
    assert.deepEqual(agentCacheLoadIds(copies), {
      "org/model": "/old/complete",
    });
  }
});

test("Agents uses the active cache regardless of discovery order or repo casing", () => {
  const active = {
    repo_id: "Org/Model",
    load_id: "Org/Model",
    active_cache: true,
  };
  const inactive = {
    repo_id: "org/model",
    load_id: "/old/rev",
    active_cache: false,
  };
  for (const copies of [
    [active, inactive],
    [inactive, active],
  ]) {
    assert.deepEqual(agentCacheLoadIds(copies), {});
  }
});

test("inactive-only Agents models choose a deterministic snapshot", () => {
  const copies = ["/z/rev", "/a/rev"].map((load_id) => ({
    repo_id: "Org/Model",
    load_id,
    active_cache: false,
  }));
  assert.deepEqual(agentCacheLoadIds(copies), { "org/model": "/a/rev" });
  assert.deepEqual(agentCacheLoadIds([...copies].reverse()), {
    "org/model": "/a/rev",
  });
});
