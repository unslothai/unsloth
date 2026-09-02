// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// mlock_applicable is what lets the panel say WHY the lock is off. Absent from
// an older backend it must read true, or a frontend ahead of its server would
// tell every user their model has nothing to pin (issue #9549).

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
installLocalStorageFake();

type Listener = (event: Event) => void;
const listeners = new Map<string, Set<Listener>>();
Object.assign(globalThis.window as object, {
  addEventListener: (type: string, fn: Listener) => {
    if (!listeners.has(type)) listeners.set(type, new Set());
    listeners.get(type)?.add(fn);
  },
  removeEventListener: (type: string, fn: Listener) => {
    listeners.get(type)?.delete(fn);
  },
  dispatchEvent: (event: Event) => {
    for (const fn of listeners.get(event.type) ?? []) fn(event);
    return true;
  },
});

const BASE = {
  keep_resident: true,
  no_ram_reserve: false,
  default_keep_resident: false,
  default_no_ram_reserve: false,
  mlock_active: false,
  reload_required: false,
  memlock_limit_bytes: null as number | null,
};

let nextBody: Record<string, unknown> = { ...BASE };

globalThis.fetch = (async () =>
  new Response(JSON.stringify(nextBody), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })) as typeof fetch;

const { loadModelMemorySettings } = await import(
  "../src/features/settings/api/model-memory.ts"
);

test("a discrete full offload maps through as not applicable", async () => {
  nextBody = { ...BASE, mlock_applicable: false };
  const settings = await loadModelMemorySettings({ force: true });
  assert.equal(settings.mlockApplicable, false);
  // The three the UI had before, none of which distinguishes this case.
  assert.equal(settings.keepResident, true);
  assert.equal(settings.mlockActive, false);
  assert.equal(settings.reloadRequired, false);
});

test("a host-resident launch maps through as applicable", async () => {
  nextBody = { ...BASE, mlock_applicable: true, mlock_active: true };
  const settings = await loadModelMemorySettings({ force: true });
  assert.equal(settings.mlockApplicable, true);
});

test("a backend that does not send the field is treated as applicable", async () => {
  nextBody = { ...BASE };
  const settings = await loadModelMemorySettings({ force: true });
  assert.equal(settings.mlockApplicable, true);
});

test("a non-governed runner does not inherit the GPU-offload reason", async () => {
  nextBody = {
    ...BASE,
    mlock_applicable: false,
    mlock_skip_reason: "ungoverned",
  };
  const settings = await loadModelMemorySettings({ force: true });
  assert.equal(settings.mlockApplicable, false);
  assert.equal(settings.mlockSkipReason, "ungoverned");
});
