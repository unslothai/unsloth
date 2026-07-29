// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { adoptResidentModelStatus } from "../src/features/hub/lib/adopt-inference-status.ts";
import {
  ggufVariantsMatch,
  residentModelIdMatches,
} from "../src/features/hub/lib/model-identity.ts";
import {
  type ResidentStatusRefreshTargets,
  subscribeResidentStatusRefresh,
} from "../src/features/hub/lib/resident-status-refresh.ts";

function fakeTargets(): ResidentStatusRefreshTargets & {
  hidden: boolean;
  fire: (target: "window" | "document", type: string) => void;
  listenerCount: () => number;
} {
  const listeners = new Map<string, Set<EventListenerOrEventListenerObject>>();
  const key = (target: string, type: string) => `${target}:${type}`;
  const make = (target: "window" | "document") => ({
    addEventListener(type: string, fn: EventListenerOrEventListenerObject) {
      const set = listeners.get(key(target, type)) ?? new Set();
      set.add(fn);
      listeners.set(key(target, type), set);
    },
    removeEventListener(type: string, fn: EventListenerOrEventListenerObject) {
      listeners.get(key(target, type))?.delete(fn);
    },
  });
  const visibility = { hidden: false };
  const state = {
    get hidden() {
      return visibility.hidden;
    },
    set hidden(next: boolean) {
      visibility.hidden = next;
    },
    window: make("window"),
    document: {
      ...make("document"),
      get hidden() {
        return visibility.hidden;
      },
    },
    fire(target: "window" | "document", type: string) {
      for (const fn of listeners.get(key(target, type)) ?? []) {
        (fn as EventListener)(new Event(type));
      }
    },
    listenerCount() {
      let total = 0;
      for (const set of listeners.values()) total += set.size;
      return total;
    },
  };
  return state as never;
}

test("coming back to the window re-reads inference status", () => {
  // An OpenAI-compatible request auto-switches the resident model whenever it
  // likes. The Hub's only other status read is its mount effect, so without this
  // the catalog and the settings page keep describing the previous model for as
  // long as the Hub stays mounted.
  const targets = fakeTargets();
  let reads = 0;
  subscribeResidentStatusRefresh(() => {
    reads += 1;
  }, targets);

  assert.equal(reads, 0, "subscribing must not read on its own");
  targets.fire("window", "focus");
  assert.equal(reads, 1);
  targets.fire("document", "visibilitychange");
  assert.equal(reads, 2);
});

test("a tab going hidden does not read", () => {
  // visibilitychange fires on the way out too, and a hidden tab has no settings
  // page to correct.
  const targets = fakeTargets();
  let reads = 0;
  subscribeResidentStatusRefresh(() => {
    reads += 1;
  }, targets);

  targets.hidden = true;
  targets.fire("document", "visibilitychange");
  assert.equal(reads, 0);

  targets.hidden = false;
  targets.fire("document", "visibilitychange");
  assert.equal(reads, 1);
});

test("an auto-switch under a mounted Hub stops hiding the live config", () => {
  // The whole point, end to end: while the Hub is mounted an OpenAI-compatible
  // request swaps the resident model. Without a second read the store still names
  // the old one, so hub-page's settingsTargetIsResident says the newly loaded
  // model is not resident, its settings page is handed loadedConfig=null, and
  // ModelConfigPage seeds the editor from saved/default values -- which Apply then
  // reloads the model with, over what the API actually selected.
  const store = {
    checkpoint: "unsloth/Qwen3-8B-GGUF" as string | null,
    checkpointIsExternal: false,
    activeGgufVariant: "Q4_K_M" as string | null,
    modelLoading: false,
  };
  // What the server reports once the API request has switched it.
  let serverStatus = {
    checkpointId: "unsloth/Llama-3.1-8B-Instruct-GGUF",
    ggufVariant: "Q8_0",
  };
  const readStatusAndAdopt = () => {
    adoptResidentModelStatus(
      serverStatus,
      { ...store },
      {
        setCheckpoint: (checkpointId, ggufVariant) => {
          store.checkpoint = checkpointId;
          store.activeGgufVariant = ggufVariant;
        },
        applyStatus: () => undefined,
      },
    );
  };

  // hub-page.tsx's settingsTargetIsResident, for the model the API just loaded.
  const settingsTargetIsResident = () =>
    residentModelIdMatches(store.checkpoint, serverStatus.checkpointId) &&
    ggufVariantsMatch(store.activeGgufVariant, serverStatus.ggufVariant);

  const targets = fakeTargets();
  subscribeResidentStatusRefresh(readStatusAndAdopt, targets);

  assert.equal(
    settingsTargetIsResident(),
    false,
    "precondition: the mount-time read predates the switch",
  );
  targets.fire("window", "focus");
  assert.equal(settingsTargetIsResident(), true);

  // A load this tab started owns the store until it settles, so a refresh landing
  // mid-switch must not re-pin the model the user is moving away from.
  store.modelLoading = true;
  serverStatus = {
    checkpointId: "unsloth/Qwen3-8B-GGUF",
    ggufVariant: "Q4_K_M",
  };
  targets.fire("window", "focus");
  assert.equal(store.checkpoint, "unsloth/Llama-3.1-8B-Instruct-GGUF");
});

test("unsubscribing stops the reads and leaves no listener behind", () => {
  const targets = fakeTargets();
  let reads = 0;
  const unsubscribe = subscribeResidentStatusRefresh(() => {
    reads += 1;
  }, targets);

  assert.equal(targets.listenerCount(), 2);
  unsubscribe();
  assert.equal(targets.listenerCount(), 0);
  targets.fire("window", "focus");
  targets.fire("document", "visibilitychange");
  assert.equal(reads, 0);
});
