// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { installLocalStorageFake } from "./helpers/kit.ts";

const { store, storage } = installLocalStorageFake();
const SUBSCRIBE_RE = /subscribeFitOnDeviceOnlyPreference/;
const SET_RE = /setFitOnDeviceOnlyPreference/;
const MIRROR_RE = /mirrorSettingToBackend/;
const RETIRED_HUB_KEY_RE = /unsloth\.hub\.fitOnDeviceOnly/;
store.set("unsloth_models_fit_on_device_only", "true");

const preference = await import("../src/lib/model-selection-preferences.ts");

test("fit-on-device uses one reactive persisted preference", () => {
  assert.equal(preference.getFitOnDeviceOnlyPreference(), true);
  const observed: boolean[] = [];
  const unsubscribe = preference.subscribeFitOnDeviceOnlyPreference((value) => {
    observed.push(value);
  });

  preference.setFitOnDeviceOnlyPreference(false);
  assert.equal(
    storage.getItem(preference.MODELS_FIT_ON_DEVICE_ONLY_KEY),
    "false",
  );
  assert.deepEqual(observed, [false]);

  unsubscribe();
  preference.setFitOnDeviceOnlyPreference(true);
  assert.deepEqual(observed, [false]);
});

test("a notification runs over the listeners subscribed when it started", () => {
  const observed: string[] = [];
  const lateSubscriptions: (() => void)[] = [];
  const unsubscribeEarly = preference.subscribeFitOnDeviceOnlyPreference(() => {
    observed.push("early");
    if (lateSubscriptions.length > 0) {
      return;
    }
    lateSubscriptions.push(
      preference.subscribeFitOnDeviceOnlyPreference(() => {
        observed.push("late");
      }),
    );
  });

  preference.setFitOnDeviceOnlyPreference(false);
  assert.deepEqual(observed, ["early"]);

  preference.setFitOnDeviceOnlyPreference(true);
  assert.deepEqual(observed, ["early", "early", "late"]);

  unsubscribeEarly();
  for (const unsubscribe of lateSubscriptions) {
    unsubscribe();
  }
});

test("Hub and the picker share the neutral preference while Chat mirrors it", () => {
  const root = fileURLToPath(new URL("../src/", import.meta.url));
  const hub = readFileSync(`${root}features/hub/hub-page.tsx`, "utf8");
  const picker = readFileSync(
    `${root}features/model-picker/components/model-selector/pickers.tsx`,
    "utf8",
  );
  const chatStore = readFileSync(
    `${root}features/chat/stores/chat-runtime-store.ts`,
    "utf8",
  );

  for (const source of [hub, picker]) {
    assert.match(source, SUBSCRIBE_RE);
    assert.match(source, SET_RE);
  }
  assert.match(chatStore, SUBSCRIBE_RE);
  assert.match(chatStore, MIRROR_RE);
  assert.doesNotMatch(hub, RETIRED_HUB_KEY_RE);
});
