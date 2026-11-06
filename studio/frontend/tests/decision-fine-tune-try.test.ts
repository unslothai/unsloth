// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { afterEach } from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();
Object.assign(window, { dispatchEvent: () => true });
Object.assign(globalThis, { HTMLElement: class {} });

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { tryFineTuneInDecisionApi } = await import(
  "../src/features/settings/api/systemone.ts"
);
const { useSettingsDialogStore } = await import(
  "../src/features/settings/stores/settings-dialog-store.ts"
);

function backendStoring(model: string) {
  const sent: { model?: string; enabled?: boolean }[] = [];
  setAuthFetchHandler((_input, init) => {
    sent.push(JSON.parse(String(init?.body)));
    return Response.json({ enabled: true, model, models: [] });
  });
  return sent;
}

afterEach(() => {
  setAuthFetchHandler(null);
  useSettingsDialogStore.setState({ decisionTryRequested: null, open: false });
});

test("a Clef run opens Try under the name the backend stored", async () => {
  const sent = backendStoring("clef-ft:clef_run_1");
  await tryFineTuneInDecisionApi("clef_run_1");
  assert.equal(sent[0].model, "laya-ft:clef_run_1");
  assert.equal(sent[0].enabled, true);
  const state = useSettingsDialogStore.getState();
  assert.equal(state.decisionTryRequested, "clef-ft:clef_run_1");
  assert.equal(state.open, true);
});

test("a Laya run still opens Try under its own name", async () => {
  backendStoring("laya-ft:laya_run_1");
  await tryFineTuneInDecisionApi("laya_run_1");
  assert.equal(
    useSettingsDialogStore.getState().decisionTryRequested,
    "laya-ft:laya_run_1",
  );
});

test("Try stays closed when the backend refuses the model", async () => {
  setAuthFetchHandler(() =>
    Response.json({ detail: "Unknown Decision API model" }, { status: 400 }),
  );
  await assert.rejects(tryFineTuneInDecisionApi("gone"));
  assert.equal(useSettingsDialogStore.getState().decisionTryRequested, null);
});
