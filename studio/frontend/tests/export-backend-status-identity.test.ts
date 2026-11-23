// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/export-store-resolver.mjs", import.meta.url);

const { useExportRuntimeStore } = await import(
  "../src/features/export/stores/export-runtime-store.ts"
);

type Status = Parameters<
  ReturnType<typeof useExportRuntimeStore.getState>["applyBackendStatus"]
>[0];

const idle = { is_export_active: false } as Status;

test("an unchanged idle status poll does not notify store listeners", () => {
  useExportRuntimeStore.getState().reset();
  useExportRuntimeStore.getState().applyBackendStatus(idle);
  const hydrated = useExportRuntimeStore.getState();
  assert.equal(hydrated.hasHydrated, true);

  let notified = 0;
  const unsubscribe = useExportRuntimeStore.subscribe(() => {
    notified += 1;
  });
  try {
    useExportRuntimeStore.getState().applyBackendStatus(idle);
    useExportRuntimeStore.getState().applyBackendStatus({ ...idle });
    assert.equal(notified, 0);
    assert.equal(useExportRuntimeStore.getState(), hydrated);

    useExportRuntimeStore
      .getState()
      .applyBackendStatus({ is_export_active: true } as Status);
    assert.equal(notified, 1, "a real change still publishes");
    assert.equal(useExportRuntimeStore.getState().backendActive, true);
  } finally {
    unsubscribe();
    useExportRuntimeStore.getState().reset();
  }
});
