// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/export-store-resolver.mjs", import.meta.url);

const { selectExportProgressPercent, isExportPanelActive } = await import(
  "../src/features/export/stores/export-runtime-store.ts"
);

type Percentable = Parameters<typeof selectExportProgressPercent>[0];

/** Only the fields the selectors read; the rest of the store is irrelevant here. */
function state(partial: Record<string, unknown>): Percentable {
  return {
    phase: "idle",
    quantTotal: 1,
    quantIndex: 0,
    isExporting: false,
    ...partial,
  } as unknown as Percentable;
}

test("idle is 0 and loading is a fixed 8", () => {
  assert.equal(selectExportProgressPercent(state({ phase: "idle" })), 0);
  assert.equal(selectExportProgressPercent(state({ phase: "loading" })), 8);
});

test("an active export never reaches 100 - the band stops at 87", () => {
  const band = (quantIndex: number, quantTotal: number) =>
    selectExportProgressPercent(state({ phase: "exporting", quantIndex, quantTotal }));

  assert.equal(band(0, 4), 15);
  assert.equal(band(2, 4), 51);
  assert.equal(band(4, 4), 87);
  // 100% must mean "finished", never "still working".
  for (let i = 0; i <= 8; i++) {
    assert.ok(band(i, 4) < 100, `quantIndex=${i} reached 100 while exporting`);
  }
});

test("quantIndex beyond quantTotal clamps instead of overshooting", () => {
  assert.equal(
    selectExportProgressPercent(state({ phase: "exporting", quantIndex: 99, quantTotal: 4 })),
    87,
  );
  assert.equal(
    selectExportProgressPercent(state({ phase: "exporting", quantIndex: -5, quantTotal: 4 })),
    15,
  );
});

test("a zero or negative quantTotal does not divide by zero", () => {
  // NaN is deliberately not covered: runExport always sets quantTotal to
  // Math.max(1, quantLevels.length) and applyBackendStatus never touches it, so
  // a non-numeric quantTotal is unreachable. Math.max(1, NaN) would be NaN.
  for (const quantTotal of [0, -1]) {
    const value = selectExportProgressPercent(
      state({ phase: "exporting", quantIndex: 1, quantTotal }),
    );
    assert.ok(Number.isFinite(value), `quantTotal=${quantTotal} produced ${value}`);
    assert.ok(value >= 15 && value <= 87, `quantTotal=${quantTotal} produced ${value}`);
  }
});

test("only success reports 100", () => {
  assert.equal(selectExportProgressPercent(state({ phase: "success" })), 100);
  for (const phase of ["idle", "loading", "exporting", "error", "canceled"]) {
    assert.notEqual(
      selectExportProgressPercent(state({ phase, quantIndex: 4, quantTotal: 4 })),
      100,
      `${phase} reported 100`,
    );
  }
});

test("error and canceled freeze near where they stopped instead of snapping to 0", () => {
  for (const phase of ["error", "canceled"]) {
    assert.equal(
      selectExportProgressPercent(state({ phase, quantIndex: 2, quantTotal: 4 })),
      51,
    );
    assert.equal(selectExportProgressPercent(state({ phase, quantIndex: 0, quantTotal: 4 })), 15);
  }
});

test("the panel stays visible for terminal phases, not just active ones", () => {
  assert.equal(isExportPanelActive(state({ phase: "idle", isExporting: false })), false);
  assert.equal(isExportPanelActive(state({ phase: "idle", isExporting: true })), true);
  for (const phase of ["loading", "exporting", "success", "error", "canceled"]) {
    assert.equal(isExportPanelActive(state({ phase, isExporting: false })), true, phase);
  }
});
