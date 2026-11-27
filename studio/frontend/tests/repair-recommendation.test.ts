// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { repairRecommendationText } = await import(
  "../src/features/settings/components/repair-recommendation.ts"
);
const { en } = await import("../src/i18n/locales/en.ts");

const strings = en.settings.general.repairInstall as Record<string, string>;
const t = (key: never, values: Record<string, string> = {}) => {
  const leaf = String(key).split(".").pop() as string;
  return strings[leaf].replace(/\{(\w+)\}/g, (_m, name: string) => values[name] ?? `{${name}}`);
};

test("an old torch names both torch versions and no xFormers upgrade", () => {
  const text = repairRecommendationText(
    { reason: "torch_older_than_managed", torch: "2.10.0+cu130", managed_torch: "2.13" },
    t,
  );
  assert.ok(text != null && text.startsWith("Repair recommended:"), text ?? "null");
  assert.match(text, /PyTorch 2\.10\.0\+cu130, older than the managed 2\.13/);
  // On sm_120 the managed xFormers 0.0.35 is not an upgrade over 0.0.34, so no wording offers it.
  assert.doesNotMatch(text, /xFormers|0\.0\.35/);
  assert.doesNotMatch(text, /\{/);
});

test("the old xFormers reason no longer renders, and no string names an xFormers version", () => {
  assert.equal(
    repairRecommendationText({ reason: "xformers_older_than_managed", torch: "2.13.0+cu130" }, t),
    null,
  );
  for (const key of ["recommendedTorch", "recommendedDriver"]) {
    assert.doesNotMatch(strings[key], /xFormers/);
  }
  assert.equal(strings.recommendedXformers, undefined);
});

test("a driver too old for the managed route asks for a driver update first", () => {
  const text = repairRecommendationText(
    { reason: "driver_older_than_managed", torch: "2.10.0+cu128", managed_torch: "2.13" },
    t,
  );
  assert.match(text ?? "", /Update the GPU driver first, then repair/);
  assert.doesNotMatch(text ?? "", /\{/);
});

test("nothing recommended, or an unknown reason, shows nothing", () => {
  assert.equal(repairRecommendationText(null, t), null);
  assert.equal(repairRecommendationText(undefined, t), null);
  assert.equal(repairRecommendationText({ reason: "something_new" }, t), null);
});

test("the Repair row renders the hint below it", async () => {
  const src = await readSrcAsync("features/settings/components/desktop-repair-control.tsx");
  assert.match(src, /repairRecommendationText\(system\.repair_recommendation, t\)/);
  assert.match(src, /below=\{/);
  // The hook runs before the early return, so hook order never changes.
  assert.ok(src.indexOf("useSystemInfo(") < src.indexOf("if (!repair || repair.isExternalServer) return null;"));
});
