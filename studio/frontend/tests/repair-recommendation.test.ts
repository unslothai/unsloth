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

test("an old torch names both versions and the managed pair", () => {
  const text = repairRecommendationText(
    {
      reason: "torch_older_than_managed",
      torch: "2.10.0+cu128",
      xformers: "0.0.33",
      managed_torch: "2.13",
      managed_xformers: "0.0.35",
    },
    t,
  );
  assert.ok(text?.startsWith("Repair recommended:"), text ?? "null");
  assert.match(text, /PyTorch 2\.10\.0\+cu128/);
  assert.match(text, /xFormers 0\.0\.35/);
  assert.doesNotMatch(text, /\{/);
});

test("an old xFormers on a current torch gets its own wording", () => {
  const text = repairRecommendationText(
    { reason: "xformers_older_than_managed", torch: "2.13.0+cu130", xformers: "0.0.33", managed_xformers: "0.0.35" },
    t,
  );
  assert.match(text ?? "", /xFormers 0\.0\.33, older than the managed 0\.0\.35/);
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
