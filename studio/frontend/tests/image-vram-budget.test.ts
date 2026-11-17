// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function setup() {
  Object.assign(globalThis, { window: new EventTarget() });
  const sent: string[] = [];
  let release!: (response: Response) => void;
  const reply = new Promise<Response>((resolve) => {
    release = resolve;
  });
  const stored = {
    fraction: 0.97,
    is_stored: false,
    default_fraction: 0.97,
    min_fraction: 0.1,
    max_fraction: 1,
    reload_required: false,
  };
  const authFetch = async (path: string, init?: RequestInit) => {
    sent.push(`${init?.method ?? "GET"} ${path}`);
    if (path === "/api/settings/vram-budget" && !init?.method) return Response.json(stored);
    if (path === "/api/settings/vram-budget") return reply;
    return Response.json({ loaded: false });
  };
  const budget = loadWithStubs<
    typeof import("../src/features/settings/api/vram-budget")
  >(new URL("../src/features/settings/api/vram-budget.ts", import.meta.url), {
    "@/features/auth": { authFetch },
    "@/lib/format-fastapi-error": {
      readFastApiError: async () => "Budget save failed",
    },
    "./settings-route-absent": {
      SettingsRouteAbsentError: class extends Error {},
    },
  });
  const images = loadWithStubs<typeof import("../src/features/images/api")>(
    new URL("../src/features/images/api.ts", import.meta.url),
    {
      "@/features/settings/api/vram-budget": budget,
      "@/features/auth": { authFetch },
      "@/lib/model-lifecycle-events": {
        withBackgroundLoadNotice: (
          _kind: string,
          _model: string,
          run: () => unknown,
        ) => run(),
      },
      "@/lib/format-fastapi-error": {},
      "@/lib/multi-model-refusal": {
        parseLoadJson: (response: Response) => response.json(),
      },
      "@/lib/unified-memory-load-refusal": {
        withAllowOversizedLoad: (body: unknown) => body,
      },
      "./lib/memory-refusal": {},
    },
  );
  const ok = () =>
    release(
      Response.json({
        fraction: 0.9,
        is_stored: true,
        default_fraction: 0.97,
        min_fraction: 0.1,
        max_fraction: 1,
        reload_required: true,
      }),
    );
  return { budget, images, sent, release, ok };
}

test("an immediate image load waits for the staged budget save", async () => {
  const { budget, images, sent, ok } = setup();
  budget.stageVramBudgetSave(0.9);
  const loading = images.loadDiffusionModel({ model_path: "example/image" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(sent, ["PUT /api/settings/vram-budget"]);
  assert.equal(budget.isVramBudgetLocked(), true);
  ok();
  await loading;
  assert.deepEqual(sent, [
    "PUT /api/settings/vram-budget",
    "POST /api/inference/images/load",
  ]);
  assert.equal(budget.isVramBudgetLocked(), false);
});

test("a rejected budget save prevents image loading and unlocks the slider", async () => {
  const { budget, images, sent, release } = setup();
  budget.stageVramBudgetSave(0.9);
  const loading = images.loadDiffusionModel({ model_path: "example/image" });
  release(Response.json({ detail: "Budget save failed" }, { status: 500 }));
  await assert.rejects(loading, /Budget save failed/);
  assert.deepEqual(sent, ["PUT /api/settings/vram-budget", "GET /api/settings/vram-budget"]);
  assert.equal(budget.isVramBudgetLocked(), false);
});

test("a failed in-flight budget save is not retried by the next image load", async () => {
  const { budget, images, sent, release } = setup();
  const saving = budget.updateVramBudgetSettings(0.9);
  const loading = images.loadDiffusionModel({ model_path: "example/image" });
  release(Response.json({ detail: "Budget save failed" }, { status: 500 }));
  await assert.rejects(saving, /Budget save failed/);
  await assert.rejects(loading, /Budget save failed/);
  assert.equal(budget.flushVramBudgetSave(), null);
  assert.deepEqual(sent, ["PUT /api/settings/vram-budget", "GET /api/settings/vram-budget"]);
});

test("image loading also waits for an in-flight reset", async () => {
  const { budget, images, sent, ok } = setup();
  const resetting = budget.updateVramBudgetSettings(null);
  const loading = images.loadDiffusionModel({ model_path: "example/image" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(sent, ["PUT /api/settings/vram-budget"]);
  ok();
  await Promise.all([resetting, loading]);
  assert.deepEqual(sent, [
    "PUT /api/settings/vram-budget",
    "POST /api/inference/images/load",
  ]);
});

test("a Hub download plan waits for the staged budget save", async () => {
  const { budget, images, sent, ok } = setup();
  budget.stageVramBudgetSave(0.9);
  const planning = images.withSettledVramBudget(() =>
    images.getDiffusionDownloadPlan({ model_path: "example/image" }),
  );
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(sent, ["PUT /api/settings/vram-budget"]);
  ok();
  await planning;
  assert.deepEqual(sent, [
    "PUT /api/settings/vram-budget",
    "POST /api/inference/images/download-plan",
  ]);
});

test("a failed budget save surfaces as VramBudgetSaveError and repaints the stored budget", async () => {
  const { budget, images, sent, release } = setup();
  const published: number[] = [];
  budget.subscribeVramBudgetSettings((next) => published.push(next.fraction));
  budget.stageVramBudgetSave(0.9);
  const planning = images.withSettledVramBudget(() =>
    images.getDiffusionDownloadPlan({ model_path: "example/image" }),
  );
  release(Response.json({ detail: "Budget save failed" }, { status: 500 }));
  await assert.rejects(planning, (error: unknown) => error instanceof images.VramBudgetSaveError);
  assert.deepEqual(sent, ["PUT /api/settings/vram-budget", "GET /api/settings/vram-budget"]);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(published, [0.97]);
});

test("the staged Hub plan settles the budget first", async () => {
  const { readFileSync } = await import("node:fs");
  const read = (path: string) => readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
  assert.match(
    read("features/images/images-page.tsx"),
    /const plan = await withSettledVramBudget\(\(\) =>\s*retiredSnapshot/,
  );
  assert.match(
    read("features/images/images-page.tsx"),
    /if \(error instanceof VramBudgetSaveError\) \{[\s\S]{0,400}?return true;/,
    "a failed budget save must not fall through to the fallback load",
  );
});

test("a dropped retry no longer counts as a staged edit", async () => {
  const { budget, release } = setup();
  const saving = budget.updateVramBudgetSettings(0.9);
  release(Response.json({ detail: "Budget save failed" }, { status: 500 }));
  await assert.rejects(saving, /Budget save failed/);
  assert.equal(budget.hasStagedVramBudgetSave(), true);
  budget.dropVramBudgetRetry();
  assert.equal(budget.hasStagedVramBudgetSave(), false);
});
