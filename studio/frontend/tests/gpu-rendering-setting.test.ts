// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type {
  GpuRenderingStatus,
  notifyRendererReady as NotifyRendererReady,
} from "../src/features/settings/api/gpu-rendering.ts";
import {
  type GpuRenderingNote,
  gpuRenderingNotes,
} from "../src/features/settings/gpu-rendering-notes.ts";
import { en } from "../src/i18n/locales/en.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const API = new URL(
  "../src/features/settings/api/gpu-rendering.ts",
  import.meta.url,
);
const PROVIDER = new URL("../src/app/provider.tsx", import.meta.url);
const APPEARANCE_TAB = new URL(
  "../src/features/settings/tabs/appearance-tab.tsx",
  import.meta.url,
);
const SEGMENTED = new URL(
  "../src/features/settings/components/gpu-rendering-segmented.tsx",
  import.meta.url,
);

const BASE: GpuRenderingStatus = {
  setting: "auto",
  configurable: true,
  restartRequired: false,
  fallbackActive: false,
  envOverride: null,
  oldWebkitWarning: null,
  oldDriverWarning: null,
};

const kinds = (notes: GpuRenderingNote[]) => notes.map((note) => note.kind);

test("GPU rendering notes follow the status", () => {
  const rows: {
    name: string;
    status: Partial<GpuRenderingStatus>;
    selected: GpuRenderingStatus["setting"];
    expected: GpuRenderingNote["kind"][];
  }[] = [
    { name: "plain auto", status: {}, selected: "auto", expected: [] },
    {
      name: "macOS shows only its note",
      status: { configurable: false, setting: "on", restartRequired: true },
      selected: "on",
      expected: ["macNote"],
    },
    {
      name: "env override",
      status: { envOverride: "WEBKIT_DISABLE_DMABUF_RENDERER" },
      selected: "auto",
      expected: ["envOverride"],
    },
    {
      name: "fallback",
      status: { fallbackActive: true },
      selected: "auto",
      expected: ["fallbackNote"],
    },
    {
      name: "old WebKit warning needs On",
      status: { oldWebkitWarning: "2.50.4" },
      selected: "auto",
      expected: [],
    },
    {
      name: "old WebKit warning with Off",
      status: { oldWebkitWarning: "2.50.4" },
      selected: "off",
      expected: [],
    },
    {
      name: "old WebKit warning with On",
      status: { oldWebkitWarning: "2.50.4", setting: "on" },
      selected: "on",
      expected: ["oldWebkitWarning"],
    },
    {
      name: "old NVIDIA driver warning needs On",
      status: { oldDriverWarning: "580" },
      selected: "auto",
      expected: [],
    },
    {
      name: "old NVIDIA driver warning with On",
      status: { oldDriverWarning: "580", setting: "on" },
      selected: "on",
      expected: ["oldDriverWarning"],
    },
    {
      name: "On without a warning",
      status: { setting: "on" },
      selected: "on",
      expected: [],
    },
    {
      name: "restart pending",
      status: { restartRequired: true, setting: "off" },
      selected: "off",
      expected: ["restartPending"],
    },
    {
      name: "everything, in display order",
      status: {
        setting: "on",
        envOverride: "WEBKIT_FORCE_SHM",
        fallbackActive: true,
        oldWebkitWarning: "2.50.4",
        restartRequired: true,
      },
      selected: "on",
      expected: [
        "envOverride",
        "fallbackNote",
        "oldWebkitWarning",
        "restartPending",
      ],
    },
  ];
  for (const row of rows) {
    assert.deepEqual(
      kinds(gpuRenderingNotes({ ...BASE, ...row.status }, row.selected)),
      row.expected,
      row.name,
    );
  }

  const notes = gpuRenderingNotes(
    { ...BASE, envOverride: "WEBKIT_FORCE_SHM", oldWebkitWarning: "2.50.4" },
    "on",
  );
  assert.deepEqual(notes, [
    { kind: "envOverride", name: "WEBKIT_FORCE_SHM" },
    { kind: "oldWebkitWarning", version: "2.50.4" },
  ]);
});

test("the note copy carries the placeholders the row fills", () => {
  const copy = en.settings.appearance.gpuRendering;
  assert.ok(copy.oldWebkitWarning.includes("{version}"));
  assert.ok(copy.oldDriverWarning.includes("{version}"));
  assert.ok(copy.envOverride.includes("{name}"));
});

type ApiModule = {
  loadGpuRendering: () => Promise<GpuRenderingStatus>;
  updateGpuRendering: (value: string) => Promise<GpuRenderingStatus>;
  relaunchToApplyGpuRendering: () => Promise<boolean>;
  notifyRendererReady: typeof NotifyRendererReady;
};

function loadApi(
  isTauri: boolean,
  invoke: (command: string, args?: unknown) => Promise<unknown>,
): ApiModule {
  return loadWithStubs<ApiModule>(API, {
    "@/lib/api-base": { isTauri },
    "@tauri-apps/api/core": { invoke },
  });
}

async function withAnimationFrames<T>(
  run: (frames: () => number) => Promise<T>,
): Promise<T> {
  const g = globalThis as { window?: unknown };
  const previous = g.window;
  let frames = 0;
  g.window = {
    requestAnimationFrame: (callback: () => void) => {
      frames += 1;
      setTimeout(callback, 0);
      return frames;
    },
  };
  try {
    return await run(() => frames);
  } finally {
    g.window = previous;
  }
}

test("the api calls the backend commands by name", async () => {
  const calls: [string, unknown][] = [];
  const api = loadApi(true, async (command, args) => {
    calls.push([command, args]);
    if (command === "relaunch_to_apply_gpu_rendering") return true;
    return BASE;
  });

  assert.deepEqual(await api.loadGpuRendering(), BASE);
  await api.updateGpuRendering("off");
  assert.equal(await api.relaunchToApplyGpuRendering(), true);
  await withAnimationFrames(async (frames) => {
    await api.notifyRendererReady();
    assert.equal(frames(), 2, "waits two frames before reporting ready");
  });

  assert.deepEqual(calls, [
    ["get_gpu_rendering", undefined],
    ["set_gpu_rendering", { value: "off" }],
    ["relaunch_to_apply_gpu_rendering", undefined],
    ["renderer_ready", undefined],
  ]);
});

test("renderer ready is a no-op outside Tauri and never throws", async () => {
  const calls: string[] = [];
  const browser = loadApi(false, async (command) => {
    calls.push(command);
  });
  await withAnimationFrames(async (frames) => {
    await browser.notifyRendererReady();
    assert.equal(frames(), 0);
  });
  assert.deepEqual(calls, []);

  const failing = loadApi(true, async () => {
    throw new Error("unknown command renderer_ready");
  });
  await withAnimationFrames(() => failing.notifyRendererReady());
});

test("the provider reports ready after every window show", () => {
  const source = readFileSync(PROVIDER, "utf8");
  const shows = source.match(/await win\.show\(\);/g) ?? [];
  const reported =
    source.match(/await win\.show\(\);\n\s*void notifyRendererReady\(\);/g) ??
    [];
  assert.equal(shows.length, 3);
  assert.equal(reported.length, shows.length);
});

test("the Rendering section renders on desktop only", () => {
  const source = readFileSync(APPEARANCE_TAB, "utf8");
  assert.match(
    source,
    /\{isTauri \? \(\s*<SettingsSection title=\{t\("settings\.appearance\.gpuRendering\.sectionTitle"\)\}>\s*<GpuRenderingRow \/>/,
  );
  // Between Preferences and the sidebar nav section.
  const preferences = source.indexOf(
    "settings.appearance.custom.preferencesTitle",
  );
  const rendering = source.indexOf(
    "settings.appearance.gpuRendering.sectionTitle",
  );
  const sidebarNav = source.indexOf("settings.appearance.sidebarNav.title");
  assert.ok(preferences < rendering && rendering < sidebarNav);
  assert.match(source, /enabled: isTauri/);
});

test("the segmented control offers the three options and can be disabled", () => {
  const source = readFileSync(SEGMENTED, "utf8");
  for (const value of ["auto", "on", "off"]) {
    assert.ok(source.includes(`value: "${value}"`), value);
  }
  assert.ok(source.includes('layoutId="gpu-rendering-pill"'));
  assert.ok(source.includes("aria-pressed={active}"));
  assert.ok(source.includes("disabled={disabled}"));
});
