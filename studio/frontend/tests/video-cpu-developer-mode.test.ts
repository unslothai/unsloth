// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Developer mode's CPU video opt-in (unslothai/unsloth#9458): off by default, enabling asks first
// with the issue's warning and the host RAM against the minimum, and while it is in effect the
// Video page keeps a "CPU (experimental)" warning up and the sidebar row stays navigable.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { videoCpuRamCheck, videoCpuToggleNeedsConfirm } = await import(
  "../src/features/settings/lib/video-cpu-warning.ts"
);
const { videoNavHint } = await import("../src/config/hardware-verdict.ts");

const GIB = 1024 ** 3;
const MIN = 32 * GIB;

test("RAM below the minimum warns, at or near it does not", () => {
  assert.deepEqual(videoCpuRamCheck(16 * GIB, MIN), {
    status: "below",
    hostGib: 16,
    minGib: 32,
  });
  assert.equal(videoCpuRamCheck(64 * GIB, MIN).status, "ok");
  // A "32 GB" machine reports ~31.3 GiB after reservations; it must not be told it is short.
  assert.equal(videoCpuRamCheck(31.3 * GIB, MIN).status, "ok");
  assert.equal(videoCpuRamCheck(29 * GIB, MIN).status, "below");
});

test("an unreadable RAM figure says nothing either way", () => {
  for (const host of [null, undefined, 0, -1, Number.NaN]) {
    const check = videoCpuRamCheck(host as number | null, MIN);
    assert.equal(check.status, "unknown");
    assert.equal(check.hostGib, null);
    assert.equal(check.minGib, 32);
  }
});

test("only turning the opt-in on asks for confirmation", () => {
  assert.equal(videoCpuToggleNeedsConfirm(true), true);
  assert.equal(videoCpuToggleNeedsConfirm(false), false);
});

test("the sidebar waives only the plain no-GPU hint, and only when opted in", () => {
  assert.equal(typeof videoNavHint(true, "no_gpu"), "string");
  assert.equal(typeof videoNavHint(true, "no_gpu", false), "string");
  assert.equal(videoNavHint(true, "no_gpu", true), undefined);
  // A GPU torch cannot reach, or an Intel Mac, stays disabled whatever the opt-in says.
  for (const reason of ["torch_cpu_build", "torch_cuda_unavailable", "intel_mac"]) {
    assert.equal(typeof videoNavHint(true, reason, true), "string", reason);
  }
});

test("the confirm dialog carries the issue's warning text verbatim", () => {
  const en = readSrc("i18n/locales/en.ts");
  assert.ok(
    en.includes(
      "CPU video generation may take hours, consume large amounts of RAM, and may freeze your system. Continue at your own risk.",
    ),
  );
  const section = readSrc("features/settings/components/developer-mode-section.tsx");
  assert.match(section, /settings\.general\.developerMode\.confirmWarning/);
  assert.match(section, /settings\.general\.developerMode\.ramBelowMinimum/);
  assert.match(section, /settings\.general\.developerMode\.performanceEstimate/);
  // Enabling goes through the dialog; only the confirm button saves true.
  assert.match(section, /videoCpuToggleNeedsConfirm\(next\)[\s\S]*setConfirmOpen\(true\)/);
  assert.match(section, /setConfirmOpen\(false\);\s*void save\(true\);/);
  // Saving refreshes both cached gates.
  assert.match(section, /usePlatformStore\.setState\(\{ videoCpuOptIn: next\.videoCpu \}\)/);
  assert.match(section, /refreshHardwareInfo\(\)/);
  // An env override decides, so the switch is not offered against it.
  assert.match(section, /disabled=\{!settings \|\| saving \|\| envForced !== null\}/);
});

test("the section is rendered for the owner only, on the System tab", () => {
  const tab = readSrc("features/settings/tabs/resources-tab.tsx");
  assert.match(tab, /\{isOwner \? <DeveloperModeSection \/> : null\}/);
  assert.doesNotMatch(readSrc("features/settings/tabs/general-tab.tsx"), /DeveloperModeSection/);
});

test("the Video page keeps a CPU (experimental) warning up while the opt-in is in effect", () => {
  const hook = readSrc("hooks/use-hardware-info.ts");
  assert.match(hook, /videoCpuExperimental: data\?\.video_cpu_experimental === true/);
  assert.match(hook, /videoCpuExperimental: false,/);
  const page = readSrc("features/video/video-page.tsx");
  assert.match(page, /cpuExperimental=\{hardware\.videoCpuExperimental\}/);
  assert.match(page, /\{cpuExperimental \? <VideoCpuExperimentalBanner \/> : null\}/);
  assert.match(page, /CPU \(experimental\)\./);
  // The gate itself is unchanged: only an explicit false hides the generator.
  assert.match(page, /hardware\.videoSupported === false/);
});

test("the health verdict carries the opt-in into the platform store", () => {
  const env = readSrc("config/env.ts");
  assert.match(env, /video_cpu_opt_in\?: boolean;/);
  assert.match(env, /videoCpuOptIn: false,/);
  assert.match(env, /previous\.videoCpuOptIn/);
  for (const file of ["components/app-sidebar.tsx", "app/routes/__root.tsx"]) {
    assert.match(
      readSrc(file),
      /videoNavHint\(chatOnlyMeasured, chatOnlyReason, videoCpuOptIn\)/,
      file,
    );
  }
});
