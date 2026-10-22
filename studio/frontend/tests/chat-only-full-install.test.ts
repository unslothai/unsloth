// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A Desktop install set up as Chat only (GGUF) becomes the full install through the manual repair,
// which reruns the installer with PyTorch: training, export, video, safetensors models and the
// image generation features beyond GGUF need it. The only pointer used to be a tooltip on a Train row that
// could not be clicked, aimed at a Settings row filed under Danger zone. Train now offers the
// upgrade directly, and Settings has an Install type row, both only where the repair can do it.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { canUpgradeToFullInstall } = await import("../src/hooks/tauri-repair-context.ts");
const { SETTINGS_SEARCH_KEYWORDS, createSettingsSearchIndex, renderedSearchEntries } =
  await import("../src/features/settings/settings-search.ts");
const { en } = await import("../src/i18n/locales/en.ts");

const managed = { repairInstall: () => Promise.resolve(), isExternalServer: false };
const external = { repairInstall: () => Promise.resolve(), isExternalServer: true };

test("only a measured no_torch verdict with a managed repair can be upgraded", () => {
  assert.equal(canUpgradeToFullInstall("no_torch", true, managed), true);
  // Before /api/health answers the verdict is a guess, and the row spins instead.
  assert.equal(canUpgradeToFullInstall("no_torch", false, managed), false);
  // Web build, startup screen, or a backend started in a terminal: no repair to run.
  assert.equal(canUpgradeToFullInstall("no_torch", true, null), false);
  assert.equal(canUpgradeToFullInstall("no_torch", true, external), false);
  // None of these is a Chat only install, so they keep their own hints (the PyTorch build
  // mismatches already point at Repair installation).
  for (const reason of [
    "no_gpu",
    "intel_mac",
    "rosetta",
    "mlx_unavailable",
    "torch_cpu_build",
    "torch_cuda_unavailable",
    "detection_failed",
    null,
  ]) {
    assert.equal(canUpgradeToFullInstall(reason, true, managed), false, String(reason));
  }
});

test("Settings search finds the row by what a Chat only user would type", () => {
  const index = createSettingsSearchIndex({ desktop: true, closeToTray: true, menuBarIcon: false });
  const entry = "settings.general.fullInstall.label";
  assert.ok(index.general.includes(entry));
  assert.ok(renderedSearchEntries(index, "general", "huggingface", true, true).includes(entry));
  // The row renders nothing on a full install, so search must not scroll to it there.
  assert.ok(!renderedSearchEntries(index, "general", "huggingface", true, false).includes(entry));
  assert.ok(
    !createSettingsSearchIndex({ desktop: false, closeToTray: false, menuBarIcon: false }).general.includes(entry),
  );

  const keywordsKey = SETTINGS_SEARCH_KEYWORDS[entry];
  assert.equal(keywordsKey, "settings.general.fullInstall.keywords");
  const keywords = en.settings.general.fullInstall.keywords;
  for (const query of ["full install", "upgrade", "add training", "train", "pytorch", "chat only", "export", "video"]) {
    assert.ok(keywords.includes(query), query);
  }
});

test("the Chat only confirm copy states the download and does not claim a reinstall", () => {
  const full = en.settings.general.fullInstall;
  assert.match(full.confirmDescription, /gigabytes/);
  assert.doesNotMatch(full.confirmDescription, /reinstall/i);
  // The upgrade is more than Train: naming only training undersold it.
  for (const unlocked of ["training", "export", "video", "safetensors", "image generation"]) {
    assert.match(full.confirmDescription, new RegExp(unlocked));
  }
  const repair = en.settings.general.repairInstall;
  assert.match(repair.confirmDescriptionChatOnly, /upgrades it to the full install with PyTorch/);
  assert.doesNotMatch(repair.confirmDescriptionChatOnly, /reinstalls/);
});

test("the Train row opens the full install dialog instead of greying out", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");
  const train = /\n    train: \{([\s\S]*?)\n    \},\n/.exec(source);
  assert.ok(train, "could not find the train row");
  const body = train[1];
  assert.match(body, /disabled: chatOnlyMeasured && !fullInstallUpgrade,/);
  assert.match(
    body,
    /if \(fullInstallUpgrade\) \{[\s\S]*?setFullInstallOpen\(true\);[\s\S]*?return;\s*\}\s*if \(chatOnlyMeasured\) return;/,
    "the upgrade branch must come before the chat-only early return",
  );
  assert.match(
    source,
    /\{fullInstallUpgrade && \(\s*<FullInstallDialog[\s\S]*?onConfirm=\{\(\) => void fullInstallUpgrade\.repairInstall\(\)\}/,
    "the dialog runs the same manual repair as Settings",
  );
});
