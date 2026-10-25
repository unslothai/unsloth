// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > General's Install type row upgrades a Desktop Chat only install through the manual
// repair; it used to be reachable only through a Train tooltip aimed at Danger zone.

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

const MAC_HINT = "Training on this Mac needs MLX, which requires macOS 14 or later.";

test("only a measured no_torch verdict with a managed repair can be upgraded", () => {
  assert.equal(canUpgradeToFullInstall("no_torch", null, true, managed), true);
  assert.equal(canUpgradeToFullInstall("no_torch", null, false, managed), false);
  assert.equal(canUpgradeToFullInstall("no_torch", null, true, null), false);
  assert.equal(canUpgradeToFullInstall("no_torch", null, true, external), false);
  // A Mac the MLX wheels skip.
  assert.equal(canUpgradeToFullInstall("no_torch", MAC_HINT, true, managed), false);
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
    assert.equal(canUpgradeToFullInstall(reason, null, true, managed), false, String(reason));
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
  for (const unlocked of ["training", "export", "video", "safetensors", "image generation"]) {
    assert.match(full.confirmDescription, new RegExp(unlocked));
  }
  const repair = en.settings.general.repairInstall;
  assert.match(repair.confirmDescriptionChatOnly, /upgrades it to the full install with PyTorch/);
  assert.doesNotMatch(repair.confirmDescriptionChatOnly, /reinstalls/);
});

test("the greyed Train row points at the Install type row, not at Danger zone", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");
  const hint = /const trainDisabledHint: string \| undefined = ([\s\S]*?\n\s*: undefined);/.exec(source);
  assert.ok(hint, "could not find trainDisabledHint");
  const forNoTorch = (canRepair: boolean, chatOnlyHint: string | null = null) =>
    new Function(
      "chatOnlyMeasured",
      "chatOnlyReason",
      "chatOnlyDetail",
      "chatOnlyHint",
      "canRepair",
      `return ${hint[1]};`,
    )(true, "no_torch", null, chatOnlyHint, canRepair) as string;
  assert.match(forNoTorch(true), /Use Upgrade to full install in Settings/);
  assert.doesNotMatch(forNoTorch(true), /Repair installation/);
  assert.match(forNoTorch(false), /--no-torch/);
  assert.equal(forNoTorch(true, MAC_HINT), MAC_HINT);
  assert.equal(forNoTorch(false, MAC_HINT), MAC_HINT);
});

test("the greyed Train row opens the full install dialog when the upgrade is offered", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");
  const row = /train: \{[\s\S]*?\n    \},\n/.exec(source);
  assert.ok(row, "could not find the train nav row");
  // Clickable (not natively disabled) only while the upgrade is offered, and greyed all the same.
  assert.match(row[0], /disabled: chatOnlyMeasured && !fullInstall,/);
  assert.match(row[0], /dimmed: chatOnlyMeasured && fullInstall !== null,/);
  assert.match(row[0], /setFullInstallOpen\(true\)/);
  assert.match(row[0], /5 GB/);
  // Never navigates or preloads Train on a Chat only install.
  assert.match(row[0], /if \(chatOnlyMeasured\) \{[\s\S]*?return;\n        \}\n        navigateFromRow/);
  assert.match(row[0], /onIntent: \(\) => \{\n        if \(chatOnlyMeasured\) return;/);
  assert.match(source, /<FullInstallDialog\s+open=\{fullInstallOpen\}/);
  assert.match(en.settings.general.fullInstall.confirmDescription, /at least 5 GB of disk space/);
});

test("the Train hint and the Export banner both carry an Install Training button", async () => {
  const sidebar = await readSrcAsync("components/app-sidebar.tsx");
  const row = /train: \{[\s\S]*?\n    \},\n/.exec(sidebar);
  assert.ok(row, "could not find the train nav row");
  assert.match(
    row[0],
    /tooltipAction: fullInstall\s*\? \{ label: "Install Training", onClick: \(\) => setFullInstallOpen\(true\) \}\s*: undefined,/,
  );
  // The inline row renders it, and a still-spinning row never offers it.
  assert.match(sidebar, /tooltipAction=\{rowState\.pending \? undefined : row\.tooltipAction\}/);
  assert.match(sidebar, /<Button size="xs" onClick=\{tooltipAction\.onClick\}>/);
  // The More flyout renders no button, so its title keeps the click guidance.
  assert.match(sidebar, /row\.tooltipAction && !rowState\.pending\s*\? `\$\{rowState\.tooltip\} Click to install it\.`/);
  // The nav provider disables hoverable tooltips; a tooltip with a button must opt back in.
  assert.match(sidebar, /interactiveTooltip=\{Boolean\(tooltip && tooltipAction\)\}/);

  const exportPage = await readSrcAsync("features/export/export-page.tsx");
  const banner = /<AlertTitle>Export unavailable<\/AlertTitle>[\s\S]*?<\/Alert>/.exec(exportPage);
  assert.ok(banner, "could not find the Export unavailable banner");
  assert.match(banner[0], /\{fullInstall && \([\s\S]*?setFullInstallOpen\(true\)[\s\S]*?Install Training/);
  assert.match(exportPage, /<FullInstallDialog\s+open=\{fullInstallOpen\}/);
});
