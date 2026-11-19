// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  canUpgradeForVideo,
  canUpgradeToFullInstall,
} from "../src/hooks/tauri-repair-context.ts";
import { readSrc } from "./helpers/kit.ts";

const VIDEO_PAGE = readSrc("features/video/video-page.tsx");
const HOOK = readSrc("hooks/use-full-install-upgrade.ts");

test("a Chat only desktop install is offered the full install on the Video page", () => {
  assert.match(
    VIDEO_PAGE,
    /useFullInstallBringsVideo\(\)/,
    "the Video page does not offer the full install",
  );
  assert.match(
    VIDEO_PAGE,
    /fullInstall !== null &&\s+hardware\.videoUnsupportedReason === "pytorch_not_installed"/,
    "the offer is not limited to a host whose only blocker is the missing PyTorch",
  );
  assert.match(
    VIDEO_PAGE,
    /<FullInstallDialog[\s\S]*?onConfirm=\{\(\) => void fullInstall\.repairInstall\(\)\}/,
    "the Video page does not reuse the full install dialog and repair",
  );
  // The full install is the route that records PyTorch; an in-place torch install would leave it Chat only.
  assert.ok(
    !/pytorch-install/.test(VIDEO_PAGE),
    "the Video page installs PyTorch in place again",
  );
});

const repair = { repairInstall: async () => {}, isExternalServer: false };
const MLX_HINT = "Training on this Mac needs MLX, which requires macOS 14 or later.";

test("the full install is not offered for video on a GPU-less host", () => {
  // It trains on the CPU there and brings no video device.
  assert.equal(
    canUpgradeForVideo("no_torch", null, true, repair, { deviceType: "linux", cpuTraining: true }),
    false,
  );
  assert.equal(
    canUpgradeForVideo("no_torch", null, true, repair, { deviceType: "linux", cpuTraining: false }),
    true,
  );
  assert.match(HOOK, /canUpgradeForVideo\(chatOnlyReason, chatOnlyHint, verdictMeasured, repair/);
});

test("a Mac the MLX wheels skip is still offered video, unlike Train", () => {
  // macOS 13 on Apple Silicon: the full install skips MLX but brings PyTorch with MPS.
  assert.equal(canUpgradeToFullInstall("no_torch", MLX_HINT, true, repair), false);
  assert.equal(
    canUpgradeForVideo("no_torch", MLX_HINT, true, repair, { deviceType: "mac", cpuTraining: false }),
    true,
  );
});

test("a closed AMD node blocks the video offer too", () => {
  const amd = "Studio cannot open /dev/kfd; add this user to the render group.";
  assert.equal(
    canUpgradeForVideo("no_torch", amd, true, repair, { deviceType: "linux", cpuTraining: false }),
    false,
  );
  assert.equal(
    canUpgradeForVideo("no_torch", null, true, null, { deviceType: "mac", cpuTraining: false }),
    false,
  );
  assert.equal(
    canUpgradeForVideo("no_torch", null, false, repair, { deviceType: "mac", cpuTraining: false }),
    false,
  );
});
