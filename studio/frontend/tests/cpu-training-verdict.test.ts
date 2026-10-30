// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A CPU-only host trains on the CPU backend (/api/health cpu_training) while chat stays GGUF-only.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { trainsOnCpu, videoNavHint } = await import("../src/config/hardware-verdict.ts");
const { en } = await import("../src/i18n/locales/en.ts");

test("only a no_gpu verdict with cpu_training trains on the CPU", () => {
  assert.equal(trainsOnCpu("no_gpu", true), true);
  assert.equal(trainsOnCpu("no_gpu", false), false);
  // no_torch with cpu_training is a Chat only install: the full install trains on CPU, this one cannot.
  for (const reason of ["no_torch", "torch_cpu_build", "torch_cuda_unavailable", "intel_mac", null]) {
    assert.equal(trainsOnCpu(reason, true), false, String(reason));
  }
});

test("a GPU-less Chat only install keeps the Video row's GPU message", () => {
  assert.equal(
    videoNavHint(true, "no_torch", false, true),
    "Video generation needs an NVIDIA or AMD GPU.",
  );
  // A Chat only install on a GPU host gets video from the full install, so its row stays open.
  assert.equal(videoNavHint(true, "no_torch", false, false), undefined);
  assert.equal(videoNavHint(false, "no_torch", false, true), undefined);
});

test("the health verdict carries cpu_training into the platform store", () => {
  const env = readSrc("config/env.ts");
  assert.match(env, /cpu_training\?: boolean;/);
  assert.match(env, /cpuTraining: false,/);
  assert.match(env, /data\.cpu_training === undefined \? previous\.cpuTraining/);
  assert.match(env, /isTrainUnavailable: \(\) => \{[\s\S]*?trainsOnCpu\(state\.chatOnlyReason, state\.cpuTraining\)/);
});

test("every Train entry point reads isTrainUnavailable, not isChatOnly", () => {
  for (const file of [
    "components/app-sidebar.tsx",
    "app/routes/__root.tsx",
    "features/studio/studio-page.tsx",
    "features/settings/tabs/data-tab.tsx",
  ]) {
    assert.match(readSrc(file), /isTrainUnavailable\(\)/, file);
  }
  assert.match(readSrc("features/studio/studio-page.tsx"), /t\("studio\.cpuTrainingNotice"\)/);
  assert.match(en.studio.cpuTrainingNotice, /runs on the CPU/);
  assert.match(en.studio.cpuTrainingNotice, /text models only/);
});
