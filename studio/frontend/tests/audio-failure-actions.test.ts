// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
const { audioFailureActions, parseAudioFailure, reportAudioFailure } =
  await import("../src/features/audio/audio-failure-actions.ts");
const { useAudioWorkspaceStore } = await import(
  "../src/features/audio/stores/audio-workspace-store.ts"
);

const OOM = {
  code: "out_of_memory",
  message: "Not enough GPU memory.",
  smaller_variants: [
    { variant: "Q4_K_M", label: "Q4_K_M", size_bytes: 1, estimate_gb: 3.1 },
    { variant: "Q3_K_M" },
  ],
};
const CHOOSE = "Choose a different model";
const CPU = "Load into CPU RAM";
const RUNTIME = [
  "runtime_missing",
  "family_unknown_to_runtime",
  "espeak_missing",
];

function actionsFor(detail: unknown, overrides: Record<string, unknown> = {}) {
  const calls: string[] = [];
  const actions = audioFailureActions(parseAudioFailure(detail), {
    deviceIsCpu: false,
    canUpdateRuntime: false,
    onUpdateRuntime: () => calls.push("update"),
    onUseCpu: () => calls.push("cpu"),
    onUseVariant: (variant: string) => calls.push(`variant:${variant}`),
    onChooseModel: () => calls.push("choose"),
    ...overrides,
  });
  for (const action of actions) action.onClick();
  return { labels: actions.map((action) => action.label), calls };
}

test("parses a detail, a body wrapping it and a deferred load error; anything else is null", () => {
  assert.deepEqual(parseAudioFailure(OOM), {
    code: "out_of_memory",
    message: "Not enough GPU memory.",
    smallerVariants: [
      { variant: "Q4_K_M", label: "Q4_K_M" },
      { variant: "Q3_K_M", label: "Q3_K_M" },
    ],
    onCpu: false,
  });
  assert.equal(parseAudioFailure({ detail: OOM })?.code, "out_of_memory");
  const deferred = { status_code: 503, detail: { code: "gpu_unavailable" } };
  assert.equal(
    parseAudioFailure({ _deferred_error: deferred })?.code,
    "gpu_unavailable",
  );
  for (const other of ["boom", null, [{ msg: "x" }], { code: "new" }, {}]) {
    assert.equal(parseAudioFailure(other), null);
  }
});

test("each code maps onto its actions; unknown codes keep today's action", () => {
  const bareOom = { code: "out_of_memory", message: "m" };
  const up = { canUpdateRuntime: true };
  const cases: [unknown, Record<string, unknown>, string[], string[]][] = [
    [OOM, {}, ["Use Q4_K_M", CPU], ["variant:Q4_K_M", "cpu"]],
    [OOM, { deviceIsCpu: true }, ["Use Q4_K_M"], ["variant:Q4_K_M"]],
    [OOM, { cpuUnsupported: true }, ["Use Q4_K_M"], ["variant:Q4_K_M"]],
    [bareOom, {}, [CPU], ["cpu"]],
    [bareOom, { deviceIsCpu: true }, [CHOOSE], ["choose"]],
    [{ ...OOM, device: "cpu" }, {}, ["Use Q4_K_M"], ["variant:Q4_K_M"]],
    [{ code: "gpu_unavailable" }, {}, [CPU], ["cpu"]],
    [{ code: "file_incomplete" }, {}, [CHOOSE], ["choose"]],
    [{ code: "something_new" }, up, [CHOOSE], ["choose"]],
    [null, up, [CHOOSE], ["choose"]],
  ];
  for (const code of RUNTIME) {
    cases.push([{ code }, up, ["Update the audio runtime"], ["update"]]);
    cases.push([{ code }, {}, [CHOOSE], ["choose"]]);
  }
  for (const [detail, overrides, labels, calls] of cases) {
    const label = `${JSON.stringify(detail)} ${JSON.stringify(overrides)}`;
    assert.deepEqual(actionsFor(detail, overrides), { labels, calls }, label);
  }
});

test("reportAudioFailure keeps the failure with the message it explains, never persisted", () => {
  const error = Object.assign(new Error("Not enough GPU memory."), {
    detail: OOM,
  });
  assert.equal(reportAudioFailure(error)?.code, "out_of_memory");
  const stored = useAudioWorkspaceStore.getState().failure;
  assert.equal(stored?.message, "Not enough GPU memory.");
  assert.equal(stored?.failure.code, "out_of_memory");
  useAudioWorkspaceStore.getState().rememberModel("speak", "org/model");
  assert.doesNotMatch(
    store.get("unsloth_audio_workspace") ?? "",
    /failure|out_of_memory/,
  );
  assert.equal(reportAudioFailure(new Error("plain")), null);
  assert.equal(useAudioWorkspaceStore.getState().failure, null);
});

test("generation hooks report failures and the page uses them only for the error on screen", () => {
  for (const hook of ["speech", "clone", "edit", "convert", "separate"]) {
    assert.match(
      readSrc(`features/audio/hooks/use-${hook}-generation.ts`),
      /updateGenerationPhase\("finishing"\);\s*reportAudioFailure\(error\);/,
    );
  }
  const page = readSrc("features/audio/audio-page.tsx");
  assert.match(
    page,
    /audioFailure\?\.message === pageGenerationError\s*\?\s*audioFailure\.failure\s*:\s*null/,
  );
  assert.match(page, /useLlamaUpdateCheck\(\{ enabled: active \}\)/);
  const slot = readSrc("features/audio/hooks/use-audio-model-slot.ts");
  assert.match(slot, /res\.audio_device_fallback === "cpu" && !wantsCpu/);
  assert.match(slot, /duration: 12000,\s*action: action &&/);
  assert.match(slot, /queued\.isGguf,\s*queued\.deviceOverride,/);
});
