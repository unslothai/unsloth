// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The applier is one object literal and the config page cannot render under Node, so these
// checks are source-level.

import assert from "node:assert/strict";
import test from "node:test";

import { getImageInputUnavailableReason } from "../src/features/chat/utils/image-input-support.ts";

import { readSrc } from "./helpers/kit.ts";

const APPLIER = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
const API_TYPES = readSrc("features/chat/types/api.ts");
const CONFIG_PAGE = readSrc(
  "features/model-picker/components/model-config-page.tsx",
);

test("both response types carry the raw disable_vision echo", () => {
  // vision_disabled_by_user is gated on having a projector, so the seed needs the load request.
  assert.equal(
    API_TYPES.match(/^ {2}disable_vision\?: boolean;$/gm)?.length,
    2,
  );
});

test("the applier seeds the switch from status, through the seed resolver", () => {
  // Unguarded, every poll would overwrite an unapplied switch; rule in shouldSeedVisionSwitch.
  assert.match(
    APPLIER,
    /status\.disable_vision !== undefined &&\s*\n\s*shouldSeedVisionSwitch\(\{/,
  );
  // An older backend omitting the field must change nothing, so no `?? false` here.
  assert.doesNotMatch(
    APPLIER,
    /disableVision: status\.disable_vision \?\? false/,
  );
});

test("the composer's own mirror of the flag stays unguarded", () => {
  // loadedVisionDisabledByUser mirrors the live load, so every poll must land on it.
  assert.match(
    APPLIER,
    /status\.vision_disabled_by_user !== undefined && \{\s*\n\s*loadedVisionDisabledByUser: status\.vision_disabled_by_user,/,
  );
});

test("the Vision row exists only in the GGUF half of Advanced Settings", () => {
  // MLX has no mmproj, so the row must not render for it.
  const lines = CONFIG_PAGE.split("\n");
  const bodyOf = (name: string): string => {
    const start = lines.findIndex((line) =>
      line.startsWith(`function ${name}(`),
    );
    assert.notEqual(start, -1, `no top-level function ${name}`);
    let end = lines.length;
    for (let i = start + 1; i < lines.length; i++) {
      if (/^(export )?function \w+\(/.test(lines[i])) {
        end = i;
        break;
      }
    }
    return lines.slice(start, end).join("\n");
  };
  const gguf = bodyOf("GgufAdvancedSettings");
  const mlx = bodyOf("MlxAdvancedSettings");
  const wiring = "checked={!config.disableVision}";

  assert.equal(
    CONFIG_PAGE.split(wiring).length - 1,
    1,
    "expected exactly one Vision switch in the file",
  );
  assert.ok(
    gguf.includes(wiring),
    "Vision switch is not in GgufAdvancedSettings",
  );
  assert.ok(
    !mlx.includes(wiring),
    "Vision switch leaked into MlxAdvancedSettings",
  );
  assert.ok(
    !mlx.includes("disableVision"),
    "MlxAdvancedSettings reads disableVision, which it cannot act on",
  );
  assert.ok(gguf.includes(">Vision</span>"));

  const gateAbove = (index: number): string => {
    for (let i = index; i >= 0; i--) {
      const line = lines[i].trim();
      // An audio-runtime GGUF launches no llama-server.
      if (
        line === "{target.isGguf && (" ||
        line === "{target.isGguf && !audioRuntimeGguf && ("
      )
        return "isGguf";
      if (line === "{!target.isGguf && (") return "!isGguf";
    }
    return "none";
  };
  const ggufAt = lines.findIndex((l) => l.includes("<GgufAdvancedSettings"));
  const mlxAt = lines.findIndex((l) => l.includes("<MlxAdvancedSettings"));
  assert.ok(ggufAt > 0 && mlxAt > 0, "one of the panels is never rendered");
  assert.equal(gateAbove(ggufAt), "isGguf");
  assert.equal(gateAbove(mlxAt), "!isGguf");
});

const VISION_GGUF = {
  id: "local/qwen3.5-4b",
  name: "Qwen3.5 4B",
  isLora: false,
  isVision: true,
  isGguf: true,
  isAudio: false,
  audioType: null,
  hasAudioInput: false,
};

function reason(overrides: Record<string, unknown> = {}) {
  return getImageInputUnavailableReason({
    activeModel: VISION_GGUF,
    isExternalModel: false,
    loadedIsMultimodal: false,
    modelLoaded: true,
    ...overrides,
  });
}

test("switching Vision off points at the switch, not at a missing mmproj", () => {
  const message = reason({ visionDisabledByUser: true });
  assert.ok(message, "attaching images should still be blocked");
  assert.match(message, /Advanced Settings/);
  assert.match(message, /Qwen3\.5 4B/);
  assert.doesNotMatch(message, /valid mmproj/);
  assert.doesNotMatch(message, /Load a vision-capable model/);
});

test("every other refusal is untouched by the new branch", () => {
  const missing = reason({ visionDisabledByUser: false });
  assert.match(missing ?? "", /valid mmproj/);
  assert.doesNotMatch(missing ?? "", /Advanced Settings/);
  assert.equal(reason(), missing);
  assert.equal(reason({ visionDisabledByUser: null }), missing);
  assert.match(
    reason({ modelLoaded: false, visionDisabledByUser: true }) ?? "",
    /Load a model before adding images/,
  );
  assert.equal(
    reason({ loadedIsMultimodal: true, visionDisabledByUser: true }),
    null,
  );
});

// The rollback must replay the baseline of the RUNNING server: the control field already holds
// the target's setting, and the gating field is false for non-image models.
test("the rollback replays the loaded vision baseline, not the control or the gate", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  const replay = runtime.slice(
    runtime.indexOf("tensor_parallel: rollbackState.loadedTensorParallel"),
  );
  const line = replay.slice(
    replay.indexOf("disable_vision:"),
    replay.indexOf("gpu_memory_mode:"),
  );
  assert.ok(
    line.includes("rollbackState.loadedDisableVision"),
    `rollback must replay the loaded baseline, got: ${line.trim()}`,
  );
  assert.ok(
    !line.includes("loadedVisionDisabledByUser"),
    "rollback must not replay the narrowed image-gating field",
  );
  assert.ok(
    !/stateBeforeUnload\.disableVision\b/.test(line),
    "rollback must not replay the control field, which the pending config overwrites",
  );
});

// applyPerModelConfigToRuntime runs before stateBeforeUnload is captured, so the snapshot's
// control holds the TARGET's setting; seed from the restored model.
test("the rollback seeds the Vision control from the restored model, not the target", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  const assignment = runtime.slice(
    runtime.indexOf("loadedSpeculativeType: rollbackSpeculativeType"),
  );
  const line = assignment.slice(
    assignment.indexOf("disableVision:"),
    assignment.indexOf("loadedVisionDisabledByUser:"),
  );
  assert.ok(
    line.includes("rollbackState.loadedDisableVision"),
    `the control must be seeded from the restored model's loaded value, got: ${line.trim()}`,
  );
  assert.ok(
    !/stateBeforeUnload\.disableVision\b/.test(line),
    "the control must not be seeded from the snapshot the pending config overwrote",
  );
  assert.ok(
    !/rollbackResponse\.disable_vision/.test(line),
    "the control must not be seeded from the echo, which is false for a text-only GGUF",
  );
});

// Vision defaults ON per model; unlike tensorParallel it must not carry over from the outgoing model.
test("an unconfigured target gets the default Vision value, not the outgoing model's", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  const decl = runtime.slice(
    runtime.indexOf("const loadDisableVision ="),
    runtime.indexOf("const loadActivePresetSource"),
  );
  assert.ok(
    decl.includes("DEFAULT_PER_MODEL_CONFIG.disableVision"),
    `a model switch must fall back to the per-model default, got: ${decl.trim()}`,
  );
  assert.ok(
    decl.includes("loadSwitchesModelOrVariant"),
    "the default must apply on a model or variant switch specifically",
  );
});

test("a compare pane with no saved config does not inherit the live Vision value", () => {
  const composer = readSrc("features/chat/shared-composer.tsx");
  const decl = composer.slice(
    composer.indexOf("const effectiveDisableVision ="),
    composer.indexOf("if (ownConfig.selectedGpuIds != null)"),
  );
  assert.ok(
    decl.includes("DEFAULT_PER_MODEL_CONFIG.disableVision"),
    `an unremembered pane must take the per-model default, got: ${decl.trim()}`,
  );
  assert.ok(
    !/fallbackDisableVision/.test(composer),
    "the store-derived fallback should be gone entirely, not just unused",
  );
});

test("the Vision row is gated out for diffusion models", () => {
  // The diffusion runner ignores disableVision and it is forced false, so the row must be gated.
  assert.match(
    CONFIG_PAGE,
    /disableVision: false,/,
    "withoutUnsupportedDiffusionSettings no longer clears disableVision",
  );

  const lines = CONFIG_PAGE.split("\n");
  const visionAt = lines.findIndex((line) =>
    line.includes("checked={!config.disableVision}"),
  );
  assert.notEqual(visionAt, -1, "no Vision switch to gate");

  // A `)}` first means the gate closed before the row.
  let nearest = "none";
  for (let i = visionAt; i >= 0; i--) {
    const line = lines[i].trim();
    if (line === "{!isDiffusion && (") {
      nearest = "!isDiffusion";
      break;
    }
    if (line === ")}") {
      nearest = "closed";
      break;
    }
  }
  assert.equal(
    nearest,
    "!isDiffusion",
    "the Vision row is not inside a !isDiffusion gate",
  );
});
