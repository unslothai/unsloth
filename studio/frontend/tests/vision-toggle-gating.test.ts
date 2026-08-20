// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the Disable Vision toggle is allowed to reach: the status reseed that
// teaches a fresh tab the running model has no projector, the MLX half of
// Advanced Settings which must never show the row, and the composer's refusal
// message when the toggle is what is in the way.
//
// The reseed and the MLX row are checked at the source. The applier is one large
// object literal with no seam to call, and the config page cannot be rendered
// (the repo has no render harness, and importing it pulls a directory import
// Node cannot resolve). Both follow the chat-template and llama-extra-args seed
// tests next door.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { getImageInputUnavailableReason } from "../src/features/chat/utils/image-input-support.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (relative: string) =>
  readFileSync(path.join(HERE, "..", relative), "utf8");

const APPLIER = read(
  "src/features/chat/lib/apply-inference-status-to-store.ts",
);
const API_TYPES = read("src/features/chat/types/api.ts");
const CONFIG_PAGE = read(
  "src/features/model-picker/components/model-config-page.tsx",
);

test("both response types carry the raw disable_vision echo", () => {
  // vision_disabled_by_user is additionally gated on the model HAVING a
  // projector, so it cannot round-trip the switch on a text-only GGUF. The seed
  // needs the request the load actually ran with, on the load response and the
  // status poll alike.
  assert.equal(
    API_TYPES.match(/^ {2}disable_vision\?: boolean;$/gm)?.length,
    2,
  );
});

test("the applier seeds the switch from status, through the seed resolver", () => {
  // Unguarded it would fight the user: every poll would stamp the running model's
  // value over a switch that was flipped but not yet applied. The rule is in
  // shouldSeedVisionSwitch, which vision-switch-seed.test.ts pins case by case;
  // this only checks the applier still routes through it.
  assert.match(
    APPLIER,
    /status\.disable_vision !== undefined &&\s*\n\s*shouldSeedVisionSwitch\(\{/,
  );
  // An older backend that omits the field must change nothing at all, so no
  // `?? false` may creep in here.
  assert.doesNotMatch(
    APPLIER,
    /disableVision: status\.disable_vision \?\? false/,
  );
});

test("the composer's own mirror of the flag stays unguarded", () => {
  // loadedVisionDisabledByUser is what the image gate reads. It mirrors the live
  // load rather than a user setting, so every poll must land on it; the seed
  // guard here would freeze the refusal string at the first read.
  assert.match(
    APPLIER,
    /status\.vision_disabled_by_user !== undefined && \{\s*\n\s*loadedVisionDisabledByUser: status\.vision_disabled_by_user,/,
  );
});

test("the Vision row exists only in the GGUF half of Advanced Settings", () => {
  // MLX is a separate engine with no mmproj to skip, so the row must not render
  // for it. Checked by exact wiring, so a second switch added elsewhere fails
  // here rather than quietly widening the control's reach.
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

  // And the GGUF half only renders under target.isGguf, so a non-GGUF target
  // shows no Vision switch at all.
  const gateAbove = (index: number): string => {
    for (let i = index; i >= 0; i--) {
      const line = lines[i].trim();
      if (line === "{target.isGguf && (") return "isGguf";
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
  // The generic copy would send someone who turned the toggle off hunting for a
  // vision model with a valid mmproj, a problem they do not have.
  assert.doesNotMatch(message, /valid mmproj/);
  assert.doesNotMatch(message, /Load a vision-capable model/);
});

test("every other refusal is untouched by the new branch", () => {
  const missing = reason({ visionDisabledByUser: false });
  assert.match(missing ?? "", /valid mmproj/);
  assert.doesNotMatch(missing ?? "", /Advanced Settings/);
  // An absent flag behaves exactly as before the toggle existed.
  assert.equal(reason(), missing);
  assert.equal(reason({ visionDisabledByUser: null }), missing);
  // No model loaded still reports the load, not the toggle.
  assert.match(
    reason({ modelLoaded: false, visionDisabledByUser: true }) ?? "",
    /Load a model before adding images/,
  );
  // And a stale echo must not disable attach for a session that DID load its
  // projector.
  assert.equal(
    reason({ loadedIsMultimodal: true, visionDisabledByUser: true }),
    null,
  );
});

// The rollback replays the previous load's settings after a failed switch. It must
// send the baseline that tracks the RUNNING server, not the control field: a pending
// per-model config is written to disableVision before the switch captures its
// baseline, so that holds the TARGET's setting and a failed switch would roll the old
// model back with the new model's Vision choice. Nor the narrowed image-gating field,
// which is false for a model that cannot do images. Source-level for the same reason
// as the reseed above: the replay sits inside one large object literal.
test("the rollback replays the loaded vision baseline, not the control or the gate", () => {
  const runtime = read("src/features/chat/hooks/use-chat-model-runtime.ts");
  const replay = runtime.slice(
    runtime.indexOf("tensor_parallel: stateBeforeUnload.loadedTensorParallel"),
  );
  const line = replay.slice(
    replay.indexOf("disable_vision:"),
    replay.indexOf("gpu_memory_mode:"),
  );
  assert.ok(
    line.includes("stateBeforeUnload.loadedDisableVision"),
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

// The other half of the same rollback: the store assignment that follows the replayed
// request. Getting the request right is not enough, because the control the user sees
// is seeded separately, and applyPerModelConfigToRuntime(pendingLoadConfig) runs BEFORE
// stateBeforeUnload is captured, so the snapshot's control field already holds the
// TARGET's setting. Seeding from it leaves the Vision row reading "off" over a restored
// model whose projector is running, and arms the next Apply to switch it off for real.
// Source-level for the same reason as the replay above.
test("the rollback seeds the Vision control from the restored model, not the target", () => {
  const runtime = read("src/features/chat/hooks/use-chat-model-runtime.ts");
  const assignment = runtime.slice(
    runtime.indexOf("loadedSpeculativeType: rollbackSpeculativeType"),
  );
  const line = assignment.slice(
    assignment.indexOf("disableVision:"),
    assignment.indexOf("loadedVisionDisabledByUser:"),
  );
  assert.ok(
    line.includes("stateBeforeUnload.loadedDisableVision"),
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

// Vision is per-model config with a default of ON, so switching to a model that saved
// no config must give it that default rather than the outgoing model's setting. It
// used to inherit, which loaded the new model text-only and did it silently: the
// dedupe comparison builds its own view of an unconfigured switch out of
// DEFAULT_PER_MODEL_CONFIG, so the two halves disagreed about the same load. This is
// where it parts company with tensorParallel, which is deliberately standing across
// models. Source-level for the same reason as the tests above.
test("an unconfigured target gets the default Vision value, not the outgoing model's", () => {
  const runtime = read("src/features/chat/hooks/use-chat-model-runtime.ts");
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
  const composer = read("src/features/chat/shared-composer.tsx");
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
  // withoutUnsupportedDiffusionSettings forces disableVision back to false and the
  // diffusion runner never reads it, so an ungated row is a switch that flips back
  // under the pointer and changes nothing if it did not. The batch rows above it are
  // gated for exactly this reason.
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

  // The nearest JSX conditional boundary above the switch. A `)}` first means the
  // gate closed before the row, i.e. the row is not inside it.
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
