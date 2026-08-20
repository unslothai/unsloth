import assert from "node:assert/strict";

import { readFileSync } from "node:fs";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { getImageInputUnavailableReason } = await import(
  "../src/features/chat/utils/image-input-support.ts"
);
const {
  isTextOnlyMmprojFallback,
  mmprojFallbackMessage,
  mmprojLoadNotice,
  loadFallbackNotice,
  CPU_FALLBACK_MESSAGE,
} = await import("../src/features/chat/utils/mmproj-fallback.ts");

const IMAGE_INPUT_AVAILABLE = /Image input remains available/;
const RELOADED_TEXT_ONLY = /reloaded this model in text-only mode/;
const PROJECTOR_FAILED = /vision projector failed to start/;
const TEXT_ONLY_MODE = /text-only mode/;
const GENERIC_IMAGE_ERROR = /cannot accept images/;

test("CPU projector fallback preserves image input and explains slower vision", () => {
  assert.equal(isTextOnlyMmprojFallback("cpu_offload"), false);
  const notice = mmprojLoadNotice("PaddleOCR", "cpu_offload");
  assert.equal(notice.title, "PaddleOCR loaded with vision on CPU");
  assert.match(notice.description, IMAGE_INPUT_AVAILABLE);
});

test("text-only projector fallback replaces the misleading generic mmproj error", () => {
  assert.equal(isTextOnlyMmprojFallback("projector_startup_failure"), true);
  assert.match(
    mmprojFallbackMessage("projector_startup_failure"),
    RELOADED_TEXT_ONLY,
  );
  const reason = getImageInputUnavailableReason({
    activeModel: {
      id: "PaddleOCR",
      name: "PaddleOCR",
      isVision: true,
      isGguf: true,
      isLora: false,
    },
    isExternalModel: false,
    loadedIsMultimodal: false,
    modelLoaded: true,
    mmprojFallbackReason: "projector_startup_failure",
  });
  assert.match(reason ?? "", PROJECTOR_FAILED);
  assert.match(reason ?? "", TEXT_ONLY_MODE);
  assert.doesNotMatch(reason ?? "", GENERIC_IMAGE_ERROR);
});

test("text-only fallback overrides stale audio VLM capabilities", () => {
  const reason = getImageInputUnavailableReason({
    activeModel: {
      id: "AudioVisionModel",
      name: "AudioVisionModel",
      isVision: true,
      isGguf: true,
      isLora: false,
      isAudio: false,
      hasAudioInput: true,
    },
    isExternalModel: false,
    loadedIsMultimodal: true,
    modelLoaded: true,
    mmprojFallbackReason: "projector_startup_failure",
  });
  assert.match(reason ?? "", PROJECTOR_FAILED);
  assert.match(reason ?? "", TEXT_ONLY_MODE);
});

function sourceBetween(path: string, start: string, end: string): string {
  const source = readFileSync(new URL(path, import.meta.url), "utf8");
  const startAt = source.indexOf(start);
  assert.notEqual(startAt, -1, `Missing source marker: ${start}`);
  const endAt = source.indexOf(end, startAt);
  assert.notEqual(endAt, -1, `Missing source marker: ${end}`);
  return source.slice(startAt, endAt);
}

test("attachment and send gates forward projector fallback state", () => {
  const attachmentGate = sourceBetween(
    "../src/features/chat/runtime-provider.tsx",
    "const unavailableReason = getImageInputUnavailableReason({",
    "if (unavailableReason)",
  );
  assert.match(
    attachmentGate,
    /mmprojFallbackReason:\s*state\.mmprojFallbackReason/,
  );

  const sendGate = sourceBetween(
    "../src/features/chat/api/chat-adapter.ts",
    "const imageGateReason = getImageInputUnavailableReason({",
    "if (imageGateReason)",
  );
  assert.match(
    sendGate,
    /mmprojFallbackReason:\s*runtime\.mmprojFallbackReason/,
  );
  const compareLoadState = sourceBetween(
    "../src/features/chat/shared-composer.tsx",
    "useChatRuntimeStore.setState({",
    "activeNativePathToken: null,",
  );
  assert.match(
    compareLoadState,
    /mmprojFallbackReason:\s*resp\.mmproj_fallback_reason\s*\?\?\s*null/,
  );

  const interactiveLoad = sourceBetween(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
    "loadResponse = await loadModel({",
    "cpuFallbackReason = loadResponse.cpu_fallback_reason",
  );
  assert.match(interactiveLoad, /force_reload:\s*forceReload/);
});

// cpu_offload has three producers: the fit estimate predicting the projector will not
// fit in VRAM, a GPU allocation failure at startup, and a bare signal crash with no
// non-projector diagnostic. So the message names the outcome and no cause at all.
test("the CPU projector message describes the placement, not one route to it", () => {
  const message = mmprojFallbackMessage("cpu_offload");
  assert.match(message, IMAGE_INPUT_AVAILABLE);
  // Untrue of the predicted pin, which never attempts a GPU load.
  assert.doesNotMatch(message, /could not start|failed|crash/i);
  // Untrue of both crash routes, and worse than vague there: it sends someone whose
  // GPU runtime is broken off to cut context and offload layers.
  assert.doesNotMatch(message, /VRAM|fit|memory pressure|out of memory/i);
});

// The combination neither load path handled. Both wrote
// `mmproj ? mmprojMessage : cpu ? cpuMessage : undefined`, so a session that lost GPU
// acceleration AND vision reported only the vision loss, and the user was never told
// the model was running on the CPU.
//
// Reachable rather than theoretical: on a CPU-fallback replay llama_cpp.py preserves
// _cpu_fallback_reason and clears _mmproj_fallback_reason, so the projector can fail
// again inside that same launch. A low-VRAM box whose Vulkan backend crashed is
// precisely where both happen.

test("a load that lost both the GPU and the projector reports both", () => {
  const notice = loadFallbackNotice(
    "PaddleOCR loaded",
    "vulkan_startup_crash",
    "projector_startup_failure",
  );
  assert.equal(notice.title, "PaddleOCR loaded on CPU, without vision");
  assert.match(notice.description ?? "", /GPU acceleration is disabled/);
  assert.match(notice.description ?? "", /text-only mode/);
  assert.equal(notice.degraded, true);
});

test("a CPU-offloaded projector under a CPU fallback does not claim vision is accelerated", () => {
  const notice = loadFallbackNotice(
    "PaddleOCR loaded",
    "vulkan_startup_crash",
    "cpu_offload",
  );
  // "on CPU" already covers the projector, so the title does not also say
  // "with vision on CPU" -- but the description must still explain both.
  assert.equal(notice.title, "PaddleOCR loaded on CPU");
  assert.match(notice.description ?? "", /GPU acceleration is disabled/);
  assert.match(notice.description ?? "", /Image input remains available/);
  assert.equal(notice.degraded, true);
});

test("each fallback alone still reads exactly as it did before", () => {
  const cpuOnly = loadFallbackNotice("X loaded", "vulkan_startup_crash", null);
  assert.equal(cpuOnly.title, "X loaded on CPU");
  assert.equal(cpuOnly.description, CPU_FALLBACK_MESSAGE);

  const offload = loadFallbackNotice("X loaded", null, "cpu_offload");
  assert.equal(offload.title, "X loaded with vision on CPU");
  assert.equal(offload.description, mmprojFallbackMessage("cpu_offload"));
  // The title the standalone helper produces, unchanged.
  assert.equal(offload.title, mmprojLoadNotice("X", "cpu_offload").title);

  const textOnly = loadFallbackNotice(
    "X loaded",
    null,
    "projector_incompatible",
  );
  assert.equal(textOnly.title, "X loaded without vision");
  assert.equal(
    textOnly.description,
    mmprojFallbackMessage("projector_incompatible"),
  );
});

test("a clean load is not degraded and carries no description", () => {
  const notice = loadFallbackNotice("X loaded", null, null);
  assert.equal(notice.title, "X loaded");
  assert.equal(notice.description, undefined);
  assert.equal(notice.degraded, false);
});
