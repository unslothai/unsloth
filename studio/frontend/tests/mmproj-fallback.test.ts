import assert from "node:assert/strict";

import { readFileSync } from "node:fs";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { getImageInputUnavailableReason } = await import(
  "../src/features/chat/utils/image-input-support.ts"
);
const { isTextOnlyMmprojFallback, mmprojFallbackMessage, mmprojLoadNotice } =
  await import("../src/features/chat/utils/mmproj-fallback.ts");

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
