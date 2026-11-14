// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

const { TranscriptionProgress } = loadWithStubs<{
  TranscriptionProgress: (props: Record<string, unknown>) => StubElement;
}>(new URL("../src/features/audio/transcription-progress.tsx", import.meta.url), {
  react: { useEffect: () => {}, useState: () => [0, () => {}] },
  "react/jsx-runtime": stubJsxRuntime(),
  "@/components/ui/button": { Button: "button" },
  "./transcription-progress.css": {},
});

/** Collect visible text from the actual component's rendered element tree. */
function renderedText(value: unknown): string {
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (Array.isArray(value)) return value.map(renderedText).join("");
  if (value && typeof value === "object" && "props" in value) {
    return renderedText((value as StubElement).props.children);
  }
  return "";
}

const render = (progress: unknown, finishedAt: number | null = null) =>
  renderedText(
    TranscriptionProgress({
      startedAt: 0,
      finishedAt,
      stopping: false,
      progress,
      onCancel: () => {},
    }),
  );

test("shows percentage from measured audio progress", () => {
  assert.match(render({ text: "hello", processed_seconds: 30, duration: 120 }), /25%/);
  assert.match(render({ text: "", processed_seconds: 0, duration: 10 }), /0%/);
  assert.match(render({ text: "", processed_seconds: 12, duration: 10 }), /100%/);
});

test("keeps phase-only progress without inventing a percentage", () => {
  assert.match(render({ text: "", phase: "transcribing" }), /Transcribing/);
  assert.doesNotMatch(render({ text: "", phase: "transcribing" }), /%/);
  assert.doesNotMatch(render({ text: "", processed_seconds: 0, duration: 0 }), /%/);
  assert.doesNotMatch(render({ text: "", processed_seconds: NaN, duration: 10 }), /%/);
});

test("does not present a progress percentage after completion", () => {
  assert.doesNotMatch(
    render({ text: "hello", processed_seconds: 30, duration: 120 }, 1000),
    /%/,
  );
});
