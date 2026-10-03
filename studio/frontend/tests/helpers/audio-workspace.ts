// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { readSrc } from "./kit.ts";

/** The Audio page and the modules split out of it, host first and then the hooks in the order the
 *  host calls them, so a pattern that spans two of them still reads in source order. */
export const AUDIO_WORKSPACE_FILES = [
  "features/audio/audio-page.tsx",
  "features/audio/hooks/use-stt-sidecar.ts",
  "features/audio/hooks/use-transcription.ts",
  "features/audio/hooks/use-audio-gallery.tsx",
  "features/audio/hooks/use-audio-model-slot.ts",
  "features/audio/hooks/use-speech-generation.ts",
  "features/audio/hooks/use-audio-handoff.ts",
  "features/audio/pages/tts-workspace.tsx",
  "features/audio/pages/speak-page.tsx",
  "features/audio/pages/music-page.tsx",
  "features/audio/pages/transcribe-page.tsx",
  "features/audio/tools/instructions-panels.tsx",
  "features/audio/tools/registry.tsx",
  "features/audio/tools/tool-panel-host.tsx",
  "features/audio/audio-workspace-constants.ts",
  "features/audio/audio-workspace-utils.ts",
  "features/audio/components/field.tsx",
] as const;

export type AudioWorkspaceFile = (typeof AUDIO_WORKSPACE_FILES)[number];

/** Every Audio workspace module as one text. Shape assertions that used to read audio-page.tsx
 *  read this, so they hold wherever the split put the code; `doesNotMatch` covers all of it. */
export function readAudioWorkspaceSource(): string {
  return AUDIO_WORKSPACE_FILES.map((file) => readSrc(file)).join("\n");
}
