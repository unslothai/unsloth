// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const NATIVE_AUDIO_TYPES = new Set([
  "higgs_tts2",
  "moss_tts_local",
  "moss_tts_nano",
  "higgs_tts3",
  "minimax_music3",
]);

// RootLayout renders AudioPage persistently (so an in-flight generation is not cancelled when leaving the tab); this route only owns the URL + auth gate.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/audio",
  staticData: { title: "Audio" },
  // An audio pick made from the chat picker arrives here as ?model= (+ ?quant=, ?ggufQuant=, and task), which the page loads and then clears.
  validateSearch: (
    search: Record<string, unknown>,
  ): {
    model?: string;
    quant?: string;
    ggufQuant?: string;
    task?: string;
    audioType?: string;
    loadId?: string;
  } => ({
    ...(typeof search.model === "string" ? { model: search.model } : {}),
    ...(typeof search.quant === "string" ? { quant: search.quant } : {}),
    ...(typeof search.ggufQuant === "string"
      ? { ggufQuant: search.ggufQuant }
      : {}),
    ...(search.task === "automatic-speech-recognition" ||
    search.task === "text-to-speech"
      ? { task: search.task }
      : {}),
    ...(typeof search.audioType === "string" &&
    NATIVE_AUDIO_TYPES.has(search.audioType)
      ? { audioType: search.audioType }
      : {}),
    ...(typeof search.loadId === "string" && search.loadId.trim()
      ? { loadId: search.loadId }
      : {}),
  }),
  beforeLoad: () => requireAuth(),
  component: () => null,
});
