// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

const modeStep: TourStep = {
  id: "mode",
  target: "audio-mode",
  title: "Generate or Transcribe",
  body: (
    <>
      Generate makes speech and music from text. Transcribe turns a recording
      into text. They use different models, so the picker above follows the
      mode.
    </>
  ),
};

const modelStep: TourStep = {
  id: "model",
  target: "audio-model",
  title: "Pick a model",
  body: (
    <>
      TTS and music models for Generate, speech recognition for Transcribe.
      Voices you fine-tuned show up under On Device.
    </>
  ),
};

const outputStep: TourStep = {
  id: "output",
  target: "audio-output",
  title: "Output",
  body: (
    <>
      Clips play here and stay in the history list beside them, ready to
      download. Transcripts appear here too, but are not kept after you leave.
    </>
  ),
};

/** The two modes swap the settings body and the footer, so each gets its own middle step. */
export function buildAudioTourSteps({
  mode,
}: {
  mode: "speak" | "transcribe";
}): TourStep[] {
  if (mode === "transcribe") {
    return [
      modeStep,
      modelStep,
      {
        id: "record",
        target: "audio-record",
        title: "Record or upload",
        body: (
          <>
            Record from your microphone and it transcribes when you stop, or
            upload a wav, mp3, m4a or webm file instead.
          </>
        ),
      },
      outputStep,
    ];
  }

  return [
    modeStep,
    modelStep,
    {
      id: "settings",
      target: "audio-settings",
      title: "Settings",
      body: (
        <>
          Your text or lyrics, plus voice and style where the model supports
          them. Length and temperature sit under Advanced.
        </>
      ),
    },
    outputStep,
  ];
}
