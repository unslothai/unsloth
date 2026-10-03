// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";
import type { AudioWorkflowId } from "../workflows";

const modeStep: TourStep = {
  id: "mode",
  target: "audio-mode",
  title: "Pick a page",
  body: (
    <>
      Open the page title to switch pages. Speak turns text into speech, Clone
      speaks in the voice from a short recording, Music makes songs and sound
      effects, and Transcribe turns a recording into text. Each lists only the
      models that can do it, so the picker above follows the page.
    </>
  ),
};

const MODEL_STEP_BODY: Record<AudioWorkflowId, string> = {
  speak:
    "Text-to-speech models, including voices you fine-tuned under On Device.",
  clone:
    "Models that can speak in the voice of a short recording you give them.",
  music: "Music models. Loading one replaces the model in the main slot.",
  transcribe:
    "Speech recognition models. They run beside your chat model, not in its place.",
};

function modelStep(workflow: AudioWorkflowId): TourStep {
  return {
    id: "model",
    target: "audio-model",
    title: "Pick a model",
    body: <>{MODEL_STEP_BODY[workflow]}</>,
  };
}

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

export function buildAudioTourSteps({
  workflow,
}: {
  workflow: AudioWorkflowId;
}): TourStep[] {
  if (workflow === "transcribe") {
    return [
      modeStep,
      modelStep(workflow),
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

  if (workflow === "clone") {
    return [
      modeStep,
      modelStep(workflow),
      {
        id: "reference",
        target: "audio-clone-reference",
        title: "Reference clip",
        body: (
          <>
            Upload, record or pick a few seconds of the voice to copy. A saved
            voice works too.
          </>
        ),
      },
      {
        id: "transcript",
        target: "audio-clone-transcript",
        title: "What's said in the clip",
        body: (
          <>
            Some models need the clip's words to match its voice. Type them or
            press Transcribe.
          </>
        ),
      },
      {
        id: "text",
        target: "audio-clone-text",
        title: "Text to speak",
        body: <>What the cloned voice should say.</>,
      },
      outputStep,
    ];
  }

  return [
    modeStep,
    modelStep(workflow),
    {
      id: "settings",
      target: "audio-settings",
      title: "Settings",
      body:
        workflow === "music" ? (
          <>
            Lyrics, plus a description of the style where the model takes one.
            Length sits under Advanced.
          </>
        ) : (
          <>
            Your text, plus voice and style where the model supports them.
            Length and temperature sit under Advanced.
          </>
        ),
    },
    outputStep,
  ];
}
