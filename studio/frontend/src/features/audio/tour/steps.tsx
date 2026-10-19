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
      speaks in the voice from a short recording, Edit changes words in a
      recording, Convert makes a recording sound like another voice, Music
      makes songs and sound effects, Separate splits a track into vocals and
      instruments, and Transcribe turns a recording into text. Each lists only
      the models that can do it, so the picker above follows the page.
    </>
  ),
};

const MAIN_SLOT = "Loading one replaces the model in the main slot.";

const MODEL_STEP_BODY: Record<AudioWorkflowId, string> = {
  speak: `Text-to-speech models, including voices you fine-tuned under On Device. ${MAIN_SLOT}`,
  clone: `Models that can speak in the voice of a short recording you give them. ${MAIN_SLOT}`,
  edit: `Models that can change words in a recording and keep the voice. ${MAIN_SLOT}`,
  convert: `Voice conversion models. ${MAIN_SLOT}`,
  music: `Music models. ${MAIN_SLOT}`,
  separate: `Source separation models. ${MAIN_SLOT}`,
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
      download. Transcripts appear here too and keep their own history.
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

  if (workflow === "separate") {
    return [
      modeStep,
      modelStep(workflow),
      {
        id: "source",
        target: "audio-separate-source",
        title: "Track",
        body: (
          <>
            Upload, record or pick the track to split. It is converted to 44.1
            kHz for the model.
          </>
        ),
      },
      {
        id: "output",
        target: "audio-output",
        title: "Stems",
        body: (
          <>
            Each stem gets its own row with solo, mute and volume, all playing
            in sync. Download one stem or all of them.
          </>
        ),
      },
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

  if (workflow === "convert") {
    return [
      modeStep,
      modelStep(workflow),
      {
        id: "source",
        target: "audio-convert-source",
        title: "Recording",
        body: (
          <>
            Upload, record or pick the speech or singing to convert. Its words
            and timing stay; only the voice changes.
          </>
        ),
      },
      {
        id: "target",
        target: "audio-convert-target",
        title: "Target voice",
        body: (
          <>
            The voice it should sound like: a short clip, a saved voice, or one
            of the model's built-in voices.
          </>
        ),
      },
      {
        ...outputStep,
        body: (
          <>
            Results play here and stay in the history list. Switch between
            Source and Converted to hear the difference at the same moment.
          </>
        ),
      },
    ];
  }

  if (workflow === "edit") {
    return [
      modeStep,
      modelStep(workflow),
      {
        id: "recording",
        target: "audio-edit-recording",
        title: "Recording",
        body: (
          <>
            Upload, record or pick up to 30 seconds of one voice from your
            history.
          </>
        ),
      },
      {
        id: "transcript",
        target: "audio-edit-transcript",
        title: "Check the transcript",
        body: (
          <>
            It fills in by itself. Fix any word the recognizer got wrong so it
            matches what's said.
          </>
        ),
      },
      {
        id: "changes",
        target: "audio-edit-changes",
        title: "Make your changes",
        body: (
          <>
            Change, add or remove words. The voice stays the same, and the
            result plays against the original.
          </>
        ),
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
