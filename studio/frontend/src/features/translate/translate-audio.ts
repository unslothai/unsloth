// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  SttModelNotDownloadedError,
  type TranscriptSegment,
  transcribeAudioBlob,
} from "@/features/chat";
import {
  applyDictationDictionary,
  requestSttDownload,
  sttModelVariant,
  useSettingsDialogStore,
  useVoiceSettingsStore,
} from "@/features/settings";
import type { useT } from "@/i18n";
import { MAX_AUDIO_SIZE_LABEL, getAudioSizeError } from "@/lib/audio-utils";
import { toast } from "@/lib/toast";
import { segmentsToSrt } from "./translate-subtitles";

/** The transcript of an audio file with the Voice settings model, as chat's audio upload makes it; null after a toast.
 *  With `timed`, an SRT file when the model reports segment timings. */
export async function transcribeAudioFile(
  file: File,
  signal: AbortSignal,
  t: ReturnType<typeof useT>,
  timed = false,
): Promise<string | null> {
  if (file.size === 0 || getAudioSizeError(file.size)) {
    toast.error(
      file.size === 0
        ? t("settings.voice.dictation.audioUploadEmpty")
        : t("settings.voice.dictation.audioUploadTooLarge", {
            size: MAX_AUDIO_SIZE_LABEL,
          }),
    );
    return null;
  }
  const settings = useVoiceSettingsStore.getState();
  const model = settings.sttModel.trim();
  if (!model) {
    toast.error(t("settings.voice.dictation.audioUploadChooseModel"), {
      action: {
        label: t("settings.voice.dictation.sttOpenVoiceSettings"),
        onClick: () => useSettingsDialogStore.getState().openDialog("voice"),
      },
    });
    return null;
  }
  const ggufVariant = sttModelVariant(model, settings.sttGgufVariant);
  try {
    let segments: TranscriptSegment[] = [];
    const transcript = await transcribeAudioBlob(file, {
      model,
      ggufVariant,
      signal,
      onSegments: timed
        ? (found) => {
            segments = found;
          }
        : undefined,
    });
    const subtitles = segmentsToSrt(
      segments.map((segment) => ({
        ...segment,
        text: applyDictationDictionary(segment.text),
      })),
    );
    const text = subtitles ?? applyDictationDictionary(transcript).trim();
    if (!text) {
      toast.error(t("settings.voice.dictation.audioUploadNoSpeech"));
    }
    return text || null;
  } catch (error) {
    if (signal.aborted) {
      return null;
    }
    if (error instanceof SttModelNotDownloadedError) {
      requestSttDownload(model, { ggufVariant });
      return null;
    }
    toast.error(
      error instanceof Error && error.message
        ? error.message
        : t("settings.voice.dictation.audioUploadFailed"),
    );
    return null;
  }
}
