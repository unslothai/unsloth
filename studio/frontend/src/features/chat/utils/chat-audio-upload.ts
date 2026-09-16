// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  MAX_AUDIO_SIZE_LABEL,
  getAudioSizeError,
} from "../../../lib/audio-utils.ts";

export interface ChatAudioUploadFence {
  generation: number;
  owner: string;
  authSessionEpoch: number;
}

export type ChatAudioUploadFileError = "empty" | "too-large";

export function chatAudioUploadFileError(
  file: Pick<File, "size">,
): ChatAudioUploadFileError | null {
  if (file.size === 0) return "empty";
  return getAudioSizeError(file.size) ? "too-large" : null;
}

export { MAX_AUDIO_SIZE_LABEL };

export function chatAudioUploadFenceMatches(
  started: ChatAudioUploadFence,
  current: ChatAudioUploadFence,
): boolean {
  return (
    started.generation === current.generation &&
    started.owner === current.owner &&
    started.authSessionEpoch === current.authSessionEpoch
  );
}

export function appendChatAudioTranscript(
  draft: string,
  transcript: string,
): string {
  const finalTranscript = transcript.trim();
  if (!finalTranscript) return draft;
  return draft ? `${draft.trimEnd()} ${finalTranscript}` : finalTranscript;
}

export type ChatAudioUploadCompletion = "committed" | "empty" | "stale";

export async function completeChatAudioUpload({
  started,
  current,
  signal,
  blocked,
  transcribe,
  commit,
}: {
  started: ChatAudioUploadFence;
  current: () => ChatAudioUploadFence;
  signal: AbortSignal;
  blocked: () => boolean;
  transcribe: () => Promise<string>;
  commit: (transcript: string) => void;
}): Promise<ChatAudioUploadCompletion> {
  const transcript = (await transcribe()).trim();
  if (
    signal.aborted ||
    blocked() ||
    !chatAudioUploadFenceMatches(started, current())
  ) {
    return "stale";
  }
  if (!transcript) return "empty";
  commit(transcript);
  return "committed";
}
