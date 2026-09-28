// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  MAX_AUDIO_SIZE_LABEL,
  getAudioSizeError,
} from "../../../lib/audio-utils.ts";
import { pollSignal } from "../../hub/lib/abort-signals.ts";

const CHAT_AUDIO_READINESS_TIMEOUT_MS = 10_000;

export interface ChatAudioUploadFence {
  generation: number;
  owner: string;
  authSessionEpoch: number;
}

export type ChatAudioReadinessRefreshResult =
  | "ran"
  | "skipped"
  | "superseded";

export class ChatAudioReadinessRefreshQueue {
  private tail: Promise<void> | null = null;
  private generation = 0;
  private activeController: AbortController | null = null;
  private readonly timeoutMs: number;

  constructor(timeoutMs: number = CHAT_AUDIO_READINESS_TIMEOUT_MS) {
    this.timeoutMs = timeoutMs;
  }

  invalidate(): void {
    this.generation += 1;
    this.activeController?.abort();
  }

  isCurrent(generation: number): boolean {
    return this.generation === generation;
  }

  async run(
    silent: boolean,
    refresh: (generation: number, signal: AbortSignal) => Promise<void>,
  ): Promise<ChatAudioReadinessRefreshResult> {
    const preceding = this.tail;
    if (silent && preceding) return "skipped";

    const generation = silent ? null : this.generation + 1;
    if (generation !== null) {
      this.generation = generation;
      this.activeController?.abort();
    }

    let release = () => {};
    const turn = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.tail = turn;

    if (preceding) await preceding;
    try {
      const activeGeneration = generation ?? this.generation + 1;
      if (generation === null) this.generation = activeGeneration;
      if (!this.isCurrent(activeGeneration)) return "superseded";
      const controller = new AbortController();
      const request = pollSignal(controller.signal, this.timeoutMs);
      this.activeController = controller;
      try {
        await refresh(activeGeneration, request.signal);
        return "ran";
      } finally {
        request.dispose();
        if (this.activeController === controller) {
          this.activeController = null;
        }
      }
    } finally {
      release();
      if (this.tail === turn) this.tail = null;
    }
  }
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
