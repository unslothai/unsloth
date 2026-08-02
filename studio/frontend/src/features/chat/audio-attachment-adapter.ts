// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUDIO_ACCEPT,
  MAX_AUDIO_SIZE,
  fileToBase64,
} from "@/lib/audio-utils";
import type {
  Attachment,
  AttachmentAdapter,
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";
import { toast } from "sonner";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

// crypto.randomUUID is undefined in non-secure contexts (HTTP over a LAN IP).
function newAttachmentId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

// Audio shares the "Add photos & files" picker. Like VisionImageAdapter,
// unsupported models are rejected at add() time with a toast.
export class AudioAttachmentAdapter implements AttachmentAdapter {
  // MIME is unreliable for some containers (m4a), so also match by
  // extension. No .webm extension: it would claim video/webm files; real
  // audio webm (MediaRecorder) always reports the audio/webm MIME.
  accept = `${AUDIO_ACCEPT},audio/x-m4a,.wav,.mp3,.m4a,.ogg,.oga,.flac`;
  private readonly attachmentIds = new Set<string>();

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    const state = useChatRuntimeStore.getState();
    const checkpoint = state.params.checkpoint;
    const activeModel = state.models.find((m) => m.id === checkpoint);
    const modelLoaded = !!checkpoint && !state.modelLoading;
    let unavailableReason: string | null = null;
    if (!modelLoaded) {
      // Mirror the image gate: flag a failed load vs "no model picked".
      unavailableReason = state.lastModelLoadError
        ? "The last model failed to load. Check the server logs, then load a model before adding audio files."
        : "Load a model before adding audio files.";
    } else if (!activeModel?.hasAudioInput) {
      const label = activeModel?.name || checkpoint || "Current model";
      unavailableReason = `${label} cannot accept audio. Load an audio-input model before attaching audio files.`;
    }
    if (unavailableReason) {
      toast.error(unavailableReason);
      throw new Error(unavailableReason);
    }
    if (file.size > MAX_AUDIO_SIZE) {
      const sizeReason = "Audio size exceeds 50MB limit";
      toast.error(sizeReason);
      throw new Error(sizeReason);
    }
    if (this.attachmentIds.size > 0 || state.pendingAudioBase64) {
      const duplicateReason = "Only one audio file can be attached per message.";
      toast.error(duplicateReason);
      throw new Error(duplicateReason);
    }

    const id = newAttachmentId();
    this.attachmentIds.add(id);
    return {
      id,
      type: "file",
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    try {
      const base64 = await fileToBase64(attachment.file);
      // Backend takes raw base64; format only satisfies the part type.
      const format = attachment.contentType === "audio/mpeg" ? "mp3" : "wav";
      return {
        id: attachment.id,
        type: "file",
        name: attachment.name,
        contentType: attachment.contentType,
        content: [{ type: "audio", audio: { data: base64, format } }],
        status: { type: "complete" },
      };
    } finally {
      this.attachmentIds.delete(attachment.id);
    }
  }

  remove(attachment: Attachment): Promise<void> {
    this.attachmentIds.delete(attachment.id);
    return Promise.resolve();
  }
}
