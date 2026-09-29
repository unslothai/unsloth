// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUDIO_ATTACHMENT_ACCEPT,
  fileToBase64,
  getAudioAddError,
  maxAudioFilesFor,
} from "@/lib/audio-utils";
import type {
  Attachment,
  AttachmentAdapter,
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";
import { toast } from "sonner";
import { externalModelLabel } from "./lib/external-model-label";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

// crypto.randomUUID is undefined in non-secure contexts (HTTP over a LAN IP).
function newAttachmentId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

const AUDIO_ADD_TOAST_ID = "audio-attachment-limit";

// A loaded model without audio rejects at add(); with none loaded, the send path checks it later.
export class AudioAttachmentAdapter implements AttachmentAdapter {
  // MIME is unreliable for some containers (m4a), so also match by extension. No .webm extension:
  // it would claim video/webm files; real audio webm (MediaRecorder) always reports the audio/webm
  // MIME. .mp4 and .m4v stay off for the same reason; only .m4a is audio-only. Not the picker list:
  // this decides routing, and .3gp is in that list only so a dialog can offer a recording.
  accept = AUDIO_ATTACHMENT_ACCEPT;
  // Pending clip sizes by id; caps cover all clips in a message.
  private readonly attachmentSizes = new Map<string, number>();

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    const state = useChatRuntimeStore.getState();
    const checkpoint = state.params.checkpoint;
    const activeModel = state.models.find((m) => m.id === checkpoint);
    const modelLoaded = !!checkpoint && !state.modelLoading;
    let unavailableReason: string | null = null;
    if (modelLoaded && !activeModel?.hasAudioInput) {
      // A connected provider's model has no row in `models`, so without the parse this named it by its
      // raw `external::` id (#8405).
      const label =
        activeModel?.name ||
        externalModelLabel(checkpoint) ||
        checkpoint ||
        "Current model";
      unavailableReason = `${label} cannot accept audio. Load an audio-input model before attaching audio files.`;
    }
    if (unavailableReason) {
      toast.error(unavailableReason);
      throw new Error(unavailableReason);
    }
    // A staged store clip would be dropped if sent alongside attachments.
    if (state.pendingAudioBase64) {
      const stagedReason =
        "Send or remove the staged audio clip before attaching more audio.";
      toast.error(stagedReason, { id: AUDIO_ADD_TOAST_ID });
      throw new Error(stagedReason);
    }
    let totalSize = 0;
    for (const size of this.attachmentSizes.values()) totalSize += size;
    const addReason = getAudioAddError(
      this.attachmentSizes.size,
      totalSize,
      file.size,
      maxAudioFilesFor(modelLoaded ? activeModel : undefined),
    );
    if (addReason) {
      // Shared id so a large batch toasts once.
      toast.error(addReason, { id: AUDIO_ADD_TOAST_ID });
      throw new Error(addReason);
    }

    const id = newAttachmentId();
    this.attachmentSizes.set(id, file.size);
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
      this.attachmentSizes.delete(attachment.id);
    }
  }

  remove(attachment: Attachment): Promise<void> {
    this.attachmentSizes.delete(attachment.id);
    return Promise.resolve();
  }
}
