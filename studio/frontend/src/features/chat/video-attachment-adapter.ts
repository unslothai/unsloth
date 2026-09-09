// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fileToBase64 } from "@/lib/audio-utils";
import {
  VIDEO_ACCEPT,
  getVideoSizeError,
  videoMimeForFile,
} from "@/lib/video-utils";
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

/** Video shares the "Add photos & files" picker, like audio. llama-server samples the clip into
 *  frames with ffmpeg, so the container is forwarded untouched. */
export class VideoAttachmentAdapter implements AttachmentAdapter {
  accept = VIDEO_ACCEPT;
  private readonly attachmentIds = new Set<string>();

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    const state = useChatRuntimeStore.getState();
    const checkpoint = state.params.checkpoint;
    const activeModel = state.models.find((m) => m.id === checkpoint);
    const modelLoaded = !!checkpoint && !state.modelLoading;
    let unavailableReason: string | null = null;
    if (!modelLoaded) {
      // Mirror the image and audio gates: a failed load reads differently from no model picked.
      unavailableReason = state.lastModelLoadError
        ? "The last model failed to load. Check the server logs, then load a model before adding video."
        : "Load a model before adding video.";
    } else if (!activeModel?.hasVideoInput) {
      const label =
        activeModel?.name ||
        externalModelLabel(checkpoint) ||
        checkpoint ||
        "Current model";
      // Three causes land here and the server does not say which, so name all three: /props reports
      // video only when all of them line up.
      unavailableReason = `${label} cannot accept video. Video needs a GGUF model whose mmproj supports video, a llama.cpp build with video enabled, and ffmpeg installed on this machine.`;
    }
    if (unavailableReason) {
      toast.error(unavailableReason);
      throw new Error(unavailableReason);
    }
    const sizeReason = getVideoSizeError(file.size);
    if (sizeReason) {
      toast.error(sizeReason);
      throw new Error(sizeReason);
    }
    // One clip per message: a second would blow the context in frames alone.
    if (this.attachmentIds.size > 0) {
      const duplicateReason = "Only one video can be attached per message.";
      toast.error(duplicateReason);
      throw new Error(duplicateReason);
    }

    const id = newAttachmentId();
    this.attachmentIds.add(id);
    return {
      id,
      type: "file",
      name: file.name,
      contentType: videoMimeForFile(file),
      file,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    try {
      const data = await fileToBase64(attachment.file);
      return {
        id: attachment.id,
        type: "file",
        name: attachment.name,
        contentType: attachment.contentType,
        content: [
          {
            type: "file",
            filename: attachment.name,
            data,
            // Normalised at pick time: the extractor keys off this, and a browser that answered "" or
            // application/octet-stream for an mkv would otherwise cost the clip silently.
            mimeType: attachment.contentType || "video/mp4",
          },
        ],
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
