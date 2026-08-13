// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { attachmentAudioSrc, isAudioAttachment } from "@/features/chat";
import { useAuiState } from "@assistant-ui/react";
import { useEffect, useState } from "react";
import { useShallow } from "zustand/shallow";

export type AttachmentPreviewKind = "image" | "audio" | "text";

export type AttachmentSource = {
  kind: AttachmentPreviewKind;
  name: string;
  contentType: string | undefined;
  file: File | undefined;
  src: string | undefined;
  text: string | undefined;
};

const useFileSrc = (file: File | undefined): string | undefined => {
  const [objectUrl, setObjectUrl] = useState<string | undefined>(undefined);

  useEffect(() => {
    if (!file) {
      setObjectUrl(undefined);
      return;
    }
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return objectUrl;
};

// Resolves what a preview can show for the attachment in scope: a composer
// attachment still holds its File, a sent one only the content parts.
export const useAttachmentSource = (): AttachmentSource => {
  const source = useAuiState(
    useShallow(({ attachment }) => {
      const parts = attachment.content ?? [];
      const file = (attachment as { file?: File }).file;
      const contentType =
        file?.type ||
        (attachment as { contentType?: string }).contentType ||
        undefined;
      const audio = parts.find((part) => part.type === "audio")?.audio;
      const isImage = attachment.type === "image";
      const isAudio =
        !isImage &&
        (!!audio || isAudioAttachment(attachment.name, contentType));
      const text = parts
        .filter((part) => part.type === "text")
        .map((part) => part.text)
        .join("\n");
      return {
        kind: (isImage
          ? "image"
          : isAudio
            ? "audio"
            : "text") as AttachmentPreviewKind,
        name: attachment.name,
        contentType,
        file,
        contentSrc: isImage
          ? parts.find((part) => part.type === "image")?.image
          : audio
            ? attachmentAudioSrc(audio, contentType, attachment.name)
            : undefined,
        text: text || undefined,
      };
    }),
  );

  const fileSrc = useFileSrc(source.kind === "text" ? undefined : source.file);

  return {
    kind: source.kind,
    name: source.name,
    contentType: source.contentType,
    file: source.file,
    src: fileSrc ?? source.contentSrc,
    text: source.text,
  };
};

export const useAttachmentImageSrc = (): string | undefined => {
  const source = useAttachmentSource();
  return source.kind === "image" ? source.src : undefined;
};
