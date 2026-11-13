// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  type AttachmentAudioPart,
  type AttachmentPreviewKind,
  type AttachmentVideoPart,
  selectAttachmentSource,
} from "@/components/assistant-ui/attachment-selection";
import { useAuiState } from "@assistant-ui/react";
import { useEffect, useState } from "react";
import { useShallow } from "zustand/shallow";

export type { AttachmentPreviewKind };

export type AttachmentSource = {
  kind: AttachmentPreviewKind;
  name: string;
  contentType: string | undefined;
  file: File | undefined;
  src: string | undefined;
  audio: AttachmentAudioPart | undefined;
  video: AttachmentVideoPart | undefined;
  text: string | undefined;
  hasOriginal: boolean;
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

export const useAttachmentSource = (): AttachmentSource => {
  const source = useAuiState(useShallow(selectAttachmentSource));

  const fileSrc = useFileSrc(
    source.kind === "text" || source.kind === "document" ? undefined : source.file,
  );

  // audio and video pass through unjoined: a clip is up to MAX_VIDEO_SIZE of base64 and every tile mounts this hook
  return {
    kind: source.kind,
    name: source.name,
    contentType: source.contentType,
    file: source.file,
    src: fileSrc ?? source.image,
    audio: source.audio,
    video: source.video,
    text: source.text,
    hasOriginal: source.hasOriginal,
  };
};

export const useAttachmentImageSrc = (): string | undefined => {
  const source = useAttachmentSource();
  return source.kind === "image" ? source.src : undefined;
};
