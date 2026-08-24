// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  type AttachmentAudioPart,
  type AttachmentPreviewKind,
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
  const source = useAuiState(useShallow(selectAttachmentSource));

  const fileSrc = useFileSrc(source.kind === "text" ? undefined : source.file);

  // audio passes through unjoined: the payload is MAX_AUDIO_SIZE of base64 and every tile mounts this hook
  return {
    kind: source.kind,
    name: source.name,
    contentType: source.contentType,
    file: source.file,
    src: fileSrc ?? source.image,
    audio: source.audio,
    text: source.text,
  };
};

export const useAttachmentImageSrc = (): string | undefined => {
  const source = useAttachmentSource();
  return source.kind === "image" ? source.src : undefined;
};
