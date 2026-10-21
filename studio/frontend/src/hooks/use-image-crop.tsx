// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactNode, useCallback, useMemo, useState } from "react";

import { ReferenceImageEditor } from "@/components/image-crop-editor";
import {
  type CropRect,
  type StagedReferenceImage,
  stageReferenceImage,
} from "@/lib/image-crop";

// Uploads keyed by their crop, outside any picker: a workflow switch remounts it, and a re-crop must start from the upload.
const cropOrigins = new Map<string, StagedReferenceImage>();
const MAX_CROP_ORIGINS = 8;

function rememberCrop(entry: StagedReferenceImage) {
  cropOrigins.delete(entry.dataUrl);
  cropOrigins.set(entry.dataUrl, entry);
  for (const key of cropOrigins.keys()) {
    if (cropOrigins.size <= MAX_CROP_ORIGINS) break;
    cropOrigins.delete(key);
  }
}

export function useImageCrop(
  value: string | null,
  onChange: (dataUrl: string) => void,
  subject: string,
): { openCrop: () => void; cropDialog: ReactNode } {
  // Keyed to the value it opened on, so a replaced or removed image closes it.
  const [cropFor, setCropFor] = useState<string | null>(null);
  const picture = useMemo(
    () => (value ? (cropOrigins.get(value) ?? stageReferenceImage(value)) : null),
    [value],
  );
  const apply = useCallback(
    (dataUrl: string, crop: CropRect | null) => {
      if (!picture) return;
      rememberCrop({ originalDataUrl: picture.originalDataUrl, dataUrl, crop });
      onChange(dataUrl);
    },
    [onChange, picture],
  );
  const openCrop = useCallback(() => setCropFor(value), [value]);
  const cropDialog =
    value !== null && cropFor === value ? (
      <ReferenceImageEditor
        open={true}
        picture={picture}
        subject={subject}
        onOpenChange={(open) => setCropFor(open ? value : null)}
        onApply={apply}
      />
    ) : null;
  return { openCrop, cropDialog };
}
