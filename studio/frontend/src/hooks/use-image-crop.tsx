// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactNode, useCallback, useMemo, useState } from "react";

import { ReferenceImageEditor } from "@/components/image-crop-editor";
import {
  type CropRect,
  type StagedReferenceImage,
  stageReferenceImage,
} from "@/lib/image-crop";

// Uploads keyed by the crop made from them, outside any picker: switching workflow remounts the
// picker, and a re-crop must still start from the upload. Bounded since entries hold data URLs.
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

/** Crop dialog state for one image value; `cropDialog` must be rendered by the caller. */
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
