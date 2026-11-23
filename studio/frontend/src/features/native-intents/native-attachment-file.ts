// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { decodeBase64Bytes } from "@/lib/base64-bytes";
import { readNativeAttachmentFile } from "./api";
import type { NativeIntent } from "./types";

export async function nativeAttachmentIntentToFile(
  intent: NativeIntent,
): Promise<File> {
  const payload = await readNativeAttachmentFile(intent.path.token);
  const bytes = decodeBase64Bytes(payload.base64);
  return new File([bytes], payload.name, {
    type: payload.mimeType,
    lastModified: intent.path.modifiedMs ?? Date.now(),
  });
}
