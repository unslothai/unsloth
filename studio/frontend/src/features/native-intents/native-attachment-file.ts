// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { readNativeAttachmentFile } from "./api";
import type { NativeIntent } from "./types";

export async function nativeAttachmentIntentToFile(
  intent: NativeIntent,
): Promise<File> {
  const payload = await readNativeAttachmentFile(intent.path.token);
  const binary = globalThis.atob(payload.base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new File([bytes], payload.name, {
    type: payload.mimeType,
    lastModified: intent.path.modifiedMs ?? Date.now(),
  });
}
