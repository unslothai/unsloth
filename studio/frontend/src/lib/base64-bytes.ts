// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type FromBase64 = (base64: string) => Uint8Array<ArrayBuffer>;

// Decodes the base64 Tauri hands back for native attachments. The atob +
// charCodeAt loop costs ~60 ms per 20 MiB on the UI thread; the native
// Uint8Array.fromBase64 (Chromium 140+, Safari 18.2+) does the same work in
// a fraction of that. Anything fromBase64 rejects goes through atob so callers
// still see atob's InvalidCharacterError.
export function decodeBase64Bytes(base64: string): Uint8Array<ArrayBuffer> {
  const fromBase64 = (Uint8Array as unknown as { fromBase64?: FromBase64 })
    .fromBase64;
  if (typeof fromBase64 === "function") {
    try {
      return fromBase64.call(Uint8Array, base64);
    } catch {
      // Fall through to atob for its error.
    }
  }
  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}
