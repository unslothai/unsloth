// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type FromBase64 = (base64: string) => Uint8Array<ArrayBuffer>;

// Inputs fromBase64 rejects go through atob so callers keep its InvalidCharacterError.
export function decodeBase64Bytes(base64: string): Uint8Array<ArrayBuffer> {
  const fromBase64 = (Uint8Array as unknown as { fromBase64?: FromBase64 })
    .fromBase64;
  if (typeof fromBase64 === "function") {
    try {
      return fromBase64.call(Uint8Array, base64);
    } catch {
      // Fall through to atob.
    }
  }
  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}
