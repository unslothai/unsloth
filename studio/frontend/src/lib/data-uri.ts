// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const DATA_URI_BASE64_RE = /(?:^|;)base64(?:;|$)/i;

export interface DecodedDataUri {
  bytes: Uint8Array;
  mimeType: string;
}

export function decodeDataUri(dataUri: string): DecodedDataUri {
  const separator = dataUri.indexOf(",");
  if (!dataUri.startsWith("data:") || separator < 0) {
    throw new Error("Invalid data URI.");
  }
  const metadata = dataUri.slice(5, separator);
  const data = dataUri.slice(separator + 1);
  const mimeType = metadata.split(";", 1)[0] || "text/plain;charset=US-ASCII";
  if (!DATA_URI_BASE64_RE.test(metadata)) {
    return {
      bytes: new TextEncoder().encode(decodeURIComponent(data)),
      mimeType,
    };
  }
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return { bytes, mimeType };
}
