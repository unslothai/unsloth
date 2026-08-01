// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const DATA_URI_BASE64_RE = /(?:^|;)base64(?:;|$)/i;
const HEX_PAIR_RE = /^[0-9a-f]{2}$/i;

export interface DecodedDataUri {
  bytes: Uint8Array;
  mimeType: string;
}

/**
 * Percent-decode to raw octets, the way a browser's own data-URL parser does.
 *
 * decodeURIComponent() cannot be used here: it insists the escapes form valid
 * UTF-8, so a legitimate binary payload such as `data:audio/wav,%FF%00%80`
 * throws URIError instead of yielding [255, 0, 128]. Percent-decoding is
 * byte-oriented, and an invalid escape is left alone rather than rejected.
 */
function percentDecodeOctets(data: string): Uint8Array {
  const encoder = new TextEncoder();
  const bytes: number[] = [];
  for (let index = 0; index < data.length; ) {
    if (
      data[index] === "%" &&
      HEX_PAIR_RE.test(data.slice(index + 1, index + 3))
    ) {
      bytes.push(Number.parseInt(data.slice(index + 1, index + 3), 16));
      index += 3;
      continue;
    }
    // Not an escape: emit the character's own UTF-8 bytes. Iterating by code
    // point keeps surrogate pairs intact.
    const character = String.fromCodePoint(data.codePointAt(index) as number);
    for (const byte of encoder.encode(character)) {
      bytes.push(byte);
    }
    index += character.length;
  }
  return Uint8Array.from(bytes);
}

export function decodeDataUri(dataUri: string): DecodedDataUri {
  const separator = dataUri.indexOf(",");
  if (!dataUri.startsWith("data:") || separator < 0) {
    throw new Error("Invalid data URI.");
  }
  const metadata = dataUri.slice(5, separator);
  const data = dataUri.slice(separator + 1);
  const mimeType = metadata.split(";", 1)[0] || "text/plain;charset=US-ASCII";
  // Percent-decoding comes first either way: a base64 payload may itself carry
  // escapes, so `data:audio/wav;base64,SGVsbG8%3D` has to become `SGVsbG8=`
  // before atob() sees it. Browsers decode that URI; passing the raw string
  // through would throw InvalidCharacterError on the %.
  const decoded = percentDecodeOctets(data);
  if (!DATA_URI_BASE64_RE.test(metadata)) {
    return { bytes: decoded, mimeType };
  }
  let payload = "";
  for (const byte of decoded) {
    payload += String.fromCharCode(byte);
  }
  const binary = atob(payload);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return { bytes, mimeType };
}
