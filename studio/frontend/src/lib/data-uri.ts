// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only a trailing `;base64` marks the payload as base64. A segment named
// `base64` anywhere else is an ordinary parameter, so
// `data:text/plain;base64;charset=utf-8,SGVsbG8=` stays literal text.
const DATA_URI_BASE64_RE = /;[ \t]*base64[ \t]*$/i;
const PERCENT_ESCAPE_RE = /%([0-9a-f]{2})/gi;
const DEFAULT_MIME_TYPE = "text/plain;charset=US-ASCII";
const PERCENT = 0x25;

export interface DecodedDataUri {
  bytes: Uint8Array;
  mimeType: string;
}

/** URL schemes are case-insensitive, so `DATA:image/png;...` is a data URI. */
export function isDataUri(url: string): boolean {
  return url.slice(0, 5).toLowerCase() === "data:";
}

/** Hex digit value, or -1. Avoids allocating a substring per character. */
function hexValue(code: number): number {
  if (code >= 0x30 && code <= 0x39) {
    return code - 0x30;
  }
  if (code >= 0x61 && code <= 0x66) {
    return code - 0x57;
  }
  if (code >= 0x41 && code <= 0x46) {
    return code - 0x37;
  }
  return -1;
}

function escapeAt(data: string, index: number): number {
  if (data.charCodeAt(index) !== PERCENT) {
    return -1;
  }
  const high = hexValue(data.charCodeAt(index + 1));
  const low = hexValue(data.charCodeAt(index + 2));
  return high < 0 || low < 0 ? -1 : high * 16 + low;
}

/**
 * Percent-decode to raw octets, the way a browser's own data-URL parser does.
 *
 * decodeURIComponent() cannot be used here: it insists the escapes form valid
 * UTF-8, so a legitimate binary payload such as `data:audio/wav,%FF%00%80`
 * throws URIError instead of yielding [255, 0, 128]. Percent-decoding is
 * byte-oriented, and an invalid escape is left alone rather than rejected.
 *
 * Everything goes into one growable buffer. Payloads here reach tens of
 * megabytes, and a per-run or per-character allocation turns an encoded SVG
 * into seconds of blocked UI and hundreds of MiB of garbage.
 */
function percentDecodeOctets(data: string): Uint8Array {
  const encoder = new TextEncoder();
  if (!data.includes("%")) {
    return encoder.encode(data);
  }

  // Escapes shrink three characters to one byte and ASCII literals are 1:1, so
  // the source length already fits all but non-ASCII payloads.
  let out = new Uint8Array(data.length);
  let length = 0;

  const reserve = (extra: number) => {
    if (length + extra <= out.length) {
      return;
    }
    let capacity = Math.max(out.length, 1);
    while (capacity < length + extra) {
      capacity *= 2;
    }
    const grown = new Uint8Array(capacity);
    grown.set(out.subarray(0, length));
    out = grown;
  };

  let literalStart = 0;
  const flushLiteral = (end: number) => {
    if (end <= literalStart) {
      return;
    }
    const literal = data.slice(literalStart, end);
    // Worst case is 3 UTF-8 bytes per BMP character; a surrogate pair is 4
    // bytes for 2 characters, so this bound holds either way.
    reserve(literal.length * 3);
    length += encoder.encodeInto(literal, out.subarray(length)).written;
  };

  let index = 0;
  while (index < data.length) {
    if (escapeAt(data, index) < 0) {
      index += 1;
      continue;
    }
    flushLiteral(index);
    let octet = escapeAt(data, index);
    while (octet >= 0) {
      reserve(1);
      out[length] = octet;
      length += 1;
      index += 3;
      octet = escapeAt(data, index);
    }
    literalStart = index;
  }
  flushLiteral(data.length);

  return out.slice(0, length);
}

function base64ToBytes(payload: string): Uint8Array {
  const binary = atob(payload);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

export function decodeDataUri(dataUri: string): DecodedDataUri {
  const separator = dataUri.indexOf(",");
  if (!isDataUri(dataUri) || separator < 0) {
    throw new Error("Invalid data URI.");
  }
  const metadata = dataUri.slice(5, separator);
  // A fragment belongs to the URL, not to the payload.
  const fragment = dataUri.indexOf("#", separator + 1);
  const data = dataUri.slice(
    separator + 1,
    fragment < 0 ? undefined : fragment,
  );

  const isBase64 = DATA_URI_BASE64_RE.test(metadata);
  const essence = (
    isBase64 ? metadata.replace(DATA_URI_BASE64_RE, "") : metadata
  )
    .split(";", 1)[0]
    .trim();
  // Anything without a slash is not a media type, so `data:base64,...` falls
  // back to the RFC 2397 default rather than reporting `base64`.
  const mimeType = essence.includes("/") ? essence : DEFAULT_MIME_TYPE;

  if (!isBase64) {
    return { bytes: percentDecodeOctets(data), mimeType };
  }
  // A base64 payload may carry its own escapes, so `data:audio/wav;base64,
  // SGVsbG8%3D` has to become `SGVsbG8=` before atob() sees it. The base64
  // alphabet is ASCII, so decoding the escapes in place is enough.
  const payload = data.includes("%")
    ? data.replace(PERCENT_ESCAPE_RE, (_match, hex: string) =>
        String.fromCharCode(Number.parseInt(hex, 16)),
      )
    : data;
  return { bytes: base64ToBytes(payload), mimeType };
}
