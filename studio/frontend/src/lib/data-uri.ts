// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only a trailing `;base64` counts; elsewhere it is an ordinary parameter.
const DATA_URI_BASE64_RE = /;[ \t]*base64[ \t]*$/i;
const PERCENT_ESCAPE_RE = /%([0-9a-f]{2})/gi;
const DEFAULT_MIME_TYPE = "text/plain;charset=US-ASCII";
const PERCENT = 0x25;
// The URL parser strips leading/trailing C0 and space and every tab/newline; escaped %0A stays payload.
const URL_WHITESPACE_RE = /[\t\n\r]/g;
const HAS_URL_WHITESPACE_RE = /[\t\n\r]/;
const DATA_SCHEME = "data:";

function isUrlWhitespace(code: number): boolean {
  return code === 0x09 || code === 0x0a || code === 0x0d;
}

function trimUrlControls(url: string): string {
  let start = 0;
  let end = url.length;
  while (start < end && url.charCodeAt(start) <= 0x20) {
    start += 1;
  }
  while (end > start && url.charCodeAt(end - 1) <= 0x20) {
    end -= 1;
  }
  return start === 0 && end === url.length ? url : url.slice(start, end);
}

function normalizeUrl(url: string): string {
  const trimmed = trimUrlControls(url);
  // Test first so the common case does not copy a multi-megabyte string.
  return HAS_URL_WHITESPACE_RE.test(trimmed)
    ? trimmed.replace(URL_WHITESPACE_RE, "")
    : trimmed;
}

export interface DecodedDataUri {
  bytes: Uint8Array;
  mimeType: string;
}

/** URL schemes are case-insensitive. */
export function isDataUri(url: string): boolean {
  // Matched in place: any number of leading controls is legal.
  let index = 0;
  while (index < url.length && url.charCodeAt(index) <= 0x20) {
    index += 1;
  }
  for (const expected of DATA_SCHEME) {
    // Tabs and newlines go anywhere, but a space does not, as in all three engines.
    while (index < url.length && isUrlWhitespace(url.charCodeAt(index))) {
      index += 1;
    }
    if (index >= url.length || url[index].toLowerCase() !== expected) {
      return false;
    }
    index += 1;
  }
  return true;
}

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

/** Byte-oriented like the browser's parser: decodeURIComponent throws on non-UTF-8 binary.
 * One growable buffer, since payloads reach tens of megabytes. */
function percentDecodeOctets(data: string): Uint8Array {
  const encoder = new TextEncoder();
  if (!data.includes("%")) {
    return encoder.encode(data);
  }

  // Escapes shrink and ASCII is 1:1, so the source length fits all but non-ASCII payloads.
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
    // At most 3 UTF-8 bytes per UTF-16 unit.
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

export function decodeDataUri(rawDataUri: string): DecodedDataUri {
  const dataUri = normalizeUrl(rawDataUri);
  const separator = dataUri.indexOf(",");
  if (!isDataUri(dataUri) || separator < 0) {
    throw new Error("Invalid data URI.");
  }
  const metadata = dataUri.slice(5, separator);
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
  // Without a slash it is not a media type; fall back to the RFC 2397 default.
  const mimeType = essence.includes("/") ? essence : DEFAULT_MIME_TYPE;

  if (!isBase64) {
    return { bytes: percentDecodeOctets(data), mimeType };
  }
  // A base64 payload may carry its own escapes (`SGVsbG8%3D`).
  const payload = data.includes("%")
    ? data.replace(PERCENT_ESCAPE_RE, (_match, hex: string) =>
        String.fromCharCode(Number.parseInt(hex, 16)),
      )
    : data;
  return { bytes: base64ToBytes(payload), mimeType };
}
