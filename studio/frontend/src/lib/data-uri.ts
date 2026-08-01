// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only a trailing `;base64` marks the payload as base64. A segment named
// `base64` anywhere else is an ordinary parameter, so
// `data:text/plain;base64;charset=utf-8,SGVsbG8=` stays literal text.
const DATA_URI_BASE64_RE = /;[ \t]*base64[ \t]*$/i;
const PERCENT_ESCAPE_RE = /%([0-9a-f]{2})/gi;
const DEFAULT_MIME_TYPE = "text/plain;charset=US-ASCII";
const PERCENT = 0x25;
// The URL parser normalises its input before parsing anything: it removes any
// leading and trailing C0 control or space, then removes every ASCII tab and
// newline. So ` data:text/plain;base64\n,...` really is base64. Only raw
// characters go; an escaped %0A or %20 is payload.
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
  // Testing first keeps the common case from copying a multi-megabyte string.
  return HAS_URL_WHITESPACE_RE.test(trimmed)
    ? trimmed.replace(URL_WHITESPACE_RE, "")
    : trimmed;
}

export interface DecodedDataUri {
  bytes: Uint8Array;
  mimeType: string;
}

/** URL schemes are case-insensitive, so `DATA:image/png;...` is a data URI. */
export function isDataUri(url: string): boolean {
  // Matched in place rather than by normalising a fixed-size prefix: any
  // number of leading controls is legal, and a window large enough for all of
  // them would be a guess. Cost is the leading run plus five characters.
  let index = 0;
  while (index < url.length && url.charCodeAt(index) <= 0x20) {
    index += 1;
  }
  for (const expected of DATA_SCHEME) {
    // Tabs and newlines are removed anywhere, the scheme included. A space is
    // not, so `da ta:` stays invalid, which is what all three engines do.
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

export function decodeDataUri(rawDataUri: string): DecodedDataUri {
  const dataUri = normalizeUrl(rawDataUri);
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
