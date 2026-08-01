// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only a trailing `;base64` marks the payload as base64. A segment named
// `base64` anywhere else is an ordinary parameter, so
// `data:text/plain;base64;charset=utf-8,SGVsbG8=` stays literal text.
const DATA_URI_BASE64_RE = /;[ \t]*base64[ \t]*$/i;
const PERCENT_ESCAPE_RE = /%([0-9a-f]{2})/gi;
const HEX_PAIR_RE = /^[0-9a-f]{2}$/i;
const DEFAULT_MIME_TYPE = "text/plain;charset=US-ASCII";

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
 *
 * Literal spans are encoded in one call rather than per character: payloads
 * here run to tens of megabytes, and a character at a time is seconds of
 * blocked UI.
 */
function percentDecodeOctets(data: string): Uint8Array {
  const encoder = new TextEncoder();
  if (!data.includes("%")) {
    return encoder.encode(data);
  }

  const chunks: Uint8Array[] = [];
  let total = 0;
  const push = (chunk: Uint8Array) => {
    chunks.push(chunk);
    total += chunk.length;
  };

  let literalStart = 0;
  let index = 0;
  while (index < data.length) {
    if (
      data[index] !== "%" ||
      !HEX_PAIR_RE.test(data.slice(index + 1, index + 3))
    ) {
      index += 1;
      continue;
    }
    if (index > literalStart) {
      push(encoder.encode(data.slice(literalStart, index)));
    }
    const octets: number[] = [];
    while (
      data[index] === "%" &&
      HEX_PAIR_RE.test(data.slice(index + 1, index + 3))
    ) {
      octets.push(Number.parseInt(data.slice(index + 1, index + 3), 16));
      index += 3;
    }
    push(Uint8Array.from(octets));
    literalStart = index;
  }
  if (data.length > literalStart) {
    push(encoder.encode(data.slice(literalStart)));
  }

  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
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
  if (!dataUri.startsWith("data:") || separator < 0) {
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
