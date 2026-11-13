// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Split out of api-monitor-page.tsx so it runs under `node --test`.

export const PAYLOAD_PREVIEW_CHARS = 12_000;
// Above this a full render stalls the page; Copy still has the whole body.
export const PAYLOAD_SHOW_ALL_MAX_CHARS = 1_000_000;

const ENDS_IN_HIGH_SURROGATE = /[\uD800-\uDBFF]$/;
const STARTS_WITH_LOW_SURROGATE = /^[\uDC00-\uDFFF]/;

// Head and tail, as SGLang's request log cuts: agent prompts put the long system prompt
// first and the turn being answered last, so a head-only cut showed only instructions.
export function payloadPreview(
  body: string,
  limit: number = PAYLOAD_PREVIEW_CHARS,
): string {
  if (body.length <= limit) {
    return body;
  }
  const half = Math.floor(limit / 2);
  let head = body.slice(0, half);
  let tail = body.slice(body.length - half);
  if (ENDS_IN_HIGH_SURROGATE.test(head)) {
    head = head.slice(0, -1);
  }
  if (STARTS_WITH_LOW_SURROGATE.test(tail)) {
    tail = tail.slice(1);
  }
  const omitted = body.length - head.length - tail.length;
  return `${head}\n\n[... ${omitted.toLocaleString("en-US")} characters omitted ...]\n\n${tail}`;
}
