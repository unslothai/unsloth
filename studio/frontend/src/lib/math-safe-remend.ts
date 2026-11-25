// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isWithinCodeBlock,
  isWithinMathBlock,
  type RemendHandler,
  type RemendOptions,
} from "remend";
import { codeSpans } from "./markdown-code-spans.ts";
import { findCodeBlockRegions } from "./latex.ts";

// Same pattern as remend's htmlTags pass, which also cuts a `<` inside math like `x_{<i}`.
const TRAILING_TAG_RE = /<[a-zA-Z/][^>]*$/;

function maskCode(text: string): string {
  const regions: Array<[number, number]> = [
    ...findCodeBlockRegions(text),
    ...codeSpans(text).map((span): [number, number] => [span.start, span.end]),
  ];
  if (regions.length === 0) return text;
  regions.sort((a, b) => a[0] - b[0]);
  let out = "";
  let cursor = 0;
  for (const [start, end] of regions) {
    if (end <= cursor) continue;
    const from = Math.max(start, cursor);
    out +=
      text.slice(cursor, from) + " ".repeat(Math.min(end, text.length) - from);
    cursor = end;
  }
  return out + text.slice(cursor);
}

export function stripIncompleteHtmlTag(text: string): string {
  const match = TRAILING_TAG_RE.exec(text);
  if (!match || isWithinCodeBlock(text, match.index)) return text;
  const head = text.slice(0, match.index);
  if (/[$\\]/.test(head) && isWithinMathBlock(maskCode(head), head.length)) {
    return text;
  }
  return head.trimEnd();
}

const MATH_SAFE_HTML_TAGS: RemendHandler = {
  name: "mathSafeHtmlTags",
  handle: stripIncompleteHtmlTag,
  priority: 10,
};

const DEFAULT_OPTIONS: RemendOptions = {
  htmlTags: false,
  handlers: [MATH_SAFE_HTML_TAGS],
};
const wrapped = new WeakMap<RemendOptions, RemendOptions>();

export function mathSafeRemendOptions(options?: RemendOptions): RemendOptions {
  if (!options) return DEFAULT_OPTIONS;
  if (options.htmlTags === false) return options;
  let safe = wrapped.get(options);
  if (!safe) {
    safe = {
      ...options,
      htmlTags: false,
      handlers: [...(options.handlers ?? []), MATH_SAFE_HTML_TAGS],
    };
    wrapped.set(options, safe);
  }
  return safe;
}
