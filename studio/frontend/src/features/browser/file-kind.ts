// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { documentKind, isMarkdown } from "@/components/file-viewer";
import { attachmentTextLanguage } from "@/features/chat";
import { mediaKind } from "./media-kind";

export const HTML_NAME = /\.(html?|xhtml)$/i;
const HTML_TYPE = /^(text\/html|application\/xhtml\+xml)\b/i;
export const TEXT_TYPE = /^(text\/|application\/(json|xml|javascript|x-yaml|yaml|toml|x-sh|sql)\b)/i;
export const TEXT_NAME =
  /\.(txt|log|md|markdown|mdx|json|jsonl|ya?ml|toml|ini|cfg|conf|csv|tsv|xml|svg|py|ipynb|js|mjs|cjs|ts|tsx|jsx|css|scss|sh|bash|zsh|rs|go|java|kt|c|cc|cpp|h|hpp|cs|rb|php|swift|sql|r|lua|pl|tex)$/i;

export { type Media, mediaKind } from "./media-kind";

export type TextFileKind = "html" | "markdown" | "code" | "text";

export function isHtml(name: string, contentType: string): boolean {
  return HTML_NAME.test(name) || HTML_TYPE.test(contentType);
}

export function textFileKind(name: string, contentType: string, plainText = false): TextFileKind | null {
  if (plainText) return "text";
  if (mediaKind(name, contentType) || documentKind(name, contentType)) return null;
  if (!(TEXT_TYPE.test(contentType) || TEXT_NAME.test(name) || HTML_NAME.test(name) || !contentType)) return null;
  if (isHtml(name, contentType)) return "html";
  if (isMarkdown(name, contentType)) return "markdown";
  return attachmentTextLanguage(name, null) ? "code" : "text";
}

// Types a blob URL shows without running anything. Anchored, so SVG or smuggled params fail.
const SAFE_TAB_TYPE =
  /^(application\/pdf|image\/(png|jpe?g|gif|webp|avif|bmp)|video\/[\w.+-]+|audio\/[\w.+-]+|text\/plain)\s*(;|$)/i;

/** Type to open a file as in the user's browser, or null if unsafe: a blob URL has Studio's origin. */
export function browserTabType(name: string, contentType: string): string | null {
  const kind = textFileKind(name, contentType);
  if (kind) return kind === "html" ? null : "text/plain";
  return SAFE_TAB_TYPE.test(contentType) ? contentType : null;
}

export function canShowFile(name: string, contentType: string): boolean {
  if (documentKind(name, contentType) || mediaKind(name, contentType)) return true;
  return TEXT_TYPE.test(contentType) || TEXT_NAME.test(name) || HTML_NAME.test(name) || !contentType;
}
