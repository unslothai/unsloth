// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Apart from file-kind.ts, which pulls in the chat feature, so zoom.ts can use it.

export type Media = "image" | "video" | "audio";

export function mediaKind(name: string, contentType: string): Media | null {
  if (/^image\//i.test(contentType) || /\.(png|jpe?g|gif|webp|avif|bmp|ico)$/i.test(name)) return "image";
  if (/^video\//i.test(contentType) || /\.(mp4|webm|mov|m4v|ogv)$/i.test(name)) return "video";
  if (/^audio\//i.test(contentType) || /\.(mp3|wav|ogg|oga|flac|m4a|aac|opus)$/i.test(name)) return "audio";
  return null;
}
