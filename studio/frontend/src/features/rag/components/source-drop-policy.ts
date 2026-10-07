// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TEXT_ATTACHMENT_EXTENSIONS } from "../../chat/text-attachment-accept.ts";
import { RAG_UPLOAD_ACCEPT } from "../types/rag.ts";

// Backend SOURCE_TEXT_EXTS mirrors this list.
export const RAG_SOURCE_UPLOAD_ACCEPT = [
  ...new Set([...RAG_UPLOAD_ACCEPT.split(","), ...TEXT_ATTACHMENT_EXTENSIONS]),
].join(",");

const ACCEPTED_UPLOAD_EXTS = new Set(RAG_SOURCE_UPLOAD_ACCEPT.split(","));

// `accept` only filters the picker; a drop can carry anything.
export function isSupportedSourceName(name: string): boolean {
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return false;
  return ACCEPTED_UPLOAD_EXTS.has(name.slice(dot).toLowerCase());
}

export function partitionSupported<T>(
  entries: T[],
  nameOf: (entry: T) => string,
): { supported: T[]; unsupported: string[] } {
  const supported: T[] = [];
  const unsupported: string[] = [];
  for (const entry of entries) {
    const name = nameOf(entry);
    if (isSupportedSourceName(name)) supported.push(entry);
    else unsupported.push(name);
  }
  return { supported, unsupported };
}
