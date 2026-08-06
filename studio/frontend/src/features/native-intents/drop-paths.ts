// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { RAG_UPLOAD_ACCEPT } from "../rag/types/rag.ts";

const DOC_EXTS = RAG_UPLOAD_ACCEPT.split(",").map((ext) => ext.trim().toLowerCase());

/** Vision chat attachments; keep in sync with `shared-composer` `IMAGE_ACCEPT`. */
export const CHAT_IMAGE_DROP_ACCEPT = ".jpg,.jpeg,.png,.webp,.gif";

const IMAGE_EXTS = CHAT_IMAGE_DROP_ACCEPT.split(",").map((ext) => ext.trim().toLowerCase());
const ATTACH_EXTS = [...DOC_EXTS, ...IMAGE_EXTS];

/** What the window actually takes, for the rejection toast and the overlay. */
export const SUPPORTED_DROP_HINT = `Supported files: ${RAG_UPLOAD_ACCEPT}, ${CHAT_IMAGE_DROP_ACCEPT}, or a single .gguf model.`;

function hasExt(path: string, ext: string): boolean {
  return path.toLowerCase().endsWith(ext);
}

export type NativeDropClass =
  | { kind: "none" }
  | { kind: "model"; path: string }
  | { kind: "docs"; paths: string[] }
  | { kind: "unsupported" };

/** What a native drag payload is, before any of it is registered with Rust. */
export function classifyDropPaths(paths: string[]): NativeDropClass {
  if (paths.length === 0) return { kind: "none" };
  const ggufs = paths.filter((path) => hasExt(path, ".gguf"));
  // One model loads; a batch of models is ambiguous, so it isn't a drop target.
  if (ggufs.length > 0) {
    return paths.length === 1 && ggufs.length === 1
      ? { kind: "model", path: ggufs[0] }
      : { kind: "unsupported" };
  }
  const attachments = paths.filter((path) =>
    ATTACH_EXTS.some((ext) => hasExt(path, ext)),
  );
  if (attachments.length === paths.length) return { kind: "docs", paths: attachments };
  return { kind: "unsupported" };
}
