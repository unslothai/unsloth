// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CHAT_IMAGE_EXTENSIONS } from "../chat/image-normalize.ts";
import {
  OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS,
  isOpenDocumentAttachmentName,
  RTF_ATTACHMENT_EXTENSIONS,
  isRtfAttachmentName,
  TOOL_ONLY_ATTACHMENT_EXTENSIONS,
  isToolOnlyAttachmentName,
} from "../chat/open-document-accept.ts";
import {
  TEXT_ATTACHMENT_EXTENSIONS,
  isTextAttachmentName,
} from "../chat/text-attachment-accept.ts";
import { RAG_UPLOAD_ACCEPT } from "../rag/types/rag.ts";
import { MAX_AUDIO_FILES } from "../../lib/audio-utils.ts";

const DOC_EXTS = RAG_UPLOAD_ACCEPT.split(",").map((ext) =>
  ext.trim().toLowerCase(),
);

// RAG types are filtered out, so a dropped .txt/.md keeps being indexed.
const TEXT_EXTS = TEXT_ATTACHMENT_EXTENSIONS.map((ext) =>
  ext.toLowerCase(),
).filter((ext) => !DOC_EXTS.includes(ext));

export function isComposerAttachmentName(path: string): boolean {
  return (
    isOpenDocumentAttachmentName(path) ||
    isRtfAttachmentName(path) ||
    isToolOnlyAttachmentName(path) ||
    isTextDropName(path)
  );
}

function isTextDropName(path: string): boolean {
  const name = nativeFileName(path).toLowerCase();
  if (!name.includes(".") && isTextAttachmentName(name)) {
    return true;
  }
  const dot = name.lastIndexOf(".");
  return dot > 0 && TEXT_EXTS.includes(name.slice(dot));
}

export const CHAT_IMAGE_DROP_ACCEPT = CHAT_IMAGE_EXTENSIONS;

const IMAGE_EXTS = CHAT_IMAGE_DROP_ACCEPT.split(",").map((ext) =>
  ext.trim().toLowerCase(),
);

/** Keep in sync with `audio-attachment-adapter.ts` `accept`. */
export const CHAT_AUDIO_DROP_ACCEPT =
  ".wav,.mp3,.m4a,.ogg,.oga,.opus,.flac,.aac,.aiff,.aif,.aifc,.caf,.wma,.amr,.mp2";

const AUDIO_EXTS = CHAT_AUDIO_DROP_ACCEPT.split(",").map((ext) =>
  ext.trim().toLowerCase(),
);

/** Keep in sync with `native_path_policy.rs` `VIDEO_ATTACHMENT_EXTS` (what ffmpeg reads). */
export const CHAT_VIDEO_DROP_ACCEPT =
  ".mp4,.m4v,.mov,.webm,.mkv,.avi,.mpg,.mpeg,.wmv,.flv,.3gp,.ogv,.m2ts";

const VIDEO_EXTS = CHAT_VIDEO_DROP_ACCEPT.split(",").map((ext) =>
  ext.trim().toLowerCase(),
);

export const SUPPORTED_DROP_HINT = `Supported files: ${RAG_UPLOAD_ACCEPT}, ${OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS}, ${RTF_ATTACHMENT_EXTENSIONS}, source and text files, ${TOOL_ONLY_ATTACHMENT_EXTENSIONS} with Code on, ${CHAT_IMAGE_DROP_ACCEPT}, up to ${MAX_AUDIO_FILES} of ${CHAT_AUDIO_DROP_ACCEPT}, one of ${CHAT_VIDEO_DROP_ACCEPT}, or a single .gguf model.`;

export function nativeFileName(path: string): string {
  const segments = path.split(/[\\/]/);
  return segments[segments.length - 1] || path;
}

function hasExt(path: string, ext: string): boolean {
  return path.toLowerCase().endsWith(ext);
}

export type NativeDropClass =
  | { kind: "none" }
  | { kind: "model"; path: string }
  | { kind: "docs"; paths: string[] }
  | { kind: "images"; paths: string[] }
  | { kind: "audio"; paths: string[] }
  | { kind: "video"; paths: string[] }
  | {
      kind: "attach";
      docs: string[];
      images: string[];
      audio: string[];
      video: string[];
    }
  | { kind: "unsupported" };

export function classifyDropPaths(paths: string[]): NativeDropClass {
  if (paths.length === 0) return { kind: "none" };
  const ggufs = paths.filter((path) => hasExt(path, ".gguf"));
  // A batch of models is ambiguous, so it is not a drop target.
  if (ggufs.length > 0) {
    return paths.length === 1 && ggufs.length === 1
      ? { kind: "model", path: ggufs[0] }
      : { kind: "unsupported" };
  }
  const docs = paths.filter(
    (path) =>
      DOC_EXTS.some((ext) => hasExt(path, ext)) ||
      isComposerAttachmentName(path),
  );
  const images = paths.filter((path) =>
    IMAGE_EXTS.some((ext) => hasExt(path, ext)),
  );
  const audio = paths.filter((path) =>
    AUDIO_EXTS.some((ext) => hasExt(path, ext)),
  );
  const video = paths.filter((path) =>
    VIDEO_EXTS.some((ext) => hasExt(path, ext)),
  );
  if (
    docs.length + images.length + audio.length + video.length !==
    paths.length
  ) {
    return { kind: "unsupported" };
  }
  // Over the per-message cap a batch would only partly attach.
  if (audio.length > MAX_AUDIO_FILES) {
    return { kind: "unsupported" };
  }
  // One video expands into many frames, so a batch would blow the context.
  if (video.length > 1) {
    return { kind: "unsupported" };
  }
  if (
    docs.length === 0 &&
    images.length === 0 &&
    audio.length === 0 &&
    video.length === 0
  ) {
    return { kind: "none" };
  }
  const kinds = [docs, images, audio, video].filter(
    (group) => group.length > 0,
  );
  if (kinds.length === 1) {
    if (docs.length > 0) return { kind: "docs", paths: docs };
    if (images.length > 0) return { kind: "images", paths: images };
    if (audio.length > 0) return { kind: "audio", paths: audio };
    return { kind: "video", paths: video };
  }
  return { kind: "attach", docs, images, audio, video };
}
