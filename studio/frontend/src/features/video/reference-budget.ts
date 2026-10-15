// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUDIO_PICKER_ACCEPT, isAudioAttachmentFile } from "../../lib/audio-utils.ts";
import { VIDEO_ACCEPT, isVideoFile } from "../../lib/video-utils.ts";

export const MAX_H3_REFERENCES = 12;

export function hasReferenceCapacity(
  images: number,
  videos: number,
  audios: number,
): boolean {
  return images + videos + audios < MAX_H3_REFERENCES;
}

export type ReferenceKind = "video" | "audio";

/** Reserved before FileReader runs, since the OS-reported MIME length is unknown. */
const DATA_URL_HEADER_BUDGET = 256;

/** The models/inference.py caps (96 MiB video, 32 MiB audio) bound the base64 STRING; subtract
 * the header and floor to a multiple of 3 so the encode has no padding. */
function rawLimitFor(base64Cap: number): number {
  return Math.floor(((base64Cap - DATA_URL_HEADER_BUDGET) * 3) / 4 / 3) * 3;
}

export const MAX_REFERENCE_BYTES: Record<ReferenceKind, number> = {
  video: rawLimitFor(96 * 1024 * 1024),
  audio: rawLimitFor(32 * 1024 * 1024),
};

/** Extensions too: browsers report "" for wma, amr, caf and others. */
export const REFERENCE_PICKER_ACCEPT: Record<ReferenceKind, string> = {
  video: `${VIDEO_ACCEPT},.ts,.mts`,
  audio: AUDIO_PICKER_ACCEPT,
};

/** Must not be narrower than the dialog's list; shown verbatim when a file is refused. */
export const REFERENCE_DROP_ACCEPT: Record<ReferenceKind, string> = {
  video: extensionsOf(REFERENCE_PICKER_ACCEPT.video),
  audio: extensionsOf(REFERENCE_PICKER_ACCEPT.audio),
};

function extensionsOf(accept: string): string {
  return accept
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.startsWith("."))
    .join(",");
}

export function referenceFileRejection(
  kind: ReferenceKind,
  file: { type: string; size: number; name?: string },
): string | null {
  // Name as well as MIME, since the browser can report an empty type.
  const named = { type: file.type, name: file.name ?? "" };
  const matches = kind === "video" ? isVideoFile(named) : isAudioAttachmentFile(named);
  if (!matches) {
    return `Please choose ${kind === "video" ? "a video" : "an audio"} file`;
  }
  if (file.size > MAX_REFERENCE_BYTES[kind]) {
    const limitMb = Math.round(MAX_REFERENCE_BYTES[kind] / (1024 * 1024));
    return `This ${kind} is too large (limit ${limitMb} MB)`;
  }
  return null;
}

/** Size-checked before FileReader: a data URL costs ~2.33x the file in memory. */
export function readReferenceFile(
  kind: ReferenceKind,
  file: File | undefined | null,
  handlers: {
    onLoaded: (dataUrl: string | null) => void;
    onError: (message: string) => void;
  },
): void {
  if (!file) return;
  const rejection = referenceFileRejection(kind, file);
  if (rejection !== null) {
    handlers.onError(rejection);
    return;
  }
  const reader = new FileReader();
  reader.onload = () =>
    handlers.onLoaded(typeof reader.result === "string" ? reader.result : null);
  reader.onerror = () => handlers.onError(`Could not read the ${kind} file`);
  reader.readAsDataURL(file);
}

export interface ReferenceSelectionClaim {
  isCurrent(): boolean;
}

export interface ReferenceSelectionGate {
  begin(): ReferenceSelectionClaim;
  invalidate(): void;
  mount(): () => void;
}

export function createReferenceSelectionGate(): ReferenceSelectionGate {
  let revision = 0;
  let live = true;
  return {
    begin() {
      revision += 1;
      const claimed = revision;
      return { isCurrent: () => live && claimed === revision };
    },
    invalidate() {
      revision += 1;
    },
    mount() {
      live = true;
      return () => {
        live = false;
        // A StrictMode remount must not revive the previous mount's claim.
        revision += 1;
      };
    },
  };
}
