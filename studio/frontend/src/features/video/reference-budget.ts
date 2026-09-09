// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUDIO_PICKER_ACCEPT, isAudioAttachmentFile } from "../../lib/audio-utils.ts";
import { VIDEO_ACCEPT, isVideoFile } from "../../lib/video-utils.ts";

/** MiniMax-H3's combined budget across picture, video and standalone-audio references. */
export const MAX_H3_REFERENCES = 12;

export function hasReferenceCapacity(
  images: number,
  videos: number,
  audios: number,
): boolean {
  return images + videos + audios < MAX_H3_REFERENCES;
}

export type ReferenceKind = "video" | "audio";

/** Room for the `data:<mime>;base64,` prefix FileReader prepends. Reserved rather than measured,
 *  because the cap has to be known before the FileReader runs and the MIME string is whatever the
 *  OS reported: `video/mp4` makes 22 characters and the long-winded ones half as long again. 256
 *  covers any of them and costs 192 bytes of the advertised limit. */
const DATA_URL_HEADER_BUDGET = 256;

/** The largest raw file of each kind whose data URL still fits the backend's cap. The caps in
 *  models/inference.py bound the STRING, not the file: 96 MiB for a reference video and 32 MiB
 *  for its soundtrack. Base64 costs 4 characters per 3 bytes, so a plain three quarters is off by
 *  the header and a file at exactly that size was accepted and then 422'd. Floor to a multiple of
 *  3 as well, so the encode is exactly (n / 3) * 4 with no padding. */
function rawLimitFor(base64Cap: number): number {
  return Math.floor(((base64Cap - DATA_URL_HEADER_BUDGET) * 3) / 4 / 3) * 3;
}

export const MAX_REFERENCE_BYTES: Record<ReferenceKind, number> = {
  video: rawLimitFor(96 * 1024 * 1024),
  audio: rawLimitFor(32 * 1024 * 1024),
};

/** What a reference file dialog should offer, per kind. Extensions ride along
 *  with the MIME types: a browser answers "" for wma, amr, caf and several
 *  others, and `${kind}/*` alone greys those out of the dialog. The audio list
 *  is the picker one, .3gp included, because the picker reads a recording's
 *  tracks once it has the file and a clip is refused then. */
export const REFERENCE_PICKER_ACCEPT: Record<ReferenceKind, string> = {
  video: VIDEO_ACCEPT,
  audio: AUDIO_PICKER_ACCEPT,
};

/**
 * The same policy for the drop zone, extensions only.
 *
 * A drop zone filters on the name before the classifier can look at the file,
 * so a list narrower than the dialog's refuses what the button accepts. It
 * carries no MIME types because the zone matches names and puts this list
 * verbatim into the message it shows when it turns a file away.
 */
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

/** Why this file cannot be staged as a reference, or null when it can. */
export function referenceFileRejection(
  kind: ReferenceKind,
  file: { type: string; size: number; name?: string },
): string | null {
  // Name as well as MIME, matching the accept list above and the native drop:
  // the same recording chosen through the button used to be refused for the
  // empty type the browser gave it.
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

/** Read one reference file into the data URL the request carries. The size check happens BEFORE the
 *  FileReader exists, because a data URL costs roughly 2.33x the file in renderer memory and is
 *  built long before the backend's 422 can arrive. H3 reference clips are 2 to 15 seconds by
 *  spec, so a 15 second 4K phone clip clears the cap routinely. */
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

/** Create a latest-wins guard for asynchronous picker reads. */
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
