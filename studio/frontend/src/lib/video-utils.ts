// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** llama-server decodes via ffmpeg. Extensions included because MIME is unreliable for mkv/mov. */
export const VIDEO_ACCEPT =
  "video/mp4,video/x-m4v,video/quicktime,video/webm,video/x-matroska,video/x-msvideo,video/mpeg,video/x-ms-wmv,video/x-flv,video/3gpp,video/ogg,video/mp2t,.mp4,.m4v,.mov,.webm,.mkv,.avi,.mpg,.mpeg,.wmv,.flv,.3gp,.ogv,.m2ts";

// Matches _MAX_VIDEO_B64_CHARS in the backend so the composer does not accept a refused clip.
const MAX_VIDEO_SIZE_MB = 64;
export const MAX_VIDEO_SIZE = MAX_VIDEO_SIZE_MB * 1024 * 1024;
export const MAX_VIDEO_SIZE_LABEL = `${MAX_VIDEO_SIZE_MB}MB`;

export function getVideoSizeError(size: number): string | null {
  return size > MAX_VIDEO_SIZE
    ? `Video size exceeds ${MAX_VIDEO_SIZE_LABEL} limit`
    : null;
}

// Mirrors the extension table in native_intents.rs (parity-tested).
const VIDEO_MIME_BY_EXTENSION: Record<string, string> = {
  ".mp4": "video/mp4",
  ".m4v": "video/x-m4v",
  ".mov": "video/quicktime",
  ".webm": "video/webm",
  ".mkv": "video/x-matroska",
  ".avi": "video/x-msvideo",
  ".mpg": "video/mpeg",
  ".mpeg": "video/mpeg",
  ".wmv": "video/x-ms-wmv",
  ".flv": "video/x-flv",
  ".3gp": "video/3gpp",
  ".ogv": "video/ogg",
  ".m2ts": "video/mp2t",
};

const VIDEO_EXTENSIONS = Object.keys(VIDEO_MIME_BY_EXTENSION);
const VIDEO_MIME_RE = /^video\//i;

/** The request builder drops non-^video/ parts, so trust the extension if the type is not video. */
export function videoMimeForFile(file: File): string {
  if (VIDEO_MIME_RE.test(file.type)) return file.type;
  const name = file.name.toLowerCase();
  for (const [ext, mime] of Object.entries(VIDEO_MIME_BY_EXTENSION)) {
    if (name.endsWith(ext)) return mime;
  }
  return "video/mp4";
}

export function isVideoFile(file: { name: string; type: string }): boolean {
  if (VIDEO_MIME_RE.test(file.type)) {
    return true;
  }
  // The extension fallback claims .3gp, which recordings share; a track read has already decided.
  if (/^audio\//i.test(file.type)) {
    return false;
  }
  const name = file.name.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => name.endsWith(ext));
}

/** Mirrors `bmff_box_payloads` in native_path_policy.rs. */
function bmffBoxPayloads(data: Uint8Array, wanted: string): Uint8Array[] {
  const payloads: Uint8Array[] = [];
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 0;
  while (data.length - offset >= 8) {
    const size32 = view.getUint32(offset);
    const type = String.fromCharCode(
      data[offset + 4]!,
      data[offset + 5]!,
      data[offset + 6]!,
      data[offset + 7]!,
    );
    let headerSize = 8;
    let boxSize = size32;
    if (size32 === 0) {
      boxSize = data.length - offset;
    } else if (size32 === 1) {
      if (data.length - offset < 16) break;
      const size64 = view.getBigUint64(offset + 8);
      if (size64 > BigInt(Number.MAX_SAFE_INTEGER)) break;
      headerSize = 16;
      boxSize = Number(size64);
    }
    if (boxSize < headerSize || boxSize > data.length - offset) break;
    if (type === wanted) {
      payloads.push(data.subarray(offset + headerSize, offset + boxSize));
    }
    offset += boxSize;
  }
  return payloads;
}

type BmffTracks = { audio: boolean; video: boolean };

function tracksInMoov(moov: Uint8Array, found: BmffTracks): void {
  for (const trak of bmffBoxPayloads(moov, "trak")) {
    for (const mdia of bmffBoxPayloads(trak, "mdia")) {
      for (const hdlr of bmffBoxPayloads(mdia, "hdlr")) {
        if (hdlr.length < 12) continue;
        const handler = String.fromCharCode(
          hdlr[8]!,
          hdlr[9]!,
          hdlr[10]!,
          hdlr[11]!,
        );
        if (handler === "soun") found.audio = true;
        else if (handler === "vide") found.video = true;
      }
    }
  }
}

/** Mirrors `is_audio_only_3gp` in native_path_policy.rs. */
export function isAudioOnly3gpBytes(raw: Uint8Array): boolean {
  const found: BmffTracks = { audio: false, video: false };
  for (const moov of bmffBoxPayloads(raw, "moov")) {
    tracksInMoov(moov, found);
  }
  return found.audio && !found.video;
}

// A track table is kilobytes; larger is not one, and reading it defeats the box walk.
const MAX_MOOV_BYTES = 8 * 1024 * 1024;
// Real containers have a handful of top-level boxes; thousands means malformed.
const MAX_TOP_LEVEL_BOXES = 64;

/** Reads only `moov` via slices, so classifying never holds the whole clip in memory. */
async function read3gpTracks(file: File): Promise<BmffTracks> {
  const found: BmffTracks = { audio: false, video: false };
  let offset = 0;
  for (let box = 0; box < MAX_TOP_LEVEL_BOXES && offset + 8 <= file.size; box++) {
    const header = new Uint8Array(
      await file.slice(offset, offset + 16).arrayBuffer(),
    );
    if (header.length < 8) break;
    const view = new DataView(
      header.buffer,
      header.byteOffset,
      header.byteLength,
    );
    const size32 = view.getUint32(0);
    const type = String.fromCharCode(
      header[4]!,
      header[5]!,
      header[6]!,
      header[7]!,
    );
    let headerSize = 8;
    let boxSize = size32;
    if (size32 === 0) {
      boxSize = file.size - offset;
    } else if (size32 === 1) {
      if (header.length < 16) break;
      const size64 = view.getBigUint64(8);
      if (size64 > BigInt(Number.MAX_SAFE_INTEGER)) break;
      headerSize = 16;
      boxSize = Number(size64);
    }
    if (boxSize < headerSize || boxSize > file.size - offset) break;
    if (type === "moov") {
      if (boxSize - headerSize > MAX_MOOV_BYTES) break;
      const moov = new Uint8Array(
        await file.slice(offset + headerSize, offset + boxSize).arrayBuffer(),
      );
      tracksInMoov(moov, found);
    }
    offset += boxSize;
  }
  return found;
}

/** Extension only: MIME comes from the same ambiguous extension and a size cap drifts per surface. */
export function needsAttachmentTrackInspection(file: File): boolean {
  return /\.(3gp|m?ts)$/i.test(file.name);
}

/** Mirrors `is_mpeg_transport_stream` in native_path_policy.rs. */
export function isMpegTransportStreamBytes(head: Uint8Array): boolean {
  return [
    [0, 188],
    [4, 192],
  ].some(([start, packet]) =>
    [0, 1, 2].every((index) => head[start! + index * packet!] === 0x47),
  );
}

/** Recordings and clips share .3gp, so read BMFF handlers and restamp as native readers do. */
export async function classifiedAttachmentFile(file: File): Promise<File> {
  if (!needsAttachmentTrackInspection(file)) {
    return file;
  }
  if (!/\.3gp$/i.test(file.name)) {
    let head: Uint8Array;
    try {
      head = new Uint8Array(await file.slice(0, 4 + 3 * 192).arrayBuffer());
    } catch {
      return file;
    }
    const corrected = isMpegTransportStreamBytes(head)
      ? "video/mp2t"
      : "text/plain";
    return corrected === file.type
      ? file
      : new File([file], file.name, {
          type: corrected,
          lastModified: file.lastModified,
        });
  }
  let tracks: BmffTracks;
  try {
    tracks = await read3gpTracks(file);
  } catch {
    return file;
  }
  // Both directions: a platform mapping .3gp to audio/3gpp says so for clips too.
  const corrected = tracks.video
    ? "video/3gpp"
    : tracks.audio
      ? "audio/3gpp"
      : null;
  if (corrected === null || corrected === file.type) {
    return file;
  }
  return new File([file], file.name, {
    type: corrected,
    lastModified: file.lastModified,
  });
}

/** One file at a time so a multi-file drop never holds more than one container's boxes. */
export async function classifiedAttachmentFiles(
  files: FileList | readonly File[],
): Promise<File[]> {
  const classified: File[] = [];
  for (const file of Array.from(files)) {
    classified.push(await classifiedAttachmentFile(file));
  }
  return classified;
}
