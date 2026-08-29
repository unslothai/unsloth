// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Containers llama-server can decode. It shells out to ffmpeg, so this is
 * what ffmpeg reads, not what the webview can play. Extensions ride along
 * because MIME is unreliable for mkv and some mov files. */
export const VIDEO_ACCEPT =
  "video/mp4,video/x-m4v,video/quicktime,video/webm,video/x-matroska,video/x-msvideo,video/mpeg,video/x-ms-wmv,video/x-flv,video/3gpp,video/ogg,.mp4,.m4v,.mov,.webm,.mkv,.avi,.mpg,.mpeg,.wmv,.flv,.3gp,.ogv";

// Matches _MAX_VIDEO_B64_CHARS in the backend, so the composer does not accept
// a clip the route refuses. The native reader's cap is a higher backstop.
const MAX_VIDEO_SIZE_MB = 64;
export const MAX_VIDEO_SIZE = MAX_VIDEO_SIZE_MB * 1024 * 1024;
export const MAX_VIDEO_SIZE_LABEL = `${MAX_VIDEO_SIZE_MB}MB`;

export function getVideoSizeError(size: number): string | null {
  return size > MAX_VIDEO_SIZE
    ? `Video size exceeds ${MAX_VIDEO_SIZE_LABEL} limit`
    : null;
}

// Mirrors the extension table in native_intents.rs, which the parity test keeps
// in step: a clip that arrives through the desktop reader and one picked in the
// browser must reach the route as the same mime type.
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
};

const VIDEO_EXTENSIONS = Object.keys(VIDEO_MIME_BY_EXTENSION);
const VIDEO_MIME_RE = /^video\//i;

/** The mime type to send a picked clip under.
 *
 * The accept list carries extensions as well as mime types because the browser's
 * answer is unreliable for mkv and some mov files, so a file the picker took on
 * its extension can arrive as "" or as application/octet-stream. Both are then
 * carried into the attachment, and the request builder only recognises a file
 * part whose mimeType matches ^video/, so the clip is dropped and the model
 * answers as though it were never attached. Trust the extension whenever the
 * browser did not say video.
 */
export function videoMimeForFile(file: File): string {
  if (VIDEO_MIME_RE.test(file.type)) return file.type;
  const name = file.name.toLowerCase();
  for (const [ext, mime] of Object.entries(VIDEO_MIME_BY_EXTENSION)) {
    if (name.endsWith(ext)) return mime;
  }
  return "video/mp4";
}

/** Whether a picked file is a video. mkv and some mov files arrive with an
 * empty MIME type, hence the extension fallback. */
export function isVideoFile(file: { name: string; type: string }): boolean {
  if (VIDEO_MIME_RE.test(file.type)) {
    return true;
  }
  // The extension fallback below claims .3gp, which a recording shares with a
  // clip. Something that read the tracks has already said which this is.
  if (/^audio\//i.test(file.type)) {
    return false;
  }
  const name = file.name.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => name.endsWith(ext));
}

/** Payloads of every box of the given type at this level. Mirrors
 *  `bmff_box_payloads` in native_path_policy.rs. */
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
      // Runs to the end of this level.
      boxSize = data.length - offset;
    } else if (size32 === 1) {
      // A 64-bit size follows the type.
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

/** The handler types a moov box's tracks declare. */
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

/** Whether a 3GP container has audio tracks and no video tracks. Mirrors
 *  `is_audio_only_3gp` in native_path_policy.rs. */
export function isAudioOnly3gpBytes(raw: Uint8Array): boolean {
  const found: BmffTracks = { audio: false, video: false };
  for (const moov of bmffBoxPayloads(raw, "moov")) {
    tracksInMoov(moov, found);
  }
  return found.audio && !found.video;
}

// A track table runs to kilobytes; anything this large is not one, and reading
// it would be the memory problem the box walk exists to avoid.
const MAX_MOOV_BYTES = 8 * 1024 * 1024;
// A container holds a handful of these: ftyp, moov, mdat, and maybe free or
// mfra. A file that reports thousands is malformed, and walking it would be
// one slice per box.
const MAX_TOP_LEVEL_BOXES = 64;

/**
 * Which track kinds a container declares, without holding it.
 *
 * The tracks live in `moov`, and the samples in `mdat` beside it, so reading the
 * file to reach a handler retains the whole clip: a 64 MB one costs 64 MB, and a
 * dropped batch costs that per file at once. This walks the top-level boxes
 * through slices and reads only `moov`, which is the same walk `bmffBoxPayloads`
 * does, one level up.
 */
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

/**
 * Whether a file's own tracks have to be read before it can be classified.
 * Cheap and synchronous, so a surface can keep its existing path for everything
 * else.
 *
 * The extension alone, and nothing about the MIME type or the size.
 *
 * Not the type, because it comes from the same ambiguous extension: a platform
 * that reports audio/3gpp for a recording reports it for a clip as well, and
 * trusting that sent the clip down the audio path.
 *
 * Not the size, because the one condition here sat at the composer's video cap
 * while the video reference surface accepts a larger file, so a recording in
 * between skipped inspection entirely. A ceiling that has to track every
 * surface's limit will fall behind one of them, and the walk below is bounded
 * by box count and by the size of the track table rather than by the file.
 */
export function needsAttachmentTrackInspection(file: File): boolean {
  return /\.3gp$/i.test(file.name);
}

/**
 * The file an attachment surface should classify, with an audio-only 3GP
 * restamped as audio.
 *
 * A voice recording and a clip share the .3gp extension, and the browser
 * answers "" or video/3gpp for both, so the name alone sends the recording to
 * whichever surface claims video: rejected outright on an audio model, and fed
 * to ffmpeg as frames on a video one. The native readers already read the BMFF
 * handlers and stamp audio/3gpp, so do the same before an adapter is picked.
 * Everything else is returned untouched, so this costs one predicate per file.
 */
export async function classifiedAttachmentFile(file: File): Promise<File> {
  if (!needsAttachmentTrackInspection(file)) {
    return file;
  }
  let tracks: BmffTracks;
  try {
    tracks = await read3gpTracks(file);
  } catch {
    // An unreadable file is left as it came; the surface reports the read.
    return file;
  }
  // Both directions, because the browser's answer comes from the same ambiguous
  // extension: a platform that maps .3gp to audio/3gpp says so for a clip too,
  // and the audio adapter is matched before the video one. Tracks it cannot
  // read decide nothing, so the file is left as it came.
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

/** The same restamping across a picked or dropped batch, one file at a time so
 *  a drop of several never has more than one container's boxes in hand. */
export async function classifiedAttachmentFiles(
  files: FileList | readonly File[],
): Promise<File[]> {
  const classified: File[] = [];
  for (const file of Array.from(files)) {
    classified.push(await classifiedAttachmentFile(file));
  }
  return classified;
}
