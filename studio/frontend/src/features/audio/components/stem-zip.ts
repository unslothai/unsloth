// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// stored zip avoids wasted compression because WAV barely compresses.

import { Zip, ZipPassThrough } from "fflate";
import { AUDIO_WORKFLOWS, clipWorkflow } from "../workflows";

const INVALID_CHARS = new Set('<>:"/\\|?*');
const MAX_TITLE_CHARS = 120;
// prompts can span paragraphs; 60 characters keeps CJK and emoji names under 255 bytes.
const MAX_CLIP_TITLE_CHARS = 60;
const FALLBACK_TITLE = "Separated track";

export function sanitizeFileNamePart(part: string, fallback: string): string {
  let out = "";
  for (const char of part) {
    const code = char.codePointAt(0) ?? 0;
    out += INVALID_CHARS.has(char) || code < 0x20 || code === 0x7f ? "_" : char;
  }
  const cleaned = out
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[. ]+$/, "");
  return cleaned || fallback;
}

function shortTitle(title: string, max: number, fallback: string): string {
  const bare = title.replace(/\.(wav|mp3|flac|ogg|m4a|aac|opus|webm)$/i, "");
  const safe = [...sanitizeFileNamePart(bare, fallback)]
    .slice(0, max)
    .join("")
    .trim();
  return safe || fallback;
}

/** the title is capped so the stem label always survives. */
export function stemFileName(title: string, label: string): string {
  const safeLabel = sanitizeFileNamePart(label, "Stem");
  return `${shortTitle(title, MAX_TITLE_CHARS, FALLBACK_TITLE)} - ${safeLabel}.wav`;
}

/** gallery clips are always WAV. */
export function clipFileName(clip: {
  prompt?: string | null;
  workflow?: string | null;
  audio_type?: string | null;
}): string {
  const workflow = clipWorkflow(clip);
  const label =
    AUDIO_WORKFLOWS.find((item) => item.id === workflow)?.label ?? "Audio";
  const prompt = (clip.prompt ?? "").replace(/\s+/g, " ");
  return `${shortTitle(prompt, MAX_CLIP_TITLE_CHARS, "Audio")} - ${label}.wav`;
}

export function stemZipName(title: string): string {
  return stemFileName(title, "stems").replace(/\.wav$/, ".zip");
}

export function uniqueStemNames(names: readonly string[]): string[] {
  const taken = new Set<string>();
  return names.map((name) => {
    let candidate = name;
    const dot = name.lastIndexOf(".");
    const stem = dot > 0 ? name.slice(0, dot) : name;
    const ext = dot > 0 ? name.slice(dot) : "";
    for (let n = 2; taken.has(candidate.toLowerCase()); n += 1)
      candidate = `${stem} (${n})${ext}`;
    taken.add(candidate.toLowerCase());
    return candidate;
  });
}

// Zip output is moved into Blob parts at this size, so it leaves the JS heap as it is written.
const ZIP_FLUSH_BYTES = 32 * 1024 * 1024;

/** Blobs are streamed in one at a time (a loader is called only when its turn comes), and the
 *  output is flushed into Blob parts, so neither all stems nor the whole archive sit in the heap. */
export async function zipStems(
  files: readonly { name: string; blob: Blob | (() => Promise<Blob>) }[],
): Promise<Blob> {
  const names = uniqueStemNames(files.map((file) => file.name));
  const parts: Blob[] = [];
  let pending: Uint8Array[] = [];
  let pendingBytes = 0;
  const flush = () => {
    if (pending.length === 0) return;
    parts.push(new Blob(pending as BlobPart[]));
    pending = [];
    pendingBytes = 0;
  };
  await new Promise<void>((resolve, reject) => {
    const zip = new Zip((error, data, final) => {
      if (error) {
        reject(error);
        return;
      }
      pending.push(data);
      pendingBytes += data.byteLength;
      if (pendingBytes >= ZIP_FLUSH_BYTES) flush();
      if (final) resolve();
    });
    const write = async () => {
      for (const [index, file] of files.entries()) {
        const blob =
          typeof file.blob === "function" ? await file.blob() : file.blob;
        const entry = new ZipPassThrough(names[index]);
        zip.add(entry);
        const reader = blob.stream().getReader();
        for (;;) {
          const { done: end, value } = await reader.read();
          if (end) break;
          entry.push(value, false);
        }
        entry.push(new Uint8Array(0), true);
      }
      zip.end();
    };
    write().catch(reject);
  });
  flush();
  return new Blob(parts, { type: "application/zip" });
}
