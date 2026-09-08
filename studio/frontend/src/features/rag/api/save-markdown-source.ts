// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { type IndexJob, terminalJobStatus } from "../types/rag";
import {
  announceProjectSourcesUpdated,
  getJob,
  invalidateProjectSources,
  uploadProjectDocument,
} from "./rag-api";

// Windows keeps these device names reserved in every directory, with or without
// an extension: "NUL.txt and NUL.tar.gz are both equivalent to NUL". The
// ISO-8859-1 superscripts count as digits in COM#/LPT#.
const RESERVED_DEVICE_NAME =
  /^(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])$/i;
// Filesystems cap a path component in bytes, not characters, so 80 CJK or emoji
// code points would overrun the usual 255. Leave room for ".md" too.
const MAX_STEM_BYTES = 180;

/** Trim to a byte budget on a code point boundary, so a cut never leaves the
 * lone half of a surrogate pair behind. */
function clampToBytes(text: string, maxBytes: number): string {
  const encoder = new TextEncoder();
  if (encoder.encode(text).length <= maxBytes) return text;
  let used = 0;
  let out = "";
  for (const char of text) {
    const size = encoder.encode(char).length;
    if (used + size > maxBytes) break;
    used += size;
    out += char;
  }
  return out;
}

/** A chat title as the filename its source is listed under. The backend stores
 * the upload under a uuid and re-sanitises this for its own metadata, so this is
 * about what the user reads in the sources panel, not about path safety. */
export function projectSourceFileName(title: string): string {
  const stem = clampToBytes(
    Array.from(title, (char) => {
      const code = char.codePointAt(0) ?? 0;
      // Control characters are not filename characters on any host.
      if (code < 0x20 || code === 0x7f) return " ";
      // Array.from yields whole code points, so a surrogate here is an unpaired
      // one the title arrived with; it has no encoding to send.
      if (code >= 0xd800 && code <= 0xdfff) return "";
      return "\\/:*?\"<>|".includes(char) ? "_" : char;
    })
      .join("")
      .replace(/\s+/g, " ")
      .trim(),
    MAX_STEM_BYTES,
  )
    // Windows drops a trailing period or space, so never send one.
    .replace(/[\s.]+$/, "");
  // The backend collapses everything outside [A-Za-z0-9._-] to "_", so a title
  // with no ASCII word character at all would be listed as "_.md". A generic
  // name at least reads as one.
  if (!/[A-Za-z0-9]/.test(stem)) return "chat.md";
  // A device name stays reserved through any extension, and Windows reads it as
  // the part before the *first* dot, so break the name there.
  const dot = stem.indexOf(".");
  const head = dot === -1 ? stem : stem.slice(0, dot);
  return RESERVED_DEVICE_NAME.test(head)
    ? `${head}_${stem.slice(head.length)}.md`
    : `${stem}.md`;
}

// A save has no chip in the sources panel to carry a "failed" state, so poll the
// ingest far enough to warn when the document will never become searchable. The
// panel does the same over SSE for the uploads it owns.
const INGEST_POLL_MS = 2_000;
const INGEST_POLL_ATTEMPTS = 150;

async function watchIngestion(
  projectId: string,
  jobId: string,
  filename: string,
): Promise<void> {
  for (let attempt = 0; attempt < INGEST_POLL_ATTEMPTS; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, INGEST_POLL_MS));
    let job: IndexJob;
    try {
      job = await getJob(jobId);
    } catch {
      // The job is unreachable; the panel's own list is the fallback.
      return;
    }
    const terminal = terminalJobStatus(job.status);
    if (!terminal) continue;
    if (terminal === "failed") {
      // The panel hides failed documents, so without this the source just never
      // appears after a success toast.
      toast.error(`Couldn't index ${filename}`, {
        description: job.error ?? "Indexing failed",
      });
    }
    announceProjectSourcesUpdated(projectId);
    return;
  }
}

/** Upload one markdown document to a project's sources. Resolves true when the
 * upload was accepted, so a caller saving several can report the count once;
 * failures toast here, because only this layer knows why. */
export async function saveMarkdownAsProjectSource(
  projectId: string,
  markdown: string,
  title: string,
  options: { quiet?: boolean } = {},
): Promise<boolean> {
  const filename = projectSourceFileName(title);
  const file = new File([markdown], filename, { type: "text/markdown" });
  // Invalidate the sources probe before the upload as well as after it (the
  // announce below): a chat sent mid-upload must not cache "no sources" for the
  // probe's TTL.
  invalidateProjectSources(projectId);
  try {
    const result = await uploadProjectDocument(projectId, file);
    if (!options.quiet) toast.success("Saved to project sources.");
    void watchIngestion(projectId, result.jobId, result.filename || filename);
    return true;
  } catch (error) {
    toast.error("Failed to save to project sources.", {
      description: error instanceof Error ? error.message : undefined,
    });
    return false;
  } finally {
    // Announce only after the upload: a refetch fired before it would just
    // re-list the rows the panel already has.
    announceProjectSourcesUpdated(projectId);
  }
}
