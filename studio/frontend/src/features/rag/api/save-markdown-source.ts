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

// Windows reserves these device names with any extension; ISO-8859-1 superscripts count as digits.
const RESERVED_DEVICE_NAME =
  /^(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])$/i;
// Filesystems cap components in bytes (usually 255), so CJK or emoji titles need a byte budget.
const MAX_STEM_BYTES = 180;

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

/** For display in the sources panel only; the backend stores under a uuid and re-sanitises. */
export function projectSourceFileName(title: string): string {
  const stem = clampToBytes(
    Array.from(title, (char) => {
      const code = char.codePointAt(0) ?? 0;
      if (code < 0x20 || code === 0x7f) return " ";
      // Array.from yields whole code points, so a surrogate here is unpaired.
      if (code >= 0xd800 && code <= 0xdfff) return "";
      return "\\/:*?\"<>|".includes(char) ? "_" : char;
    })
      .join("")
      .replace(/\s+/g, " ")
      .trim(),
    MAX_STEM_BYTES,
  )
    // Windows drops a trailing period or space.
    .replace(/[\s.]+$/, "");
  // The backend collapses non [A-Za-z0-9._-] characters to "_", so use a generic name instead.
  if (!/[A-Za-z0-9]/.test(stem)) return "chat.md";
  // Windows reads the device name as the part before the first dot.
  const dot = stem.indexOf(".");
  const head = dot === -1 ? stem : stem.slice(0, dot);
  return RESERVED_DEVICE_NAME.test(head)
    ? `${head}_${stem.slice(head.length)}.md`
    : `${stem}.md`;
}

// A save has no chip to show failure, so poll the ingest to warn when indexing fails.
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
      return;
    }
    const terminal = terminalJobStatus(job.status);
    if (!terminal) continue;
    if (terminal === "failed") {
      // The panel hides failed documents, so the source would silently never appear.
      toast.error(`Couldn't index ${filename}`, {
        description: job.error ?? "Indexing failed",
      });
    }
    announceProjectSourcesUpdated(projectId);
    return;
  }
}

export async function saveMarkdownAsProjectSource(
  projectId: string,
  markdown: string,
  title: string,
  options: { quiet?: boolean } = {},
): Promise<boolean> {
  const filename = projectSourceFileName(title);
  const file = new File([markdown], filename, { type: "text/markdown" });
  // Invalidate before and after the upload: a chat sent mid-upload must not cache "no sources".
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
    announceProjectSourcesUpdated(projectId);
  }
}
