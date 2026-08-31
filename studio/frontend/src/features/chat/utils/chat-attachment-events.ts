// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Notifies loaded chat runtimes when the Data tab deletes a stored attachment.
 * Without this, the active thread's in-memory repository still holds the
 * attachment, and any later repo-to-storage sync (e.g. deleting a message in
 * that thread) writes it back, undoing the deletion.
 */

import forge from "node-forge";

export type ChatAttachmentDeletedEvent = {
  messageId: string;
  attachmentId: string;
};

const CONTENT_PART_ID_PREFIX = "content-part-sha256-";
const URI_SCHEME_RE = /^[A-Za-z][A-Za-z0-9+.-]*:/;

function isLocallyStoredBlob(value: string): boolean {
  const candidate = value.trimStart();
  if (!candidate) return false;
  if (candidate.slice(0, 5).toLowerCase() === "data:") return true;
  if (candidate.startsWith("//") || candidate.startsWith("\\\\")) {
    return false;
  }
  return !URI_SCHEME_RE.test(candidate);
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value
      .map((item) => (item === undefined ? "null" : stableJson(item)))
      .join(",")}]`;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .filter((key) => record[key] !== undefined)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/** Canonical payload used to detect whether an async hash still describes the
 * current message content. */
export function chatContentPartAttachmentSignature(
  part: unknown,
): string | null {
  if (!part || typeof part !== "object") return null;
  const record = part as Record<string, unknown>;
  let payload: ["image" | "audio", unknown] | null = null;
  if (
    typeof record.image === "string" &&
    record.image.slice(0, 5).toLowerCase() === "data:"
  ) {
    payload = ["image", record.image];
  } else if (
    typeof record.audio === "string" &&
    isLocallyStoredBlob(record.audio)
  ) {
    payload = ["audio", record.audio];
  } else if (record.audio && typeof record.audio === "object") {
    const data = (record.audio as Record<string, unknown>).data;
    if (typeof data === "string" && isLocallyStoredBlob(data)) {
      payload = ["audio", record.audio];
    }
  }
  if (!payload) return null;

  return stableJson(payload);
}

/** Mirrors the backend's stable content-part identity without adding private
 * metadata to the message payload sent to inference. */
export async function chatContentPartAttachmentIdFromSignature(
  signature: string,
): Promise<string> {
  let hex: string | null = null;
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    try {
      const digest = await subtle.digest(
        "SHA-256",
        new TextEncoder().encode(signature),
      );
      hex = Array.from(new Uint8Array(digest), (byte) =>
        byte.toString(16).padStart(2, "0"),
      ).join("");
    } catch {
      // Fall through to the pure-JS implementation below. Some embedded
      // browsers expose crypto.subtle but reject it outside a secure context.
    }
  }
  if (hex === null) {
    const digest = forge.md.sha256.create();
    digest.update(signature, "utf8");
    hex = digest.digest().toHex();
  }
  return `${CONTENT_PART_ID_PREFIX}${hex}`;
}

type Listener = (event: ChatAttachmentDeletedEvent) => void | Promise<void>;

const listeners = new Set<Listener>();

export function onChatAttachmentDeleted(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function emitChatAttachmentDeleted(
  event: ChatAttachmentDeletedEvent,
): void {
  for (const listener of [...listeners]) {
    void listener(event);
  }
}
