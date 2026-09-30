// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { AttachmentSource } from "@/components/assistant-ui/use-attachment-source";
import { attachmentBodyText, fetchChatAttachmentBlob, parseAttachmentText } from "@/features/chat";
import { openFileInBrowser } from "@/features/browser";
import { toast } from "@/lib/toast";
import { useAuiState } from "@assistant-ui/react";
import type { FC, PropsWithChildren } from "react";
import { AttachmentBrowserOpenContext } from "./attachment-browser-open-context";

type Opened = { blob: Blob; plainText?: boolean };

// Documents and text open as tabs; media keeps the lightbox.
const OPENS_IN_BROWSER: ReadonlySet<AttachmentSource["kind"]> = new Set(["document", "text"]);

// What the attachment holds locally; null when only the stored original can be opened.
function localLoader(source: AttachmentSource): (() => Promise<Opened>) | null {
  const { file, text } = source;
  // Copied: the tab can outlive the composer's File.
  if (file) return () => file.arrayBuffer().then((data) => ({ blob: new Blob([data], { type: file.type }) }));
  switch (source.kind) {
    case "document":
      // Text pulled out of a document that was sent without its original.
      return !source.hasOriginal && text !== undefined
        ? () => Promise.resolve({ blob: new Blob([attachmentBodyText(text)], { type: "text/plain" }), plainText: true })
        : null;
    default: {
      if (text === undefined) return null;
      return () => {
        const parsed = parseAttachmentText(text);
        return Promise.resolve({
          blob: new Blob([parsed.text], { type: parsed.label ? "text/plain" : source.contentType || "text/plain" }),
          plainText: Boolean(parsed.label),
        });
      };
    }
  }
}

function opener(source: AttachmentSource, attachmentId: string, load: () => Promise<Opened>) {
  return () =>
    void load()
      .then(({ blob, plainText }) =>
        openFileInBrowser({
          blob,
          name: source.name || "attachment",
          contentType: source.contentType || blob.type,
          plainText,
          key: `${attachmentId}:${source.name}`,
        }),
      )
      .catch(() => toast.error(`Could not open ${source.name || "attachment"}`));
}

const SentOriginalProvider: FC<PropsWithChildren<{ source: AttachmentSource; attachmentId: string }>> = ({
  source,
  attachmentId,
  children,
}) => {
  const messageId = useAuiState(({ message }) => message.id);
  const open = opener(source, attachmentId, () =>
    fetchChatAttachmentBlob(messageId, attachmentId).then((blob) => ({ blob })),
  );
  return <AttachmentBrowserOpenContext.Provider value={open}>{children}</AttachmentBrowserOpenContext.Provider>;
};

export const AttachmentBrowserOpenProvider: FC<PropsWithChildren<{ source: AttachmentSource }>> = ({
  source,
  children,
}) => {
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
  if (!OPENS_IN_BROWSER.has(source.kind)) return children;
  const load = localLoader(source);
  if (load) {
    return (
      <AttachmentBrowserOpenContext.Provider value={opener(source, attachmentId, load)}>
        {children}
      </AttachmentBrowserOpenContext.Provider>
    );
  }
  // Only sent documents have a stored original to fetch.
  if (source.kind === "document") {
    return (
      <SentOriginalProvider source={source} attachmentId={attachmentId}>
        {children}
      </SentOriginalProvider>
    );
  }
  return children;
};
