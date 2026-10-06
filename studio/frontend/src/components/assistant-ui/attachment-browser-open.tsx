// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { AttachmentSource } from "@/components/assistant-ui/use-attachment-source";
import { authFetch } from "@/features/auth";
import { attachmentBodyText, fetchChatAttachmentBlob, parseAttachmentText } from "@/features/chat";
import { openFileInBrowser } from "@/features/browser";
import { isStudioUrl } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { useAuiState } from "@assistant-ui/react";
import { Slot } from "radix-ui";
import {
  type ComponentProps,
  type FC,
  type PropsWithChildren,
  type ReactElement,
  useCallback,
  useLayoutEffect,
  useRef,
} from "react";
import { AttachmentBrowserOpenContext } from "./attachment-browser-open-context";
import { FileContextMenu } from "./link-context-menu";

type Opened = { blob: Blob; plainText?: boolean };

// Documents and text open as tabs; media keeps the lightbox.
const OPENS_IN_BROWSER: ReadonlySet<AttachmentSource["kind"]> = new Set(["document", "text"]);

function localLoader(source: AttachmentSource): (() => Promise<Opened>) | null {
  const { file, text } = source;
  // Copied: the tab can outlive the composer's File.
  if (file) return () => file.arrayBuffer().then((data) => ({ blob: new Blob([data], { type: file.type }) }));
  switch (source.kind) {
    case "document":
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

function opener(source: AttachmentSource, id: string, load: () => Promise<Opened>) {
  return () =>
    void load()
      .then(({ blob, plainText }) => {
        const name = source.name || "attachment";
        // Extracted text, not the original bytes: name and type it as text, as the viewer saves it.
        openFileInBrowser({
          blob,
          name: plainText ? `${name.replace(/\.[^.]+$/, "")}.txt` : name,
          contentType: plainText ? "text/plain" : source.contentType || blob.type,
          plainText,
          key: `${id}:${source.name}`,
        });
      })
      .catch(() => toast.error(`Could not open ${source.name || "attachment"}`));
}

/** Provides `open` under a stable identity, so the attachment's consumers don't re-render with it. */
const OpenerProvider: FC<PropsWithChildren<{ open: () => void }>> = ({ open, children }) => {
  const openRef = useRef(open);
  useLayoutEffect(() => {
    openRef.current = open;
  });
  const stable = useCallback(() => openRef.current(), []);
  return <AttachmentBrowserOpenContext.Provider value={stable}>{children}</AttachmentBrowserOpenContext.Provider>;
};

const SentOriginalProvider: FC<PropsWithChildren<{ source: AttachmentSource; attachmentId: string }>> = ({
  source,
  attachmentId,
  children,
}) => {
  const messageId = useAuiState(({ message }) => message.id);
  // Attachment ids are only unique within their message.
  const open = opener(source, `${messageId}:${attachmentId}`, () =>
    fetchChatAttachmentBlob(messageId, attachmentId).then((blob) => ({ blob })),
  );
  return <OpenerProvider open={open}>{children}</OpenerProvider>;
};

export const AttachmentBrowserOpenProvider: FC<PropsWithChildren<{ source: AttachmentSource }>> = ({
  source,
  children,
}) => {
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
  if (!OPENS_IN_BROWSER.has(source.kind)) return children;
  const load = localLoader(source);
  if (load) {
    return <OpenerProvider open={opener(source, attachmentId, load)}>{children}</OpenerProvider>;
  }
  if (source.kind === "document") {
    return (
      <SentOriginalProvider source={source} attachmentId={attachmentId}>
        {children}
      </SentOriginalProvider>
    );
  }
  return children;
};

function blobLoader(source: AttachmentSource): (() => Promise<Blob>) | null {
  const { file, src } = source;
  if (file) return () => Promise.resolve(file);
  const local = localLoader(source);
  if (local) return () => local().then(({ blob }) => blob);
  if (src) {
    // Only Studio's own URLs get the sign-in; an image linked from elsewhere must not receive it.
    const request = /^(blob|data):/i.test(src)
      ? () => fetch(src)
      : isStudioUrl(src)
        ? () => authFetch(src)
        : () => fetch(src, { credentials: "omit" });
    return () =>
      request().then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.blob();
      });
  }
  return null;
}

type MenuProps = { source: AttachmentSource; children: ReactElement } & Omit<ComponentProps<"button">, "children">;

const AttachmentMenu: FC<MenuProps & { load: () => Promise<Blob> }> = ({
  source,
  load,
  children,
  onContextMenu,
  ...rest
}) => {
  // Open clicks the chip, so the open-in-browser setting and a missing panel apply.
  const chip = useRef<HTMLElement | null>(null);
  return (
    <FileContextMenu
      file={{ name: source.name || "attachment", contentType: source.contentType, load, open: () => chip.current?.click() }}
      {...rest}
      onContextMenu={(event) => {
        chip.current = event.currentTarget;
        onContextMenu?.(event as Parameters<NonNullable<typeof onContextMenu>>[0]);
      }}
    >
      {children}
    </FileContextMenu>
  );
};

const SentOriginalMenu: FC<MenuProps> = (props) => {
  const messageId = useAuiState(({ message }) => message.id);
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
  return <AttachmentMenu {...props} load={() => fetchChatAttachmentBlob(messageId, attachmentId)} />;
};

export const AttachmentFileContextMenu: FC<MenuProps> = ({ source, children, ...rest }) => {
  const load = blobLoader(source);
  if (load) {
    return (
      <AttachmentMenu source={source} load={load} {...rest}>
        {children}
      </AttachmentMenu>
    );
  }
  if (source.kind === "document" && source.hasOriginal) {
    return (
      <SentOriginalMenu source={source} {...rest}>
        {children}
      </SentOriginalMenu>
    );
  }
  return <Slot.Root {...rest}>{children}</Slot.Root>;
};
