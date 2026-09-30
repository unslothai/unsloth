// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  ContextMenu,
  ContextMenuCheckboxItem,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { authFetch } from "@/features/auth";
import {
  browserPanelAvailable,
  openFileInBrowser,
  openUrlInBrowserPanel,
  saveLinkAs,
  textFileKind,
  useBrowserPrefsStore,
  useBrowserStore,
} from "@/features/browser";
import { startLibraryChat } from "@/features/library";
import { useT } from "@/i18n";
import { apiUrl, isTauri } from "@/lib/api-base";
import { copyToClipboard, copyToClipboardFrom } from "@/lib/copy-to-clipboard";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { openExternalLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import {
  ArrowUpRight01Icon,
  BubbleChatAddIcon,
  Copy01Icon,
  Download01Icon,
  File01Icon,
  FolderOpenIcon,
  InternetIcon,
  Link01Icon,
  LinkSquare02Icon,
  PaintBoardIcon,
  SquareArrowUpRightIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import type { ComponentProps, ReactElement, ReactNode } from "react";
import { sandboxFilePath, sandboxRoutePrefix } from "./sandbox-files";

const CONTENT_CLASS = "min-w-60 rounded-[20px] p-1.5";

// Passed through to the trigger, so a menu can sit inside another trigger (a dialog's) that uses asChild.
type TriggerProps = Omit<ComponentProps<typeof ContextMenuTrigger>, "asChild" | "children">;
const SUB_CLASS = "min-w-52 rounded-[20px] p-1.5";

function ItemIcon({ icon }: { icon: IconSvgElement }) {
  return <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />;
}

async function copyWithToast(copy: Promise<boolean>, done: string, failed: string): Promise<void> {
  if (await copy) toast.success(done);
  else toast.error(failed);
}

/** Whether the backend runs on this machine, so its file manager and apps are the user's. */
function backendIsLocal(): boolean {
  if (isTauri) return true;
  try {
    const { hostname } = new URL(apiUrl("/"), window.location.href);
    return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";
  } catch {
    return false;
  }
}

function revealLabelKey() {
  const platform = typeof navigator === "undefined" ? "" : navigator.userAgent;
  if (/Mac/i.test(platform)) return "linkMenu.revealFinder" as const;
  if (/Windows/i.test(platform)) return "linkMenu.revealExplorer" as const;
  return "linkMenu.revealFolder" as const;
}

/** Right-click menu for a web link: open it here or outside, copy it, or save what it points at. */
export function WebLinkContextMenu({
  href,
  children,
  ...triggerProps
}: { href: string; children: ReactElement } & TriggerProps) {
  const t = useT();
  const openLinksInBrowser = useBrowserPrefsStore((state) => state.openLinksInBrowser);
  if (!/^https?:\/\//i.test(href)) return children;
  const inPanel = browserPanelAvailable();
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild={true} className="select-text" {...triggerProps}>
        {children}
      </ContextMenuTrigger>
      <ContextMenuContent className={CONTENT_CLASS}>
        {inPanel ? (
          <ContextMenuItem onSelect={() => openUrlInBrowserPanel(href)}>
            <ItemIcon icon={InternetIcon} />
            {t("linkMenu.openInBrowser")}
          </ContextMenuItem>
        ) : null}
        <ContextMenuItem onSelect={() => openExternalLink(href)}>
          <ItemIcon icon={LinkSquare02Icon} />
          {t("linkMenu.openExternal")}
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem
          onSelect={() =>
            void copyWithToast(copyToClipboard(href), t("linkMenu.linkCopied"), t("linkMenu.copyFailed"))
          }
        >
          <ItemIcon icon={Link01Icon} />
          {t("linkMenu.copyLink")}
        </ContextMenuItem>
        <ContextMenuItem
          onSelect={() =>
            void saveLinkAs(href).catch((error) => {
              if (!isDownloadCancelled(error)) toast.error(t("linkMenu.saveFailed"));
            })
          }
        >
          <ItemIcon icon={Download01Icon} />
          {t("linkMenu.saveLinkAs")}
        </ContextMenuItem>
        {inPanel ? (
          <>
            <ContextMenuSeparator />
            <ContextMenuCheckboxItem
              checked={openLinksInBrowser}
              onCheckedChange={(value) => useBrowserPrefsStore.getState().setOpenLinksInBrowser(value)}
            >
              {t("linkMenu.alwaysInBrowser")}
            </ContextMenuCheckboxItem>
          </>
        ) : null}
      </ContextMenuContent>
    </ContextMenu>
  );
}

/** A file a chat shows: an attachment, or one a tool wrote to the chat's folder. */
export type ContextFile = {
  name: string;
  contentType?: string;
  load: () => Promise<Blob>;
  /** What a click on it does; opening it in the browser otherwise. */
  open?: () => void;
  /** Set for a file in the chat's folder, which has a path on disk. */
  sandbox?: { sessionId: string; file: string };
};

async function postSandbox(sessionId: string, action: "open" | "reveal", file: string): Promise<void> {
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  const separator = query ? "&" : "?";
  const response = await authFetch(`${prefix}/${action}${query}${separator}file=${encodeURIComponent(file)}`, {
    method: "POST",
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    throw new Error(typeof body?.detail === "string" ? body.detail : `HTTP ${response.status}`);
  }
}

async function sandboxAbsolutePath(sessionId: string, file: string): Promise<string> {
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  const response = await authFetch(`${prefix}${query}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const { path } = (await response.json()) as { path?: unknown };
  if (typeof path !== "string" || !path) throw new Error("No folder");
  const separator = path.includes("\\") && !path.includes("/") ? "\\" : "/";
  return `${path.replace(/[\\/]+$/, "")}${separator}${file.split("/").join(separator)}`;
}

/** Fetch a file from the chat's folder. */
export function loadSandboxFile(sessionId: string, file: string): Promise<Blob> {
  return authFetch(sandboxFilePath(sessionId, file)).then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.blob();
  });
}

// Types a blob URL can show without running anything in Studio's origin.
const SAFE_TAB_TYPE = /^(application\/pdf|image\/(png|jpe?g|gif|webp|avif|bmp)|video\/|audio\/|text\/plain)/i;

/** Right-click menu for a file link, as a desktop file manager has it. */
export function FileContextMenu({
  file,
  children,
  ...triggerProps
}: { file: ContextFile; children: ReactNode } & TriggerProps) {
  const t = useT();
  const navigate = useNavigate();
  const openInCanvas = useBrowserStore((state) => state.openInCanvas);
  const contentType = file.contentType ?? "";
  const kind = textFileKind(file.name, contentType);
  // Spreadsheet text (CSV/TSV) previews as a sheet but copies as the text it is.
  const copyable = kind !== null || /\.(csv|tsv)$/i.test(file.name) || /^text\//i.test(contentType);
  const local = file.sandbox !== undefined && backendIsLocal();
  const failed = (key: "linkMenu.openFailed" | "linkMenu.revealFailed" | "linkMenu.saveFailed") => () =>
    toast.error(t(key, { name: file.name }));

  const openInBrowser = () =>
    void file
      .load()
      .then((blob) =>
        openFileInBrowser({
          blob,
          name: file.name,
          contentType: contentType || blob.type,
          key: file.sandbox ? `sandbox:${file.sandbox.sessionId}:${file.sandbox.file}` : undefined,
        }),
      )
      .catch(failed("linkMenu.openFailed"));
  const open = () => (file.open ? file.open() : openInBrowser());
  const openInNewChat = () =>
    void file
      .load()
      .then((blob) =>
        startLibraryChat(navigate, { files: [new File([blob], file.name, { type: contentType || blob.type })] }),
      )
      .catch(failed("linkMenu.openFailed"));
  const openCanvas = () =>
    void file
      .load()
      .then((blob) => blob.text())
      .then((code) => openInCanvas?.({ title: file.name.replace(/\.[^.]+$/, ""), code }))
      .catch(failed("linkMenu.openFailed"));
  // A tab in the user's own browser; text is retyped so it shows as text rather than runs.
  const tabType = kind && kind !== "html" ? "text/plain" : contentType;
  const openInTab = () =>
    void file
      .load()
      .then((blob) => {
        const url = URL.createObjectURL(new Blob([blob], { type: tabType || blob.type }));
        window.open(url, "_blank", "noopener,noreferrer");
        window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
      })
      .catch(failed("linkMenu.openFailed"));
  const saveAs = () =>
    void file
      .load()
      .then((blob) => downloadFile(blob, file.name, contentType || blob.type || undefined))
      .catch((error) => {
        if (!isDownloadCancelled(error)) failed("linkMenu.saveFailed")();
      });
  const sandbox = file.sandbox;

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild={true} className="select-text" {...triggerProps}>
        {children}
      </ContextMenuTrigger>
      <ContextMenuContent className={CONTENT_CLASS}>
        <ContextMenuItem onSelect={open}>
          <ItemIcon icon={File01Icon} />
          {t("linkMenu.openFile")}
        </ContextMenuItem>
        {local && sandbox ? (
          <ContextMenuItem
            onSelect={() => void postSandbox(sandbox.sessionId, "open", sandbox.file).catch(failed("linkMenu.openFailed"))}
          >
            <ItemIcon icon={SquareArrowUpRightIcon} />
            {t("linkMenu.openDefaultApp")}
          </ContextMenuItem>
        ) : null}
        <ContextMenuSub>
          <ContextMenuSubTrigger className="gap-2.5 rounded-[11px]">
            <ItemIcon icon={ArrowUpRight01Icon} />
            {t("linkMenu.openWith")}
          </ContextMenuSubTrigger>
          <ContextMenuSubContent className={SUB_CLASS}>
            {browserPanelAvailable() ? (
              <ContextMenuItem onSelect={openInBrowser}>
                <ItemIcon icon={InternetIcon} />
                {t("linkMenu.unslothBrowser")}
              </ContextMenuItem>
            ) : null}
            {kind === "html" && openInCanvas ? (
              <ContextMenuItem onSelect={openCanvas}>
                <ItemIcon icon={PaintBoardIcon} />
                {t("linkMenu.canvas")}
              </ContextMenuItem>
            ) : null}
            <ContextMenuItem onSelect={openInNewChat}>
              <ItemIcon icon={BubbleChatAddIcon} />
              {t("linkMenu.newChat")}
            </ContextMenuItem>
            {/* A blob URL can't be handed to another app from the desktop app. */}
            {!isTauri && (kind ? kind !== "html" : SAFE_TAB_TYPE.test(contentType)) ? (
              <ContextMenuItem onSelect={openInTab}>
                <ItemIcon icon={LinkSquare02Icon} />
                {t("linkMenu.browserTab")}
              </ContextMenuItem>
            ) : null}
          </ContextMenuSubContent>
        </ContextMenuSub>
        <ContextMenuItem onSelect={saveAs}>
          <ItemIcon icon={Download01Icon} />
          {t("linkMenu.saveAs")}
        </ContextMenuItem>
        {(local && sandbox) || copyable ? <ContextMenuSeparator /> : null}
        {local && sandbox ? (
          <ContextMenuItem
            onSelect={() =>
              void copyWithToast(
                copyToClipboardFrom(() => sandboxAbsolutePath(sandbox.sessionId, sandbox.file)),
                t("linkMenu.pathCopied"),
                t("linkMenu.copyFailed"),
              )
            }
          >
            <ItemIcon icon={Copy01Icon} />
            {t("linkMenu.copyPath")}
          </ContextMenuItem>
        ) : null}
        {copyable ? (
          <ContextMenuItem
            onSelect={() =>
              void copyWithToast(
                copyToClipboardFrom(() => file.load().then((blob) => blob.text())),
                t("linkMenu.contentsCopied"),
                t("linkMenu.copyFailed"),
              )
            }
          >
            <ItemIcon icon={Copy01Icon} />
            {t("linkMenu.copyContents")}
          </ContextMenuItem>
        ) : null}
        {local && sandbox ? (
          <ContextMenuItem
            onSelect={() =>
              void postSandbox(sandbox.sessionId, "reveal", sandbox.file).catch(failed("linkMenu.revealFailed"))
            }
          >
            <ItemIcon icon={FolderOpenIcon} />
            {t(revealLabelKey())}
          </ContextMenuItem>
        ) : null}
      </ContextMenuContent>
    </ContextMenu>
  );
}
