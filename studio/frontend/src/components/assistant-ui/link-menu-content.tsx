// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ContextMenuCheckboxItem,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
} from "@/components/ui/context-menu";
import { authFetch } from "@/features/auth";
import {
  browserPanelAvailable,
  browserTabType,
  openFileInBrowser,
  openUrlInBrowserPanel,
  saveLinkAs,
  textFileKind,
  useBrowserPrefsStore,
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
  SquareArrowUpRightIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { sandboxRoutePrefix } from "./sandbox-files";

import type { ContextFile } from "./link-context-menu";

const CONTENT_CLASS = "min-w-60 rounded-[20px] p-1.5";

const SUB_CLASS = "min-w-52 rounded-[20px] p-1.5";

function ItemIcon({ icon }: { icon: IconSvgElement }) {
  return <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />;
}

async function copyWithToast(copy: Promise<boolean>, done: string, failed: string): Promise<void> {
  if (await copy) toast.success(done);
  else toast.error(failed);
}

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

export function WebLinkMenuContent({ href }: { href: string }) {
  const t = useT();
  const openLinksInBrowser = useBrowserPrefsStore((state) => state.openLinksInBrowser);
  // Read on open: the link may render before the chat registers the panel.
  const inPanel = browserPanelAvailable();
  return (
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
  );
}

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

export function FileMenuContent({ file }: { file: ContextFile }) {
  const t = useT();
  const navigate = useNavigate();
  const contentType = file.contentType ?? "";
  const kind = textFileKind(file.name, contentType);
  // CSV/TSV previews as a sheet but copies as text.
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
  // A tab in the user's own browser; text is retyped so it shows as text rather than runs.
  const tabType = browserTabType(file.name, contentType);
  const openInTab = () => {
    if (!tabType) return;
    // Opened now, while the click still counts for the popup blocker.
    const tab = window.open("", "_blank");
    if (tab) tab.opener = null;
    void file
      .load()
      .then((blob) => {
        const url = URL.createObjectURL(new Blob([blob], { type: tabType }));
        if (tab) tab.location.href = url;
        window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
      })
      .catch(() => {
        tab?.close();
        failed("linkMenu.openFailed")();
      });
  };
  const saveAs = () =>
    void file
      .load()
      .then((blob) => downloadFile(blob, file.name, contentType || blob.type || undefined))
      .catch((error) => {
        if (!isDownloadCancelled(error)) failed("linkMenu.saveFailed")();
      });
  const sandbox = file.sandbox;

  return (
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
          <ContextMenuItem onSelect={openInNewChat}>
            <ItemIcon icon={BubbleChatAddIcon} />
            {t("linkMenu.newChat")}
          </ContextMenuItem>
          {/* A blob URL can't be handed to another app from the desktop app. */}
          {!isTauri && tabType ? (
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
  );
}
