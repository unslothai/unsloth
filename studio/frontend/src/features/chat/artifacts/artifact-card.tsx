// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  type FileViewMode,
  openHtmlInBrowser,
  openReactInBrowser,
  useBrowserStore,
  useShownHtmlView,
} from "@/features/browser";
import { authFetch } from "@/features/auth";
import { apiUrl, isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { openExternalLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useAuiState } from "@assistant-ui/react";
import {
  Copy01Icon,
  Download01Icon,
  InternetIcon,
  LinkSquare02Icon,
  SourceCodeIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { type ReactFenceLang, reactComponentName } from "./html-fences";
import { needsNode, noteNodeAvailability, prepareReactPreview } from "./react-preview/react-preview";
import {
  hasAutoOpenedArtifact,
  rememberAutoOpenedArtifact,
  useChatArtifactsStore,
} from "./store";
import {
  type ChatArtifact,
  type ChatArtifactKind,
  type ChatArtifactSource,
  createChatArtifact,
  getArtifactFilename,
  htmlDocumentTitle,
} from "./types";

function openArtifactInBrowser(artifact: ChatArtifact, view: FileViewMode): void {
  openHtmlInBrowser({
    key: artifact.id,
    name: getArtifactFilename({ title: htmlDocumentTitle(artifact.code) ?? artifact.title }),
    code: artifact.code,
    view,
  });
}

// Same slot for every glyph so labels line up.
function MenuGlyph({ icon, className }: { icon: typeof InternetIcon; className?: string }) {
  return (
    <span aria-hidden={true} className="flex size-4.5 shrink-0 items-center justify-center">
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className={cn("size-4", className)} />
    </span>
  );
}

// Its message is shown as is, in place of the generic one.
class PageOpenError extends Error {}

/** Open in the user's default browser, served sandboxed by the backend. The web tab opens first
 *  so the popup blocker allows it. */
async function openInDefaultBrowser(page: string | (() => Promise<string>)): Promise<void> {
  const tab = isTauri ? null : window.open("", "_blank");
  if (tab) tab.opener = null;
  try {
    const html = typeof page === "string" ? page : await page();
    const response = await authFetch(apiUrl("/api/inference/artifact-preview-page"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        html,
        allow_network: useChatRuntimeStore.getState().allowArtifactNetworkAccess,
      }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { path } = (await response.json()) as { path: string };
    const url = new URL(apiUrl(path), window.location.href).href;
    if (tab) tab.location.href = url;
    else openExternalLink(url);
  } catch (error) {
    tab?.close();
    toast.error(error instanceof PageOpenError ? error.message : "Couldn't open the page in your browser.");
  }
}

/** The compiled page for a React component; the same page the Unsloth Browser runs. */
function reactPage(code: string, lang: ReactFenceLang, title: string): () => Promise<string> {
  return async () => {
    const prepared = await prepareReactPreview({ source: code, lang, title });
    noteNodeAvailability(prepared);
    if (prepared.status === "ready") return prepared.html;
    if (prepared.status === "compile-error") {
      throw new PageOpenError("This component didn't compile. Open it in the Unsloth Browser to see why.");
    }
    if (needsNode(prepared.reason)) {
      throw new PageOpenError("React previews need Node.js. Install it, then re-run Unsloth setup.");
    }
    throw new PageOpenError("Couldn't prepare this preview.");
  };
}

// The subtitle says "HTML preview", so untitled pages get a different title.
function displayFallbackTitle(title: string): string {
  const generic = /^HTML preview(?: (\d+))?$/.exec(title);
  if (!generic) return title;
  return generic[1] ? `Untitled page ${generic[1]}` : "Untitled page";
}

function displayReactFallbackTitle(title: string): string {
  const generic = /^React preview(?: (\d+))?$/.exec(title);
  if (!generic) return title;
  return generic[1] ? `Untitled component ${generic[1]}` : "Untitled component";
}

export function ArtifactCard({
  code,
  title,
  source,
  sourceToolCallId,
  sourceMessageId,
  className,
  autoOpen = false,
  isStreaming = false,
  kind = "html",
  lang = "tsx",
}: {
  code: string;
  title?: string | null;
  source: ChatArtifactSource;
  sourceToolCallId?: string | null;
  sourceMessageId?: string | null;
  className?: string;
  autoOpen?: boolean;
  isStreaming?: boolean;
  // A React card never opens on its own: opening it compiles and runs the component.
  kind?: ChatArtifactKind;
  lang?: ReactFenceLang;
}) {
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const messageIdFromContext = useAuiState(({ message }) => message.id);
  const threadIdFromContext = useAuiState(
    ({ threads }) => threads.mainThreadId,
  );
  const artifactThreadId = threadIdFromContext ?? activeThreadId ?? null;
  const artifact = useMemo<ChatArtifact>(
    () =>
      createChatArtifact({
        code,
        title,
        source,
        sourceMessageId: sourceMessageId ?? messageIdFromContext ?? null,
        sourceToolCallId: sourceToolCallId ?? null,
        threadId: artifactThreadId,
        isStreaming,
      }),
    [
      artifactThreadId,
      code,
      isStreaming,
      messageIdFromContext,
      source,
      sourceMessageId,
      sourceToolCallId,
      title,
    ],
  );
  const isReact = kind === "react";
  const reactUnavailable = useChatArtifactsStore((state) => state.reactPreviewUnavailable);
  const shownView = useShownHtmlView(isReact ? `react:${artifact.id}` : artifact.id);
  // Only React cards: an HTML card's code is never scanned for a component name.
  const reactTitle = useMemo(
    () =>
      isReact
        ? (reactComponentName(artifact.code) ?? displayReactFallbackTitle(artifact.title))
        : null,
    [artifact.code, artifact.title, isReact],
  );
  const open = (view: FileViewMode) =>
    isReact && reactTitle !== null
      ? openReactInBrowser({
          key: artifact.id,
          name: getArtifactFilename({ title: reactTitle }, lang),
          code: artifact.code,
          view,
        })
      : openArtifactInBrowser(artifact, view);
  // Once per mount, and only when the page is complete.
  const autoOpenAttemptedRef = useRef(false);

  useEffect(() => {
    if (isReact) return;
    if (!autoOpen || isStreaming || autoOpenAttemptedRef.current) return;
    if (artifact.code.trim().length === 0) return;
    autoOpenAttemptedRef.current = true;
    if (hasAutoOpenedArtifact(artifact.id)) return;
    rememberAutoOpenedArtifact(artifact.id);
    openArtifactInBrowser(artifact, "preview");
  }, [artifact, autoOpen, isReact, isStreaming]);

  const pageTitle =
    reactTitle ?? htmlDocumentTitle(artifact.code) ?? displayFallbackTitle(artifact.title);
  const filename = isReact
    ? getArtifactFilename({ title: pageTitle }, lang)
    : getArtifactFilename({ title: pageTitle });
  const showing = shownView !== null;
  const toggle = (view: FileViewMode) => {
    if (shownView === view) {
      useBrowserStore.getState().closePanel();
      return;
    }
    open(view);
  };
  const download = () =>
    void downloadFile(
      artifact.code,
      filename,
      isReact ? "text/plain;charset=utf-8" : "text/html;charset=utf-8",
    ).catch((error) => {
      if (!isDownloadCancelled(error)) {
        toast.error(isReact ? "Couldn't save the code." : "Couldn't save the HTML.");
      }
    });
  const copy = () =>
    void copyToClipboard(artifact.code).then(
      (ok) => ok && toast.success(isReact ? "Code copied" : "HTML copied"),
    );
  const openDefault = () =>
    void openInDefaultBrowser(isReact ? reactPage(artifact.code, lang, pageTitle) : artifact.code);

  return (
    <div className={cn("my-2 w-full max-w-xl", className)}>
      <div
        className={cn(
          "group/artifact-card relative flex min-h-[calc(72px*var(--ui-space-scale,1))] items-center gap-3.5 overflow-hidden rounded-2xl border border-border/70 bg-muted/15 py-3 pl-3 pr-3.5 transition-colors dark:border-transparent dark:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)]",
          !isStreaming &&
            "hover:bg-muted/25 dark:hover:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)]",
        )}
      >
        {isStreaming ? (
          <span
            aria-hidden={true}
            className="artifact-card-shimmer pointer-events-none absolute inset-0 z-0 motion-reduce:hidden"
          />
        ) : null}
        <button
          type="button"
          disabled={isStreaming}
          onClick={() => toggle("preview")}
          // State rides on aria-expanded, not the name: the startup-bundle harness counts cards by name.
          aria-expanded={shownView === "preview"}
          aria-label={`Open ${artifact.title} preview`}
          className="absolute inset-0 z-0 cursor-pointer rounded-2xl focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-default"
        />
        <span
          aria-hidden={true}
          className="pointer-events-none relative flex size-[calc(48px*var(--ui-space-scale,1))] shrink-0 items-center justify-center rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] dark:bg-[color-mix(in_oklab,var(--foreground)_calc(11%*var(--contrast-wash-gain,1)),transparent)]"
        >
          <HugeiconsIcon
            icon={isReact ? SourceCodeIcon : InternetIcon}
            strokeWidth={1.75}
            className="size-6 text-control-accent"
          />
        </span>
        <span className="pointer-events-none relative grid min-w-0 flex-1 gap-1">
          <span className="truncate text-ui-15 font-medium leading-tight text-foreground">
            {pageTitle}
          </span>
          <span className="truncate text-sm leading-tight text-muted-foreground">
            {isStreaming ? (
              <span className="shimmer motion-reduce:animate-none">Generating…</span>
            ) : isReact ? (
              reactUnavailable ? "React preview · Needs Node.js" : "React preview"
            ) : (
              "HTML preview"
            )}
          </span>
        </span>
        <div
          className={cn(
            "relative z-10 flex h-9 shrink-0 items-stretch gap-px overflow-hidden rounded-full border border-border/80 bg-background/60 text-sm text-foreground dark:border-transparent dark:bg-transparent",
            isStreaming && "opacity-50",
          )}
        >
          <button
            type="button"
            disabled={isStreaming}
            onClick={() => open("preview")}
            aria-pressed={showing}
            className="cursor-pointer pl-3.5 pr-2.5 transition-colors hover:bg-muted/60 focus-visible:bg-muted/60 focus-visible:outline-none disabled:cursor-default disabled:hover:bg-transparent dark:bg-[color-mix(in_oklab,var(--foreground)_calc(12%*var(--contrast-wash-gain,1)),transparent)] dark:hover:bg-[color-mix(in_oklab,var(--foreground)_calc(17%*var(--contrast-wash-gain,1)),transparent)] dark:focus-visible:bg-[color-mix(in_oklab,var(--foreground)_calc(17%*var(--contrast-wash-gain,1)),transparent)]"
          >
            Open in
          </button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild={true}>
              <button
                type="button"
                disabled={isStreaming}
                aria-label="More ways to open"
                className="flex cursor-pointer items-center border-l border-border/80 pl-1.5 pr-2.5 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground focus-visible:bg-muted/60 focus-visible:outline-none aria-expanded:bg-muted/60 disabled:cursor-default dark:border-l-0 dark:bg-[color-mix(in_oklab,var(--foreground)_calc(12%*var(--contrast-wash-gain,1)),transparent)] dark:hover:bg-[color-mix(in_oklab,var(--foreground)_calc(17%*var(--contrast-wash-gain,1)),transparent)] dark:focus-visible:bg-[color-mix(in_oklab,var(--foreground)_calc(17%*var(--contrast-wash-gain,1)),transparent)] dark:aria-expanded:bg-[color-mix(in_oklab,var(--foreground)_calc(17%*var(--contrast-wash-gain,1)),transparent)]"
              >
                <HugeiconsIcon
                  icon={ChevronDownStandardIcon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="end"
              sideOffset={8}
              className="min-w-56 rounded-[20px] p-1.5 [&_[data-slot=dropdown-menu-separator]]:mx-3 [&_[data-slot=dropdown-menu-separator]]:my-1.5"
            >
              <DropdownMenuItem onSelect={() => open("preview")}>
                <MenuGlyph icon={InternetIcon} />
                Unsloth Browser
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={openDefault}>
                <MenuGlyph icon={LinkSquare02Icon} />
                Default browser
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => open("source")}>
                {/* </> draws small, so one size up. */}
                <MenuGlyph icon={SourceCodeIcon} className="size-4.5" />
                Code view
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={copy}>
                <MenuGlyph icon={Copy01Icon} />
                {isReact ? "Copy code" : "Copy HTML"}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={download}>
                <MenuGlyph icon={Download01Icon} />
                {isReact ? "Download code" : "Download HTML"}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  );
}
