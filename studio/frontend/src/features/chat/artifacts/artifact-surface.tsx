// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { createCodePlugin } from "@/components/assistant-ui/code-plugin";
import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import {
  unslothDarkTheme,
  unslothLightTheme,
} from "@/components/assistant-ui/code-themes";
import { MascotImg } from "@/components/mascot-img";
import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { CopyIcon, EyeIcon, Maximize2Icon, XIcon } from "lucide-react";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Streamdown } from "streamdown";
import { ArtifactHtmlFrame, type ArtifactViewMode } from "./html-frame";
import { useChatArtifactsStore } from "./store";
import type { ChatArtifact } from "./types";
import { getArtifactFilename } from "./types";

const COPY_RESET_MS = 2000;
const artifactSourceCodePlugin = createCodePlugin({
  themes: [unslothLightTheme, unslothDarkTheme],
});

function buildHtmlFence(source: string): string {
  const longestBacktickRun = Math.max(
    2,
    ...(source.match(/`+/g) ?? []).map((match) => match.length),
  );
  const fence = "`".repeat(longestBacktickRun + 1);
  return `${fence}html\n${source}\n${fence}`;
}
// Sandboxed canvas iframes are deliberately outside the overlay focus trap:
// granting same-origin sandbox privileges would weaken isolation, so reaching
// interactive canvas content via keyboard is a known sandbox limitation.
const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
  ).filter(
    (element) =>
      !element.hasAttribute("disabled") &&
      element.getAttribute("aria-hidden") !== "true" &&
      element.tabIndex !== -1,
  );
}

function ArtifactLoadingLine() {
  return (
    <div className="absolute inset-x-0 bottom-0 h-[2.5px] overflow-hidden bg-border/45">
      <span
        aria-hidden={true}
        className="artifact-loading-line block h-full rounded-full motion-reduce:hidden"
      />
    </div>
  );
}

function ArtifactGeneratingPanel() {
  return (
    <div className="flex h-full min-h-0 flex-col items-center justify-center bg-muted/10 px-6 text-center">
      <div className="max-w-[30ch] space-y-1.5">
        <MascotImg
          src="Sloth emojis/sloth w pc transparent.png"
          aria-hidden={true}
          className="mx-auto mb-3 size-20 object-contain"
        />
        <p className="text-sm font-medium text-foreground">
          Building canvas preview…
        </p>
        <p className="text-xs leading-relaxed text-muted-foreground">
          The preview will appear here when the HTML is ready.
        </p>
      </div>
    </div>
  );
}

export function ArtifactSurface({
  artifact,
  variant,
  onClose,
  onOpenFullscreen,
}: {
  artifact: ChatArtifact;
  variant: "panel" | "overlay";
  onClose: () => void;
  onOpenFullscreen?: () => void;
}) {
  const [viewMode, setViewMode] = useState<ArtifactViewMode>("preview");
  // Follow the view the opener asked for (Preview vs Code button), per artifact.
  const requestedView = useChatArtifactsStore((state) => state.requestedView);
  const [copied, setCopied] = useState(false);
  const copyResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const surfaceRef = useRef<HTMLElement>(null);
  const previousFocusRef = useRef<Element | null>(null);
  const filename = getArtifactFilename(artifact);
  const sourceMarkdown = useMemo(
    () => buildHtmlFence(artifact.code),
    [artifact.code],
  );
  const hasArtifactCode = artifact.code.trim().length > 0;
  const isLoadingArtifact = Boolean(artifact.isStreaming);
  const effectiveViewMode = isLoadingArtifact ? "preview" : viewMode;

  useEffect(() => {
    return () => {
      if (copyResetRef.current) clearTimeout(copyResetRef.current);
    };
  }, []);

  useEffect(() => {
    setViewMode(requestedView);
  }, [artifact.id, requestedView]);

  useEffect(() => {
    if (variant !== "overlay") return;
    previousFocusRef.current = document.activeElement;
    const timeoutId = window.setTimeout(() => {
      const surface = surfaceRef.current;
      if (!surface) return;
      const firstFocusable = getFocusableElements(surface)[0];
      if (firstFocusable) {
        firstFocusable.focus();
      } else {
        surface.focus();
      }
    }, 0);
    return () => {
      window.clearTimeout(timeoutId);
      const previousFocus = previousFocusRef.current;
      if (previousFocus instanceof HTMLElement) previousFocus.focus();
    };
  }, [variant]);

  const handleCopy = async () => {
    if (!(await copyToClipboard(artifact.code))) return;
    setCopied(true);
    if (copyResetRef.current) clearTimeout(copyResetRef.current);
    copyResetRef.current = setTimeout(() => {
      setCopied(false);
      copyResetRef.current = null;
    }, COPY_RESET_MS);
  };

  const handleDialogKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (variant !== "overlay") return;
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = getFocusableElements(event.currentTarget);
    if (focusable.length === 0) {
      event.preventDefault();
      event.currentTarget.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const content = (
    <section
      ref={surfaceRef}
      role={variant === "overlay" ? "dialog" : undefined}
      aria-modal={variant === "overlay" ? true : undefined}
      tabIndex={variant === "overlay" ? -1 : undefined}
      onKeyDown={handleDialogKeyDown}
      className={cn(
        "relative flex min-h-0 flex-col bg-background",
        variant === "panel"
          ? "artifact-panel-shell mx-2 mt-[80px] mb-8 h-[calc(100%_-_112px)] overflow-visible rounded-[28px] border-t border-border/70 bg-card/95"
          : "h-[min(92vh,900px)] w-[min(96vw,1200px)] overflow-hidden rounded-2xl border border-border shadow-xl",
      )}
      aria-label={`${artifact.title} canvas`}
    >
      <header
        className={cn(
          "relative flex shrink-0 items-center justify-between gap-3 px-2.5 py-2",
          variant === "panel" && "rounded-t-[28px]",
        )}
      >
        <div
          className="flex items-center gap-1 rounded-full bg-muted/40 p-0.5"
          role="tablist"
          aria-label="Canvas view"
        >
          {(["preview", "source"] as const).map((mode) => {
            const isPreview = mode === "preview";
            const Icon = isPreview ? EyeIcon : CodeToggleIcon;
            return (
              <button
                key={mode}
                type="button"
                role="tab"
                disabled={isLoadingArtifact && !isPreview}
                onClick={() => setViewMode(mode)}
                className={cn(
                  "flex size-8 items-center justify-center rounded-full text-muted-foreground transition-colors",
                  effectiveViewMode === mode
                    ? "bg-background text-foreground shadow-sm"
                    : "hover:bg-background/70 hover:text-foreground",
                  isLoadingArtifact &&
                    !isPreview &&
                    "cursor-not-allowed opacity-50",
                )}
                aria-label={
                  isPreview ? "Preview canvas" : "View canvas source"
                }
                aria-selected={effectiveViewMode === mode}
                aria-pressed={effectiveViewMode === mode}
                title={
                  isPreview
                    ? "Preview"
                    : isLoadingArtifact
                      ? "Source available when generation finishes"
                      : "Source"
                }
              >
                <Icon className="size-4" />
              </button>
            );
          })}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-8"
            disabled={isLoadingArtifact || !hasArtifactCode}
            onClick={() => {
              // Route through the native save dialog on desktop; the plain
              // blob-anchor download is silently dropped by the Tauri WebView2.
              void downloadFile(
                artifact.code,
                filename,
                "text/html;charset=utf-8",
              ).catch((err) => {
                if (!isDownloadCancelled(err)) {
                  toast.error("Failed to save canvas HTML");
                }
              });
            }}
            aria-label="Download canvas HTML"
          >
            <HugeiconsIcon icon={Download01Icon} className="size-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-8"
            disabled={isLoadingArtifact || !hasArtifactCode}
            onClick={handleCopy}
            aria-label="Copy canvas HTML"
          >
            {copied ? (
              <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-4" />
            ) : (
              <CopyIcon className="size-4" />
            )}
          </Button>
          {variant === "panel" && onOpenFullscreen ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="size-8"
              onClick={onOpenFullscreen}
              aria-label="Open canvas fullscreen"
            >
              <Maximize2Icon className="size-4" />
            </Button>
          ) : null}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-8"
            onClick={onClose}
            aria-label="Close canvas"
          >
            <XIcon className="size-4" />
          </Button>
        </div>
        {isLoadingArtifact ? (
          <ArtifactLoadingLine />
        ) : (
          <div className="absolute inset-x-0 bottom-0 h-px bg-border" />
        )}
      </header>

      <div
        className={cn(
          "min-h-0 flex-1 overflow-hidden bg-background",
          variant === "panel" && "rounded-b-[28px]",
        )}
      >
        {isLoadingArtifact ? (
          <ArtifactGeneratingPanel />
        ) : effectiveViewMode === "preview" ? (
          <ArtifactHtmlFrame
            key={artifact.id}
            code={artifact.code}
            title={artifact.title}
            fill={true}
            // Network mode only for tool-rendered canvases, never fences.
            allowNetworkAccess={artifact.source === "tool"}
            className="h-full"
          />
        ) : (
          <div className="h-full overflow-auto text-xs leading-relaxed [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!gap-0 [&_[data-streamdown=code-block]]:!rounded-none [&_[data-streamdown=code-block]]:!border-0 [&_[data-streamdown=code-block]]:!bg-transparent [&_[data-streamdown=code-block]]:!p-0 [&_[data-streamdown=code-block-body]]:!border-0 [&_[data-streamdown=code-block-body]]:!bg-transparent [&_[data-streamdown=code-block-body]]:!p-0 [&_pre]:!m-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:text-xs [&_pre]:leading-relaxed [&_code]:text-xs">
            <Streamdown
              mode="streaming"
              plugins={{ code: artifactSourceCodePlugin }}
              controls={{ code: false }}
              shikiTheme={[unslothLightTheme, unslothDarkTheme]}
            >
              {sourceMarkdown}
            </Streamdown>
          </div>
        )}
      </div>
    </section>
  );

  if (variant === "overlay") {
    return (
      <div
        className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-4 backdrop-blur-sm"
        onMouseDown={(event) => {
          if (event.target === event.currentTarget) onClose();
        }}
      >
        {content}
      </div>
    );
  }

  return content;
}
