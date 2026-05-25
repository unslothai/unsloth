// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import {
  CheckIcon,
  CopyIcon,
  DownloadIcon,
  FileTextIcon,
  Maximize2Icon,
  XIcon,
} from "lucide-react";
import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import { ArtifactHtmlFrame, type ArtifactViewMode } from "./html-frame";
import type { ChatArtifact } from "./types";
import { getArtifactFilename } from "./types";

const COPY_RESET_MS = 2000;
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

function downloadTextFile(filename: string, text: string): void {
  const blob = new Blob([text], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
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
  const [copied, setCopied] = useState(false);
  const copyResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const surfaceRef = useRef<HTMLElement>(null);
  const previousFocusRef = useRef<Element | null>(null);
  const filename = getArtifactFilename(artifact);

  useEffect(() => {
    return () => {
      if (copyResetRef.current) clearTimeout(copyResetRef.current);
    };
  }, []);

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
        "flex min-h-0 flex-col overflow-hidden border border-border bg-background shadow-xl",
        variant === "panel"
          ? "h-full w-full rounded-none border-y-0 border-r-0"
          : "h-[min(92vh,900px)] w-[min(96vw,1200px)] rounded-2xl",
      )}
      aria-label={`${artifact.title} artifact`}
    >
      <header className="flex shrink-0 items-center gap-3 border-b border-border px-3 py-2">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <FileTextIcon className="size-4" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-foreground">
            {artifact.title}
          </p>
          <p className="text-xs text-muted-foreground">
            HTML artifact ·{" "}
            {artifact.source === "tool" ? "tool call" : "fenced fallback"}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-8"
            onClick={() => downloadTextFile(filename, artifact.code)}
            aria-label="Download artifact HTML"
          >
            <DownloadIcon className="size-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-8"
            onClick={handleCopy}
            aria-label="Copy artifact HTML"
          >
            {copied ? (
              <CheckIcon className="size-4" />
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
              aria-label="Open artifact fullscreen"
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
            aria-label="Close artifact"
          >
            <XIcon className="size-4" />
          </Button>
        </div>
      </header>

      <div className="flex shrink-0 items-center gap-1 border-b border-border px-3 py-2">
        {(["preview", "source"] as const).map((mode) => (
          <button
            key={mode}
            type="button"
            onClick={() => setViewMode(mode)}
            className={cn(
              "rounded-lg px-3 py-1.5 text-xs font-medium transition-colors",
              viewMode === mode
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
            aria-pressed={viewMode === mode}
          >
            {mode === "preview" ? "Preview" : "Source"}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-hidden bg-background">
        {viewMode === "preview" ? (
          <ArtifactHtmlFrame
            key={artifact.id}
            code={artifact.code}
            title={artifact.title}
            fill={true}
            className="h-full"
          />
        ) : (
          <pre className="h-full overflow-auto p-4 text-xs leading-relaxed text-foreground whitespace-pre">
            <code>{artifact.code}</code>
          </pre>
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
