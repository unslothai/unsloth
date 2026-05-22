// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import DOMPurify from "dompurify";
import { CodeIcon, EyeIcon, Maximize2Icon, Minimize2Icon } from "lucide-react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

export type HtmlSvgLanguage = "html" | "svg";

export type HtmlSvgRendererProps = {
  language: HtmlSvgLanguage;
  source: string;
  // Pre-rendered syntax-highlighted code view. Optional; if omitted the
  // renderer falls back to a plain <pre><code> block.
  codeView?: ReactNode;
  // When the markdown stream is still arriving the fence may be partial.
  // In that case we force the Code tab and disable the toggle controls.
  isIncomplete?: boolean;
};

const DEFAULT_PREVIEW_HEIGHT = 500;
const POPOUT_HEIGHT_VH = 80;
const COPY_RESET_MS = 2000;

// Conservative regex covering the same vectors we previously stripped before
// DOMPurify was wired in. DOMPurify itself does the heavy lifting; this is
// belt-and-braces so a quick visual inspection of the source still catches the
// usual XSS hot spots.
const HEURISTIC_UNSAFE_SVG_RE =
  /<script[\s>]|<foreignObject[\s>]|<iframe[\s>]|<embed[\s>]|<object[\s>]/i;

// SVGs may legitimately reference fonts/images via href, so we keep those.
// We strip every event handler attribute (on*) and any tag DOMPurify would
// otherwise allow that could escape the SVG sandbox.
const SVG_PURIFY_CONFIG = {
  USE_PROFILES: { svg: true, svgFilters: true },
  FORBID_TAGS: ["script", "foreignObject", "iframe", "embed", "object"],
  // DOMPurify already drops on* handlers when USE_PROFILES is set, but we
  // call this out explicitly so the intent is obvious to future readers.
  ALLOW_DATA_ATTR: false,
};

/** Strip every XML processing instruction and disallowed node from an SVG. */
export function sanitizeSvgSource(source: string): string {
  // Drop XML declarations -- DOMPurify keeps them but some renderers choke.
  const stripped = source.replace(/^\s*<\?xml[^?]*\?>\s*/i, "");
  // First pass: regex screen. We do NOT bail out -- DOMPurify will still
  // produce a safe string -- but logging here helps debugging.
  if (HEURISTIC_UNSAFE_SVG_RE.test(stripped)) {
    // eslint-disable-next-line no-console
    console.debug("SVG renderer: stripping unsafe nodes before sanitize");
  }
  return DOMPurify.sanitize(stripped, SVG_PURIFY_CONFIG);
}

function useCopyState() {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  const flash = useCallback(() => {
    setCopied(true);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => {
      setCopied(false);
      timeoutRef.current = null;
    }, COPY_RESET_MS);
  }, []);

  return { copied, flash };
}

function CopyButton({ source }: { source: string }) {
  const { copied, flash } = useCopyState();
  return (
    <button
      type="button"
      title="Copy code"
      aria-label="Copy code"
      className={cn(
        "flex size-8 cursor-pointer items-center justify-center rounded-[10px]",
        "text-chat-icon-fg transition-all hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
      )}
      onClick={async () => {
        if (await copyToClipboard(source)) flash();
      }}
    >
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        strokeWidth={1.75}
        className="size-icon"
      />
    </button>
  );
}

type TabKey = "preview" | "code";

function TabButton({
  active,
  disabled,
  icon,
  label,
  onSelect,
}: {
  active: boolean;
  disabled?: boolean;
  icon: ReactNode;
  label: string;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      disabled={disabled}
      onClick={onSelect}
      className={cn(
        "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
        active
          ? "bg-background text-foreground shadow-sm"
          : "text-muted-foreground hover:text-foreground",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function SvgPreview({ source }: { source: string }) {
  const safe = useMemo(() => sanitizeSvgSource(source), [source]);
  return (
    <div
      data-testid="html-svg-renderer-svg-preview"
      className="flex justify-center bg-white p-4 dark:bg-neutral-100"
      // biome-ignore lint/security/noDangerouslySetInnerHtml: sanitized by DOMPurify above.
      dangerouslySetInnerHTML={{ __html: safe }}
    />
  );
}

function HtmlPreview({
  source,
  popped,
}: {
  source: string;
  popped: boolean;
}) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  // Tiny helper script that posts the document height back to the parent so
  // we can right-size the iframe. Communication is one-way and the iframe
  // cannot read parent.document because we never grant allow-same-origin.
  const srcDoc = useMemo(
    () =>
      `${source}<script>(()=>{const post=()=>parent.postMessage({htmlPreviewHeight:document.documentElement.scrollHeight},"*");window.addEventListener("load",post);new ResizeObserver(post).observe(document.documentElement);})();</script>`,
    [source],
  );

  const [autoHeight, setAutoHeight] = useState<number | null>(null);

  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.source !== iframeRef.current?.contentWindow) return;
      const raw = (e.data as { htmlPreviewHeight?: unknown })
        ?.htmlPreviewHeight;
      if (typeof raw === "number" && Number.isFinite(raw)) {
        setAutoHeight(Math.max(100, raw));
      }
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, []);

  // In the docked view we cap at DEFAULT_PREVIEW_HEIGHT; in the popout we
  // let the iframe fill the modal panel.
  const iframeHeight = popped
    ? "100%"
    : Math.min(autoHeight ?? DEFAULT_PREVIEW_HEIGHT, DEFAULT_PREVIEW_HEIGHT);

  return (
    <iframe
      ref={iframeRef}
      data-testid="html-svg-renderer-iframe"
      title="HTML preview"
      srcDoc={srcDoc}
      // SECURITY: allow-scripts only. We do NOT grant allow-same-origin or
      // allow-top-navigation, so the iframe cannot read parent.document,
      // navigate the host page, or escape its origin.
      sandbox="allow-scripts"
      style={{
        width: "100%",
        height: iframeHeight,
        border: "none",
        display: "block",
        background: "white",
      }}
    />
  );
}

export function HtmlSvgRenderer({
  language,
  source,
  codeView,
  isIncomplete,
}: HtmlSvgRendererProps) {
  // Stream-in: while the fence is still being filled in we lock to the Code
  // tab so users see the in-flight tokens rather than a flashing preview.
  const lockedToCode = Boolean(isIncomplete);
  const [tab, setTab] = useState<TabKey>("preview");
  const [popped, setPopped] = useState(false);

  const activeTab: TabKey = lockedToCode ? "code" : tab;

  // Escape key exits the popout view.
  useEffect(() => {
    if (!popped) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPopped(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [popped]);

  const codeFallback = useMemo(
    () =>
      codeView ?? (
        <pre className="overflow-x-auto rounded-md bg-muted/40 p-3 text-xs">
          <code>{source}</code>
        </pre>
      ),
    [codeView, source],
  );

  const preview =
    language === "svg" ? (
      <SvgPreview source={source} />
    ) : (
      <HtmlPreview source={source} popped={popped} />
    );

  const previewLabel = language === "svg" ? "SVG preview" : "HTML preview";

  return (
    <div
      data-testid="html-svg-renderer"
      data-language={language}
      data-active-tab={activeTab}
      className="my-4 overflow-hidden rounded-xl border border-border"
    >
      <div
        role="tablist"
        aria-label={`${language.toUpperCase()} fence view`}
        className="flex items-center justify-between gap-2 border-b border-border bg-muted/40 px-2 py-1.5"
      >
        <div className="flex items-center gap-1 rounded-lg bg-muted p-0.5">
          <TabButton
            active={activeTab === "preview"}
            disabled={lockedToCode}
            icon={<EyeIcon className="size-3.5" />}
            label="Preview"
            onSelect={() => setTab("preview")}
          />
          <TabButton
            active={activeTab === "code"}
            icon={<CodeIcon className="size-3.5" />}
            label="Code"
            onSelect={() => setTab("code")}
          />
        </div>
        <div className="flex items-center gap-1">
          {activeTab === "code" && !codeView ? (
            // When a custom code view is supplied it typically renders its
            // own copy/download chrome (see CodeBlockActions in
            // markdown-text.tsx). We only surface the fallback Copy button
            // when we are using the plain <pre> fallback below.
            <CopyButton source={source} />
          ) : null}
          {activeTab === "preview" && language === "html" ? (
            <button
              type="button"
              title={popped ? "Exit pop out" : "Pop out preview"}
              aria-label={popped ? "Exit pop out" : "Pop out preview"}
              className={cn(
                "flex size-8 cursor-pointer items-center justify-center rounded-[10px]",
                "text-chat-icon-fg transition-all hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
              )}
              onClick={() => setPopped((p) => !p)}
            >
              {popped ? (
                <Minimize2Icon className="size-4" />
              ) : (
                <Maximize2Icon className="size-4" />
              )}
            </button>
          ) : null}
        </div>
      </div>

      <div data-testid="html-svg-renderer-body" className="bg-background">
        {activeTab === "preview" ? (
          popped && language === "html" ? (
            <>
              {/* Keep layout stable behind the modal. */}
              <div
                style={{ height: DEFAULT_PREVIEW_HEIGHT }}
                aria-hidden={true}
              />
              <div
                className="fixed inset-0 z-50 flex flex-col bg-background/80 backdrop-blur-sm"
                onClick={(e) => {
                  if (e.target === e.currentTarget) setPopped(false);
                }}
              >
                <div className="flex items-center justify-end px-4 py-2">
                  <button
                    type="button"
                    className="flex items-center gap-1.5 rounded-md border border-border bg-background px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    onClick={() => setPopped(false)}
                    title="Exit pop out (Esc)"
                  >
                    <Minimize2Icon className="size-4" />
                    Exit pop out
                  </button>
                </div>
                <div
                  className="mx-4 mb-4 flex-1 overflow-hidden rounded-lg border border-border bg-background"
                  style={{ maxHeight: `${POPOUT_HEIGHT_VH}vh` }}
                >
                  {preview}
                </div>
              </div>
            </>
          ) : (
            <div aria-label={previewLabel}>{preview}</div>
          )
        ) : (
          <div data-testid="html-svg-renderer-code" className="min-w-0">
            {codeFallback}
          </div>
        )}
      </div>
    </div>
  );
}

// ---- Fence helpers (exported for tests + reuse from markdown-text.tsx) ----

export type CodeFenceInfo = {
  language: string | null;
  source: string;
};

const CODE_FENCE_RE = /^```([^\r\n`]*)\r?\n([\s\S]*?)\r?\n?```$/;

export function parseCodeFence(blockContent: string): CodeFenceInfo | null {
  const match = blockContent.trimEnd().match(CODE_FENCE_RE);
  if (!match) return null;
  return {
    language: match[1]?.trim() || null,
    source: match[2],
  };
}

export function isSvgFence(fence: CodeFenceInfo): boolean {
  const lang = fence.language?.toLowerCase() ?? "";
  if (lang === "svg") return true;
  if (lang === "xml" || lang === "html") {
    const trimmed = fence.source.trimStart();
    if (trimmed.startsWith("<svg")) return true;
    if (trimmed.startsWith("<?xml") && trimmed.includes("<svg")) return true;
  }
  return false;
}

export function isHtmlFence(fence: CodeFenceInfo): boolean {
  const lang = fence.language?.toLowerCase() ?? "";
  return lang === "html" && !isSvgFence(fence);
}
