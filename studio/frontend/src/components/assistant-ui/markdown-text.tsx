// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { ArtifactCard, useChatRuntimeStore } from "@/features/chat";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { preprocessLaTeX } from "@/lib/latex";
import { openLink } from "@/lib/open-link";
import { INTERNAL, useAuiState, useMessagePartText } from "@assistant-ui/react";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { createMathPlugin } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { DownloadIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Block, type BlockProps, Streamdown } from "streamdown";
import { createCodePlugin } from "./code-plugin";
import "katex/dist/katex.min.css";
import { AudioPlayer } from "./audio-player";
import { unslothDarkTheme, unslothLightTheme } from "./code-themes";

const math = createMathPlugin({ singleDollarTextMath: true });
const code = createCodePlugin({
  themes: [unslothLightTheme, unslothDarkTheme],
});
const { withSmoothContextProvider } = INTERNAL;

const STREAMDOWN_COMPONENTS = {
  a: ({ href, children, ...props }: React.ComponentProps<"a">) => (
    <a
      href={href}
      rel="noopener noreferrer"
      className="text-primary underline underline-offset-2 decoration-primary/40 hover:decoration-primary transition-colors cursor-pointer"
      onClick={(e) => {
        if (href && openLink(href)) {
          e.preventDefault();
        }
      }}
      {...props}
    >
      {children}
    </a>
  ),
};
const COPY_RESET_MS = 2000;
const MERMAID_SOURCE_RE = /```mermaid\s*([\s\S]*?)```/i;
const CODE_FENCE_RE = /^```([^\r\n`]*)\r?\n([\s\S]*?)\r?\n?```$/;
const ACTION_PANEL_CLASS =
  "pointer-events-auto flex shrink-0 items-center gap-1";
const ACTION_BUTTON_CLASS =
  "flex size-8 cursor-pointer items-center justify-center rounded-[10px] text-chat-icon-fg transition-all hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover disabled:cursor-not-allowed disabled:opacity-50";

type CodeFence = {
  language: string | null;
  source: string;
};

type ToolCallPartLike = {
  type?: string;
  toolName?: string;
  args?: unknown;
  result?: unknown;
};

function isRenderableRenderHtmlToolPart(part: unknown): boolean {
  const toolPart = part as ToolCallPartLike;
  if (toolPart.type !== "tool-call" || toolPart.toolName !== "render_html") {
    return false;
  }
  if (
    typeof toolPart.result === "string" &&
    toolPart.result.startsWith("Error:")
  ) {
    return false;
  }
  if (
    typeof toolPart.result === "string" &&
    toolPart.result.startsWith("Rendered HTML artifact")
  ) {
    return true;
  }
  const args = toolPart.args as { code?: unknown } | undefined;
  return typeof args?.code === "string" && args.code.trim().length > 0;
}

function getMermaidSource(blockContent: string): string | null {
  const source = blockContent.match(MERMAID_SOURCE_RE)?.[1]?.trim();
  return source && source.length > 0 ? source : null;
}

function getCodeFence(blockContent: string): CodeFence | null {
  const match = blockContent.trimEnd().match(CODE_FENCE_RE);
  if (!match) {
    return null;
  }

  return {
    language: match[1]?.trim() || null,
    source: match[2],
  };
}

function getCodeFilename(language: string | null) {
  const extByLanguage: Record<string, string> = {
    bash: "sh",
    javascript: "js",
    js: "js",
    json: "json",
    jsx: "jsx",
    markdown: "md",
    md: "md",
    python: "py",
    py: "py",
    shell: "sh",
    sh: "sh",
    sql: "sql",
    ts: "ts",
    tsx: "tsx",
    typescript: "ts",
    svg: "svg",
    yaml: "yml",
    yml: "yml",
  };

  const normalized = language?.toLowerCase();
  const fallbackExt = normalized?.replace(/[^a-z0-9]+/g, "-");
  const ext = normalized
    ? extByLanguage[normalized] || fallbackExt || "txt"
    : "txt";
  return `snippet.${ext}`;
}

function isSvgFence(codeFence: CodeFence): boolean {
  const lang = codeFence.language?.toLowerCase() ?? "";
  if (lang === "svg") return true;
  if (lang === "xml" || lang === "html") {
    const trimmed = codeFence.source.trimStart();
    // Match <svg directly or <?xml ...?> followed by <svg
    if (trimmed.startsWith("<svg")) return true;
    if (trimmed.startsWith("<?xml") && trimmed.includes("<svg")) return true;
  }
  return false;
}

function isHtmlFence(codeFence: CodeFence): boolean {
  const lang = codeFence.language?.toLowerCase() ?? "";
  return lang === "html" && !isSvgFence(codeFence);
}

function isFullHtmlDocument(source: string): boolean {
  const trimmed = source.trimStart();
  return /^<!doctype\s+html\b/i.test(trimmed) || /^<html[\s>]/i.test(trimmed);
}

const UNSAFE_SVG_RE =
  /<script[\s>]|on\w+\s*=|javascript:|<foreignObject[\s>]|<iframe[\s>]|<embed[\s>]|<object[\s>]/i;

function sanitizeSvg(source: string): string | null {
  if (UNSAFE_SVG_RE.test(source)) return null;
  // Strip XML declaration (<?xml ...?>) -- not needed for data URI
  // rendering and can cause issues with some renderers.
  return source.replace(/^\s*<\?xml[^?]*\?>\s*/i, "");
}

function SvgPreview({ source }: { source: string }) {
  const dataUri = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(source)}`;
  return (
    <div className="mt-2 flex justify-center rounded-lg border border-border bg-white p-4 dark:bg-neutral-100">
      <img
        src={dataUri}
        alt="SVG preview"
        style={{ maxWidth: "100%", maxHeight: 512 }}
      />
    </div>
  );
}

function downloadTextFile(filename: string, text: string): void {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function useCopiedState() {
  const [copied, setCopied] = useState(false);
  const resetTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (resetTimeoutRef.current) {
        clearTimeout(resetTimeoutRef.current);
      }
    };
  }, []);

  const showCopied = () => {
    setCopied(true);
    if (resetTimeoutRef.current) {
      clearTimeout(resetTimeoutRef.current);
    }
    resetTimeoutRef.current = setTimeout(() => {
      setCopied(false);
      resetTimeoutRef.current = null;
    }, COPY_RESET_MS);
  };

  return { copied, showCopied };
}

function MermaidCopyButton({ source }: { source: string }) {
  const { copied, showCopied } = useCopiedState();

  return (
    <button
      type="button"
      className="absolute top-3.5 right-20 z-20 cursor-pointer text-muted-foreground transition-all hover:text-foreground"
      title="Copy Mermaid source"
      onClick={async () => {
        if (!(await copyToClipboard(source))) {
          return;
        }
        showCopied();
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

function CodeBlockActions({
  disabled,
  language,
  source,
}: {
  disabled: boolean;
  language: string | null;
  source: string;
}) {
  const { copied, showCopied } = useCopiedState();

  return (
    <div className="pointer-events-none absolute top-3 right-3 z-20 flex items-center justify-end">
      <div className={ACTION_PANEL_CLASS}>
        <button
          type="button"
          className={ACTION_BUTTON_CLASS}
          title="Copy code"
          disabled={disabled}
          onClick={async () => {
            if (!(await copyToClipboard(source))) {
              return;
            }
            showCopied();
          }}
        >
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            strokeWidth={1.75}
            className="size-icon"
          />
        </button>
        <button
          type="button"
          className={ACTION_BUTTON_CLASS}
          title="Download file"
          disabled={disabled}
          onClick={() => {
            downloadTextFile(getCodeFilename(language), source);
          }}
        >
          <DownloadIcon className="size-icon" />
        </button>
      </div>
    </div>
  );
}

function StreamdownBlock(props: BlockProps) {
  const shouldCollapseHtmlArtifacts = useChatRuntimeStore(
    (state) => state.artifactsEnabled || state.collapseHtmlArtifacts,
  );
  const messageHasRenderableRenderHtmlTool = useAuiState(({ message }) =>
    message.parts.some(isRenderableRenderHtmlToolPart),
  );
  const hasMermaidFence = props.content.includes("```mermaid");
  const mermaidSource = getMermaidSource(props.content);
  const codeFence = getCodeFence(props.content);

  if (props.isIncomplete && hasMermaidFence) {
    return (
      <div className="my-4 flex h-48 items-center justify-center rounded-xl border border-border bg-muted/30 text-sm text-muted-foreground animate-pulse">
        Loading diagram...
      </div>
    );
  }

  if (props.isIncomplete && codeFence && isSvgFence(codeFence)) {
    return (
      <div className="relative isolate">
        <div className="my-4 rounded-xl border border-border bg-muted/30 p-4">
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            svg
          </div>
          <pre className="overflow-x-auto text-xs text-muted-foreground whitespace-pre-wrap break-all">
            <code>{codeFence.source}</code>
          </pre>
        </div>
      </div>
    );
  }

  if (
    shouldCollapseHtmlArtifacts &&
    !messageHasRenderableRenderHtmlTool &&
    props.isIncomplete &&
    codeFence &&
    isHtmlFence(codeFence) &&
    isFullHtmlDocument(codeFence.source)
  ) {
    return (
      <div className="my-4 flex h-48 items-center justify-center rounded-xl border border-border bg-muted/30 text-sm text-muted-foreground animate-pulse">
        Loading artifact preview...
      </div>
    );
  }

  if (mermaidSource) {
    return (
      <div className="relative isolate">
        <Block {...props} />
        <MermaidCopyButton source={mermaidSource} />
      </div>
    );
  }

  if (codeFence) {
    const svgSource =
      !props.isIncomplete && isSvgFence(codeFence)
        ? sanitizeSvg(codeFence.source)
        : null;
    const htmlSource =
      shouldCollapseHtmlArtifacts &&
      !messageHasRenderableRenderHtmlTool &&
      !props.isIncomplete &&
      isHtmlFence(codeFence) &&
      isFullHtmlDocument(codeFence.source)
        ? codeFence.source
        : null;
    if (htmlSource) {
      return (
        <ArtifactCard code={htmlSource} title="HTML preview" source="fence" />
      );
    }

    return (
      <>
        <div className="relative isolate">
          <Block {...props} />
          <CodeBlockActions
            disabled={props.isIncomplete}
            language={codeFence.language}
            source={codeFence.source}
          />
        </div>
        {svgSource && <SvgPreview source={svgSource} />}
      </>
    );
  }

  return <Block {...props} />;
}
const AUDIO_PLAYER_RE = /<audio-player\s+src="([^"]+)"\s*\/>/;

// Coalesce markdown re-parses to one per animation frame while streaming: the
// runtime notifies on every token (hundreds/sec) and the monitor can't paint
// that fast. When not streaming we return live text rather than the throttled
// state, so the final text never lags and a reused instance (parts are keyed by
// index) shows a completed message's text immediately instead of a stale frame.
function useRafCoalescedText(text: string, isStreaming: boolean): string {
  const [displayed, setDisplayed] = useState(text);
  const pendingRef = useRef(text);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    pendingRef.current = text;
    if (!isStreaming) {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      return;
    }
    if (rafRef.current === null) {
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        setDisplayed(pendingRef.current);
      });
    }
  }, [text, isStreaming]);

  // Unmount cleanup. Cancel the in-flight rAF and null the handle so a
  // StrictMode remount isn't gated out by a stale id. Kept separate from the
  // scheduling effect so it doesn't cancel mid-stream and defeat the throttle.
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, []);

  if (isStreaming && text.startsWith(displayed)) {
    return displayed;
  }
  return text;
}

const MarkdownTextImpl = () => {
  const { text, status } = useMessagePartText();
  const displayText = useRafCoalescedText(text, status.type === "running");
  const processedText = useMemo(
    () => preprocessLaTeX(displayText),
    [displayText],
  );

  const audioMatch = displayText.match(AUDIO_PLAYER_RE);
  if (audioMatch) {
    return <AudioPlayer src={audioMatch[1]} />;
  }

  return (
    <div data-status={status.type} className="min-w-0 max-w-full">
      <Streamdown
        mode="streaming"
        isAnimating={status.type === "running"}
        plugins={{ code, math, mermaid }}
        components={STREAMDOWN_COMPONENTS}
        controls={{
          code: false,
          mermaid: {
            fullscreen: true,
            download: true,
            copy: false,
            panZoom: true,
          },
        }}
        shikiTheme={[unslothLightTheme, unslothDarkTheme]}
        BlockComponent={StreamdownBlock}
      >
        {processedText}
      </Streamdown>
    </div>
  );
};

export const MarkdownText = withSmoothContextProvider(MarkdownTextImpl);
