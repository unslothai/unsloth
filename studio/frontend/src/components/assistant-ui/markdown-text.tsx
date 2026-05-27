// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { preprocessLaTeX } from "@/lib/latex";
import { openLink } from "@/lib/open-link";
import { INTERNAL, useMessagePartText } from "@assistant-ui/react";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { createCodePlugin } from "./code-plugin";
import { createMathPlugin } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { DownloadIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Block, type BlockProps, Streamdown } from "streamdown";
import "katex/dist/katex.min.css";
import { AudioPlayer } from "./audio-player";
import { unslothDarkTheme, unslothLightTheme } from "./code-themes";
import {
  HtmlSvgRenderer,
  isHtmlFence,
  isSvgFence,
  parseCodeFence,
  parseIncompleteCodeFence,
  type CodeFenceInfo,
} from "./html-svg-renderer";

const math = createMathPlugin({ singleDollarTextMath: true });
const code = createCodePlugin({
  themes: [unslothLightTheme, unslothDarkTheme],
});
const { withSmoothContextProvider } = INTERNAL;

const STREAMDOWN_COMPONENTS = {
  a: ({
    href,
    children,
    ...props
  }: React.ComponentProps<"a">) => (
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
const ACTION_PANEL_CLASS =
  "pointer-events-auto flex shrink-0 items-center gap-1";
const ACTION_BUTTON_CLASS =
  "flex size-8 cursor-pointer items-center justify-center rounded-[10px] text-chat-icon-fg transition-all hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover disabled:cursor-not-allowed disabled:opacity-50";

function getMermaidSource(blockContent: string): string | null {
  const source = blockContent.match(MERMAID_SOURCE_RE)?.[1]?.trim();
  return source && source.length > 0 ? source : null;
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

function renderHighlightedCode(props: BlockProps, codeFence: CodeFenceInfo) {
  return (
    <div className="relative isolate">
      <Block {...props} />
      <CodeBlockActions
        disabled={props.isIncomplete}
        language={codeFence.language}
        source={codeFence.source}
      />
    </div>
  );
}

function StreamdownBlock(props: BlockProps) {
  const hasMermaidFence = props.content.includes("```mermaid");
  const mermaidSource = getMermaidSource(props.content);
  // Fall back to parseIncompleteCodeFence for both still-streaming AND
  // finished-but-unclosed fences (small LLMs routinely drop the closing ```).
  const codeFence =
    parseCodeFence(props.content) ?? parseIncompleteCodeFence(props.content);

  if (props.isIncomplete && hasMermaidFence) {
    return (
      <div className="my-4 flex h-48 items-center justify-center rounded-xl border border-border bg-muted/30 text-sm text-muted-foreground animate-pulse">
        Loading diagram...
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
    const svg = isSvgFence(codeFence);
    const html = !svg && isHtmlFence(codeFence);
    if (svg || html) {
      // Tabbed Preview/Code; locks to Code while streaming.
      return (
        <HtmlSvgRenderer
          language={svg ? "svg" : "html"}
          source={codeFence.source}
          codeView={renderHighlightedCode(props, codeFence)}
          isIncomplete={props.isIncomplete}
        />
      );
    }
    return renderHighlightedCode(props, codeFence);
  }

  return <Block {...props} />;
}
const AUDIO_PLAYER_RE = /<audio-player\s+src="([^"]+)"\s*\/>/;

const MarkdownTextImpl = () => {
  const { text, status } = useMessagePartText();
  const processedText = useMemo(() => preprocessLaTeX(text), [text]);

  const audioMatch = text.match(AUDIO_PLAYER_RE);
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
