// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { INTERNAL, useMessagePartText } from "@assistant-ui/react";
import { Copy02Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { DownloadIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Block, type BlockProps, Streamdown } from "streamdown";
import "katex/dist/katex.min.css";
import { AudioPlayer } from "./audio-player";

const { withSmoothContextProvider } = INTERNAL;
const COPY_RESET_MS = 2000;
const MERMAID_SOURCE_RE = /```mermaid\s*([\s\S]*?)```/i;
const CODE_FENCE_RE = /^```([^\r\n`]*)\r?\n([\s\S]*?)\r?\n?```$/;
const ACTION_PANEL_CLASS =
  "pointer-events-auto flex shrink-0 items-center gap-2 rounded-md border border-sidebar bg-sidebar/80 px-1.5 py-1 supports-[backdrop-filter]:bg-sidebar/70 supports-[backdrop-filter]:backdrop-blur";
const ACTION_BUTTON_CLASS =
  "cursor-pointer p-1 text-muted-foreground transition-all hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50";

type CodeFence = {
  language: string | null;
  source: string;
};

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
      onClick={() => {
        if (!copyToClipboard(source)) {
          return;
        }
        showCopied();
      }}
    >
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy02Icon}
        className="size-5"
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
    <div className="pointer-events-none absolute top-3.5 right-3 z-20 flex items-center justify-end">
      <div className={ACTION_PANEL_CLASS}>
        <button
          type="button"
          className={ACTION_BUTTON_CLASS}
          title="Copy code"
          disabled={disabled}
          onClick={() => {
            if (!copyToClipboard(source)) {
              return;
            }
            showCopied();
          }}
        >
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy02Icon}
            className="size-3.5"
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
          <DownloadIcon className="size-3.5" />
        </button>
      </div>
    </div>
  );
}

function StreamdownBlock(props: BlockProps) {
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

  if (mermaidSource) {
    return (
      <div className="relative isolate">
        <Block {...props} />
        <MermaidCopyButton source={mermaidSource} />
      </div>
    );
  }

  if (codeFence) {
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

  return <Block {...props} />;
}
const AUDIO_PLAYER_RE = /<audio-player\s+src="([^"]+)"\s*\/>/;

const MarkdownTextImpl = () => {
  const { text, status } = useMessagePartText();

  const audioMatch = text.match(AUDIO_PLAYER_RE);
  if (audioMatch) {
    return <AudioPlayer src={audioMatch[1]} />;
  }

  return (
    <div data-status={status.type}>
      <Streamdown
        mode="streaming"
        isAnimating={status.type === "running"}
        plugins={{ code, math, mermaid }}
        controls={{
          code: false,
          mermaid: {
            fullscreen: true,
            download: true,
            copy: false,
            panZoom: true,
          },
        }}
        shikiTheme={["github-light", "github-dark"]}
        BlockComponent={StreamdownBlock}
      >
        {text}
      </Streamdown>
    </div>
  );
};

export const MarkdownText = withSmoothContextProvider(MarkdownTextImpl);
