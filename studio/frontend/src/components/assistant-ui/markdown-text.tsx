"use client";

import { INTERNAL, useMessagePartText } from "@assistant-ui/react";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Copy02Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { DownloadIcon } from "lucide-react";
import { Block, type BlockProps, Streamdown } from "streamdown";
import { useEffect, useRef, useState } from "react";
import "katex/dist/katex.min.css";
import { AudioPlayer } from "./audio-player";

const { withSmoothContextProvider } = INTERNAL;

function getMermaidSource(blockContent: string): string | null {
  const source = blockContent.match(/```mermaid\s*([\s\S]*?)```/i)?.[1]?.trim();
  return source && source.length > 0 ? source : null;
}

function getCodeFence(
  blockContent: string,
): { language: string | null; source: string } | null {
  const match = blockContent
    .trimEnd()
    .match(/^```([^\n`]*)\n([\s\S]*?)\n?```$/);
  if (!match) return null;

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
  const ext = normalized ? extByLanguage[normalized] || normalized : "txt";
  return `snippet.${ext}`;
}

function downloadTextFile(filename: string, text: string) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

const COPY_RESET_MS = 2000;

function MermaidCopyButton({ source }: { source: string }) {
  const [copied, setCopied] = useState(false);
  const resetTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (resetTimeoutRef.current) {
        clearTimeout(resetTimeoutRef.current);
      }
    };
  }, []);

  return (
    <button
      type="button"
      className="absolute top-3.5 right-20 z-20 cursor-pointer text-muted-foreground transition-all hover:text-foreground"
      title="Copy Mermaid source"
      onClick={() => {
        if (!copyToClipboard(source)) return;
        setCopied(true);
        if (resetTimeoutRef.current) {
          clearTimeout(resetTimeoutRef.current);
        }
        resetTimeoutRef.current = setTimeout(() => {
          setCopied(false);
          resetTimeoutRef.current = null;
        }, COPY_RESET_MS);
      }}
    >
      <HugeiconsIcon icon={copied ? Tick02Icon : Copy02Icon} className="size-5" />
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
  const [copied, setCopied] = useState(false);
  const resetTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (resetTimeoutRef.current) {
        clearTimeout(resetTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="pointer-events-none absolute top-3.5 right-3 z-20 flex items-center justify-end">
      <div className="pointer-events-auto flex shrink-0 items-center gap-2 rounded-md border border-sidebar bg-sidebar/80 px-1.5 py-1 supports-[backdrop-filter]:bg-sidebar/70 supports-[backdrop-filter]:backdrop-blur">
        <button
          type="button"
          className="cursor-pointer p-1 text-muted-foreground transition-all hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
          title="Copy code"
          disabled={disabled}
          onClick={() => {
            if (!copyToClipboard(source)) return;
            setCopied(true);
            if (resetTimeoutRef.current) {
              clearTimeout(resetTimeoutRef.current);
            }
            resetTimeoutRef.current = setTimeout(() => {
              setCopied(false);
              resetTimeoutRef.current = null;
            }, COPY_RESET_MS);
          }}
        >
          <HugeiconsIcon icon={copied ? Tick02Icon : Copy02Icon} className="size-3.5" />
        </button>
        <button
          type="button"
          className="cursor-pointer p-1 text-muted-foreground transition-all hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
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
