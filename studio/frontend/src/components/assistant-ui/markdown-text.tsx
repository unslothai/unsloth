"use client";

import { INTERNAL, useMessagePartText } from "@assistant-ui/react";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Copy02Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { Block, type BlockProps, Streamdown } from "streamdown";
import { useEffect, useRef, useState } from "react";
import "katex/dist/katex.min.css";
import { AudioPlayer } from "./audio-player";

const { withSmoothContextProvider, useSmoothStatus } = INTERNAL;

function getMermaidSource(blockContent: string): string | null {
  const source = blockContent.match(/```mermaid\s*([\s\S]*?)```/i)?.[1]?.trim();
  return source && source.length > 0 ? source : null;
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

function StreamdownBlock(props: BlockProps) {
  const hasMermaidFence = props.content.includes("```mermaid");
  const mermaidSource = getMermaidSource(props.content);

  if (props.isIncomplete && hasMermaidFence) {
    return (
      <div className="my-4 flex h-48 items-center justify-center rounded-xl border border-border bg-muted/30 text-sm text-muted-foreground animate-pulse">
        Loading diagram...
      </div>
    );
  }

  if (mermaidSource) {
    return (
      <div className="relative">
        <Block {...props} />
        <MermaidCopyButton source={mermaidSource} />
      </div>
    );
  }

  return <Block {...props} />;
}
const AUDIO_PLAYER_RE = /<audio-player\s+src="([^"]+)"\s*\/>/;

const MarkdownTextImpl = () => {
  const { text } = useMessagePartText();
  const status = useSmoothStatus();

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
