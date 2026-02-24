import { cn } from "@/lib/utils";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { memo, type ReactElement } from "react";
import { Streamdown } from "streamdown";
import "katex/dist/katex.min.css";

const MARKDOWN_PLUGINS = { code, math, mermaid } as const;

type MarkdownPreviewProps = {
  markdown: string;
  className?: string;
  plain?: boolean;
};

function MarkdownPreviewImpl({
  markdown,
  className,
  plain = false,
}: MarkdownPreviewProps): ReactElement {
  return (
    <div
      className={cn(
        plain
          ? "nodrag h-full w-full overflow-auto p-2 text-xs leading-relaxed"
          : "nodrag max-h-56 overflow-auto rounded-md border border-border/60 bg-muted/20 p-2 text-xs leading-relaxed",
        className,
      )}
    >
      <Streamdown mode="static" plugins={MARKDOWN_PLUGINS} controls={false}>
        {markdown.trim() ? markdown : "_Empty note_"}
      </Streamdown>
    </div>
  );
}

export const MarkdownPreview = memo(MarkdownPreviewImpl);
