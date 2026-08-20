


import { markdownPluginNeeds } from "@/lib/markdown-plugins";
import { openLink } from "@/lib/open-link";
import { safeMarkdownUrl } from "@/lib/safe-markdown-url";
import { scheduleIdleTask } from "@/lib/schedule-idle-task";
import { cn } from "@/lib/utils";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import {
  type ComponentProps,
  type ReactElement,
  memo,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Streamdown } from "streamdown";
import "katex/dist/katex.min.css";

type MarkdownPlugins = NonNullable<
  ComponentProps<typeof Streamdown>["plugins"]
>;
const MARKDOWN_COMPONENTS = {
  a: ({ href, children, ...props }: ComponentProps<"a">) => (
    <a
      href={href}
      rel="noopener noreferrer"
      className="cursor-pointer text-primary underline decoration-primary/40 underline-offset-2 transition-colors hover:decoration-primary"
      onClick={(event) => {
        if (href && openLink(href)) {
          event.preventDefault();
        }
      }}
      {...props}
    >
      {children}
    </a>
  ),
};

type MarkdownPreviewProps = {
  markdown: string;
  className?: string;
  plain?: boolean;
  /**
   * Parse on the next idle callback so the surrounding UI paints first. For a document that
   * arrives whole and is big enough to stall - a finished research report - the wait is the same
   * either way, but the window stays interactive through it.
   */
  defer?: boolean;
};

function MarkdownPreviewImpl({
  markdown,
  className,
  plain = false,
  defer = false,
}: MarkdownPreviewProps): ReactElement {
  // Math and mermaid over a document with neither still cost a pass per node, and shiki over a
  // very long one costs more than it is worth; the report lands in one synchronous commit.
  const plugins = useMemo<MarkdownPlugins>(() => {
    const needs = markdownPluginNeeds(markdown);
    const next: MarkdownPlugins = {};
    if (needs.code) next.code = code;
    if (needs.math) next.math = math;
    if (needs.mermaid) next.mermaid = mermaid;
    return next;
  }, [markdown]);
  // Readiness belongs to the markdown value, not the component: resetting it from an effect is
  // one commit late, so the new document is parsed synchronously and thrown away - the stall
  // `defer` exists to avoid, paid twice. Deriving it during render keeps it out of Streamdown.
  const [readyMarkdown, setReadyMarkdown] = useState<string | null>(null);
  const ready = !defer || readyMarkdown === markdown;
  useEffect(() => {
    if (!defer) {
      return;
    }
    return scheduleIdleTask(() => setReadyMarkdown(markdown), 200);
  }, [defer, markdown]);
  const markdownClassName =
    "w-full max-w-none min-w-0 space-y-2 [overflow-wrap:anywhere] [&_*]:max-w-none [&_p]:w-full [&_ul]:w-full [&_ol]:w-full [&_li]:w-full [&_h1]:w-full [&_h2]:w-full [&_h3]:w-full [&_h4]:w-full [&_h5]:w-full [&_h6]:w-full [&_pre]:w-full [&_table]:w-full [&_p]:break-words [&_li]:break-words [&_code]:break-words [&_pre]:whitespace-pre-wrap [&_pre]:break-words";

  return (
    <div
      className={cn(
        plain
          ? "h-full w-full min-w-0 overflow-auto p-2 text-xs leading-relaxed pointer-events-none select-none"
          : "nodrag max-h-56 w-full min-w-0 overflow-auto rounded-md border border-border/60 bg-muted/20 p-2 text-xs leading-relaxed",
        className,
      )}
    >
      {ready ? (
        <Streamdown
          mode="static"
          plugins={plugins}
          components={MARKDOWN_COMPONENTS}
          urlTransform={safeMarkdownUrl}
          controls={false}
          className={markdownClassName}
        >
          {markdown.trim() ? markdown : "_Empty note_"}
        </Streamdown>
      ) : (
        <div className={markdownClassName} aria-busy="true" />
      )}
    </div>
  );
}

export const MarkdownPreview = memo(MarkdownPreviewImpl);
