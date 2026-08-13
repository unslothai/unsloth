// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
   * Paint the surrounding UI first and parse on the next idle callback. For documents that
   * arrive in one piece and are large enough that the parse is a visible stall - a finished
   * deep research report, above all - the wait is the same either way, but the rest of the
   * window stays interactive through it.
   */
  defer?: boolean;
};

function MarkdownPreviewImpl({
  markdown,
  className,
  plain = false,
  defer = false,
}: MarkdownPreviewProps): ReactElement {
  // Wiring math and mermaid into a document containing neither still costs a pass over every
  // node; shiki over a very long document costs more than it is worth. Both are decided here
  // rather than always-on, because the report lands in a single synchronous commit.
  const plugins = useMemo<MarkdownPlugins>(() => {
    const needs = markdownPluginNeeds(markdown);
    const next: MarkdownPlugins = {};
    if (needs.code) next.code = code;
    if (needs.math) next.math = math;
    if (needs.mermaid) next.mermaid = mermaid;
    return next;
  }, [markdown]);
  const [ready, setReady] = useState(!defer);
  useEffect(() => {
    if (!defer) {
      setReady(true);
      return;
    }
    setReady(false);
    return scheduleIdleTask(() => setReady(true), 200);
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
