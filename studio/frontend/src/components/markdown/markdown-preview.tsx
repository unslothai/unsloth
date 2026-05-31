// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
      <Streamdown
        mode="static"
        plugins={MARKDOWN_PLUGINS}
        controls={false}
        className={markdownClassName}
      >
        {markdown.trim() ? markdown : "_Empty note_"}
      </Streamdown>
    </div>
  );
}

export const MarkdownPreview = memo(MarkdownPreviewImpl);
