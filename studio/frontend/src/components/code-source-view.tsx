// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import { Streamdown } from "streamdown";

import { createCodePlugin } from "@/components/assistant-ui/code-plugin";
import { unslothDarkTheme, unslothLightTheme } from "@/components/assistant-ui/code-themes";
import { cn } from "@/lib/utils";

const sourceCodePlugin = createCodePlugin({
  themes: [unslothLightTheme, unslothDarkTheme],
});

// Past either, highlighting a whole file on the main thread freezes the page for seconds; the
// source shows plain, still line-numbered.
const MAX_HIGHLIGHT_CHARS = 200_000;
const MAX_HIGHLIGHT_LINES = 5_000;

function lineCount(source: string): number {
  let lines = 1;
  for (let index = source.indexOf("\n"); index !== -1; index = source.indexOf("\n", index + 1)) {
    lines += 1;
  }
  return lines;
}

/** Monospace with a line-number gutter, for source too large to highlight. */
function PlainSource({ code, lines, className }: { code: string; lines: number; className?: string }) {
  const numbers = useMemo(
    () => Array.from({ length: lines }, (_, index) => index + 1).join("\n"),
    [lines],
  );
  return (
    <div className={cn("h-full overflow-auto font-mono text-xs leading-relaxed", className)}>
      <div className="flex min-w-max">
        <pre aria-hidden="true" className="m-0 select-none pr-4 text-right text-muted-foreground/60">
          {numbers}
        </pre>
        <pre className="m-0 flex-1">{code}</pre>
      </div>
    </div>
  );
}

function buildFence(source: string, language: string): string {
  const longestBacktickRun = Math.max(
    2,
    ...(source.match(/`+/g) ?? []).map((match) => match.length),
  );
  const fence = "`".repeat(longestBacktickRun + 1);
  return `${fence}${language}\n${source}\n${fence}`;
}

/** Read-only source, highlighted and line-numbered, styled as the chat canvas shows its source. */
export function CodeSourceView({
  code,
  language,
  className,
}: {
  code: string;
  language: string;
  className?: string;
}) {
  const lines = useMemo(() => lineCount(code), [code]);
  const plain = code.length > MAX_HIGHLIGHT_CHARS || lines > MAX_HIGHLIGHT_LINES;
  const markdown = useMemo(
    () => (plain ? "" : buildFence(code, language)),
    [plain, code, language],
  );
  if (plain) return <PlainSource code={code} lines={lines} className={className} />;
  return (
    <div
      className={cn(
        "h-full overflow-auto text-xs leading-relaxed [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!gap-0 [&_[data-streamdown=code-block]]:!rounded-none [&_[data-streamdown=code-block]]:!border-0 [&_[data-streamdown=code-block]]:!bg-transparent [&_[data-streamdown=code-block]]:!p-0 [&_[data-streamdown=code-block-body]]:!border-0 [&_[data-streamdown=code-block-body]]:!bg-transparent [&_[data-streamdown=code-block-body]]:!p-0 [&_pre]:!m-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:text-xs [&_pre]:leading-relaxed [&_code]:text-xs",
        className,
      )}
    >
      <Streamdown
        // Whole, unchanging source: nothing to stream.
        mode="static"
        plugins={{ code: sourceCodePlugin }}
        controls={{ code: false }}
        shikiTheme={[unslothLightTheme, unslothDarkTheme]}
      >
        {markdown}
      </Streamdown>
    </div>
  );
}
