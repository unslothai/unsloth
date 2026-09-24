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
  const markdown = useMemo(() => buildFence(code, language), [code, language]);
  return (
    <div
      className={cn(
        "h-full overflow-auto text-xs leading-relaxed [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!gap-0 [&_[data-streamdown=code-block]]:!rounded-none [&_[data-streamdown=code-block]]:!border-0 [&_[data-streamdown=code-block]]:!bg-transparent [&_[data-streamdown=code-block]]:!p-0 [&_[data-streamdown=code-block-body]]:!border-0 [&_[data-streamdown=code-block-body]]:!bg-transparent [&_[data-streamdown=code-block-body]]:!p-0 [&_pre]:!m-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:text-xs [&_pre]:leading-relaxed [&_code]:text-xs",
        className,
      )}
    >
      <Streamdown
        mode="streaming"
        plugins={{ code: sourceCodePlugin }}
        controls={{ code: false }}
        shikiTheme={[unslothLightTheme, unslothDarkTheme]}
      >
        {markdown}
      </Streamdown>
    </div>
  );
}
