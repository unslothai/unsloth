"use client";

import { INTERNAL, useMessagePartText } from "@assistant-ui/react";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { Streamdown } from "streamdown";
import "katex/dist/katex.min.css";

const { withSmoothContextProvider, useSmoothStatus } = INTERNAL;

const MarkdownTextImpl = () => {
  const { text } = useMessagePartText();
  const status = useSmoothStatus();

  return (
    <div data-status={status.type}>
      <Streamdown
        mode="streaming"
        isAnimating={status.type === "running"}
        plugins={{ code, math, mermaid }}
        controls={true}
        shikiTheme={["github-light", "github-dark"]}
      >
        {text}
      </Streamdown>
    </div>
  );
};

export const MarkdownText = withSmoothContextProvider(MarkdownTextImpl);
