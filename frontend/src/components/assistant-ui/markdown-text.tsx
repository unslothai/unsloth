"use client";

import { INTERNAL } from "@assistant-ui/react";
import { StreamdownTextPrimitive } from "@assistant-ui/react-streamdown";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import "katex/dist/katex.min.css";

const { withSmoothContextProvider } = INTERNAL;

const MarkdownTextImpl = () => {
  return (
    <StreamdownTextPrimitive
      plugins={{ code, math, mermaid }}
      controls={true}
    />
  );
};

export const MarkdownText = withSmoothContextProvider(MarkdownTextImpl);
