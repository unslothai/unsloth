// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HighlightResult } from "@streamdown/code";
import {
  type CSSProperties,
  createElement,
  useEffect,
  useState,
} from "react";
import { CodeBlockContainer, CodeBlockHeader } from "streamdown";

import { shouldAutoHighlightStreamingCode } from "./streaming-code-policy.ts";

type OversizedStreamingCodeBlockProps = {
  isFenceOpen: boolean;
  language: string | null;
  prepareHighlighted?: (
    source: string,
    language: string | null,
    onReady: (result: HighlightResult) => void,
  ) => () => void;
  source: string;
};

type PreparedHighlight = {
  language: string | null;
  result: HighlightResult;
  source: string;
};

const LINE_NUMBER_CLASS =
  "block before:content-[counter(line)] before:inline-block before:[counter-increment:line] before:w-6 before:mr-4 before:text-ui-13 before:text-right before:text-muted-foreground/50 before:font-mono before:select-none";

function parseRootStyle(rootStyle: HighlightResult["rootStyle"]): CSSProperties {
  if (!rootStyle) return {};
  const style: Record<string, string> = {};
  for (const declaration of rootStyle.split(";")) {
    const separator = declaration.indexOf(":");
    if (separator <= 0) continue;
    const property = declaration.slice(0, separator).trim();
    const value = declaration.slice(separator + 1).trim();
    if (property && value) style[property] = value;
  }
  return style as CSSProperties;
}

function tokenStyle(
  token: HighlightResult["tokens"][number][number],
): CSSProperties {
  const style: Record<string, string> = {};
  if (token.color) style["--sdm-c"] = token.color;
  if (token.bgColor) style["--sdm-tbg"] = token.bgColor;
  for (const [property, value] of Object.entries(token.htmlStyle ?? {})) {
    if (property === "color") style["--sdm-c"] = value;
    else if (property === "background-color") style["--sdm-tbg"] = value;
    else style[property] = value;
  }
  return style as CSSProperties;
}


function CodeBlockBody({
  language,
  result,
}: {
  language: string;
  result: HighlightResult;
}) {
  const rootStyle = {
    ...(result.bg ? { "--sdm-bg": result.bg } : {}),
    ...(result.fg ? { "--sdm-fg": result.fg } : {}),
    ...parseRootStyle(result.rootStyle),
  } as CSSProperties;

  return createElement(
    "div",
    {
      className:
        "overflow-x-auto rounded-md border border-border bg-background p-4 text-sm",
      "data-language": language,
      "data-streamdown": "code-block-body",
    },
    createElement(
      "pre",
      {
        className:
          "bg-[var(--sdm-bg,inherit)] dark:bg-[var(--shiki-dark-bg,var(--sdm-bg,inherit))]",
        style: rootStyle,
      },
      createElement(
        "code",
        { className: "[counter-increment:line_0] [counter-reset:line]" },
        result.tokens.map((line, lineIndex) =>
          createElement(
            "span",
            { className: LINE_NUMBER_CLASS, key: lineIndex },
            line.length === 0 ||
              (line.length === 1 && line[0].content === "")
              ? "\n"
              : line.map((token, tokenIndex) => {
                  const hasBackground = Boolean(
                    token.bgColor || token.htmlStyle?.["background-color"],
                  );
                  return createElement(
                    "span",
                    {
                      className: [
                        "text-[var(--sdm-c,inherit)]",
                        "dark:text-[var(--shiki-dark,var(--sdm-c,inherit))]",
                        hasBackground ? "bg-[var(--sdm-tbg)]" : "",
                        hasBackground
                          ? "dark:bg-[var(--shiki-dark-bg,var(--sdm-tbg))]"
                          : "",
                      ]
                        .filter(Boolean)
                        .join(" "),
                      style: tokenStyle(token),
                      ...token.htmlAttrs,
                      key: `${token.offset ?? tokenIndex}:${tokenIndex}`,
                    },
                    token.content,
                  );
                }),
          ),
        ),
      ),
    ),
  );
}

function PlainCodeBlock({
  isFenceOpen,
  language,
  source,
}: Omit<OversizedStreamingCodeBlockProps, "prepareHighlighted">) {
  const label = language ?? "";
  return createElement(
    CodeBlockContainer,
    { isIncomplete: isFenceOpen, language: label },
    createElement(CodeBlockHeader, { language: label }),
    createElement(
      "div",
      {
        className:
          "overflow-x-auto rounded-md border border-border bg-background p-4 text-sm text-foreground",
        "data-language": label,
        "data-streamdown": "code-block-body",
      },
      createElement(
        "pre",
        { className: "bg-transparent" },
        createElement(
          "code",
          { className: "font-mono whitespace-pre" },
          source,
        ),
      ),
    ),
  );
}

function PreparedCodeBlock({
  language,
  result,
}: {
  language: string | null;
  result: HighlightResult;
}) {
  const label = language ?? "";
  return createElement(
    CodeBlockContainer,
    { isIncomplete: false, language: label },
    createElement(CodeBlockHeader, { language: label }),
    createElement(CodeBlockBody, { language: label, result }),
  );
}

// The exact canonical source is tokenized in bounded tasks while the plain block
// stays mounted. The resulting tree is rendered directly, so Streamdown cannot
// trim a final line ending or issue a second highlighter request. No synthetic
// source character or shared provenance registry is needed.
export function OversizedStreamingCodeBlock({
  isFenceOpen,
  language,
  prepareHighlighted,
  source,
}: OversizedStreamingCodeBlockProps) {
  const [prepared, setPrepared] = useState<PreparedHighlight | null>(null);
  const autoHighlight = shouldAutoHighlightStreamingCode(source);

  useEffect(() => {
    if (
      isFenceOpen ||
      !autoHighlight ||
      !prepareHighlighted ||
      (prepared?.source === source && prepared.language === language)
    ) {
      return;
    }
    return prepareHighlighted(source, language, (result) => {
      setPrepared({ language, result, source });
    });
  }, [
    autoHighlight,
    isFenceOpen,
    language,
    prepareHighlighted,
    prepared?.language,
    prepared?.source,
    source,
  ]);

  if (
    isFenceOpen ||
    !autoHighlight ||
    prepared?.source !== source ||
    prepared.language !== language
  ) {
    return createElement(PlainCodeBlock, { isFenceOpen, language, source });
  }
  return createElement(PreparedCodeBlock, {
    language,
    result: prepared.result,
  });
}
