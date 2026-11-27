// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import githubLight from "@shikijs/themes/github-light";
import { bundledLanguages } from "shiki/langs";
import { createHighlighter, normalizeLanguage, TOKENIZE_LIMITS } from "../../../components/assistant-ui/code-plugin.ts";

const pdfTheme = {
  ...githubLight,
  name: "unsloth-pdf",
  tokenColors: [
    ...(githubLight.tokenColors ?? []),
    { scope: ["string"], settings: { foreground: "#287447" } },
    { scope: ["support.type.property-name", "string.quoted.double.json"], settings: { foreground: "#287447" } },
    { scope: ["support.type.property-name.json"], settings: { foreground: "#9A531B" } },
    { scope: ["keyword", "storage"], settings: { foreground: "#8150A0" } },
    { scope: ["constant.numeric", "constant.language"], settings: { foreground: "#166C86" } },
    { scope: ["comment"], settings: { foreground: "#6E7B73", fontStyle: "italic" } },
  ],
};

export async function highlightPdfCode(body: string): Promise<string> {
  const template = document.createElement("template");
  template.innerHTML = body;
  const fences = template.content.querySelectorAll<HTMLElement>("pre > code[class^='language-']");
  if (!fences.length) return body;

  const highlighter = await createHighlighter({ themes: [pdfTheme], langs: [] });
  try {
    for (const fence of fences) {
      const language = normalizeLanguage(fence.className.slice("language-".length));
      if (!Object.hasOwn(bundledLanguages, language)) continue;
      try {
        await highlighter.loadLanguage(language);
        const { tokens } = highlighter.codeToTokens(fence.textContent ?? "", {
          lang: language,
          theme: "unsloth-pdf",
          ...TOKENIZE_LIMITS,
        });
        const colored = document.createDocumentFragment();
        for (const [index, line] of tokens.entries()) {
          if (index) colored.append("\n");
          for (const token of line) {
            const span = document.createElement("span");
            if (token.color) span.style.color = token.color;
            if (token.fontStyle && token.fontStyle & 1) span.style.fontStyle = "italic";
            if (token.fontStyle && token.fontStyle & 2) span.style.fontWeight = "bold";
            span.textContent = token.content;
            colored.append(span);
          }
        }
        fence.replaceChildren(colored);
      } catch {
        // Keep the original code when a grammar cannot be loaded or tokenized.
      }
    }
    return template.innerHTML;
  } finally {
    highlighter.dispose();
  }
}
