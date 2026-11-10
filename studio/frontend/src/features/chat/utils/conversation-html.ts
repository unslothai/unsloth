// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Marked, type Tokens } from "marked";
import {
  type ConversationMarkdownMessage,
  roleLabel,
} from "./conversation-markdown.ts";

export const CONVERSATION_HTML_FORMAT = "html";
export const CONVERSATION_HTML_LABEL = "HTML";
export const CONVERSATION_HTML_EXTENSION = "html";
export const CONVERSATION_HTML_MIME_TYPE = "text/html";
export const CONVERSATION_PDF_FORMAT = "pdf";
export const CONVERSATION_PDF_LABEL = "PDF";
export const CONVERSATION_PDF_EXTENSION = "pdf";
export const CONVERSATION_PDF_MIME_TYPE = "application/pdf";

export type NamedConversationHtml = {
  readonly title: string;
  readonly messages: readonly ConversationMarkdownMessage[];
};

export function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

// The thinking wrapper is the only markup the markdown export emits on purpose; all other raw HTML is shown as text.
const ALLOWED_RAW_TAG_PATTERN = /&lt;(\/?)(details|summary)&gt;/g;
const DATA_IMAGE_PATTERN =
  /^data:image\/(?:png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=\s]+$/i;

function safeHref(raw: string): string | null {
  const value = raw.trim();
  if (value.startsWith("#")) return value;
  try {
    const url = new URL(value);
    return ["http:", "https:", "mailto:"].includes(url.protocol)
      ? url.href
      : null;
  } catch {
    return null;
  }
}

const markdown = new Marked({
  async: false,
  gfm: true,
  renderer: {
    html({ text }: Tokens.HTML | Tokens.Tag): string {
      return escapeHtml(text).replace(ALLOWED_RAW_TAG_PATTERN, "<$1$2>");
    },
    link({ href, title, tokens }: Tokens.Link): string {
      const text = this.parser.parseInline(tokens);
      const url = safeHref(href);
      if (!url) return text;
      const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
      return `<a href="${escapeHtml(url)}"${titleAttr} rel="noopener noreferrer">${text}</a>`;
    },
    image({ href, text }: Tokens.Image): string {
      if (DATA_IMAGE_PATTERN.test(href)) {
        return `<img src="${escapeHtml(href)}" alt="${escapeHtml(text)}">`;
      }
      // A remote image would phone home when the file is opened, so keep it as a link.
      const url = safeHref(href);
      const label = escapeHtml(text || "image");
      return url
        ? `<a href="${escapeHtml(url)}" rel="noopener noreferrer">[${label}]</a>`
        : `[${label}]`;
    },
  },
});

export function markdownToHtml(text: string): string {
  return markdown.parse(text) as string;
}

function roleClass(role: string): string {
  return ["user", "assistant", "system", "tool"].includes(role)
    ? role
    : "other";
}

function renderMessages(
  messages: readonly ConversationMarkdownMessage[],
): string {
  return messages
    .filter(({ content }) => content.trim())
    .map(
      ({ role, content }) =>
        `<section class="message ${roleClass(role)}">\n<h3 class="role">${escapeHtml(roleLabel(role))}</h3>\n<div class="content">\n${markdownToHtml(content)}</div>\n</section>`,
    )
    .join("\n");
}

export function buildConversationHtmlBody(
  conversations: readonly NamedConversationHtml[],
): string {
  const rendered = conversations
    .map(({ title, messages }) => ({ title, body: renderMessages(messages) }))
    .filter(({ body }) => body);
  return rendered
    .map(({ title, body }) =>
      rendered.length > 1
        ? `<article class="conversation">\n<h2 class="conversation-title">${escapeHtml(title)}</h2>\n${body}\n</article>`
        : `<article class="conversation">\n${body}\n</article>`,
    )
    .join("\n<hr>\n");
}

export const CONVERSATION_EXPORT_CSS = `
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #1f2328; }
h1 { font-size: 18pt; margin: 0 0 4pt 0; }
.meta { color: #6e7781; font-size: 9pt; margin: 0 0 16pt 0; }
.conversation-title { font-size: 14pt; margin: 18pt 0 8pt 0; }
.message { margin: 0 0 14pt 0; padding: 8pt 12pt; border-left: 3pt solid #d0d7de; }
.message.user { background-color: #f6f8fa; border-left-color: #8c959f; }
.message.system { border-left-color: #bf8700; }
.message.tool { border-left-color: #8250df; }
.role { font-size: 9pt; text-transform: uppercase; letter-spacing: 0.04em; color: #57606a; margin: 0 0 4pt 0; }
.content h1, .content h2, .content h3, .content h4 { font-size: 12pt; margin: 10pt 0 4pt 0; }
p { margin: 0 0 6pt 0; }
pre { background-color: #f6f8fa; padding: 6pt 8pt; font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt; white-space: pre-wrap; word-wrap: break-word; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9.5pt; }
blockquote { margin: 0 0 6pt 0; padding: 0 0 0 10pt; border-left: 2pt solid #d0d7de; color: #57606a; }
table { border-collapse: collapse; margin: 0 0 6pt 0; }
th, td { border: 1px solid #d0d7de; padding: 3pt 6pt; text-align: left; }
details { margin: 0 0 6pt 0; color: #57606a; }
summary { font-style: italic; }
img { max-width: 100%; }
a { color: #0969da; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 18pt 0; }
`;

/** A standalone file: inline styles only, and a CSP so nothing in the chat can run or load. */
export function buildConversationHtmlDocument(
  title: string,
  body: string,
  exportedAt: Date = new Date(),
): string {
  const safeTitle = escapeHtml(
    title.replace(/\s+/g, " ").trim() || "Conversation",
  );
  return `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="Unsloth">
<title>${safeTitle}</title>
<style>${CONVERSATION_EXPORT_CSS}
body { max-width: 820px; margin: 32px auto; padding: 0 20px; }
@media print { body { margin: 0 auto; } .message { break-inside: avoid-page; } }
</style>
</head>
<body>
<header>
<h1>${safeTitle}</h1>
<p class="meta">Exported from Unsloth on ${escapeHtml(exportedAt.toLocaleString())}</p>
</header>
${body}
</body>
</html>
`;
}
