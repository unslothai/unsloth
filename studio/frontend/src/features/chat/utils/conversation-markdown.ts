export type ConversationMarkdownMessage = {
  readonly role: string;
  readonly content: string;
};

export const CONVERSATION_MARKDOWN_FORMAT = "markdown";
export const CONVERSATION_MARKDOWN_LABEL = "Markdown";
export const CONVERSATION_MARKDOWN_EXTENSION = "md";
export const CONVERSATION_MARKDOWN_MIME_TYPE = "text/markdown";

const ROLE_LABELS: Readonly<Record<string, string>> = {
  assistant: "Assistant",
  system: "System",
  user: "User",
};

function roleLabel(role: string): string {
  const knownLabel = ROLE_LABELS[role];
  if (knownLabel) {
    return knownLabel;
  }
  return role.length > 0
    ? `${role[0]?.toUpperCase()}${role.slice(1)}`
    : "Message";
}

// A message split into renderable pieces. The training exports flatten
// everything to plain text, which is fine for JSONL/CSV but leaves tool calls
// as a one-line JSON blob and thinking as literal [thinking] markers. Markdown
// is a document format, so it renders each piece on its own terms.
export type ConversationMarkdownBlock =
  | { readonly kind: "text"; readonly text: string }
  | { readonly kind: "thinking"; readonly text: string }
  | {
      readonly kind: "tool-call";
      readonly name: string;
      readonly args?: unknown;
      readonly result?: unknown;
    }
  | { readonly kind: "attachment"; readonly label: string };

// Long enough to not be closed early by backticks inside the body.
function fence(body: string, language = ""): string {
  const longestRun = [...body.matchAll(/`+/g)].reduce(
    (max, [run]) => Math.max(max, run.length),
    0,
  );
  const ticks = "`".repeat(Math.max(3, longestRun + 1));
  return `${ticks}${language}\n${body}\n${ticks}`;
}

function inlineCode(value: string): string {
  const longestRun = [...value.matchAll(/`+/g)].reduce(
    (max, [run]) => Math.max(max, run.length),
    0,
  );
  const ticks = "`".repeat(longestRun + 1);
  // Padding keeps a leading or trailing backtick from closing the span.
  const pad = value.startsWith("`") || value.endsWith("`") ? " " : "";
  return `${ticks}${pad}${value}${pad}${ticks}`;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// A multi-line string is almost always source: html, a patch, a script. Fence
// it as-is so it reads as code instead of a JSON string full of \n. No language
// tag, since guessing one wrong is worse than none.
function renderValue(label: string, value: unknown): string[] {
  if (typeof value === "string") {
    if (value.includes("\n")) return [`**${label}:**`, fence(value)];
    // Prose stays prose; anything that could be read as markup or a fence
    // becomes an inline code span so it renders as itself.
    const inert = /[<`]/.test(value) ? inlineCode(value) : value;
    return [`**${label}:** ${inert}`];
  }
  if (value === undefined) return [];
  return [`**${label}:**`, fence(JSON.stringify(value, null, 2), "json")];
}

function renderBlock(block: ConversationMarkdownBlock): string {
  if (block.kind === "text") {
    return block.text.trim();
  }
  if (block.kind === "thinking") {
    // Collapsed so a transcript reads as the conversation first, with the
    // reasoning still there for anyone who wants it.
    return `<details>\n<summary>thinking</summary>\n\n${block.text.trim()}\n\n</details>`;
  }
  if (block.kind === "attachment") {
    return block.label;
  }
  const parts: string[] = [`**tool call:** \`${block.name}\``];
  if (isPlainObject(block.args)) {
    for (const [key, value] of Object.entries(block.args)) {
      parts.push(...renderValue(key, value));
    }
  } else if (block.args !== undefined) {
    parts.push(...renderValue("args", block.args));
  }
  if (block.result !== undefined) {
    parts.push(...renderValue("result", block.result));
  }
  return parts.join("\n\n");
}

export function renderConversationBlocks(
  blocks: readonly ConversationMarkdownBlock[],
): string {
  return blocks
    .map((block) => renderBlock(block))
    .filter((rendered) => rendered.length > 0)
    .join("\n\n");
}

export function buildConversationMarkdown(
  messages: readonly ConversationMarkdownMessage[],
): string {
  const sections = messages.flatMap(({ role, content }) => {
    if (!content.trim()) {
      return [];
    }
    const label = roleLabel(role);
    return [`## ${label}\n\n${content}`];
  });
  return sections.length > 0 ? `${sections.join("\n\n")}\n` : "";
}
