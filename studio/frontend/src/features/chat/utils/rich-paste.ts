// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface PasteNode {
  nodeType: number;
  nodeName: string;
  childNodes: ArrayLike<PasteNode>;
  textContent: string | null;
  getAttribute?: (name: string) => string | null;
}

type Ctx = { pre: boolean; bold: boolean; italic: boolean };

const STRUCTURE =
  /<a\s[^>]*\bhref|<(?:table|ul|ol|h[1-6]|blockquote|strong|em|del|s|b|i)[\s>]/i;
const SKIP = new Set([
  "SCRIPT", "STYLE", "HEAD", "TITLE", "META", "LINK", "NOSCRIPT", "TEMPLATE",
  "SVG", "BUTTON", "INPUT", "SELECT", "TEXTAREA", "IMG", "PICTURE", "VIDEO",
  "AUDIO", "IFRAME", "OBJECT", "CANVAS",
]);
const BLOCK = new Set([
  "P", "DIV", "SECTION", "ARTICLE", "HEADER", "FOOTER", "MAIN", "ASIDE",
  "NAV", "FIGURE", "FIGCAPTION", "DL", "DT", "DD", "BODY", "HTML", "CAPTION",
]);
const INDENT = "\u0001";
const FENCE_OPEN = "\u0002";
const FENCE_CLOSE = "\u0003";

function attr(node: PasteNode, name: string): string {
  return node.getAttribute?.(name) ?? "";
}

function elements(node: PasteNode): PasteNode[] {
  return Array.from(node.childNodes).filter((child) => child.nodeType === 1);
}

function children(node: PasteNode, ctx: Ctx): string {
  return Array.from(node.childNodes)
    .map((child) => convert(child, ctx))
    .join("");
}

const FENCED = new RegExp(`${FENCE_OPEN}[\\s\\S]*?${FENCE_CLOSE}`, "g");

function clean(text: string): string {
  const fences: string[] = [];
  return text
    .replace(FENCED, (block) => `${FENCE_OPEN}${fences.push(block) - 1}${FENCE_CLOSE}`)
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
    .replace(new RegExp(`${FENCE_OPEN}(\\d+)${FENCE_CLOSE}`, "g"), (_, i: string) => fences[Number(i)]);
}

function wrap(marker: string, inner: string): string {
  const body = inner.trim();
  if (!body) return inner;
  const lead = inner.slice(0, inner.length - inner.trimStart().length);
  const trail = inner.slice(inner.trimEnd().length);
  return `${lead}${marker}${body}${marker}${trail}`;
}

function safeHref(raw: string): string | null {
  const href = raw.trim();
  if (!/^(?:https?:|mailto:)/i.test(href)) return null;
  return href.replace(/ /g, "%20").replace(/\(/g, "%28").replace(/\)/g, "%29");
}

function codeSpan(text: string): string {
  const longest = Math.max(0, ...(text.match(/`+/g) ?? []).map((run) => run.length));
  const ticks = "`".repeat(longest + 1);
  const pad = text.startsWith("`") || text.endsWith("`") ? " " : "";
  return `${ticks}${pad}${text}${pad}${ticks}`;
}

function preBlock(node: PasteNode): string {
  const code = elements(node).find((child) => child.nodeName.toUpperCase() === "CODE");
  const cls = `${attr(node, "class")} ${code ? attr(code, "class") : ""}`;
  const lang = /(?:language|lang)-([\w+#-]+)/.exec(cls)?.[1] ?? "";
  const text = (node.textContent ?? "").replace(/\n$/, "");
  const fence = "`".repeat(Math.max(3, ...(text.match(/`{3,}/g) ?? []).map((run) => run.length + 1)));
  return `\n\n${FENCE_OPEN}${fence}${lang}\n${text}\n${fence}${FENCE_CLOSE}\n\n`;
}

function list(node: PasteNode, ctx: Ctx, ordered: boolean): string {
  let index = Number.parseInt(attr(node, "start"), 10) || 1;
  const lines: string[] = [];
  for (const item of elements(node)) {
    const tag = item.nodeName.toUpperCase();
    if (tag === "UL" || tag === "OL") {
      for (const line of list(item, ctx, tag === "OL").split("\n"))
        lines.push(`${INDENT.repeat(ordered ? 3 : 2)}${line}`);
      continue;
    }
    const marker = ordered ? `${index++}.` : "-";
    // Blank lines are kept only inside fenced code.
    let fenced = false;
    const body = clean(children(item, ctx))
      .split("\n")
      .filter((line) => {
        const keep = fenced || line.trim() !== "";
        if (line.includes(FENCE_OPEN)) fenced = true;
        if (line.includes(FENCE_CLOSE)) fenced = false;
        return keep;
      });
    const [first = "", ...rest] = body;
    lines.push(`${marker} ${first}`);
    for (const line of rest) lines.push(line ? `${INDENT.repeat(marker.length + 1)}${line}` : "");
  }
  return lines.join("\n");
}

function tableRows(node: PasteNode): PasteNode[] {
  const rows: PasteNode[] = [];
  for (const child of elements(node)) {
    const tag = child.nodeName.toUpperCase();
    if (tag === "TR") rows.push(child);
    else if (tag === "THEAD" || tag === "TBODY" || tag === "TFOOT") rows.push(...tableRows(child));
  }
  return rows;
}

function span(cell: PasteNode, name: string, max: number): number {
  const value = Number.parseInt(attr(cell, name), 10);
  return Number.isFinite(value) && value > 0 ? Math.min(value, max) : 1;
}

function table(node: PasteNode, ctx: Ctx): string {
  const physical = tableRows(node);
  // Lay merged cells onto the grid so every value stays under its own header.
  const rows: string[][] = physical.map(() => []);
  physical.forEach((row, r) => {
    let c = 0;
    for (const cell of elements(row).filter((child) => /^T[DH]$/i.test(child.nodeName))) {
      while (rows[r][c] !== undefined) c++;
      const text = clean(children(cell, ctx))
        .replace(new RegExp(`[${INDENT}${FENCE_OPEN}${FENCE_CLOSE}]`, "g"), "")
        .replace(/\s*\n\s*/g, " ")
        .replace(/\|/g, "\\|");
      const cols = span(cell, "colspan", 1000);
      const down = span(cell, "rowspan", physical.length - r);
      for (let dr = 0; dr < down; dr++)
        for (let dc = 0; dc < cols; dc++) rows[r + dr][c + dc] = dr === 0 && dc === 0 ? text : "";
      c += cols;
    }
  });
  const width = Math.max(0, ...rows.map((row) => row.length));
  if (rows.length === 0 || width === 0) return children(node, ctx);
  const line = (cells: string[]) =>
    `| ${Array.from({ length: width }, (_, i) => cells[i] ?? "").join(" | ")} |`;
  const [head, ...body] = rows;
  return [line(head), `| ${Array(width).fill("---").join(" | ")} |`, ...body.map(line)].join("\n");
}

function convert(node: PasteNode, ctx: Ctx): string {
  if (node.nodeType === 3) {
    const text = node.textContent ?? "";
    return ctx.pre ? text : text.replace(/\s+/g, " ");
  }
  if (node.nodeType !== 1) return "";
  const tag = node.nodeName.toUpperCase();
  if (SKIP.has(tag)) return "";
  const style = attr(node, "style").toLowerCase();
  switch (tag) {
    case "BR":
      return "\n";
    case "HR":
      return "\n\n---\n\n";
    case "A": {
      const raw = attr(node, "href").trim();
      const href = safeHref(raw);
      const inner = children(node, ctx);
      if (!href) return inner;
      const text = inner.trim();
      const lead = inner.slice(0, inner.length - inner.trimStart().length);
      const trail = inner.slice(inner.trimEnd().length);
      if (!text || text === raw || text === href || `mailto:${text}` === raw) return `${lead}${href}${trail}`;
      return `${lead}[${text.replace(/([[\]])/g, "\\$1")}](${href})${trail}`;
    }
    case "STRONG":
    case "B":
      // Google Docs wraps every copy in <b style="font-weight:normal">.
      if (ctx.bold || /font-weight:\s*(?:normal|[1-4]00)/.test(style)) return children(node, ctx);
      return wrap("**", children(node, { ...ctx, bold: true }));
    case "EM":
    case "I":
      if (ctx.italic) return children(node, ctx);
      return wrap("*", children(node, { ...ctx, italic: true }));
    case "DEL":
    case "S":
    case "STRIKE":
      return wrap("~~", children(node, ctx));
    case "CODE":
      return ctx.pre ? children(node, ctx) : codeSpan(node.textContent ?? "");
    case "PRE":
      return preBlock(node);
    case "H1":
    case "H2":
    case "H3":
    case "H4":
    case "H5":
    case "H6":
      return `\n\n${"#".repeat(Number(tag[1]))} ${clean(children(node, { ...ctx, bold: true })).replace(/\n+/g, " ")}\n\n`;
    case "BLOCKQUOTE":
      return `\n\n${clean(children(node, ctx))
        .split("\n")
        .map((line) => (line ? `> ${line}` : ">"))
        .join("\n")}\n\n`;
    case "UL":
    case "OL":
      return `\n\n${list(node, ctx, tag === "OL")}\n\n`;
    case "TABLE":
      return `\n\n${table(node, ctx)}\n\n`;
    case "SPAN": {
      let inner = children(node, {
        ...ctx,
        bold: ctx.bold || /font-weight:\s*(?:bold|[6-9]00)/.test(style),
        italic: ctx.italic || /font-style:\s*italic/.test(style),
      });
      if (!ctx.italic && /font-style:\s*italic/.test(style)) inner = wrap("*", inner);
      if (!ctx.bold && /font-weight:\s*(?:bold|[6-9]00)/.test(style)) inner = wrap("**", inner);
      return inner;
    }
    default:
      return BLOCK.has(tag) ? `\n\n${children(node, ctx)}\n\n` : children(node, ctx);
  }
}

export function htmlNodeToMarkdown(root: PasteNode): string {
  return clean(convert(root, { pre: false, bold: false, italic: false })).replace(
    new RegExp(`[${INDENT}${FENCE_OPEN}${FENCE_CLOSE}]`, "g"),
    (mark) => (mark === INDENT ? " " : ""),
  );
}

const squash = (text: string) => text.replace(/\s+/g, " ").trim();

export function richPasteMarkdown(
  data: Pick<DataTransfer, "getData" | "types"> | null | undefined,
  parse: (html: string) => PasteNode | null = parseHtml,
): string | null {
  if (!data) return null;
  // Skip VS Code and other editors' syntax-colored HTML.
  if (Array.from(data.types ?? []).includes("vscode-editor-data")) return null;
  const html = data.getData("text/html");
  if (!html || !STRUCTURE.test(html)) return null;
  const root = parse(html);
  if (!root) return null;
  const markdown = htmlNodeToMarkdown(root);
  const plain = data.getData("text/plain");
  if (!markdown || squash(markdown) === squash(plain)) return null;
  // Restore the copied text's edge whitespace, which conversion trims.
  return `${/^\s*/.exec(plain)?.[0] ?? ""}${markdown}${/\s*$/.exec(plain)?.[0] ?? ""}`;
}

function parseHtml(html: string): PasteNode | null {
  if (typeof DOMParser === "undefined") return null;
  // Inert document: nothing in it runs or loads.
  return new DOMParser().parseFromString(html, "text/html").body as unknown as PasteNode;
}

export function insertAtCaret(
  input: HTMLTextAreaElement,
  text: string,
  fallback: (next: string, caret: number) => void,
): void {
  input.focus();
  // execCommand is the only insertion that joins the native undo stack.
  if (document.execCommand?.("insertText", false, text)) return;
  const { selectionStart, selectionEnd, value } = input;
  fallback(value.slice(0, selectionStart) + text + value.slice(selectionEnd), selectionStart + text.length);
}
