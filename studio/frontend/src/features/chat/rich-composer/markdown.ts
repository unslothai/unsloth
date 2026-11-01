// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type Mark,
  type Node,
  Schema,
  Slice,
} from "../../../vendor/prosemirror-model/index.js";

export const composerSchema = new Schema({
  nodes: {
    doc: { content: "block+" },
    paragraph: {
      content: "inline*",
      group: "block",
      parseDOM: [{ tag: "p" }],
      toDOM: () => ["p", 0],
    },
    // biome-ignore lint/style/useNamingConvention: ProseMirror node names
    code_block: {
      content: "text*",
      marks: "",
      group: "block",
      code: true,
      defining: true,
      attrs: { language: { default: "" } },
      parseDOM: [{ tag: "pre", preserveWhitespace: "full" }],
      toDOM: (node) => [
        "pre",
        { "data-language": node.attrs.language || null },
        ["code", 0],
      ],
    },
    text: { group: "inline" },
    // biome-ignore lint/style/useNamingConvention: ProseMirror node names
    hard_break: {
      inline: true,
      group: "inline",
      selectable: false,
      linebreakReplacement: true,
      leafText: () => "\n",
      parseDOM: [{ tag: "br" }],
      toDOM: () => ["br"],
    },
  },
  // Schema order is nesting order: code stays innermost.
  marks: {
    bold: {
      parseDOM: [{ tag: "strong" }, { tag: "b" }],
      toDOM: () => ["strong", 0],
    },
    italic: { parseDOM: [{ tag: "em" }, { tag: "i" }], toDOM: () => ["em", 0] },
    strike: { parseDOM: [{ tag: "s" }, { tag: "del" }], toDOM: () => ["s", 0] },
    code: {
      inclusive: false,
      parseDOM: [{ tag: "code" }],
      toDOM: () => ["code", 0],
    },
  },
});

const {
  paragraph,
  code_block: codeBlock,
  hard_break: hardBreak,
} = composerSchema.nodes;
const MARKERS: Record<string, string> = {
  bold: "**",
  italic: "*",
  strike: "~~",
};
const DELIMITERS = [
  ["**", "bold"],
  ["~~", "strike"],
  ["*", "italic"],
] as const;
const OPEN_FENCE = /^```([^\s`]*)$/;
const CLOSE_FENCE = "```";
// Delimiter matching is quadratic on unmatched markers.
const MAX_STYLED_LINE = 2000;

type Run = { text: string; marks: readonly string[] };

function codeSpan(text: string): string {
  const longest = Math.max(
    0,
    ...(text.match(/`+/g) ?? []).map((run) => run.length),
  );
  const ticks = "`".repeat(longest + 1);
  const pad = text.startsWith("`") || text.endsWith("`") ? " " : "";
  return `${ticks}${pad}${text}${pad}${ticks}`;
}

function serializeCodeBlock(language: string, text: string): string {
  return text
    ? `\`\`\`${language}\n${text}\n${CLOSE_FENCE}`
    : `\`\`\`${language}\n${CLOSE_FENCE}`;
}

const isSpace = (char: string | undefined) =>
  char !== undefined && /\s/.test(char);

type Piece = Run & { run: number; at: number };

// `**a **` is not emphasis, so edge spaces move outside the markers.
function expelWhitespace(runs: Run[]): Piece[] {
  const out: Piece[] = [];
  runs.forEach((run, index) => {
    if (run.marks.length === 0 || run.marks.includes("code")) {
      out.push({ ...run, run: index, at: 0 });
      return;
    }
    const prev = runs[index - 1]?.marks ?? [];
    const next = runs[index + 1]?.marks ?? [];
    const lead = /^\s*/.exec(run.text)?.[0] ?? "";
    if (lead.length === run.text.length) {
      const marks = run.marks.filter(
        (m) => prev.includes(m) && next.includes(m),
      );
      out.push({ text: run.text, marks, run: index, at: 0 });
      return;
    }
    const trail = /\s*$/.exec(run.text)?.[0] ?? "";
    const body = run.text.slice(lead.length, run.text.length - trail.length);
    if (lead)
      out.push({
        text: lead,
        marks: run.marks.filter((m) => prev.includes(m)),
        run: index,
        at: 0,
      });
    out.push({ text: body, marks: run.marks, run: index, at: lead.length });
    if (trail) {
      const marks = run.marks.filter((m) => next.includes(m));
      out.push({
        text: trail,
        marks,
        run: index,
        at: lead.length + body.length,
      });
    }
  });
  return out;
}

const sameMarks = (a: readonly string[], b: readonly string[]) =>
  a.length === b.length && a.every((mark, i) => mark === b[i]);

function serializeLine(
  runs: Run[],
  at?: (run: number, runOffset: number, offset: number, length: number) => void,
): string {
  let out = "";
  const open: string[] = [];
  const closeTo = (depth: number) => {
    while (open.length > depth) out += MARKERS[open.pop() as string];
  };
  for (const piece of expelWhitespace(runs)) {
    const wanted = piece.marks.filter((mark) => mark !== "code");
    let keep = 0;
    while (keep < open.length && wanted.includes(open[keep])) keep++;
    closeTo(keep);
    for (const mark of wanted) {
      if (!open.includes(mark)) {
        out += MARKERS[mark];
        open.push(mark);
      }
    }
    const span = piece.marks.includes("code")
      ? codeSpan(piece.text)
      : piece.text;
    at?.(
      piece.run,
      piece.at,
      out.length + (span.length - piece.text.length) / 2,
      piece.text.length,
    );
    out += span;
  }
  closeTo(0);
  return out;
}

const markNames = (marks: readonly Mark[]) =>
  marks.map((mark) => mark.type.name);

type Segment = { pos: number; offset: number; length: number };

export type Serialized = {
  text: string;
  segments: Segment[];
};

const cache = new WeakMap<Node, Serialized>();

export function serializeDoc(doc: Node): Serialized {
  const hit = cache.get(doc);
  if (hit) return hit;
  let text = "";
  const segments: Segment[] = [];
  doc.forEach((block, blockOffset, index) => {
    if (index > 0) text += "\n";
    const start = blockOffset + 1;
    if (block.type === codeBlock) {
      const fence = `\`\`\`${block.attrs.language}\n`;
      segments.push({
        pos: start,
        offset: text.length + fence.length,
        length: block.content.size,
      });
      text += serializeCodeBlock(block.attrs.language, block.textContent);
      return;
    }
    segments.push({ pos: start, offset: text.length, length: 0 });
    let line: Run[] = [];
    let linePositions: number[] = [];
    const flush = () => {
      const base = text.length;
      text += serializeLine(line, (run, runOffset, offset, length) =>
        segments.push({
          pos: linePositions[run] + runOffset,
          offset: base + offset,
          length,
        }),
      );
      line = [];
      linePositions = [];
    };
    block.forEach((child, childOffset) => {
      if (child.type === hardBreak) {
        flush();
        segments.push({
          pos: start + childOffset,
          offset: text.length,
          length: 1,
        });
        text += "\n";
        return;
      }
      if (!child.isText) return;
      const marks = markNames(child.marks);
      const last = line[line.length - 1];
      if (last && sameMarks(last.marks, marks)) {
        last.text += child.text ?? "";
      } else {
        line.push({ text: child.text ?? "", marks });
        linePositions.push(start + childOffset);
      }
    });
    flush();
  });
  const result = { text, segments };
  cache.set(doc, result);
  return result;
}

export function serializeMarkdown(doc: Node): string {
  return serializeDoc(doc).text;
}

// A boundary caret maps before the closing marker, so an `@` token stays contiguous.
export function docPosToOffset(doc: Node, pos: number): number {
  const { segments, text } = serializeDoc(doc);
  let best = 0;
  for (const segment of segments) {
    if (pos < segment.pos) break;
    if (pos <= segment.pos + segment.length)
      return segment.offset + (pos - segment.pos);
    best = segment.offset + segment.length;
  }
  return Math.min(best, text.length);
}

export function offsetToDocPos(doc: Node, offset: number): number {
  const { segments } = serializeDoc(doc);
  for (const segment of segments) {
    if (offset < segment.offset) return segment.pos;
    if (offset <= segment.offset + segment.length)
      return segment.pos + (offset - segment.offset);
  }
  const last = segments[segments.length - 1];
  return last ? last.pos + last.length : 1;
}

function closerAt(line: string, from: number, delimiter: string): number {
  for (
    let j = line.indexOf(delimiter, from);
    j !== -1;
    j = line.indexOf(delimiter, j + 1)
  ) {
    if (isSpace(line[j - 1]) || line[j + delimiter.length] === delimiter[0])
      continue;
    if (delimiter === "*" && line[j - 1] === "*") continue;
    return j;
  }
  return -1;
}

function parseInline(line: string, marks: readonly string[], out: Run[]): void {
  let literal = "";
  const push = (text: string, runMarks: readonly string[]) => {
    const last = out[out.length - 1];
    if (last && sameMarks(last.marks, runMarks)) last.text += text;
    else if (text) out.push({ text, marks: runMarks });
  };
  const flush = () => {
    push(literal, marks);
    literal = "";
  };
  let i = 0;
  scan: while (i < line.length) {
    const char = line[i];
    if (char === "\\" && i + 1 < line.length) {
      literal += line.slice(i, i + 2);
      i += 2;
      continue;
    }
    if (char === "`" && line[i + 1] !== "`" && line[i - 1] !== "`") {
      const end = line.indexOf("`", i + 1);
      if (end > i + 1 && line[end + 1] !== "`") {
        flush();
        push(line.slice(i + 1, end), [...marks, "code"]);
        i = end + 1;
        continue;
      }
    }
    for (const [delimiter, mark] of DELIMITERS) {
      if (marks.includes(mark) || !line.startsWith(delimiter, i)) continue;
      const after = line[i + delimiter.length];
      if (
        after === undefined ||
        isSpace(after) ||
        (delimiter === "*" && after === "*")
      )
        continue;
      const end = closerAt(line, i + delimiter.length + 1, delimiter);
      if (end === -1) continue;
      flush();
      const inner = [...marks, mark].sort(
        (a, b) =>
          Object.keys(MARKERS).indexOf(a) - Object.keys(MARKERS).indexOf(b),
      );
      parseInline(line.slice(i + delimiter.length, end), inner, out);
      i = end + delimiter.length;
      continue scan;
    }
    literal += char;
    i++;
  }
  flush();
}

function lineNodes(line: string, styled: boolean): Node[] {
  if (!line) return [];
  if (styled && line.length <= MAX_STYLED_LINE) {
    const runs: Run[] = [];
    parseInline(line, [], runs);
    if (serializeLine(runs) === line) {
      return runs.map((run) =>
        composerSchema.text(
          run.text,
          run.marks.map((name) => composerSchema.marks[name].create()),
        ),
      );
    }
  }
  return [composerSchema.text(line)];
}

function paragraphOf(lines: string[], styled: boolean): Node {
  const content: Node[] = [];
  lines.forEach((line, index) => {
    if (index > 0) content.push(hardBreak.create());
    content.push(...lineNodes(line, styled));
  });
  return paragraph.create(null, content);
}

function parseBlocks(markdown: string, styled: boolean): Node {
  const lines = markdown.split("\n");
  const blocks: Node[] = [];
  let pending: string[] | null = null;
  const flush = () => {
    if (pending) blocks.push(paragraphOf(pending, styled));
    pending = null;
  };
  for (let i = 0; i < lines.length; i++) {
    const open = styled ? OPEN_FENCE.exec(lines[i]) : null;
    const close = open ? lines.indexOf(CLOSE_FENCE, i + 1) : -1;
    if (open && close !== -1) {
      const body = lines.slice(i + 1, close).join("\n");
      // One empty line and no lines would both read back as an empty block.
      if (
        serializeCodeBlock(open[1], body) ===
        lines.slice(i, close + 1).join("\n")
      ) {
        flush();
        blocks.push(
          codeBlock.create(
            { language: open[1] },
            body ? composerSchema.text(body) : null,
          ),
        );
        i = close;
        continue;
      }
    }
    pending ??= [];
    pending.push(lines[i]);
  }
  flush();
  if (blocks.length === 0) blocks.push(paragraph.create());
  return composerSchema.node("doc", null, blocks);
}

export function parseMarkdown(markdown: string): Node {
  const styled = parseBlocks(markdown, true);
  if (serializeMarkdown(styled) === markdown) return styled;
  return parseBlocks(markdown, false);
}

export function markdownSlice(markdown: string): Slice {
  const doc = parseMarkdown(markdown);
  const first = doc.firstChild;
  if (doc.childCount === 1 && first?.type === paragraph)
    return new Slice(first.content, 0, 0);
  return new Slice(
    doc.content,
    first?.type === paragraph ? 1 : 0,
    doc.lastChild?.type === paragraph ? 1 : 0,
  );
}

export function sliceToMarkdown(slice: Slice): string {
  const first = slice.content.firstChild;
  try {
    const doc = first?.isInline
      ? composerSchema.node("doc", null, [
          paragraph.create(null, slice.content),
        ])
      : composerSchema.node("doc", null, slice.content);
    return serializeMarkdown(doc);
  } catch {
    return slice.content.textBetween(0, slice.content.size, "\n", "\n");
  }
}
