// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No imports: executor.ts loads this without the file viewers.

export type FileTabKind = "preview" | "text" | "document" | "image" | "media" | "other";

export type FileEdit = { oldText?: string; newText?: string; replaceAll?: boolean; content?: string };

export const MAX_EDIT_BYTES = 5 * 1024 * 1024;

const READ_CHARS = 20_000;
const LINE_CHARS = 2_000;

export function numberedLines(text: string, start: number): { text: string; note: string } {
  const lines = text.split(/\r?\n/);
  if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
  const total = lines.length;
  if (start > total) return { text: "", note: `The file has ${total} line${total === 1 ? "" : "s"}.` };
  const out: string[] = [];
  let used = 0;
  let end = start - 1;
  for (let index = start - 1; index < total; index++) {
    const raw = lines[index] ?? "";
    const line = raw.length > LINE_CHARS ? `${raw.slice(0, LINE_CHARS)} … (line cut; ${raw.length} characters)` : raw;
    const row = `${index + 1}\t${line}`;
    if (out.length && used + row.length > READ_CHARS) break;
    out.push(row);
    used += row.length + 1;
    end = index + 1;
  }
  const rest = end < total ? ` browser_read_file with start_line ${end + 1} reads on.` : "";
  return { text: out.join("\n"), note: `Lines ${start}-${end} of ${total}.${rest}` };
}

const count = (source: string, part: string) => source.split(part).length - 1;

/** Errors are this tool's own words: they reach the model outside the untrusted-content fence. */
export function applyFileEdit(source: string, edit: FileEdit): { text: string; line: number } | { error: string } {
  if (edit.content !== undefined) return { text: edit.content, line: 1 };
  let old = edit.oldText ?? "";
  let next = edit.newText ?? "";
  if (!old) return { error: "old_text is empty." };
  if (!source.includes(old) && source.includes("\r\n") && old.includes("\n")) {
    old = old.replace(/\r?\n/g, "\r\n");
    next = next.replace(/\r?\n/g, "\r\n");
  }
  const found = count(source, old);
  if (found === 0) {
    return { error: "old_text is not in the file. Read it again with browser_read_file and copy the text exactly, without the line numbers." };
  }
  if (found > 1 && !edit.replaceAll) {
    return { error: `old_text appears ${found} times. Add the text around it so it matches once, or set replace_all.` };
  }
  const line = count(source.slice(0, source.indexOf(old)), "\n") + 1;
  return { text: edit.replaceAll ? source.split(old).join(next) : source.replace(old, () => next), line };
}
