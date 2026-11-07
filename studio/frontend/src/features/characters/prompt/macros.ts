// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Card macros: the ones character cards in the wild rely on, resolved the way other card apps do. */
export interface MacroContext {
  char: string;
  user: string;
  /** What {{original}} stands for inside a card's prompt override. */
  original?: string;
  /** Every character in the chat, for {{group}}. */
  group?: string[];
  now?: Date;
  random?: () => number;
}

const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

// Trimmed first: \s* on both sides of a long whitespace run backtracks polynomially.
function roll(spec: string, random: () => number): string {
  const match = /^(\d*)d(\d+)\s*([+-]\s*\d+)?$/i.exec(spec.trim());
  if (!match) {
    const sides = Number.parseInt(spec, 10);
    return Number.isFinite(sides) && sides > 0 ? String(1 + Math.floor(random() * sides)) : spec;
  }
  const count = Math.min(100, Math.max(1, Number.parseInt(match[1] || "1", 10)));
  const sides = Math.max(1, Number.parseInt(match[2], 10));
  let total = match[3] ? Number.parseInt(match[3].replace(/\s+/g, ""), 10) : 0;
  for (let i = 0; i < count; i += 1) total += 1 + Math.floor(random() * sides);
  return String(total);
}

function choices(body: string): string[] {
  // {{random::a::b}} allows commas inside options; {{random:a,b}} is the short form.
  if (body.startsWith(":")) return body.slice(1).split("::");
  return body.split(/(?<!\\),/).map((part) => part.replace(/\\,/g, ","));
}

/** Removes {{// comments}}; indexOf rather than a lazy regex, which rescans to the end for each unclosed one. */
function stripComments(text: string): string {
  let out = "";
  let from = 0;
  for (;;) {
    const open = text.indexOf("{{//", from);
    const close = open < 0 ? -1 : text.indexOf("}}", open + 4);
    if (close < 0) return out + text.slice(from);
    out += text.slice(from, open);
    from = close + 2;
  }
}

export function applyMacros(text: string, context: MacroContext): string {
  if (!text) return "";
  const now = context.now ?? new Date();
  const random = context.random ?? Math.random;
  const group = context.group ?? [context.char];
  // Names go in through callbacks: as replacement strings, "$&" or "$'" in a name would expand.
  let out = stripComments(text)
    .replace(/<BOT>/gi, () => context.char)
    .replace(/<USER>/gi, () => context.user);
  out = out.replace(/\{\{([^{}]*)\}\}/g, (whole, inner: string) => {
    const raw = inner.trim();
    const lower = raw.toLowerCase();
    switch (lower) {
      case "char":
        return context.char;
      case "user":
        return context.user;
      case "original":
        return context.original ?? "";
      case "group":
      case "charifnotgroup":
        return lower === "charifnotgroup" && group.length <= 1 ? context.char : group.join(", ");
      case "time":
        return now.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
      case "date":
        return now.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
      case "weekday":
        return WEEKDAYS[now.getDay()];
      case "isotime":
        return `${pad(now.getHours())}:${pad(now.getMinutes())}`;
      case "isodate":
        return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
      case "newline":
        return "\n";
      case "noop":
      case "trim":
        return whole;
      default:
        break;
    }
    const colon = raw.indexOf(":");
    if (colon > 0) {
      const name = raw.slice(0, colon).trim().toLowerCase();
      const rest = raw.slice(colon + 1);
      if (name === "random" || name === "pick") {
        const options = choices(rest).map((option) => option.trim());
        return options.length ? options[Math.floor(random() * options.length)] : "";
      }
      if (name === "roll") return roll(rest, random);
    }
    return whole;
  });
  // Split rather than /\s*{{trim}}\s*/g, which is quadratic on a long whitespace run.
  const pieces = out.split(/\{\{\s*trim\s*\}\}/i);
  const trimmed = pieces
    .map((piece, index) => {
      const start = index > 0 ? piece.trimStart() : piece;
      return index < pieces.length - 1 ? start.trimEnd() : start;
    })
    .join("");
  return trimmed.replace(/\{\{\s*noop\s*\}\}/gi, "");
}

/** Only the name macros, for text that must read the same on every send. */
export function applyNameMacros(text: string, char: string, user: string): string {
  return text.replace(/\{\{\s*char\s*\}\}|<BOT>/gi, () => char).replace(/\{\{\s*user\s*\}\}|<USER>/gi, () => user);
}
