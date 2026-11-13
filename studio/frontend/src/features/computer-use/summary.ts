// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A computer call in words for its card: the user approves what they can read, not JSON. */

/** core/inference/computer_use: the server runs it, so the turn needs no live tab. */
export const COMPUTER_USE_TOOL = "computer";

type Raw = Record<string, unknown>;

const text = (value: unknown): string => (typeof value === "string" ? value : "");
const num = (value: unknown): string =>
  typeof value === "number" || (typeof value === "string" && value.trim() !== "" && !Number.isNaN(Number(value)))
    ? String(Math.round(Number(value)))
    : "?";

function point(raw: Raw, prefix = ""): string {
  const pair = raw.coordinate ?? raw.coordinates;
  if (!prefix && Array.isArray(pair) && pair.length === 2) return `(${num(pair[0])}, ${num(pair[1])})`;
  return `(${num(raw[`${prefix}x`])}, ${num(raw[`${prefix}y`])})`;
}

function one(raw: Raw): string {
  const action = text(raw.action || raw.type).toLowerCase();
  switch (action) {
    case "screenshot":
      return "Look at the screen";
    case "zoom":
      return "Look closer at part of the screen";
    case "click":
    case "left_click":
      return `Click at ${point(raw)}`;
    case "double_click":
      return `Double-click at ${point(raw)}`;
    case "right_click":
      return `Right-click at ${point(raw)}`;
    case "move":
    case "mouse_move":
      return `Move the pointer to ${point(raw)}`;
    case "drag":
    case "left_click_drag":
      return `Drag from ${point(raw)} to ${point(raw, "to_")}`;
    case "scroll": {
      const direction = text(raw.direction || raw.scroll_direction) || "down";
      return `Scroll ${direction}${"x" in raw || "coordinate" in raw ? ` at ${point(raw)}` : ""}`;
    }
    case "type":
      return `Type "${text(raw.text)}"${raw.submit === true ? " and press Enter" : ""}`;
    case "key":
    case "keypress": {
      const keys = raw.keys ?? raw.key ?? raw.text;
      return `Press ${Array.isArray(keys) ? keys.join("+") : text(keys)}`;
    }
    case "wait":
      return "Wait";
    case "open_app":
      return `Open ${text(raw.name) || "an application"}`;
    default:
      return action ? `Do: ${action}` : "Use the computer";
  }
}

function parseCall(argsText: string | undefined): Raw | null {
  if (!argsText) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(argsText);
  } catch {
    return null;
  }
  return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Raw) : null;
}

const WHY_CHARS = 80;

/** The model's own few words on what a call is for, as one short line, or "". It is the model's
 *  claim, so a card waiting on Allow shows the steps instead. */
export function computerCallWhy(argsText: string | undefined): string {
  const why = text(parseCall(argsText)?.why).replace(/\s+/g, " ").trim();
  return why.length > WHY_CHARS ? `${why.slice(0, WHY_CHARS - 1)}…` : why;
}

/** One line per action, in the order they run; empty when the arguments are still streaming in. */
export function describeComputerCall(argsText: string | undefined): string[] {
  const raw = parseCall(argsText);
  if (!raw) return [];
  const chain = [raw.then, raw.actions, raw.steps].find(Array.isArray) as unknown[] | undefined;
  const first = raw.action || raw.type ? [raw] : [];
  const rest = (chain ?? []).filter((item): item is Raw => !!item && typeof item === "object" && !Array.isArray(item));
  return [...first, ...rest].map(one);
}

const FENCE = /<\/?untrusted_screen_text>/g;

/** A computer result without the fence the backend puts around window titles; a result with a
 *  screenshot arrives as { text, images }. */
export function plainComputerResult(result: unknown): unknown {
  if (typeof result === "string") return result.replace(FENCE, "");
  if (result && typeof result === "object" && typeof (result as { text?: unknown }).text === "string") {
    return { ...(result as object), text: (result as { text: string }).text.replace(FENCE, "") };
  }
  return result;
}

/** Why a computer call asks, from the backend's `computer_use` field on tool_start: what is
 *  risky about it, and the apps it would be the first to use in this chat. */
export type ComputerUseAsk = { risks: string[]; apps: string[] };

const strings = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string").slice(0, 8) : [];

export function computerUseAsk(value: unknown): ComputerUseAsk | undefined {
  if (!value || typeof value !== "object") return undefined;
  const { risks, apps } = value as { risks?: unknown; apps?: unknown };
  if (!Array.isArray(risks) && !Array.isArray(apps)) return undefined;
  return { risks: strings(risks), apps: strings(apps) };
}

/** The one app "Always allow" would add, or null: a risky call, or one that opens several apps
 *  at once, is answered for this time only. */
export function alwaysAllowableApp(ask: ComputerUseAsk | undefined): string | null {
  return ask && ask.risks.length === 0 && ask.apps.length === 1 ? ask.apps[0] : null;
}
