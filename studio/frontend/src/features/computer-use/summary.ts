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

/** One line per action, in the order they run; empty when the arguments are still streaming in. */
export function describeComputerCall(argsText: string | undefined): string[] {
  if (!argsText) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(argsText);
  } catch {
    return [];
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return [];
  const raw = parsed as Raw;
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

/** Why a computer call asks, from the backend's `computer_use` field on tool_start. */
export type ComputerUseAsk = { risks: string[] };

export function computerUseAsk(value: unknown): ComputerUseAsk | undefined {
  if (!value || typeof value !== "object") return undefined;
  const risks = (value as { risks?: unknown }).risks;
  if (!Array.isArray(risks)) return undefined;
  return { risks: risks.filter((risk): risk is string => typeof risk === "string").slice(0, 8) };
}
