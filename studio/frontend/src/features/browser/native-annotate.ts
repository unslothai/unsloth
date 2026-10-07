// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Annotate in a desktop native view: commands go through `browser_view_annotate`
 *  (src-tauri/src/browser_webview.rs) and reports are polled, since the view can't message back. */

import { type AnnotateEvent, parseFrameMessage } from "./frame-message";
import { callNative } from "./native-support";
import type { FrameCommand } from "./page-frame";

// Hover outlines draw in the page; this only paces marks and scrolled rects.
const POLL_MS = 80;

type NativeCommand =
  | { command: "install"; code: string }
  | { command: "start"; color: string }
  | { command: "stop" }
  | { command: "forget"; id: number }
  | { command: "number"; numbers: Array<[number, number]> }
  | { command: "poll" };

type Answer = { installed: boolean; events: unknown[] };

function nativeCommand(command: FrameCommand): NativeCommand | null {
  switch (command.command) {
    case "annotateInstall":
      return { command: "install", code: command.code };
    case "annotate":
      return command.on ? { command: "start", color: command.color ?? "" } : { command: "stop" };
    case "annotateForget":
      return { command: "forget", id: command.id };
    case "annotateNumbers":
      return { command: "number", numbers: command.numbers };
    default:
      return null;
  }
}

function answerOf(value: unknown): Answer | null {
  if (!value || typeof value !== "object") return null;
  const { installed, events } = value as Record<string, unknown>;
  return typeof installed === "boolean" && Array.isArray(events) ? { installed, events } : null;
}

/** Drives `tabId`'s page until `stop`. Reports `ready` when a navigation drops the code, so the layer reinstalls it. */
export function startNativeAnnotate(
  tabId: string,
  listener: (event: AnnotateEvent) => void,
): { send: (command: FrameCommand) => void; stop: () => void } {
  let live = true;
  // Null until the first answer; `ready` fires on installed -> not installed.
  let installed: boolean | null = null;
  // Serial, so install lands before start.
  let queue: Promise<void> = Promise.resolve();

  const deliver = (answer: Answer) => {
    if (!live) return;
    if (installed === true && !answer.installed) listener({ kind: "ready" });
    installed = answer.installed;
    // The page can write to the queue too, so validate like frame messages.
    for (const event of answer.events) {
      if (!live) return;
      const data = event && typeof event === "object" ? { ...event, source: "unsloth-browser" } : null;
      const message = parseFrameMessage(data);
      if (message?.type === "annotate") listener(message.event);
    }
  };

  const run = (command: NativeCommand) => {
    queue = queue.then(() =>
      live || command.command === "stop"
        ? callNative<unknown>("browser_view_annotate", { tabId, command }).then(
            (value) => {
              const answer = answerOf(value);
              if (answer) deliver(answer);
            },
            // No view yet or mid-navigation: the next poll retries.
            () => undefined,
          )
        : undefined,
    );
    return queue;
  };

  let polling = false;
  const timer = window.setInterval(() => {
    if (polling) return;
    polling = true;
    void run({ command: "poll" }).finally(() => {
      polling = false;
    });
  }, POLL_MS);

  return {
    send: (command) => {
      const native = nativeCommand(command);
      if (native) void run(native);
    },
    stop: () => {
      window.clearInterval(timer);
      void run({ command: "stop" });
      live = false;
    },
  };
}
