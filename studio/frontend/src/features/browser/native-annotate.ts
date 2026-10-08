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

// One queue per tab across channels, so a closing channel's stop can't land after a new one's start.
const tabQueues = new Map<string, Promise<void>>();

/** Drives `tabId`'s page until `stop`. Reports `ready` when a navigation drops the code, so the layer reinstalls it. */
export function startNativeAnnotate(
  tabId: string,
  listener: (event: AnnotateEvent) => void,
): { send: (command: FrameCommand) => void; stop: () => void } {
  let live = true;
  // Null until the first answer; `ready` fires on installed -> not installed.
  let installed: boolean | null = null;
  // Install or start calls that failed or found no code, replayed after the next answer.
  const failed = new Map<"install" | "start", NativeCommand>();

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

  const run = (command: NativeCommand): Promise<void> => {
    const call = () =>
      callNative<unknown>("browser_view_annotate", { tabId, command }).then(
        (value) => {
          const answer = answerOf(value);
          if (answer) deliver(answer);
          // A start that found no code (its install failed) goes again behind the install.
          if (live && command.command === "start" && answer && !answer.installed) failed.set("start", command);
          replay();
        },
        () => {
          if (live && (command.command === "install" || command.command === "start")) {
            failed.set(command.command, command);
          }
        },
      );
    // Serial, so install lands before start. A closed channel only sends its stop.
    const next = (tabQueues.get(tabId) ?? Promise.resolve()).then(() =>
      live || command.command === "stop" ? call() : undefined,
    );
    tabQueues.set(tabId, next);
    void next.finally(() => {
      if (tabQueues.get(tabId) === next) tabQueues.delete(tabId);
    });
    return next;
  };

  const replay = () => {
    if (!live) return;
    const install = failed.get("install");
    const start = failed.get("start");
    if (install) {
      failed.delete("install");
      void run(install);
    }
    // Start only behind its install or once the code is in, so a navigating page can't loop it.
    if (start && (install || installed)) {
      failed.delete("start");
      void run(start);
    }
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
      if (!native) return;
      if (native.command === "stop") failed.delete("start");
      void run(native);
    },
    stop: () => {
      window.clearInterval(timer);
      void run({ command: "stop" });
      live = false;
    },
  };
}
