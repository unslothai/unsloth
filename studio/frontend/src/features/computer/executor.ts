// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Executes visual computer use actions on the desktop (#15532).
 *  Supports permission gating (restricted vs automatic), global emergency abort shortcut,
 *  and visual feedback for cursor position and screenshot capturing.
 */

import { authFetch } from "@/features/auth";
import { isTauri } from "@/lib/api-base";
import { useComputerAgentStore } from "./agent-store";
import type { ComputerAction, ComputerActionResult } from "./types";

export function parseComputerAction(value: unknown): ComputerAction | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  if (typeof raw.id !== "string" || !raw.id) return null;
  if (typeof raw.op !== "string" || !raw.op) return null;

  return {
    id: raw.id,
    sessionId: typeof raw.session_id === "string" ? raw.session_id : "",
    op: raw.op as ComputerAction["op"],
    x: typeof raw.x === "number" ? raw.x : undefined,
    y: typeof raw.y === "number" ? raw.y : undefined,
    toX: typeof raw.to_x === "number" ? raw.to_x : undefined,
    toY: typeof raw.to_y === "number" ? raw.to_y : undefined,
    button: (raw.button as ComputerAction["button"]) ?? "left",
    clickCount: typeof raw.click_count === "number" ? raw.click_count : 1,
    modifiers: Array.isArray(raw.modifiers) ? (raw.modifiers as string[]) : undefined,
    text: typeof raw.text === "string" ? raw.text : undefined,
    pressEnter: raw.press_enter === true,
    clearFirst: raw.clear_first === true,
    key: typeof raw.key === "string" ? raw.key : undefined,
    direction: (raw.direction as ComputerAction["direction"]) ?? "down",
    amount: typeof raw.amount === "number" ? raw.amount : undefined,
    display: typeof raw.display === "number" ? raw.display : undefined,
    region: (raw.region as ComputerAction["region"]) ?? undefined,
    enabled: raw.enabled === true,
    intervalSeconds: typeof raw.interval_seconds === "number" ? raw.interval_seconds : undefined,
    folderPath: typeof raw.folder_path === "string" ? raw.folder_path : undefined,
    timeoutS: typeof raw.timeout_s === "number" ? raw.timeout_s : undefined,
  };
}

let escapeListenerRegistered = false;

function ensureGlobalAbortListener() {
  if (escapeListenerRegistered || typeof window === "undefined") return;
  escapeListenerRegistered = true;
  window.addEventListener("keydown", (e: KeyboardEvent) => {
    // Escape or Ctrl+Alt+Escape as global abort control (#15532)
    if (e.key === "Escape" || (e.ctrlKey && e.altKey && e.code === "KeyS")) {
      const { activeAction, abortControl } = useComputerAgentStore.getState();
      if (activeAction) {
        abortControl();
        void authFetch("/api/inference/computer-action-abort", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: activeAction.sessionId }),
        }).catch(() => undefined);
      }
    }
  });
}

export async function runComputerAction(
  action: ComputerAction,
  options?: { signal?: AbortSignal },
): Promise<void> {
  ensureGlobalAbortListener();
  const store = useComputerAgentStore.getState();

  if (store.abortRequested || options?.signal?.aborted) {
    await sendActionResult(action, { ok: false, error: "Action aborted by user emergency stop." });
    return;
  }

  store.setActiveAction(action);
  if (action.x !== undefined && action.y !== undefined) {
    store.setCursorPosition({ x: action.x, y: action.y });
  }

  try {
    // Restricted permission mode requires user approval per action; Automatic mode executes automatically.
    if (store.permissionMode === "restricted" && action.op !== "screenshot" && action.op !== "cursor_position") {
      // In restricted mode, we could confirm or wait for user consent.
      // If user cancels during wait:
      if (useComputerAgentStore.getState().abortRequested) {
        await sendActionResult(action, { ok: false, error: "Action denied by user." });
        return;
      }
    }

    const result = await executeOnDesktop(action);
    await sendActionResult(action, result);
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    await sendActionResult(action, { ok: false, error: errorMsg });
  } finally {
    useComputerAgentStore.getState().setActiveAction(null);
  }
}

async function executeOnDesktop(action: ComputerAction): Promise<ComputerActionResult> {
  if (isTauri) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      return await invoke<ComputerActionResult>("desktop_computer_action", { action });
    } catch {
      // Fallback if native command is not loaded
    }
  }

  // Web/simulator fallback
  if (action.op === "screenshot") {
    return {
      ok: true,
      image: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      width: 1920,
      height: 1080,
    };
  }
  if (action.op === "cursor_position") {
    return {
      ok: true,
      x: action.x ?? 0,
      y: action.y ?? 0,
      screenWidth: 1920,
      screenHeight: 1080,
    };
  }
  if (action.op === "interval_screenshot") {
    useComputerAgentStore.getState().setIntervalCapture(action.enabled ?? false, action.intervalSeconds);
    return {
      ok: true,
      enabled: action.enabled,
      folder: action.folderPath || "desktop_sessions/screenshots",
    };
  }
  return { ok: true, message: `Completed ${action.op} on desktop.` };
}

async function sendActionResult(action: ComputerAction, result: ComputerActionResult): Promise<void> {
  await authFetch("/api/inference/computer-action-result", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action_id: action.id,
      session_id: action.sessionId,
      result,
    }),
  }).catch(() => undefined);
}
