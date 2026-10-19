// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  ORB_WIDGET_ACTION_EVENT,
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_LABEL,
  ORB_WIDGET_POSITION_KEY,
  ORB_WIDGET_QUERY,
  ORB_WIDGET_READY_EVENT,
  ORB_WIDGET_STATE_EVENT,
  ORB_WIDGET_WIDTH,
  isOrbWidgetAction,
  parseSavedPosition,
  placeOrbWidget,
  type Area,
  type OrbWidgetSnapshot,
} from "./orb-widget";
import { requestVoicePauseToggle, requestVoiceStopSpeaking } from "./voice-loop-bridge";
import { voiceTrace } from "./voice-trace";

function snapshot(): OrbWidgetSnapshot {
  const s = useChatRuntimeStore.getState();
  return {
    orbState: s.voiceOrbState,
    paused: s.voicePaused,
    muted: s.voiceMuted,
    active: s.voiceMode === "active",
  };
}

let opening = false;

function toArea(m: {
  scaleFactor: number;
  workArea: { position: { x: number; y: number }; size: { width: number; height: number } };
}): Area {
  const k = m.scaleFactor || 1;
  return {
    x: m.workArea.position.x / k,
    y: m.workArea.position.y / k,
    width: m.workArea.size.width / k,
    height: m.workArea.size.height / k,
  };
}

/**
 * Swap the main window for the floating orb. The main window is hidden, not
 * closed: the voice loop lives in its webview and keeps running there, so the
 * widget only mirrors state and forwards its buttons back over events.
 */
export async function openOrbWidget(): Promise<void> {
  if (opening) return;
  opening = true;
  try {
    const [{ WebviewWindow }, win, { emitTo, listen }, { invoke }] = await Promise.all([
      import("@tauri-apps/api/webviewWindow"),
      import("@tauri-apps/api/window"),
      import("@tauri-apps/api/event"),
      import("@tauri-apps/api/core"),
    ]);
    const main = win.getCurrentWindow();
    const existing = await WebviewWindow.getByLabel(ORB_WIDGET_LABEL);
    if (existing) {
      await existing.setFocus().catch(() => undefined);
      await main.hide();
      return;
    }

    let saved = null;
    try {
      saved = parseSavedPosition(localStorage.getItem(ORB_WIDGET_POSITION_KEY));
    } catch {
      // Storage blocked: open in the default corner.
    }
    const [monitors, current] = await Promise.all([
      win.availableMonitors().catch(() => []),
      win.currentMonitor().catch(() => null),
    ]);
    const at = placeOrbWidget(saved, monitors.map(toArea), current ? toArea(current) : null);

    const widget = new WebviewWindow(ORB_WIDGET_LABEL, {
      url: `index.html?${ORB_WIDGET_QUERY}`,
      title: "Unsloth voice",
      width: ORB_WIDGET_WIDTH,
      height: ORB_WIDGET_HEIGHT,
      ...(at ? { x: at.x, y: at.y } : { center: true }),
      resizable: false,
      maximizable: false,
      minimizable: false,
      decorations: false,
      transparent: true,
      // The native shadow outlines the whole transparent square on macOS.
      shadow: false,
      alwaysOnTop: true,
      skipTaskbar: true,
      visibleOnAllWorkspaces: true,
      focus: false,
      // macOS: the first click on the (unfocused) widget presses the button instead of only focusing.
      acceptFirstMouse: true,
    });
    // Subscribed before any other await, so a fast create cannot slip past it.
    const settled = new Promise<void>((resolve) => {
      void widget.once("tauri://created", () => resolve());
      void widget.once("tauri://error", () => resolve());
      setTimeout(resolve, 3000);
    });

    const unlisteners: Array<() => void> = [];
    const push = () => {
      void emitTo(ORB_WIDGET_LABEL, ORB_WIDGET_STATE_EVENT, snapshot()).catch(() => undefined);
    };
    let restored = false;
    const restore = () => {
      if (restored) return;
      restored = true;
      for (const off of unlisteners.splice(0)) off();
      void invoke("reveal_main_window").catch(() => main.show());
    };

    unlisteners.push(
      useChatRuntimeStore.subscribe((s, prev) => {
        if (s.voiceMode === "off" && prev.voiceMode !== "off") {
          // Voice ended from somewhere else: nothing left for the widget to show.
          void widget.destroy().catch(() => undefined);
          return;
        }
        if (
          s.voiceOrbState !== prev.voiceOrbState ||
          s.voicePaused !== prev.voicePaused ||
          s.voiceMuted !== prev.voiceMuted ||
          s.voiceMode !== prev.voiceMode
        ) {
          push();
        }
      }),
    );
    unlisteners.push(await listen(ORB_WIDGET_READY_EVENT, push));
    unlisteners.push(
      await listen<unknown>(ORB_WIDGET_ACTION_EVENT, ({ payload }) => {
        if (!isOrbWidgetAction(payload)) return;
        voiceTrace("ui.orbWidget", { action: payload });
        if (payload === "toggle-pause") requestVoicePauseToggle();
        else if (payload === "stop") requestVoiceStopSpeaking();
        else void widget.destroy().catch(restore);
      }),
    );
    // Every way out (the back button, voice ending, a close shortcut) lands here.
    unlisteners.push(await widget.once("tauri://destroyed", restore));

    await settled;
    if (!(await WebviewWindow.getByLabel(ORB_WIDGET_LABEL))) {
      restore();
    } else if (!restored) {
      useChatRuntimeStore.getState().setVoiceOrbCollapsed(true);
      await main.hide();
    }
  } catch (error) {
    voiceTrace("ui.orbWidget.error", { error: String(error) });
  } finally {
    opening = false;
  }
}
