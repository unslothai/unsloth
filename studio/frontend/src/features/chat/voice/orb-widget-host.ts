// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { reacquireMic } from "@/features/chat/adapters/studio-voice-loop-adapter";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { decideToolCall } from "@/features/chat/tool-decision";
import { translate } from "@/i18n";
import { toolBrief, toolLabel } from "./approval-brief";
import { SPECTRUM_SIDE } from "./orb-config";
import { onMainFocus, returnFromWidget, type WidgetReturn } from "./orb-widget-return";
import {
  type Area,
  ORB_WIDGET_ACTION_EVENT,
  ORB_WIDGET_APPROVAL_EVENT,
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_LABEL,
  ORB_WIDGET_POSITION_KEY,
  ORB_WIDGET_QUERY,
  ORB_WIDGET_READY_EVENT,
  ORB_WIDGET_SPECTRUM_EVENT,
  ORB_WIDGET_STATE_EVENT,
  ORB_WIDGET_WIDTH,
  type OrbWidgetApproval,
  type OrbWidgetLabels,
  type OrbWidgetSnapshot,
  type OrbWidgetTheme,
  type Point,
  isOrbWidgetAction,
  isOrbWidgetDecision,
  packSpectrum,
  parseSavedPosition,
  pendingApproval,
  placeAtCursor,
  placeOrbWidget,
} from "./orb-widget";
import { describeToolCall, requestVoicePauseToggle, requestVoiceStopSpeaking } from "./voice-loop-bridge";
import { SPECTRUM_BANDS, onSpectrumFrame, readSpectrum } from "./voice-spectrum";
import { voiceTrace } from "./voice-trace";

function approval(): OrbWidgetApproval | null {
  const s = useChatRuntimeStore.getState();
  // Only the voiced chat's calls: the store also holds Compare panes' and background chats'.
  const pending = pendingApproval(s.toolConfirmations, s.activeThreadId, (id) => describeToolCall(id) !== null);
  if (!pending) return null;
  return {
    id: pending.id,
    line: translate("orbWidget.usingTool", { tool: toolLabel(pending.id) }),
    brief: toolBrief(pending.id),
    more: pending.more > 0 ? translate("orbWidget.more", { n: String(pending.more) }) : null,
    allow: translate("browser.agent.allow"),
    deny: translate("browser.agent.deny"),
  };
}

function labels(): OrbWidgetLabels {
  return {
    pause: translate("voiceOrb.bar.pause"),
    resume: translate("voiceOrb.bar.resume"),
    stop: translate("voiceOrb.bar.stop"),
    back: translate("voiceOrb.bar.back"),
  };
}

function theme(): OrbWidgetTheme {
  const root = document.documentElement;
  return {
    dark: root.classList.contains("dark"),
    palette: root.getAttribute("data-palette"),
  };
}

function snapshot(): OrbWidgetSnapshot {
  const s = useChatRuntimeStore.getState();
  return {
    orbState: s.voiceOrbState,
    paused: s.voicePaused,
    muted: s.voiceMuted,
    active: s.voiceMode === "active",
    approval: s.voiceMode === "active" ? approval() : null,
    labels: labels(),
    theme: theme(),
  };
}

function watchLook(push: () => void): () => void {
  if (typeof MutationObserver === "undefined") return () => undefined;
  let last = JSON.stringify([theme(), labels()]);
  const observer = new MutationObserver(() => {
    const next = JSON.stringify([theme(), labels()]);
    if (next === last) return;
    last = next;
    push();
  });
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class", "data-palette", "lang"],
  });
  return () => observer.disconnect();
}

type Rect = {
  position: { x: number; y: number };
  size: { width: number; height: number };
};

function toArea(r: Rect, scale: number): Area {
  const k = scale || 1;
  return {
    x: r.position.x / k,
    y: r.position.y / k,
    width: r.size.width / k,
    height: r.size.height / k,
  };
}

const spectrumBands = new Float32Array(SPECTRUM_BANDS);
const FEED_MS = 15;

function streamSpectrum(
  emitTo: (target: string, event: string, payload: unknown) => Promise<void>,
  shown: () => boolean,
): () => void {
  let sentAt = Number.NEGATIVE_INFINITY;
  const send = (only?: "input" | "output") => {
    if (!shown()) return;
    const orbState = useChatRuntimeStore.getState().voiceOrbState;
    const side = orbState ? SPECTRUM_SIDE[orbState] : undefined;
    if (!side || (only && only !== side)) return;
    const now = performance.now();
    if (now - sentAt < FEED_MS) return;
    sentAt = now;
    const level = readSpectrum(side, spectrumBands);
    void emitTo(ORB_WIDGET_LABEL, ORB_WIDGET_SPECTRUM_EVENT, packSpectrum(side, level, spectrumBands)).catch(
      () => undefined,
    );
  };
  // Fallback only: the audio clock drives frames, since the hidden main window's timers are slowed.
  let timer: ReturnType<typeof setInterval> | null = null;
  const stopTimer = () => {
    if (timer !== null) clearInterval(timer);
    timer = null;
  };
  const tick = () => {
    if (!shown()) stopTimer();
    else send();
  };
  const offAudio = onSpectrumFrame((side) => {
    if (timer === null && shown()) timer = setInterval(tick, 16);
    send(side);
  });
  return () => {
    offAudio();
    stopTimer();
  };
}

type Tauri = {
  WebviewWindow: typeof import("@tauri-apps/api/webviewWindow").WebviewWindow;
  win: typeof import("@tauri-apps/api/window");
  emitTo: typeof import("@tauri-apps/api/event").emitTo;
  listen: typeof import("@tauri-apps/api/event").listen;
  invoke: typeof import("@tauri-apps/api/core").invoke;
};

let tauri: Promise<Tauri> | null = null;
function loadTauri(): Promise<Tauri> {
  tauri ??= Promise.all([
    import("@tauri-apps/api/webviewWindow"),
    import("@tauri-apps/api/window"),
    import("@tauri-apps/api/event"),
    import("@tauri-apps/api/core"),
  ]).then(([{ WebviewWindow }, win, { emitTo, listen }, { invoke }]) => ({
    WebviewWindow,
    win,
    emitTo,
    listen,
    invoke,
  }));
  return tauri;
}

const mode: WidgetReturn = { floating: false };
let warm: Promise<boolean> | null = null;

function showMain(t: Tauri): void {
  void t
    .invoke("reveal_main_window")
    .catch(() => t.win.getCurrentWindow().show())
    // WebKit can leave the capture track open but silent across the swap: start the mic over.
    .finally(() => reacquireMic("back"));
}

function reveal(t: Tauri): void {
  returnFromWidget(mode, (c) => useChatRuntimeStore.getState().setVoiceOrbCollapsed(c), () => showMain(t));
}

export function prewarmOrbWidget(): Promise<boolean> {
  warm ??= (async () => {
    const t = await loadTauri();
    let widget = await t.WebviewWindow.getByLabel(ORB_WIDGET_LABEL);
    if (!widget) {
      widget = new t.WebviewWindow(ORB_WIDGET_LABEL, {
        url: `index.html?${ORB_WIDGET_QUERY}`,
        title: "Unsloth voice",
        width: ORB_WIDGET_WIDTH,
        height: ORB_WIDGET_HEIGHT,
        visible: false,
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
        // macOS: the first click on the unfocused widget presses the button instead of only focusing.
        acceptFirstMouse: true,
      });
      const created = await new Promise<boolean>((resolve) => {
        void widget!.once("tauri://created", () => resolve(true));
        void widget!.once("tauri://error", () => resolve(false));
        setTimeout(() => resolve(false), 5000);
      });
      if (!created) {
        warm = null;
        return false;
      }
    }
    const push = () => {
      void t.emitTo(ORB_WIDGET_LABEL, ORB_WIDGET_STATE_EVENT, snapshot()).catch(() => undefined);
    };
    const offs: Array<() => void> = [];
    offs.push(
      useChatRuntimeStore.subscribe((s, prev) => {
        if (s.voiceMode === "off" && prev.voiceMode !== "off") {
          if (mode.floating) reveal(t);
          return;
        }
        if (
          s.voiceOrbState !== prev.voiceOrbState ||
          s.voicePaused !== prev.voicePaused ||
          s.voiceMuted !== prev.voiceMuted ||
          s.voiceMode !== prev.voiceMode ||
          s.activeThreadId !== prev.activeThreadId ||
          // Asked, or answered from anywhere: the chat card, the popup, or voice cancelling the run.
          s.toolConfirmations !== prev.toolConfirmations
        ) {
          push();
        }
      }),
    );
    offs.push(watchLook(push));
    offs.push(await t.listen(ORB_WIDGET_READY_EVENT, push));
    // A Dock click, the tray or the app menu restore the main window natively; its focus is how this side hears.
    offs.push(streamSpectrum(t.emitTo, () => mode.floating));
    offs.push(
      await t.win.getCurrentWindow().onFocusChanged(({ payload: focused }) => {
        const back = onMainFocus(
          focused,
          mode,
          (c) => useChatRuntimeStore.getState().setVoiceOrbCollapsed(c),
          () => showMain(t),
        );
        if (back) voiceTrace("ui.orbWidget.backByFocus");
      }),
    );
    offs.push(
      await t.listen<unknown>(ORB_WIDGET_ACTION_EVENT, ({ payload }) => {
        if (!isOrbWidgetAction(payload)) return;
        voiceTrace("ui.orbWidget", { action: payload });
        if (payload === "toggle-pause") requestVoicePauseToggle();
        else if (payload === "stop") requestVoiceStopSpeaking();
        else reveal(t);
      }),
    );
    offs.push(
      await t.listen<unknown>(ORB_WIDGET_APPROVAL_EVENT, ({ payload }) => {
        if (!isOrbWidgetDecision(payload)) return;
        voiceTrace("ui.orbWidget.approval", { decision: payload.decision });
        void decideToolCall(payload.id, payload.decision).then((outcome) => {
          if (outcome !== "ok") voiceTrace("ui.orbWidget.approvalFailed", { outcome });
          push();
        });
      }),
    );
    offs.push(
      await widget.once("tauri://destroyed", () => {
        for (const off of offs.splice(0)) off();
        warm = null;
        if (mode.floating) reveal(t);
      }),
    );
    push();
    return true;
  })();
  return warm;
}

export async function openOrbWidget(cursor: Point | null = null): Promise<void> {
  const t0 = performance.now();
  try {
    const wasWarm = warm !== null;
    if (!(await prewarmOrbWidget())) return;
    const t = await loadTauri();
    const [monitors, current] = await Promise.all([
      t.win.availableMonitors().catch(() => []),
      t.win.currentMonitor().catch(() => null),
    ]);
    const screens = monitors.map((m) => ({
      ...toArea(m, m.scaleFactor),
      visibleBottom: (m.workArea.position.y + m.workArea.size.height) / (m.scaleFactor || 1),
    }));
    let saved = null;
    try {
      saved = parseSavedPosition(localStorage.getItem(ORB_WIDGET_POSITION_KEY));
    } catch {
    }
    const at =
      (cursor ? placeAtCursor(cursor, screens) : null) ??
      placeOrbWidget(saved, screens, current ? toArea(current.workArea, current.scaleFactor) : null);
    mode.floating = true;
    useChatRuntimeStore.getState().setVoiceOrbCollapsed(true);
    // One native call places and shows the widget and fades the main window in the same turn.
    await t.invoke("enter_orb_widget_mode", {
      x: at?.x ?? null,
      y: at?.y ?? null,
    });
    voiceTrace("ui.orbWidget.open", {
      ms: Math.round(performance.now() - t0),
      warm: wasWarm,
    });
  } catch (error) {
    voiceTrace("ui.orbWidget.error", { error: String(error) });
    if (mode.floating) void loadTauri().then(reveal);
  }
}
