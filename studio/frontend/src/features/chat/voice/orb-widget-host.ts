// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { decideToolCall } from "@/features/chat/tool-decision";
import { reacquireMic } from "@/features/chat/adapters/studio-voice-loop-adapter";
import { knowledgeBaseToolName, toolArgText, webSearchToolName } from "@/components/assistant-ui/tool-arg-text";
import { translate } from "@/i18n";
import {
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
  isOrbWidgetAction,
  isOrbWidgetDecision,
  packSpectrum,
  pendingApproval,
  parseSavedPosition,
  placeAtCursor,
  placeOrbWidget,
  type Area,
  type OrbWidgetApproval,
  type OrbWidgetSnapshot,
  type Point,
} from "./orb-widget";
import { SPECTRUM_SIDE } from "@/components/assistant-ui/voice-orb";
import { SPECTRUM_BANDS, readSpectrum } from "./voice-spectrum";
import { describeToolCall, requestVoicePauseToggle, requestVoiceStopSpeaking } from "./voice-loop-bridge";
import { voiceTrace } from "./voice-trace";

function hostOf(url: string): string {
  try {
    return new URL(/^[a-z]+:\/\//i.test(url) ? url : `https://${url}`).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

/** What the chat card's header says for the call, as near as its arguments allow. */
function toolLabel(toolCallId: string): string {
  const call = describeToolCall(toolCallId);
  const name = call?.toolName ?? "";
  const args = (call?.args ?? {}) as Record<string, unknown>;
  const query = toolArgText(args.query).trim();
  if (name === "web_search") {
    const url = toolArgText(args.url).trim();
    const pattern = toolArgText(args.pattern).trim();
    return webSearchToolName({
      isRunning: true,
      isFindInPage: Boolean(url && pattern),
      isUrlFetch: Boolean(url && !pattern),
      isImageOnly: false,
      foundImages: false,
      displayDomain: hostOf(url),
      pattern,
      query,
      imageLabel: "",
    });
  }
  if (name === "search_knowledge_base") return knowledgeBaseToolName({ isRunning: true, query });
  return toolArgText(name) || toolCallId;
}

function approval(): OrbWidgetApproval | null {
  const pending = pendingApproval(useChatRuntimeStore.getState().toolConfirmations);
  if (!pending) return null;
  return {
    id: pending.id,
    line: translate("orbWidget.usingTool", { tool: toolLabel(pending.id) }),
    more: pending.more > 0 ? translate("orbWidget.more", { n: String(pending.more) }) : null,
    allow: translate("browser.agent.allow"),
    deny: translate("browser.agent.deny"),
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
  };
}

type Rect = { position: { x: number; y: number }; size: { width: number; height: number } };

function toArea(r: Rect, scale: number): Area {
  const k = scale || 1;
  return { x: r.position.x / k, y: r.position.y / k, width: r.size.width / k, height: r.size.height / k };
}

const spectrumBands = new Float32Array(SPECTRUM_BANDS);

/** ~30 frames a second of the talking side's bands, only while one side is talking. */
function streamSpectrum(
  emitTo: (target: string, event: string, payload: unknown) => Promise<void>,
  shown: () => boolean,
): () => void {
  const timer = setInterval(() => {
    if (!shown()) return;
    const orbState = useChatRuntimeStore.getState().voiceOrbState;
    const side = orbState ? SPECTRUM_SIDE[orbState] : undefined;
    if (!side) return;
    const level = readSpectrum(side, spectrumBands);
    void emitTo(ORB_WIDGET_LABEL, ORB_WIDGET_SPECTRUM_EVENT, packSpectrum(side, level, spectrumBands)).catch(
      () => undefined,
    );
  }, 33);
  return () => clearInterval(timer);
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

let floating = false;
let warm: Promise<boolean> | null = null;

function reveal(t: Tauri): void {
  floating = false;
  void t.invoke("reveal_main_window")
    .catch(() => t.win.getCurrentWindow().show())
    // WebKit can leave the capture track open but silent across the swap: start the mic over.
    .finally(() => reacquireMic("back"));
}

/**
 * Create the widget hidden, page loaded and listening, as soon as voice is in use, so a
 * hold only has to place and show it. It is hidden again, not destroyed, on the way back.
 */
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
        // macOS: the first click on the (unfocused) widget presses the button instead of only focusing.
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
          // Voice ended from somewhere else: nothing left for the widget to show.
          if (floating) reveal(t);
          return;
        }
        if (
          s.voiceOrbState !== prev.voiceOrbState ||
          s.voicePaused !== prev.voicePaused ||
          s.voiceMuted !== prev.voiceMuted ||
          s.voiceMode !== prev.voiceMode ||
          // Asked, or answered from anywhere: the chat card, the popup, or voice cancelling the run.
          s.toolConfirmations !== prev.toolConfirmations
        ) {
          push();
        }
      }),
    );
    offs.push(await t.listen(ORB_WIDGET_READY_EVENT, push));
    offs.push(streamSpectrum(t.emitTo, () => floating));
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
        // The chat card's own call; the popup leaves when the store entry does.
        void decideToolCall(payload.id, payload.decision).then((outcome) => {
          if (outcome !== "ok") voiceTrace("ui.orbWidget.approvalFailed", { outcome });
          push();
        });
      }),
    );
    // Closed some other way (a close shortcut): bring the app back, build a new one next time.
    offs.push(
      await widget.once("tauri://destroyed", () => {
        for (const off of offs.splice(0)) off();
        warm = null;
        if (floating) reveal(t);
      }),
    );
    push();
    return true;
  })();
  return warm;
}

/**
 * Swap the main window for the floating orb, centred on `cursor` (screen points). The
 * voice loop lives in the main webview and keeps running there (enter_orb_widget_mode
 * keeps it capturing), so the widget only mirrors state and forwards its buttons back.
 */
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
    const screens = monitors.map((m) => toArea(m, m.scaleFactor));
    let saved = null;
    try {
      saved = parseSavedPosition(localStorage.getItem(ORB_WIDGET_POSITION_KEY));
    } catch {
      // Storage blocked: open in the default corner.
    }
    // The cursor wins; the saved spot is only for an open with no pointer behind it.
    const at =
      (cursor ? placeAtCursor(cursor, screens) : null) ??
      placeOrbWidget(saved, screens, current ? toArea(current.workArea, current.scaleFactor) : null);
    floating = true;
    useChatRuntimeStore.getState().setVoiceOrbCollapsed(true);
    // One native call places and shows the widget and fades the main window in the same turn.
    await t.invoke("enter_orb_widget_mode", { x: at?.x ?? null, y: at?.y ?? null });
    voiceTrace("ui.orbWidget.open", { ms: Math.round(performance.now() - t0), warm: wasWarm });
  } catch (error) {
    floating = false;
    voiceTrace("ui.orbWidget.error", { error: String(error) });
  }
}
