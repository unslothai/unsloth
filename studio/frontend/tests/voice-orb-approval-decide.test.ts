// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The orb's approval surfaces driven for real: decideToolCall against a stubbed backend and
// store, the orb's card with its buttons pressed, and the floating widget's host with its
// Tauri events fired. Nothing here reads the source as text.

import assert from "node:assert/strict";
import test from "node:test";

import * as orbWidget from "../src/features/chat/voice/orb-widget.ts";
import * as orbWidgetReturn from "../src/features/chat/voice/orb-widget-return.ts";
import { type StubElement, loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

type Entry = { approvalId: string; sessionId: string; autoAllowKey: string; imageDisclosure?: unknown };
type Result = "ok" | "retry" | "gone" | "missing";
type Decide = (id: string, decision: "allow" | "deny", always?: string) => Promise<Result>;

class GoneError extends Error {}

function fakeStore(confirmations: Record<string, Entry>, extra: Record<string, unknown> = {}) {
  const log: string[] = [];
  const state: Record<string, unknown> = {
    toolConfirmations: { ...confirmations },
    allowToolAlways: (key: string, name: string) => log.push(`always ${key} ${name}`),
    clearToolConfirmation: (id: string) => {
      log.push(`clear ${id}`);
      const next = { ...(state.toolConfirmations as Record<string, Entry>) };
      delete next[id];
      state.toolConfirmations = next;
    },
    ...extra,
  };
  const store = Object.assign(<T>(select: (s: typeof state) => T) => select(state), {
    getState: () => state,
    subscribe: () => () => undefined,
  });
  return { store, state, log };
}

function loadDecide(confirmations: Record<string, Entry>, resolve: () => Promise<boolean>) {
  const { store, log } = fakeStore(confirmations);
  const mod = loadWithStubs<{ decideToolCall: Decide }>(
    new URL("../src/features/chat/tool-decision.ts", import.meta.url),
    {
      "./api/chat-api": {
        ToolApprovalGoneError: GoneError,
        resolveToolConfirmation: async (sessionId: string, approvalId: string, decision: string) => {
          log.push(`post ${sessionId} ${approvalId} ${decision}`);
          return resolve();
        },
      },
      "./stores/chat-runtime-store": { useChatRuntimeStore: store },
    },
  );
  return { decide: mod.decideToolCall, log };
}

const call = { approvalId: "ap1", sessionId: "sess", autoAllowKey: "sess:A" };

test("Always allow is granted only after the backend took the decision", async () => {
  const { decide, log } = loadDecide({ c1: call }, async () => true);
  assert.equal(await decide("c1", "allow", "web_search"), "ok");
  assert.deepEqual(log, ["post sess ap1 allow", "always sess:A web_search", "clear c1"]);
});

test("a plain Allow or a Deny never grants Always allow", async () => {
  for (const [decision, always] of [
    ["allow", undefined],
    ["deny", "web_search"],
  ] as const) {
    const { decide, log } = loadDecide({ c1: call }, async () => true);
    assert.equal(await decide("c1", decision, always), "ok");
    assert.deepEqual(log, [`post sess ap1 ${decision}`, "clear c1"]);
  }
});

test("a post that fails leaves the call answerable and grants nothing", async () => {
  for (const resolve of [async () => false, async () => Promise.reject(new Error("network"))]) {
    const { decide, log } = loadDecide({ c1: call }, resolve);
    assert.equal(await decide("c1", "allow", "web_search"), "retry");
    assert.deepEqual(log, ["post sess ap1 allow"]);
  }
});

test("an approval the backend no longer holds is gone, not retryable", async () => {
  const { decide, log } = loadDecide({ c1: call }, async () => Promise.reject(new GoneError()));
  assert.equal(await decide("c1", "allow", "web_search"), "gone");
  assert.deepEqual(log, ["post sess ap1 allow"]);
});

test("a call already answered elsewhere is missing and posts nothing", async () => {
  const { decide, log } = loadDecide({}, async () => true);
  assert.equal(await decide("c1", "allow"), "missing");
  assert.deepEqual(log, []);
});

// --- the orb's card ---

function renderCard(opts: { name: string | null; disclosure?: boolean }) {
  const calls: unknown[][] = [];
  const { store } = fakeStore({
    c1: { ...call, ...(opts.disclosure ? { imageDisclosure: { count: 1 } } : {}) },
  });
  const Button = (props: Record<string, unknown>) => props;
  const mod = loadWithStubs<{ OrbApprovalCard: (p: { id: string; more: number }) => StubElement }>(
    new URL("../src/components/assistant-ui/orb-approval-card.tsx", import.meta.url),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: { useState: (v: unknown) => [v, () => undefined] },
      "@/components/ui/button": { Button },
      "@/features/chat/stores/chat-runtime-store": { useChatRuntimeStore: store },
      "@/features/chat/tool-decision": {
        decideToolCall: (...args: unknown[]) => {
          calls.push(args);
          return Promise.resolve("ok");
        },
      },
      "@/features/chat/voice/approval-brief": { toolBrief: () => "brief" },
      "@/features/chat/voice/orb-widget": orbWidget,
      "@/features/chat/voice/voice-loop-bridge": {
        describeToolCall: () => (opts.name === null ? null : { toolName: opts.name, args: {} }),
      },
      "@/i18n": { useT: () => (key: string) => key },
    },
  );
  const tree = mod.OrbApprovalCard({ id: "c1", more: 0 });
  const buttons: StubElement[] = [];
  const walk = (node: unknown) => {
    if (Array.isArray(node)) return node.forEach(walk);
    if (!node || typeof node !== "object") return;
    const el = node as StubElement;
    if (el.type === Button) buttons.push(el);
    walk(el.props?.children);
  };
  walk(tree);
  const press = (label: string) => {
    const b = buttons.find((x) => x.props.children === label);
    assert.ok(b, `no ${label} button`);
    (b.props.onClick as () => void)();
  };
  return { labels: buttons.map((b) => b.props.children), press, calls };
}

test("the orb card's buttons make the chat card's decision, Always allow naming the tool", () => {
  const card = renderCard({ name: "web_search" });
  assert.deepEqual(card.labels, [
    "browser.agent.allow",
    "browser.agentSettings.alwaysAllow",
    "browser.agent.deny",
  ]);
  card.press("browser.agent.allow");
  card.press("browser.agentSettings.alwaysAllow");
  card.press("browser.agent.deny");
  assert.deepEqual(card.calls, [
    ["c1", "allow", undefined],
    ["c1", "allow", "web_search"],
    ["c1", "deny", undefined],
  ]);
});

test("with the call's tool unknown, Always allow is not offered rather than granting \"tool\"", () => {
  const card = renderCard({ name: null });
  assert.deepEqual(card.labels, ["browser.agent.allow", "browser.agent.deny"]);
  assert.equal(renderCard({ name: "" }).labels.length, 2);
});

test("a call that sends an image is never offered Always allow", () => {
  assert.equal(renderCard({ name: "web_search", disclosure: true }).labels.length, 2);
});

// --- the floating widget's host ---

// The host reads the theme off <html> on every push, decisions included.
const g = globalThis as { document?: unknown };
if (!("document" in g)) {
  g.document = {
    documentElement: {
      classList: { contains: (c: string) => c === "dark" },
      getAttribute: (n: string) => (n === "data-palette" ? "matcha" : null),
    },
  };
  test.after(() => Reflect.deleteProperty(g, "document"));
}

async function loadHost(confirmations: Record<string, Entry>, inVoiceThread: string[]) {
  const listeners = new Map<string, (e: { payload: unknown }) => void>();
  const emitted: { event: string; payload: unknown }[] = [];
  const decided: unknown[][] = [];
  const { store, state } = fakeStore(confirmations, {
    voiceOrbState: "thinking",
    voicePaused: false,
    voiceMuted: false,
    voiceMode: "active",
    activeThreadId: "A",
  });
  const onces = new Map<string, () => void>();
  const widget = {
    once: async (event: string, fn: () => void) => {
      onces.set(event, fn);
      return () => undefined;
    },
  };
  const tauri = {
    "@tauri-apps/api/webviewWindow": { WebviewWindow: { getByLabel: async () => widget } },
    "@tauri-apps/api/window": {
      getCurrentWindow: () => ({ onFocusChanged: async () => () => undefined }),
    },
    "@tauri-apps/api/event": {
      emitTo: async (_target: string, event: string, payload: unknown) => {
        emitted.push({ event, payload });
      },
      listen: async (event: string, fn: (e: { payload: unknown }) => void) => {
        listeners.set(event, fn);
        return () => undefined;
      },
    },
    "@tauri-apps/api/core": { invoke: async () => undefined },
  };
  const host = loadWithStubs<{ prewarmOrbWidget: () => Promise<boolean> }>(
    new URL("../src/features/chat/voice/orb-widget-host.ts", import.meta.url),
    {
      ...tauri,
      "@/features/chat/adapters/studio-voice-loop-adapter": { reacquireMic: () => undefined },
      "@/features/chat/stores/chat-runtime-store": { useChatRuntimeStore: store },
      "@/features/chat/tool-decision": {
        decideToolCall: (...args: unknown[]) => {
          decided.push(args);
          return Promise.resolve("ok");
        },
      },
      "@/i18n": { translate: (key: string) => `t:${key}` },
      "./approval-brief": { toolBrief: (id: string) => `brief ${id}`, toolLabel: (id: string) => id },
      "./orb-config": { SPECTRUM_SIDE: {} },
      "./orb-widget": orbWidget,
      "./orb-widget-return": orbWidgetReturn,
      "./voice-loop-bridge": {
        describeToolCall: (id: string) => (inVoiceThread.includes(id) ? { toolName: "x", args: {} } : null),
        requestVoicePauseToggle: () => undefined,
        requestVoiceStopSpeaking: () => undefined,
      },
      "./voice-spectrum": { SPECTRUM_BANDS: 4, onSpectrumFrame: () => () => undefined, readSpectrum: () => 0 },
      "./voice-trace": { voiceTrace: () => undefined },
    },
  );
  assert.equal(await host.prewarmOrbWidget(), true);
  const last = () => emitted.filter((e) => e.event === orbWidget.ORB_WIDGET_STATE_EVENT).at(-1)
    ?.payload as orbWidget.OrbWidgetSnapshot;
  // Closing the widget tears the host down, its spectrum timer included.
  test.after(() => onces.get("tauri://destroyed")?.());
  return { listeners, last, decided, state };
}

test("the widget's popup shows the voiced chat's call, not another pane's, with translated labels and the theme", async () => {
  const { last } = await loadHost(
    { "call-compare-pane-B": { ...call, autoAllowKey: "sess:B" }, "call-voice-A": call },
    [],
  );
  const snap = last();
  assert.equal(snap.approval?.id, "call-voice-A");
  assert.equal(snap.approval?.more, null);
  assert.deepEqual(snap.labels, {
    pause: "t:voiceOrb.bar.pause",
    resume: "t:voiceOrb.bar.resume",
    stop: "t:voiceOrb.bar.stop",
    back: "t:voiceOrb.bar.back",
  });
  assert.deepEqual(snap.theme, { dark: true, palette: "matcha" });
});

test("only another pane's call waiting: the widget shows no popup", async () => {
  const { last } = await loadHost({ "call-compare-pane-B": { ...call, autoAllowKey: "sess:B" } }, []);
  assert.equal(last().approval, null);
});

test("Allow and Deny on the popup make the chat card's decision, never an Always allow", async () => {
  const { listeners, decided } = await loadHost({ "call-voice-A": call }, []);
  const fire = listeners.get(orbWidget.ORB_WIDGET_APPROVAL_EVENT);
  assert.ok(fire);
  fire({ payload: { id: "call-voice-A", decision: "allow" } });
  fire({ payload: { id: "call-voice-A", decision: "deny" } });
  fire({ payload: { id: "call-voice-A", decision: "always" } });
  fire({ payload: "allow" });
  assert.deepEqual(decided, [
    ["call-voice-A", "allow"],
    ["call-voice-A", "deny"],
  ]);
});
