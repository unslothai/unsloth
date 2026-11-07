// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { LogicalSize } from "@tauri-apps/api/dpi";
import { setApiBase } from "@/lib/api-base";
import { useBrowserStore } from "@/features/browser/store";
import { startNativeViews, hasNativeView, focusPanel, whenNativeViewShown } from "@/features/browser/native-view";
import { callNative, grantLocalOrigin, useNativeBrowser } from "@/features/browser/native-support";
import { captureHiddenPage } from "@/features/browser/capture";
import { agentChat, patchAgentChat, endBrowserAgentRun, stopBrowserAgent, useBrowserAgentStore } from "@/features/browser/agent/agent-store";
import { runBrowserAction, type BrowserAction } from "@/features/browser/agent/executor";
import { setBrowserPanelAvailable } from "@/features/browser/panel-availability";
import "@/index.css";

const api = "http://127.0.0.1:19044";
const pages = "http://127.0.0.1:19045";
const a = "single:native-a";
const b = "single:native-b";
const facts: Record<string, unknown> = { transport: "real Tauri IPC", model: "none" };
const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const assert = (ok: unknown, detail: string) => { if (!ok) throw new Error(detail); };
let stopViews: (() => void) | undefined;
let observeNative: (value: boolean) => void;
let sequence = 0;

async function checkpoint(name: string, value: unknown) {
  facts[name] = value;
  await fetch(`${api}/progress`, { method: "POST", body: JSON.stringify({ name, value }) });
}

async function until(test: () => boolean | Promise<boolean>, detail: string, timeout = 30_000) {
  const end = Date.now() + timeout;
  while (!(await test())) {
    assert(Date.now() < end, detail);
    await pause(100);
  }
}

async function action(chat: string, op: string, extra: Partial<BrowserAction> = {}, mustPass = true) {
  const id = `native-${++sequence}`;
  await runBrowserAction({ id, sessionId: chat, op, policy: { allowedDomains: [], blockedDomains: [] }, ...extra }, {
    runKey: `run-${chat}`, chat, scope: chat, fullAccess: true, userText: "Test my own local page and keep its draft",
  });
  const result = await fetch(`${api}/result/${id}`).then((response) => response.json());
  await checkpoint(id, { chat, op, result });
  if (!result.ok) await checkpoint(`failure-${id}`, useBrowserStore.getState().tabs.map(({ id, chat, nativeError, loading }) => ({ id, chat, nativeError, loading })));
  if (mustPass) assert(result.ok, `${chat} ${op}: ${JSON.stringify(result)}`);
  return result;
}

async function open(chat: string, letter: string) {
  useBrowserStore.getState().showChat(chat);
  useBrowserStore.getState().newTab();
  const id = useBrowserStore.getState().activeTabId!;
  await grantLocalOrigin(id, `${pages}/${letter}`);
  useBrowserStore.getState().navigate(id, { url: `${pages}/${letter}` });
  patchAgentChat(chat, { tabId: id, runKey: null });
  const result = await action(chat, "snapshot");
  assert(result.text.includes(`Native ${letter}`), `Wrong page for ${chat}`);
  return { id, token: result.text.match(/TOKEN:[a-z0-9-]+/i)?.[0], text: result.text };
}

function ref(text: string, role: string, name: string) {
  const line = text.split("\n").find((value) => value.includes(role) && value.includes(name));
  const found = line?.match(/\[(r\d+)\]/)?.[1];
  assert(found, `Missing ${role} ${name}: ${text}`);
  return found!;
}

async function probe() {
  setApiBase(19044);
  useBrowserAgentStore.setState({ enabled: true, scriptAccess: "ask" });
  const window = getCurrentWindow();
  await window.setResizable(true);
  await window.setSize(new LogicalSize(1440, 900));
  await window.show();
  await window.setFocus();
  await until(() => useNativeBrowser.getState().enabled, "Native browser unsupported or bridge unavailable");
  await checkpoint("native_supported", true);
  setBrowserPanelAvailable(true);
  stopViews = startNativeViews();
  const first = await open(a, "A");
  const second = await open(b, "B");
  assert(first.token && second.token && first.token !== second.token, "Missing independent native document tokens");
  await checkpoint("targets", { a: first.id, b: second.id });
  const foreground = document.getElementById("foreground") as HTMLInputElement;
  await window.setFocus();
  await focusPanel(second.id);
  foreground.value = "B unsaved draft";
  foreground.focus();
  await until(() => document.hasFocus() && document.activeElement === foreground, "Main webview could not acquire keyboard focus");
  await checkpoint("foreground_before_a", { value: foreground.value, focused: document.activeElement === foreground, documentFocused: document.hasFocus() });
  await action(a, "type", { ref: ref(first.text, "textbox", "Draft note"), text: "A background value" });
  await action(a, "click", { ref: ref(first.text, "button", "Apply note") });
  await checkpoint("foreground_after_a", { value: foreground.value, focused: document.activeElement === foreground, documentFocused: document.hasFocus() });
  assert(foreground.value === "B unsaved draft" && document.activeElement === foreground && document.hasFocus(), "A stole B's draft or foreground focus");
  const aAfter = await action(a, "snapshot");
  const bAfter = await action(b, "snapshot");
  assert(aAfter.text.includes(first.token) && aAfter.text.includes("Applied: A background value"), "A changed document or failed background input");
  assert(bAfter.text.includes(second.token) && bAfter.text.includes("B original"), "A changed B's native page");
  await checkpoint("background_actions_isolated", true);
  const tab = useBrowserStore.getState().tabs.find((item) => item.id === first.id)!;
  await focusPanel(second.id);
  foreground.focus();
  const other = useBrowserStore.getState().tabs.find((item) => item.id === second.id)!;
  const images = await Promise.all([captureHiddenPage(tab), captureHiddenPage(tab), captureHiddenPage(other)]);
  assert(images.every((image) => image && image.size > 100), "Native parallel capture returned no PNG");
  await checkpoint("foreground_after_capture", { focused: document.activeElement === foreground, documentFocused: document.hasFocus() });
  assert(document.activeElement === foreground && document.hasFocus(), "Capturing A stole B's keyboard focus");
  for (let i = 0; i < images.length; i++) {
    await fetch(`${api}/capture-${i}.png`, { method: "POST", body: images[i] });
  }
  assert((await action(b, "snapshot")).text.includes(second.token), "Observing A changed B's document");
  await checkpoint("parallel_capture_preserves_b", images.map((image) => image!.size));
  observeNative(true);
  await until(async () => !(await whenNativeViewShown(second.id, 50)), "B's native view did not hide for observation");
  await focusPanel(second.id);
  foreground.focus();
  await until(() => document.hasFocus() && document.activeElement === foreground, "Observation did not return focus to Studio");
  await action(b, "type", { ref: ref(bAfter.text, "textbox", "Draft note"), text: "B while observing A" });
  const observedAction = await action(b, "click", { ref: ref(bAfter.text, "button", "Apply note") });
  assert(observedAction.text.includes("Applied: B while observing A"), "B could not act while its own page was hidden");
  assert(foreground.value === "B unsaved draft" && document.activeElement === foreground && document.hasFocus(), "B's hidden native page stole observer focus");
  await checkpoint("observer_mode_background_input_isolated", true);
  observeNative(false);
  await until(() => whenNativeViewShown(second.id, 50), "B's native view did not return after observation");
  endBrowserAgentRun(`run-${a}`);
  endBrowserAgentRun(`run-${b}`);
  stopViews();
  stopViews = undefined;
  await pause(500);
  await checkpoint("retained_after_reply_and_panel_close", { a: hasNativeView(first.id), b: hasNativeView(second.id) });
  assert(hasNativeView(first.id) && hasNativeView(second.id), "Owned native pages closed after reply and panel close");
  stopViews = startNativeViews();
  useBrowserStore.getState().showChat(b);
  patchAgentChat(b, { runKey: `run-${b}` });
  stopBrowserAgent(b);
  const continued = await action(a, "snapshot");
  assert(continued.text.includes(first.token), "Stopping B lost A's page");
  await checkpoint("stop_b_preserves_a", true);
  patchAgentChat(b, { tabId: first.id, stoppedRun: null });
  const foreign = await action(b, "snapshot", {}, false);
  assert(!foreign.ok, "B could control A's target");
  patchAgentChat(b, { tabId: second.id });
  await checkpoint("foreign_target_rejected", true);
  useBrowserStore.getState().showChat(a);
  assert((await action(a, "snapshot")).text.includes(first.token), "Returning to A recreated its document");
  useBrowserStore.getState().closeTab(first.id);
  await until(() => !hasNativeView(first.id), "Closing A's tab did not release its view");
  await checkpoint("close_tab_releases_view", true);
  return true;
}

function Fixture() {
  const state = useBrowserStore();
  const [observing, setObserving] = useState(false);
  observeNative = setObserving;
  useEffect(() => {
    void probe().then(() => ({ passed: true }), (error) => ({ passed: false, error: String(error) }))
      .then(async (result) => {
        stopViews?.();
        await fetch(`${api}/complete`, { method: "POST", body: JSON.stringify({ ...facts, ...result, platform: navigator.userAgent }) });
      });
  }, []);
  return <main style={{ padding: 24 }}><h1>Native parallel browser verification</h1><label>Foreground chat draft<input id="foreground" /></label><p>Selected chat: {state.chat}</p><section data-browser-area style={{ position: "fixed", left: 600, right: 24, top: 120, bottom: 24 }}>{!observing && <div data-native-page={state.activeTabId ?? ""} style={{ width: "100%", height: "100%" }} />}</section></main>;
}

window.addEventListener("error", (event) => { void checkpoint("window_error", event.message); });
window.addEventListener("unhandledrejection", (event) => { void checkpoint("unhandled_rejection", String(event.reason)); });
createRoot(document.getElementById("root")!).render(<Fixture />);
