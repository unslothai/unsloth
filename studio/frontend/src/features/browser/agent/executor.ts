// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Runs the model's browser actions (core/inference/browser_agent.py) in the agent's tab and posts
 *  each result back. Every action goes through the gates in gate.ts first. */

import { authFetch } from "@/features/auth";
import { isTauri } from "@/lib/api-base";
import { screenshotPage } from "../capture";
import { hasNativeView, nativeAction } from "../native-view";
import { callNative, useNativeBrowser } from "../native-support";
import { requestFrameAgent } from "../page-frame";
import { browserPanelAvailable } from "../panel-availability";
import { type BrowserTab, currentEntry, useBrowserStore } from "../store";
import { askUser, useBrowserAgentStore } from "./agent-store";
import { type Gate, type SitePolicy, gateElement, gateNavigation, hostOfUrl, policyBlocks } from "./gate";
import { type AgentAnswer, type ElementInfo, pageAgentSource } from "./page-agent";

export type BrowserAction = {
  id: string;
  sessionId: string;
  op: string;
  url?: string;
  ref?: string;
  text?: string;
  submit?: boolean;
  value?: string;
  direction?: "up" | "down";
  key?: string;
  fullPage?: boolean;
  query?: string;
  policy: SitePolicy;
};

/** The chat run an action belongs to, and what its user asked for. */
export type ActionContext = { runKey: string; userText: string; signal?: AbortSignal };

type Result = { ok: boolean; error?: string; url?: string; title?: string; text?: string; note?: string; image?: string };

const LOAD_TIMEOUT_MS = 30_000;
const SETTLE_QUIET_MS = 350;
const SETTLE_MAX_MS = 3_000;
const PAGE_READY_MS = 4_000;
export const STOPPED = "The user stopped the browser agent. Do not use browser tools again unless they ask.";

const str = (value: unknown, max: number): string | undefined =>
  typeof value === "string" && value.length <= max ? value : undefined;
const strings = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string").slice(0, 100) : [];

/** The `browser_action` of a tool_output event, checked: it reaches here off the chat stream. */
export function parseBrowserAction(value: unknown): BrowserAction | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const id = str(raw.id, 64);
  const op = str(raw.op, 32);
  if (!id || !op) return null;
  return {
    id,
    sessionId: str(raw.session_id, 256) ?? "",
    op,
    url: str(raw.url, 8192),
    ref: str(raw.ref, 16),
    text: str(raw.text, 8192),
    submit: raw.submit === true,
    value: str(raw.value, 512),
    direction: raw.direction === "up" ? "up" : "down",
    key: str(raw.key, 32),
    fullPage: raw.full_page === true,
    query: str(raw.query, 256),
    policy: { allowedDomains: strings(raw.allowed_domains), blockedDomains: strings(raw.blocked_domains) },
  };
}

// Sites approved or reached this session, and the links on the page the agent last read.
const knownHosts = new Set<string>();
let pageLinks = new Set<string>();

let queue: Promise<void> = Promise.resolve();
// A replayed stream frame must not click twice.
const seen = new Set<string>();

/** Runs actions one at a time, in the order the model asked for them. */
export function runBrowserAction(action: BrowserAction, context: ActionContext): Promise<void> {
  if (seen.has(action.id)) return queue;
  seen.add(action.id);
  const next = queue.then(async () => {
    let result: Result;
    try {
      result = await perform(action, context);
    } catch (error) {
      result = { ok: false, error: error instanceof Error ? error.message : String(error) };
    } finally {
      useBrowserAgentStore.setState({ working: false });
    }
    await postResult(action, result).catch(() => undefined);
  });
  queue = next.catch(() => undefined);
  return next;
}

async function postResult(action: BrowserAction, result: Result): Promise<void> {
  await authFetch("/api/inference/browser-action-result", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action_id: action.id, session_id: action.sessionId, result }),
  });
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function agentTab(): BrowserTab | null {
  const { tabId } = useBrowserAgentStore.getState();
  return useBrowserStore.getState().tabs.find((tab) => tab.id === tabId) ?? null;
}

function tabUrl(tab: BrowserTab): string {
  const entry = currentEntry(tab);
  return entry.kind === "web" ? (tab.displayUrl ?? entry.url) : "";
}

/** The agent's tab in front, so the user sees what it does and its page is mounted. */
function showTab(tab: BrowserTab): void {
  const store = useBrowserStore.getState();
  if (store.activeTabId !== tab.id) store.activateTab(tab.id);
  if (!store.open) store.openPanel();
}

async function waitForLoad(tabId: string, signal?: AbortSignal): Promise<BrowserTab | null> {
  const started = Date.now();
  // A navigation marks the tab loading at once; a native one may report a beat later.
  await sleep(150);
  while (Date.now() - started < LOAD_TIMEOUT_MS) {
    if (signal?.aborted) return null;
    const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId);
    if (!tab) return null;
    if (!tab.loading) return tab;
    await sleep(100);
  }
  return useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId) ?? null;
}

function checkedAnswer(value: unknown): AgentAnswer | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  if (typeof raw.ok !== "boolean") return null;
  const info = raw.info && typeof raw.info === "object" ? (raw.info as Record<string, unknown>) : null;
  return {
    ok: raw.ok,
    error: str(raw.error, 2000),
    text: typeof raw.text === "string" ? raw.text.slice(0, 200_000) : undefined,
    links: Array.isArray(raw.links)
      ? raw.links.filter((link): link is string => typeof link === "string" && link.length <= 8192).slice(0, 400)
      : [],
    note: str(raw.note, 1000),
    navigate: str(raw.navigate, 8192),
    quietMs: typeof raw.quietMs === "number" ? raw.quietMs : undefined,
    title: typeof raw.title === "string" ? raw.title.slice(0, 300) : undefined,
    info: info
      ? {
          role: String(info.role ?? "").slice(0, 40),
          name: String(info.name ?? "").slice(0, 200),
          tag: String(info.tag ?? "").slice(0, 40),
          type: String(info.type ?? "").slice(0, 40),
          href: String(info.href ?? "").slice(0, 8192),
          autocomplete: String(info.autocomplete ?? "").slice(0, 200),
          formMethod: String(info.formMethod ?? "").slice(0, 10),
          formAction: String(info.formAction ?? "").slice(0, 8192),
          formHasPassword: info.formHasPassword === true,
          submits: info.submits === true,
        }
      : undefined,
  };
}

/** One command in the tab's page, installing the agent there first if this page lacks it. */
async function pageCall(tab: BrowserTab, op: Record<string, unknown>): Promise<AgentAnswer> {
  const native = useNativeBrowser.getState().enabled;
  const started = Date.now();
  while (true) {
    let reply: { installed: boolean; result: unknown } | null = null;
    if (native) {
      if (hasNativeView(tab.id)) {
        const call = (code?: string) =>
          callNative<{ installed: boolean; result: unknown }>("browser_view_agent", {
            tabId: tab.id,
            command: { op, ...(code ? { code } : {}) },
          }).catch(() => null);
        reply = await call();
        if (reply && !reply.installed) reply = await call(pageAgentSource());
      }
    } else {
      reply = await requestFrameAgent(tab.id, op);
      if (reply && !reply.installed) reply = await requestFrameAgent(tab.id, op, pageAgentSource());
    }
    const answer = reply?.installed ? checkedAnswer(reply.result) : null;
    if (answer) return answer;
    // The page may still be mounting after a navigation.
    if (Date.now() - started > PAGE_READY_MS) {
      throw new Error("the page is not answering; it may still be loading, or it is not a web page. Try browser_snapshot again.");
    }
    await sleep(250);
  }
}

/** Until the page stops changing after an action, following any navigation it started. */
async function settle(tabId: string, urlBefore: string, signal?: AbortSignal): Promise<BrowserTab | null> {
  await sleep(120);
  const started = Date.now();
  while (Date.now() - started < SETTLE_MAX_MS) {
    const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId);
    if (!tab) return null;
    if (tab.loading || tabUrl(tab) !== urlBefore) return waitForLoad(tabId, signal);
    const status = await pageCall(tab, { op: "status" }).catch(() => null);
    if (!status || (status.quietMs ?? 0) >= SETTLE_QUIET_MS) return tab;
    await sleep(120);
  }
  return useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId) ?? null;
}

async function pass(gate: Gate, signal?: AbortSignal): Promise<string | null> {
  if (gate.kind === "allow") return null;
  if (gate.kind === "block") return `Not done: ${gate.reason}.`;
  const { ask, host, name, detail } = gate;
  const allowed = await askUser({ ask, host, name, detail }, signal);
  return allowed ? null : `The user declined: ${gate.prompt.replace(/\?$/, "")}. Do something else, or ask them.`;
}

/** The page as the model reads it after an action. */
async function observe(tab: BrowserTab, note?: string, op: Record<string, unknown> = { op: "snapshot" }): Promise<Result> {
  const entry = currentEntry(tab);
  if (entry.kind !== "web") return { ok: false, error: "No web page is open. Use browser_navigate first." };
  if (tab.pageError || tab.nativeError) {
    return {
      ok: false,
      error: `${tabUrl(tab)} could not be loaded (the site refused, needs a check the panel can't pass, or is unreachable).`,
    };
  }
  const answer = await pageCall(tab, op);
  if (!answer.ok) return { ok: false, error: answer.error ?? "the page could not be read." };
  if (op.op === "snapshot" || op.op === "find") pageLinks = new Set(answer.links ?? []);
  const fresh = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tab.id) ?? tab;
  return {
    ok: true,
    url: tabUrl(fresh),
    title: fresh.customTitle ?? fresh.title,
    text: answer.text,
    note: [note, answer.note].filter(Boolean).join(" ") || undefined,
  };
}

/** Opens `url` in the agent's tab, making the tab if it is gone. */
async function open(url: string, signal?: AbortSignal): Promise<BrowserTab | null> {
  const store = useBrowserStore.getState();
  const existing = agentTab();
  if (existing) {
    store.navigate(existing.id, { url }, { replace: false });
    showTab(existing);
    return waitForLoad(existing.id, signal);
  }
  store.openUrl(url, { newTab: true });
  const tabId = useBrowserStore.getState().activeTabId;
  if (!tabId) return null;
  useBrowserAgentStore.setState({ tabId });
  return waitForLoad(tabId, signal);
}

function navigationContext(action: BrowserAction, context: ActionContext) {
  return { policy: action.policy, knownHosts, pageLinks, userText: context.userText };
}

async function perform(action: BrowserAction, context: ActionContext): Promise<Result> {
  const { signal } = context;
  if (useBrowserAgentStore.getState().stoppedRun === context.runKey) return { ok: false, error: STOPPED };
  if (!browserPanelAvailable()) {
    return { ok: false, error: "the browser panel can't be shown in this window; ask the user to widen it or open the chat." };
  }
  useBrowserAgentStore.setState({ working: true, runKey: context.runKey });

  if (action.op === "navigate") {
    const url = action.url ?? "";
    const refused = await pass(gateNavigation(url, navigationContext(action, context)), signal);
    if (refused) return { ok: false, error: refused };
    knownHosts.add(hostOfUrl(url));
    const tab = await open(url, signal);
    if (!tab) return { ok: false, error: "the browser tab closed." };
    return observeLoaded(tab);
  }

  const tab = agentTab();
  if (!tab) return { ok: false, error: "No page is open yet. Call browser_navigate first." };
  showTab(tab);
  const before = tabUrl(tab);

  if (action.op === "back" || action.op === "forward" || action.op === "reload") {
    const native = useNativeBrowser.getState().enabled;
    const store = useBrowserStore.getState();
    if (native && action.op !== "reload" && tab.nativeHistory?.[action.op]) nativeAction(tab.id, action.op);
    else if (native && action.op === "reload") nativeAction(tab.id, "reload");
    else if (action.op === "back") store.goBack(tab.id);
    else if (action.op === "forward") store.goForward(tab.id);
    else store.reload(tab.id);
    const loaded = await waitForLoad(tab.id, signal);
    return loaded ? observeLoaded(loaded) : { ok: false, error: "the browser tab closed." };
  }

  const ready = tab.loading ? await waitForLoad(tab.id, signal) : tab;
  if (!ready) return { ok: false, error: "the browser tab closed." };

  switch (action.op) {
    case "snapshot":
      return observe(ready, undefined, { op: "snapshot", fullPage: action.fullPage, ...(action.ref ? { ref: action.ref } : {}) });
    case "find":
      return observe(ready, undefined, { op: "find", query: action.query ?? "" });
    case "scroll": {
      const moved = await pageCall(ready, { op: "scroll", direction: action.direction, ...(action.ref ? { ref: action.ref } : {}) });
      if (!moved.ok) return { ok: false, error: moved.error };
      const after = await settle(ready.id, before, signal);
      return after ? observe(after, moved.note) : { ok: false, error: "the browser tab closed." };
    }
    case "screenshot":
      return screenshot(ready);
  }

  // Acting on an element: what it is decides whether the user is asked first.
  const ref = action.ref ?? "";
  const described = await pageCall(ready, { op: "describe", ref });
  if (!described.ok || !described.info) return { ok: false, error: described.error ?? `${ref} could not be found.` };
  const info: ElementInfo = described.info;
  const gates: Gate[] = [];
  if (action.op === "click" || action.op === "select" || action.op === "type") {
    gates.push(gateElement(action.op, info, before, action.policy));
  }
  // Enter in a field sends its form, so it is held to what a submit button would be.
  const submits = (action.op === "type" && action.submit) || (action.op === "press" && action.key === "Enter");
  if (submits && info.formAction) gates.push(gateElement("submit", info, before, action.policy));
  for (const gate of gates) {
    const refused = await pass(gate, signal);
    if (refused) return { ok: false, error: refused };
  }

  let note: string | undefined;
  if (action.op === "click") {
    const clicked = await pageCall(ready, { op: "click", ref, directLinks: !useNativeBrowser.getState().enabled });
    if (!clicked.ok) return { ok: false, error: clicked.error };
    if (clicked.navigate) {
      // A link: opened in this tab, under the same rules as any other address.
      const blocked = policyBlocks(clicked.navigate, action.policy);
      if (blocked) return { ok: false, error: `Not done: ${blocked}.` };
      knownHosts.add(hostOfUrl(clicked.navigate));
      useBrowserStore.getState().navigate(ready.id, { url: clicked.navigate, from: before }, { replace: false });
      const loaded = await waitForLoad(ready.id, signal);
      return loaded ? observeLoaded(loaded) : { ok: false, error: "the browser tab closed." };
    }
    note = clicked.note;
  } else if (action.op === "type") {
    const typed = await pageCall(ready, { op: "type", ref, text: action.text ?? "", submit: action.submit === true });
    if (!typed.ok) return { ok: false, error: typed.error };
    note = typed.note;
  } else if (action.op === "select") {
    const chosen = await pageCall(ready, { op: "select", ref, value: action.value ?? "" });
    if (!chosen.ok) return { ok: false, error: chosen.error };
    note = chosen.note;
  } else if (action.op === "press") {
    const pressed = await pageCall(ready, { op: "press", key: action.key ?? "" });
    if (!pressed.ok) return { ok: false, error: pressed.error };
    note = pressed.note;
  } else {
    return { ok: false, error: `unknown browser action ${action.op}.` };
  }
  const after = await settle(ready.id, before, signal);
  if (!after) return { ok: false, error: "the browser tab closed." };
  const now = tabUrl(after);
  if (now && now !== before) {
    const blocked = policyBlocks(now, action.policy);
    if (blocked) return { ok: false, error: `The page went to ${hostOfUrl(now)}, but ${blocked}. Go back.` };
    knownHosts.add(hostOfUrl(now));
    return observeLoaded(after, note);
  }
  return observe(after, note);
}

async function observeLoaded(tab: BrowserTab, note?: string): Promise<Result> {
  const entry = currentEntry(tab);
  if (entry.kind === "web" && tab.documentType && !/html/i.test(tab.documentType)) {
    return {
      ok: true,
      url: tabUrl(tab),
      title: tab.title,
      note: `${note ? `${note} ` : ""}This address is a file (${tab.documentType}), shown to the user; the browser tools can't read it. web_search with this url can.`,
    };
  }
  return observe(tab, note);
}

async function screenshot(tab: BrowserTab): Promise<Result> {
  if (!isTauri) return { ok: false, error: "screenshots of the browser panel need the desktop app; use browser_snapshot." };
  const pages = document.querySelectorAll<HTMLElement>("[data-browser-page]");
  const element = pages[pages.length - 1];
  if (!element) return { ok: false, error: "the browser panel is not on screen." };
  const blob = await screenshotPage(tab, element);
  if (!blob) return { ok: false, error: "the screenshot failed." };
  const image = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
  return { ok: true, url: tabUrl(tab), title: tab.title, image };
}
