// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Runs the model's browser actions (core/inference/browser_agent.py) in the agent's tab and posts
 *  each result back. Every action goes through the gates in gate.ts first. */

import { authFetch } from "@/features/auth";
import { isTauri } from "@/lib/api-base";
import { screenshotPage } from "../capture";
import { useBrowserHistoryStore } from "../history-store";
import { hasNativeView, nativeAction } from "../native-view";
import { callNative, useNativeBrowser } from "../native-support";
import { requestFrameAgent } from "../page-frame";
import { browserPanelAvailable } from "../panel-availability";
import { type BrowserTab, currentEntry, useBrowserStore } from "../store";
import { askUser, setActiveAction, useBrowserAgentStore } from "./agent-store";
import { type Gate, type Question, type SitePolicy, gateElement, gateNavigation, hostOfUrl, navigationBlocks } from "./gate";
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
  limit?: number;
  /** How long the server waits for this answer. */
  timeoutS?: number;
  policy: SitePolicy;
};

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
    limit: typeof raw.limit === "number" && Number.isInteger(raw.limit) ? raw.limit : undefined,
    timeoutS: typeof raw.timeout_s === "number" && raw.timeout_s > 0 ? Math.min(raw.timeout_s, 3600) : undefined,
    policy: { allowedDomains: strings(raw.allowed_domains), blockedDomains: strings(raw.blocked_domains) },
  };
}

// Sites approved or reached this session, and the links on the page the agent last read.
const knownHosts = new Set<string>();
let pageLinks = new Set<string>();

const TEXT_ROLES = new Set(["textbox", "searchbox", "password", "combobox", "spinbutton"]);

let queue: Promise<void> = Promise.resolve();
// A replayed stream frame must not click twice.
const seen = new Set<string>();

/** Runs actions one at a time, in the order the model asked for them. */
export function runBrowserAction(action: BrowserAction, context: ActionContext): Promise<void> {
  if (seen.has(action.id)) return queue;
  seen.add(action.id);
  // The server's wait starts when it sends the action, not when an earlier one lets this one run.
  const expires = action.timeoutS ? Date.now() + (action.timeoutS - 2) * 1000 : null;
  const next = queue.then(async () => {
    // Ends with the chat run, with Stop, or when the server stops waiting: a question still open
    // then must not be answerable, or a late Allow would act for nobody.
    const controller = new AbortController();
    const abort = () => controller.abort();
    context.signal?.addEventListener("abort", abort, { once: true });
    if (context.signal?.aborted) abort();
    if (expires !== null && expires <= Date.now()) abort();
    const deadline = expires !== null ? setTimeout(abort, Math.max(0, expires - Date.now())) : null;
    setActiveAction(controller);
    let result: Result;
    try {
      result = await perform(action, { ...context, signal: controller.signal });
    } catch (error) {
      result = { ok: false, error: error instanceof Error ? error.message : String(error) };
    } finally {
      if (deadline) clearTimeout(deadline);
      context.signal?.removeEventListener("abort", abort);
      setActiveAction(null);
      useBrowserAgentStore.setState({ working: false });
    }
    if (useBrowserAgentStore.getState().stoppedRun === context.runKey) result = { ok: false, error: STOPPED };
    await postResult(action, result);
  });
  queue = next.catch(() => undefined);
  return next;
}

/** Retried: a lost answer leaves the tool waiting for its whole timeout. */
async function postResult(action: BrowserAction, result: Result): Promise<void> {
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const response = await authFetch("/api/inference/browser-action-result", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action_id: action.id, session_id: action.sessionId, result }),
      });
      // 404: the server already gave up on this action; nothing to deliver.
      if (response.ok || (response.status >= 400 && response.status < 500)) return;
    } catch {
      // Network or auth refresh: try again.
    }
    await sleep(500 * 2 ** attempt);
  }
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

// Each newly installed page numbers its refs in its own range (page-agent.ts `base`, `span`),
// within the server's 9 digits: r1, r100001, r200001...
let nextBase = 1;
const REF_SPAN = 100_000;
const LAST_BASE = 9_000 * REF_SPAN;

/** One command in the tab's page, installing the agent there first if this page lacks it. */
async function pageCall(tab: BrowserTab, command: Record<string, unknown>): Promise<AgentAnswer> {
  const native = useNativeBrowser.getState().enabled;
  const started = Date.now();
  const fresh = () => {
    const base = nextBase;
    nextBase = nextBase > LAST_BASE ? 1 : nextBase + REF_SPAN;
    return { ...command, base, span: REF_SPAN };
  };
  while (true) {
    let reply: { installed: boolean; result: unknown } | null = null;
    if (native) {
      if (hasNativeView(tab.id)) {
        const call = (op: Record<string, unknown>, code?: string) =>
          callNative<{ installed: boolean; result: unknown }>("browser_view_agent", {
            tabId: tab.id,
            command: { op, ...(code ? { code } : {}) },
          }).catch(() => null);
        reply = await call(command);
        if (reply && !reply.installed) reply = await call(fresh(), pageAgentSource());
      }
    } else {
      reply = await requestFrameAgent(tab.id, command);
      if (reply && !reply.installed) reply = await requestFrameAgent(tab.id, fresh(), pageAgentSource());
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
  const allowed = await askInPanel({ ask, host, name, detail }, signal);
  return allowed ? null : `The user declined: ${gate.prompt.replace(/\?$/, "")}. Do something else, or ask them.`;
}

/** Questions show in the panel's agent bar, so the panel has to be on screen to ask. */
function askInPanel(question: Question, signal?: AbortSignal): Promise<boolean> {
  const store = useBrowserStore.getState();
  if (!store.open) store.openPanel();
  return askUser(question, signal);
}

async function observe(tab: BrowserTab, note?: string, op: Record<string, unknown> = { op: "snapshot" }): Promise<Result> {
  const entry = currentEntry(tab);
  if (entry.kind !== "web") return { ok: false, error: "No web page is open. Use browser_navigate first." };
  if (tab.pageError || tab.nativeError) {
    return {
      ok: false,
      // The host only: an error is outside the untrusted-content fence, and a path is the page's words.
      error: `${hostOfUrl(tabUrl(tab)) || "The page"} could not be loaded (the site refused, needs a check the panel can't pass, or is unreachable).`,
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

const permissions = () => useBrowserAgentStore.getState().permissions;

function navigationContext(action: BrowserAction, context: ActionContext) {
  const tab = agentTab();
  return {
    policy: action.policy,
    permissions: permissions(),
    currentUrl: tab ? tabUrl(tab) : "",
    knownHosts,
    pageLinks,
    userText: context.userText,
  };
}

async function perform(action: BrowserAction, context: ActionContext): Promise<Result> {
  const { signal } = context;
  if (useBrowserAgentStore.getState().stoppedRun === context.runKey) return { ok: false, error: STOPPED };
  if (signal?.aborted) return { ok: false, error: "the action expired before it could run." };
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
    return arrive(tab, url, action, context);
  }

  if (action.op === "history") return readHistory(action, signal);

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
    return loaded ? arrive(loaded, before, action, context) : { ok: false, error: "the browser tab closed." };
  }

  const ready = tab.loading ? await waitForLoad(tab.id, signal) : tab;
  if (!ready) return { ok: false, error: "the browser tab closed." };
  const moved = await stillAllowed(ready, action, context);
  if (moved) return { ok: false, error: moved };

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
    gates.push(gateElement(action.op, info, before, action.policy, permissions()));
  }
  // Enter or Space on a focused button or link is a click on it, so it asks what a click would.
  const activates = action.op === "press" && (action.key === "Enter" || action.key === "Space");
  if (activates && !TEXT_ROLES.has(info.role)) gates.push(gateElement("click", info, before, action.policy, permissions()));
  // Enter in a field sends its form, so it is held to what a submit button would be.
  const submits =
    (action.op === "type" && action.submit) ||
    (action.op === "press" && action.key === "Enter" && TEXT_ROLES.has(info.role)) ||
    ((action.op === "click" || activates) && info.submits);
  if (submits && info.formAction) {
    // A submit button's own click gate already asks what a submit would.
    if (!info.submits) gates.push(gateElement("submit", info, before, action.policy, permissions()));
    // A form sending to another site carries what was typed there: a new site asks, as a navigation would.
    if (hostOfUrl(info.formAction) !== hostOfUrl(before)) gates.push(gateNavigation(info.formAction, navigationContext(action, context)));
  }
  for (const gate of gates) {
    const refused = await pass(gate, signal);
    if (refused) return { ok: false, error: refused };
  }
  if (signal?.aborted) return { ok: false, error: STOPPED };
  if (submits && info.formAction) knownHosts.add(hostOfUrl(info.formAction));

  let note: string | undefined;
  if (action.op === "click") {
    const clicked = await pageCall(ready, { op: "click", ref, directLinks: !useNativeBrowser.getState().enabled });
    if (!clicked.ok) return { ok: false, error: clicked.error };
    if (clicked.navigate) {
      // A link: opened in this tab, under the same rules as any other address.
      const blocked = navigationBlocks(clicked.navigate, action.policy, permissions());
      if (blocked) return { ok: false, error: `Not done: ${blocked}.` };
      knownHosts.add(hostOfUrl(clicked.navigate));
      useBrowserStore.getState().navigate(ready.id, { url: clicked.navigate, from: before }, { replace: false });
      const loaded = await waitForLoad(ready.id, signal);
      return loaded ? arrive(loaded, clicked.navigate, action, context) : { ok: false, error: "the browser tab closed." };
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
  if (now && now !== before) return arrive(after, before, action, context, note);
  return observe(after, note);
}

/** Where a navigation ended up, checked before the model reads it: a redirect or a script can land
 *  on a blocked site, or on a new one the user never approved. */
async function arrive(
  tab: BrowserTab,
  requested: string,
  action: BrowserAction,
  context: ActionContext,
  note?: string,
): Promise<Result> {
  const url = tabUrl(tab);
  const host = hostOfUrl(url);
  if (url) {
    const blocked = navigationBlocks(url, action.policy, permissions());
    if (blocked) return { ok: false, error: `The page went to ${host}, but ${blocked}. Go back.` };
    if (host && host !== hostOfUrl(requested) && !knownHosts.has(host)) {
      const gate = gateNavigation(url, { ...navigationContext(action, context), currentUrl: requested });
      const refused = await pass(gate, context.signal);
      if (refused) return { ok: false, error: `The page went to ${host}. ${refused}` };
    }
    if (host) knownHosts.add(host);
  }
  return observeLoaded(tab, note);
}

/** The page the tab shows now, before the model reads or acts on it: the page may have moved on
 *  its own, or the user declined where a redirect landed and it is still loaded. */
async function stillAllowed(tab: BrowserTab, action: BrowserAction, context: ActionContext): Promise<string | null> {
  const url = tabUrl(tab);
  const host = hostOfUrl(url);
  if (!host) return null;
  const blocked = navigationBlocks(url, action.policy, permissions());
  if (blocked) return `The page is on ${host}, but ${blocked}. Use browser_navigate to go elsewhere.`;
  if (knownHosts.has(host)) return null;
  const refused = await pass(gateNavigation(url, { ...navigationContext(action, context), currentUrl: "" }), context.signal);
  if (refused) return `The page is on ${host}. ${refused}`;
  knownHosts.add(host);
  return null;
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
  // What the user typed into card, code and password fields stays out of the picture, as out of the text.
  const masked = await pageCall(tab, { op: "mask", on: true }).catch(() => null);
  if (!masked?.ok) return { ok: false, error: "the page could not be prepared for a screenshot; use browser_snapshot." };
  let blob: Blob | null;
  try {
    await sleep(100);
    blob = await screenshotPage(tab, element);
  } finally {
    await pageCall(tab, { op: "mask", on: false }).catch(() => null);
  }
  if (!blob) return { ok: false, error: "the screenshot failed." };
  const image = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
  return { ok: true, url: tabUrl(tab), title: tab.title, image };
}

const HISTORY_LIMIT = 50;

/** The browser_history tool: the user's visits, by their Settings choice (never, ask, allow). */
async function readHistory(action: BrowserAction, signal?: AbortSignal): Promise<Result> {
  const access = useBrowserAgentStore.getState().historyAccess;
  if (access === "never") return { ok: false, error: "The user doesn't let the agent read their browsing history." };
  if (access === "ask") {
    const query = action.query ?? "";
    const allowed = await askInPanel({ ask: "history", host: "", name: query, detail: query }, signal);
    if (!allowed) return { ok: false, error: "The user declined to share their browsing history. Do something else, or ask them." };
  }
  const terms = (action.query ?? "").toLowerCase().split(/\s+/).filter(Boolean);
  const limit = Math.min(Math.max(action.limit ?? 20, 1), HISTORY_LIMIT);
  const visits = useBrowserHistoryStore
    .getState()
    .history.filter((visit) => {
      const haystack = `${visit.title} ${visit.url}`.toLowerCase();
      return terms.every((term) => haystack.includes(term));
    })
    .slice(0, limit);
  const lines = visits.map((visit) => `${new Date(visit.visitedAt).toISOString().slice(0, 16).replace("T", " ")}  ${visit.title || "(untitled)"}  ${visit.url}`);
  return { ok: true, text: lines.length ? lines.join("\n") : "No pages in the browsing history match." };
}
