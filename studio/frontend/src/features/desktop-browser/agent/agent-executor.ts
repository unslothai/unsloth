// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isBrowserToolName } from "@/lib/browser-tool-names";
import {
  type BrowserAgentReply,
  type BrowserSession,
  type BrowserSnapshot,
  resolveAddress,
} from "../browser-session";
import {
  type BrowserApprovalChoice,
  isRunsChatOnScreen,
  useDesktopBrowserStore,
  waitForBrowserSession,
} from "../browser-store";
import {
  escapePageText,
  joinResult,
  pageBlock,
  quoteForSummary,
  siteOf,
} from "./agent-format";
import {
  type BrowserActionKind,
  type SensitiveKind,
  approvalNeeded,
  asSensitiveKind,
  sensitiveDetail,
} from "./agent-policy";
import {
  type ClientToolImage,
  claimClientTool,
  sendClientToolResult,
} from "./client-tool-api";

/** one browser call the backend parked for this client (browser_tools.run_browser_tool). */
export type BrowserClientRequest = {
  request_id: string;
  tool: string;
  arguments: Record<string, unknown>;
  /** room the result has in the model's window, when the backend measured it. */
  max_chars?: number;
  max_tokens?: number;
};

export type BrowserRunContext = {
  /** the chat request's session_id, which scopes the claim and the result. */
  backendSessionId: string;
  /** the tool card this call renders as; approvals show on it. */
  partId: string;
  threadId: string | null;
  /** the chat on screen when the run was sent (browser-store `viewKey`). */
  viewKey: string | null;
  permissionMode: string;
  signal?: AbortSignal;
  /** stops the whole run, for the pane's Stop button. */
  stop?: () => void;
};

type Outcome = { text: string; images?: ClientToolImage[] };

// about 1.5k tokens: room for a busy page's viewport without crowding a small model's window.
const SNAPSHOT_CHARS = 6000;
const READ_CHARS = 8000;
// below this a snapshot cannot list a page's controls; the backend's cap takes over instead.
const MIN_PAGE_CHARS = 1200;
// the action summary and notes around the page text.
const RESULT_OVERHEAD_CHARS = 400;

/** how much page text a result may carry: characters to ask for, and the tokens the backend keeps. */
type PageSize = { chars: number; tokens?: number };

function pageSize(cap: number, request: BrowserClientRequest): PageSize {
  const room = request.max_chars;
  const chars =
    room === undefined
      ? cap
      : Math.max(MIN_PAGE_CHARS, Math.min(cap, room - RESULT_OVERHEAD_CHARS));
  return request.max_tokens === undefined
    ? { chars }
    : { chars, tokens: request.max_tokens - RESULT_OVERHEAD_CHARS / 4 };
}

// ascii at about 2.5 characters a token (link-heavy pages measure 2.35-2.85) and anything else at one, as cjk costs
function tokenEstimate(text: string): number {
  let tokens = 0;
  for (let i = 0; i < text.length; i++) {
    tokens += text.charCodeAt(i) < 128 ? 0.4 : 1;
  }
  return tokens;
}

/** refetches smaller when the text exceeds the backend's token cap, whose cut would drop the trailing refs and read offset. */
async function sizedPageReply(
  fetch: (maxChars: number, retake: boolean) => Promise<BrowserAgentReply>,
  size: PageSize,
): Promise<BrowserAgentReply> {
  const reply = await fetch(size.chars, false);
  const text = String(reply.text ?? "");
  if (!reply.ok || size.tokens === undefined) return reply;
  const cost = tokenEstimate(text);
  if (cost <= size.tokens || text.length <= MIN_PAGE_CHARS) return reply;
  return fetch(
    Math.max(MIN_PAGE_CHARS, Math.floor((text.length * size.tokens) / cost)),
    true,
  );
}
const QUIET_MS = 350;
const SETTLE_MS = 3000;
// how long a click or key gets to start a navigation before it is taken not to have caused one.
const ACTION_START_MS = 700;
const LOAD_MS = 20_000;
const SCREENSHOT_EDGE = 1024;

export function parseBrowserClientRequest(
  value: unknown,
): BrowserClientRequest | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  if (typeof raw.request_id !== "string" || !raw.request_id) return null;
  if (!isBrowserToolName(raw.tool)) return null;
  const args =
    raw.arguments && typeof raw.arguments === "object"
      ? (raw.arguments as Record<string, unknown>)
      : {};
  const finite = (v: unknown) =>
    typeof v === "number" && Number.isFinite(v) ? v : undefined;
  const maxChars = finite(raw.max_chars);
  const maxTokens = finite(raw.max_tokens);
  return {
    request_id: raw.request_id,
    tool: raw.tool,
    arguments: args,
    ...(maxChars === undefined ? {} : { max_chars: maxChars }),
    ...(maxTokens === undefined ? {} : { max_tokens: maxTokens }),
  };
}

let queue: Promise<void> = Promise.resolve();

/** calls run one at a time, but each is claimed on arrival so one waiting its turn does not look unanswered to the backend. */
export function runBrowserClientRequest(
  request: BrowserClientRequest,
  context: BrowserRunContext,
): Promise<void> {
  const claimed = claimClientTool(
    context.backendSessionId,
    request.request_id,
  ).catch(() => false);
  const run = queue.then(async () => {
    if (await claimed) await handle(request, context);
  });
  queue = run.catch(() => {});
  return run;
}

async function handle(
  request: BrowserClientRequest,
  context: BrowserRunContext,
): Promise<void> {
  let outcome: Outcome;
  try {
    outcome = await execute(request, context);
  } catch (error) {
    outcome = {
      text: context.signal?.aborted
        ? "Error: the user stopped the browser action."
        : `Error: ${errorText(error)}`,
    };
  } finally {
    const store = useDesktopBrowserStore.getState();
    store.setActivity(null);
    store.setApproval(context.partId, null);
  }
  if (outcome.text.includes(OTHER_CHAT)) leftChat(context);
  await sendClientToolResult(
    context.backendSessionId,
    request.request_id,
    outcome.text,
    outcome.images,
  ).catch(() => false);
}

function errorText(error: unknown): string {
  const text = error instanceof Error ? error.message : String(error);
  return text.replace(/^Error:\s*/, "") || "the browser action failed";
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("Stopped"));
      return;
    }
    const timer = window.setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(new Error("Stopped"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function refArg(value: unknown): string {
  // small models write the ref the way the snapshot shows it: "[e12]", "e12" or just 12.
  const text = String(value ?? "")
    .trim()
    .replace(/^\[|\]$/g, "")
    .replace(/^ref[=:]\s*/i, "");
  return /^\d+$/.test(text) ? `e${text}` : text;
}

function boolArg(value: unknown, fallback: boolean): boolean {
  if (value === true || value === "true") return true;
  if (value === false || value === "false") return false;
  return fallback;
}

/** waits until the page stops changing: loaded, no DOM mutations and no new requests for a beat. */
async function settle(
  session: BrowserSession,
  signal?: AbortSignal,
  maxMs = SETTLE_MS,
): Promise<void> {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    const nav = await session.refresh().catch(() => null);
    if (!nav?.loading) {
      const status = await session.agent("status", {}).catch(() => null);
      // no runtime (an error page, about:blank) has nothing left to wait for.
      if (!status?.ok) return;
      if (
        status.readyState !== "loading" &&
        numberOr(status.msSinceMutation, Number.POSITIVE_INFINITY) >=
          QUIET_MS &&
        numberOr(status.msSinceResource, Number.POSITIVE_INFINITY) >= QUIET_MS
      ) {
        return;
      }
    }
    await sleep(150, signal);
  }
}

/** which document the pane shows: its URL changes before a load starts, the runtime's id after. */
type PageMark = { url: string; docId: string | null };

async function pageMark(
  session: BrowserSession,
  nav: BrowserSnapshot | null,
): Promise<PageMark> {
  const status = await session.agent("status", {}).catch(() => null);
  return {
    url: nav?.url ?? "",
    docId: status?.ok && typeof status.docId === "string" ? status.docId : null,
  };
}

function sameDocument(a: string, b: string): boolean {
  try {
    const left = new URL(a);
    const right = new URL(b);
    left.hash = "";
    right.hash = "";
    return left.href === right.href;
  } catch {
    return false;
  }
}

/** "navigation" waits for a new document or an error; "action" gives a click or key a moment to start a load, then settles in place. */
async function waitForLoad(
  session: BrowserSession,
  before: PageMark,
  signal: AbortSignal | undefined,
  mode: "navigation" | "action",
): Promise<BrowserSnapshot | null> {
  let nav: BrowserSnapshot | null = null;
  const startBy =
    Date.now() + (mode === "navigation" ? LOAD_MS : ACTION_START_MS);
  while (Date.now() < startBy) {
    nav = await session.refresh().catch(() => null);
    if (nav?.loading || nav?.error) break;
    const mark = await pageMark(session, nav);
    if (mark.docId && before.docId && mark.docId !== before.docId) break;
    if (mode === "action" && nav && nav.url !== before.url) break;
    await sleep(100, signal);
  }
  // on WebKitGTK the URL moves when a load starts but loading is reported only once it commits, so wait for the location to catch up.
  const committedBy = Date.now() + LOAD_MS;
  while (nav && before.docId && Date.now() < committedBy) {
    const status = await session.agent("status", {}).catch(() => null);
    if (
      !status?.ok ||
      status.docId !== before.docId ||
      typeof status.url !== "string" ||
      sameDocument(status.url, nav.url)
    ) {
      break;
    }
    await sleep(150, signal);
    nav = await session.refresh().catch(() => nav);
  }
  return untilLoaded(session, nav, signal);
}

async function untilLoaded(
  session: BrowserSession,
  nav: BrowserSnapshot | null,
  signal: AbortSignal | undefined,
): Promise<BrowserSnapshot | null> {
  const doneBy = Date.now() + LOAD_MS;
  while (nav?.loading && Date.now() < doneBy) {
    await sleep(150, signal);
    nav = await session.refresh().catch(() => nav);
  }
  await settle(session, signal);
  return session.refresh().catch(() => nav);
}

async function snapshotBlock(
  session: BrowserSession,
  size: PageSize = { chars: SNAPSHOT_CHARS },
): Promise<string> {
  const reply: BrowserAgentReply = await sizedPageReply(
    (maxChars, retake) => session.agent("snapshot", { maxChars, retake }),
    size,
  ).catch((error: unknown) => ({ ok: false, error: errorText(error) }));
  if (!reply.ok) {
    return `(The page could not be read: ${untrusted(reply.error ?? "unknown error", 200)})`;
  }
  noteRefs(reply);
  const text = String(reply.text ?? "");
  // repeated beside the page, not only in the system prompt: small models otherwise tell the user to sign in and stop.
  return joinResult(
    pageBlock("snapshot", text),
    PASSWORD_FIELD.test(text) ? SIGN_IN_HINT : null,
  );
}

// refs restart at e1 in every document, so one from the page before a navigation names a different element
let refsDocument: string | null = null;

function noteRefs(reply: BrowserAgentReply): void {
  if (reply.ok && typeof reply.docId === "string") refsDocument = reply.docId;
}

const PASSWORD_FIELD = /\(password\b/;
const SIGN_IN_HINT =
  "This page has a password field. If the user needs to sign in, call browser_handoff so they can type it.";

function popupNote(
  before: BrowserSnapshot | null,
  after: BrowserSnapshot | null,
): string | null {
  const popup = after?.popupUrl;
  if (!popup || popup === before?.popupUrl) return null;
  return `The page tried to open a new window (${untrusted(popup, 300)}). It was not opened; use browser_navigate to open it in this browser.`;
}

async function failure(
  session: BrowserSession,
  reply: BrowserAgentReply,
  size?: PageSize,
): Promise<Outcome> {
  const message = `Error: ${untrusted(reply.error ?? "the page could not do that", 300)}`;
  // the model needs the page as it is now to recover: what moved, or what covers the target.
  if (
    reply.code === "stale_ref" ||
    reply.code === "not_found" ||
    reply.code === "intercepted"
  ) {
    return {
      text: joinResult(message, await snapshotBlock(session, size)),
    };
  }
  return { text: message };
}

/** calls `leave` once the user switches chats, since a question or handoff left behind would hold the queue other chats wait in. */
function whenChatLeft(
  context: BrowserRunContext,
  leave: () => void,
): () => void {
  return useDesktopBrowserStore.subscribe(() => {
    if (!onSameChat(context)) leave();
  });
}

function ask(
  context: BrowserRunContext,
  question: string,
  detail: string | null,
  site: string | null,
): Promise<BrowserApprovalChoice> {
  const store = useDesktopBrowserStore.getState();
  return new Promise((resolve) => {
    const onAbort = () => finish("deny");
    let stopWatching = () => {};
    const finish = (choice: BrowserApprovalChoice) => {
      context.signal?.removeEventListener("abort", onAbort);
      stopWatching();
      useDesktopBrowserStore.getState().setApproval(context.partId, null);
      resolve(choice);
    };
    if (context.signal?.aborted) {
      resolve("deny");
      return;
    }
    context.signal?.addEventListener("abort", onAbort, { once: true });
    stopWatching = whenChatLeft(context, onAbort);
    store.setApproval(context.partId, {
      question,
      detail,
      site,
      resolve: finish,
    });
  });
}

/** asks when the policy says so, with `action` reading after "wants to" (`click button "Buy"`); false when the user declines. */
async function permitted(
  context: BrowserRunContext,
  kind: BrowserActionKind,
  sensitive: SensitiveKind | null,
  site: string | null,
  action: string,
): Promise<boolean> {
  guard(context);
  const sessionKey = context.threadId ?? context.backendSessionId;
  const store = useDesktopBrowserStore.getState();
  const siteAllowed =
    site !== null && (store.allowedSites[sessionKey] ?? []).includes(site);
  const need = approvalNeeded({
    mode: context.permissionMode,
    kind,
    sensitive,
    siteAllowed,
  });
  if (!need.ask) return true;
  // a "no" ends the request's browser work, since asking again would make the user refuse every retry a small model makes.
  if (deniedRuns.has(runOf(context))) {
    context.stop?.();
    return false;
  }
  const sentence = `${action.charAt(0).toUpperCase()}${action.slice(1)}`;
  const question =
    need.reason === "new-site" && site
      ? `Let the model use ${site}?`
      : `${sentence}${site && kind !== "navigate" ? ` on ${site}` : ""}?`;
  const detail =
    need.reason === "new-site"
      ? `It wants to ${action}.`
      : sensitiveDetail(sensitive);
  const previous = store.activity;
  store.setActivity(
    previous ? { ...previous, label: "Waiting for your approval" } : null,
  );
  const choice = await ask(
    context,
    question,
    detail,
    need.reason === "sensitive" ? null : site,
  );
  useDesktopBrowserStore.getState().setActivity(previous);
  // an answer that came because the run stopped or the user left the chat is not a "no".
  guard(context);
  if (choice === "allow-site" && site) {
    useDesktopBrowserStore.getState().allowSite(sessionKey, site);
  }
  if (choice === "deny") deniedRuns.add(runOf(context));
  return choice !== "deny";
}

const deniedRuns = new WeakSet<object>();

/** one run of the model: every browser call it makes shares the run's end signal. */
function runOf(context: BrowserRunContext): object {
  return context.signal ?? context;
}

function waitForUser(
  context: BrowserRunContext,
  reason: string,
): Promise<boolean> {
  const store = useDesktopBrowserStore.getState();
  return new Promise((resolve) => {
    let stopWatching = () => {};
    const finish = (done: boolean) => {
      context.signal?.removeEventListener("abort", onAbort);
      stopWatching();
      const state = useDesktopBrowserStore.getState();
      state.setHandoff(context.partId, null);
      state.setActivity(null);
      resolve(done);
    };
    const onAbort = () => finish(false);
    if (context.signal?.aborted) {
      resolve(false);
      return;
    }
    context.signal?.addEventListener("abort", onAbort, { once: true });
    stopWatching = whenChatLeft(context, onAbort);
    const done = () => finish(true);
    store.setHandoff(context.partId, { reason, done });
    store.setActivity({
      threadId: context.threadId,
      label: reason ? `Your turn: ${reason}` : "Your turn in the browser",
      stop: context.stop,
      done,
    });
  });
}

const DECLINED: Outcome = {
  text: "Error: the user declined this browser action. Do not retry it; ask the user how to proceed.",
};

/** page-supplied text placed outside the untrusted block: one line, bounded, no wrapper tags. */
function untrusted(value: unknown, max = 120): string {
  return quoteForSummary(escapePageText(String(value ?? "")), max);
}

function describe(reply: BrowserAgentReply, ref: string): string {
  return typeof reply.description === "string" && reply.description
    ? untrusted(reply.description)
    : ref;
}

const OTHER_CHAT =
  "the browser now belongs to another chat the user switched to. Stop using it and tell the user.";

/** runs already told the browser moved on: a second call ends the run, since small models otherwise retry until their tool budget runs out. */
const toldOtherChat = new WeakSet<object>();

function leftChat(context: BrowserRunContext): void {
  const run = runOf(context);
  if (toldOtherChat.has(run)) context.stop?.();
  else toldOtherChat.add(run);
}

function onSameChat(context: BrowserRunContext): boolean {
  return isRunsChatOnScreen(context.threadId, context.viewKey);
}

/** last check before anything changes the page, since Stop or a chat switch may have come while the agent was reading. */
function guard(context: BrowserRunContext): void {
  if (context.signal?.aborted) throw new Error("Stopped");
  if (!onSameChat(context)) throw new Error(OTHER_CHAT);
}

async function blockedAddress(
  session: BrowserSession,
  url: string,
): Promise<string | null> {
  const reply = await session.agent("check_url", { url }).catch(() => null);
  return reply && !reply.ok ? untrusted(reply.error, 200) : null;
}

/** the tail shared by page-changing actions: where it went, a popup it tried, a blocked private address, and the page as it is now. */
async function actionResult(
  session: BrowserSession,
  nav: BrowserSnapshot | null,
  after: BrowserSnapshot | null,
  summary: string,
  snapshotChars: PageSize,
): Promise<Outcome> {
  const moved = Boolean(after?.url && after.url !== nav?.url);
  if (moved && after) {
    const blocked = await blockedAddress(session, after.url);
    if (blocked) {
      await session.goAction("back").catch(() => {});
      return { text: `Error: ${blocked}. The browser went back.` };
    }
  }
  return {
    text: joinResult(
      summary,
      moved && after
        ? `The page changed to ${untrusted(after.url, 300)}.`
        : null,
      popupNote(nav, after),
      await snapshotBlock(session, snapshotChars),
    ),
  };
}

async function execute(
  request: BrowserClientRequest,
  context: BrowserRunContext,
): Promise<Outcome> {
  const args = request.arguments;
  const { signal } = context;
  const snapshotChars = pageSize(SNAPSHOT_CHARS, request);
  if (!onSameChat(context)) return { text: `Error: ${OTHER_CHAT}` };
  if (request.tool === "browser_handoff") {
    const reason = quoteForSummary(String(args.reason ?? ""), 300);
    const session = await waitForBrowserSession(15_000, signal);
    if (!(await waitForUser(context, reason))) {
      guard(context);
      return {
        text: "Error: the user stopped before finishing in the browser.",
      };
    }
    // "I'm done" often comes while the page the user just submitted is still loading.
    await untilLoaded(
      session,
      await session.refresh().catch(() => null),
      signal,
    );
    return {
      text: joinResult(
        `The user is done${reason ? ` (${reason})` : ""}. Continue the task from the page as it is now.`,
        await snapshotBlock(session, snapshotChars),
      ),
    };
  }
  const store = useDesktopBrowserStore.getState();
  store.setActivity({
    threadId: context.threadId,
    label: "Opening the browser",
    stop: context.stop,
  });
  const session = await waitForBrowserSession(15_000, signal);
  const nav = await session.refresh().catch(() => session.snapshot);
  const before = await pageMark(session, nav);
  if (
    refArg(args.ref) !== "" &&
    refsDocument &&
    before.docId &&
    before.docId !== refsDocument
  ) {
    return {
      text: joinResult(
        "Error: the page has changed since that ref was given, so it no longer names the same element. Use a ref from this snapshot.",
        await snapshotBlock(session, pageSize(SNAPSHOT_CHARS, request)),
      ),
    };
  }
  const site = siteOf(nav?.url);
  const setLabel = (label: string) =>
    useDesktopBrowserStore
      .getState()
      .setActivity({ threadId: context.threadId, label, stop: context.stop });

  switch (request.tool) {
    case "browser_navigate": {
      const target = String(args.url ?? "").trim();
      if (!target) return { text: "Error: browser_navigate needs a url." };
      const history = target.toLowerCase();
      if (history === "back" || history === "forward" || history === "reload") {
        const action =
          history === "reload" ? "reload the page" : `go ${history}`;
        if (!(await permitted(context, "navigate", null, site, action)))
          return DECLINED;
        guard(context);
        setLabel(history === "reload" ? "Reloading" : `Going ${history}`);
        await session.goAction(history);
        // back and forward can stay in the document (a single-page app's history), a reload cannot.
        const after = await waitForLoad(
          session,
          before,
          signal,
          history === "reload" ? "navigation" : "action",
        );
        const summary =
          history === "reload" ? "Reloaded the page." : `Went ${history}.`;
        return actionResult(session, nav, after, summary, snapshotChars);
      }
      let url: string;
      try {
        url = resolveAddress(target);
      } catch (error) {
        return { text: `Error: ${errorText(error)}` };
      }
      const blocked = await blockedAddress(session, url);
      if (blocked) return { text: `Error: ${blocked}.` };
      const action = `open ${untrusted(url, 200)}`;
      if (!(await permitted(context, "navigate", null, siteOf(url), action)))
        return DECLINED;
      guard(context);
      setLabel(`Opening ${siteOf(url) ?? url}`);
      await session.go(url);
      const after = await waitForLoad(
        session,
        before,
        signal,
        nav?.url && sameDocument(nav.url, url) ? "action" : "navigation",
      );
      if (after?.error) return { text: `Error: ${untrusted(after.error)}.` };
      const title = after?.title ? ` (${untrusted(after.title, 80)})` : "";
      // measured from the address it was sent to: a redirect elsewhere is checked like any move.
      return actionResult(
        session,
        nav ? { ...nav, url } : nav,
        after,
        `Opened ${untrusted(after?.url || url, 300)}${title}.`,
        snapshotChars,
      );
    }
    case "browser_snapshot": {
      setLabel("Reading the page");
      await settle(session, signal);
      return {
        text: joinResult(
          `The page as it is now${site ? ` on ${site}` : ""}.`,
          await snapshotBlock(session, snapshotChars),
        ),
      };
    }
    case "browser_click": {
      const ref = refArg(args.ref);
      if (!ref)
        return { text: "Error: browser_click needs a ref from the snapshot." };
      const info = await session.agent("inspect", { ref });
      if (!info.ok) return failure(session, info, snapshotChars);
      const description = describe(info, ref);
      const sensitive = asSensitiveKind(info.sensitive);
      const action = `click ${description}`;
      if (!(await permitted(context, "click", sensitive, site, action)))
        return DECLINED;
      guard(context);
      setLabel(`Clicking ${quoteForSummary(description, 48)}`);
      const reply = await session.agent("click", { ref });
      if (!reply.ok) return failure(session, reply, snapshotChars);
      const after = await waitForLoad(session, before, signal, "action");
      return actionResult(
        session,
        nav,
        after,
        `Clicked ${description}.`,
        snapshotChars,
      );
    }
    case "browser_type": {
      const ref = refArg(args.ref);
      if (!ref)
        return { text: "Error: browser_type needs a ref from the snapshot." };
      const text = String(args.text ?? "");
      const submit = boolArg(args.submit, false);
      const clear = boolArg(args.clear, true);
      const info = await session.agent("inspect", { ref });
      if (!info.ok) return failure(session, info, snapshotChars);
      const description = describe(info, ref);
      const sensitive = asSensitiveKind(info.sensitive);
      // passwords and card numbers are the user's to type at any permission level; the field decides, not its form.
      if (info.secret === true) {
        return {
          text: `Error: ${description} is a ${sensitive === "password" ? "password" : "card"} field. Do not type it yourself; call browser_handoff so the user can enter it.`,
        };
      }
      // typing alone changes nothing a site can see; Enter acts for the form's submit button.
      const submits = submit
        ? await session
            .agent("activation", { ref, key: "Enter" })
            .catch(() => null)
        : null;
      const gate = submits?.ok ? asSensitiveKind(submits.sensitive) : null;
      const action = `type "${quoteForSummary(text)}" into ${description}${submit ? " and press Enter" : ""}`;
      if (!(await permitted(context, "type", gate, site, action)))
        return DECLINED;
      guard(context);
      setLabel(`Typing into ${quoteForSummary(description, 40)}`);
      const reply = await session.agent("type", { ref, text, submit, clear });
      if (!reply.ok) return failure(session, reply, snapshotChars);
      let after: BrowserSnapshot | null;
      if (submit) {
        after = await waitForLoad(session, before, signal, "action");
      } else {
        await settle(session, signal, 1500);
        after = session.snapshot;
      }
      return actionResult(
        session,
        nav,
        after,
        `Typed "${quoteForSummary(text)}" into ${description}${submit ? " and pressed Enter" : ""}.`,
        snapshotChars,
      );
    }
    case "browser_select": {
      const ref = refArg(args.ref);
      const option = String(args.option ?? "");
      if (!ref || !option) {
        return { text: "Error: browser_select needs a ref and an option." };
      }
      const info = await session.agent("inspect", { ref });
      if (!info.ok) return failure(session, info, snapshotChars);
      const description = describe(info, ref);
      const action = `choose "${quoteForSummary(option)}" in ${description}`;
      if (!(await permitted(context, "select", null, site, action)))
        return DECLINED;
      guard(context);
      setLabel(`Choosing ${quoteForSummary(option, 40)}`);
      let reply = await session.agent("select", { ref, option });
      // a custom dropdown renders its options after the click that opens it.
      if (!reply.ok && reply.code === "options_pending") {
        await settle(session, signal, 1500);
        guard(context);
        reply = await session.agent("select", { ref, option });
      }
      if (!reply.ok) return failure(session, reply, snapshotChars);
      const after = await waitForLoad(session, before, signal, "action");
      const selected =
        typeof reply.selected === "string" && reply.selected
          ? untrusted(reply.selected, 60)
          : quoteForSummary(option);
      return actionResult(
        session,
        nav,
        after,
        `Selected "${selected}" in ${description}.`,
        snapshotChars,
      );
    }
    case "browser_press_key": {
      const key = String(args.key ?? "").trim();
      if (!key) return { text: "Error: browser_press_key needs a key." };
      // the Enter and Space keys act on the focused element, so they are asked about like a click on it.
      const target = await session
        .agent("activation", { key })
        .catch(() => null);
      const sensitive = target?.ok ? asSensitiveKind(target.sensitive) : null;
      const focused =
        target?.ok &&
        typeof target.description === "string" &&
        target.description
          ? ` on ${untrusted(target.description)}`
          : "";
      const action = `press ${quoteForSummary(key, 24)}${focused}`;
      if (!(await permitted(context, "press", sensitive, site, action)))
        return DECLINED;
      guard(context);
      setLabel(`Pressing ${quoteForSummary(key, 24)}`);
      const reply = await session.agent("press", { key });
      if (!reply.ok) return failure(session, reply, snapshotChars);
      const after = await waitForLoad(session, before, signal, "action");
      return actionResult(
        session,
        nav,
        after,
        `Pressed ${quoteForSummary(key, 24)}.`,
        snapshotChars,
      );
    }
    case "browser_scroll": {
      const direction =
        String(args.direction ?? "down").toLowerCase() === "up" ? "up" : "down";
      const ref = args.ref == null || args.ref === "" ? null : refArg(args.ref);
      setLabel(`Scrolling ${direction}`);
      const reply = await session.agent(
        "scroll",
        ref ? { direction, ref } : { direction },
      );
      if (!reply.ok) return failure(session, reply, snapshotChars);
      await settle(session, signal, 1500);
      const edge =
        direction === "down" && reply.atBottom === true
          ? " It is at the bottom."
          : direction === "up" && reply.atTop === true
            ? " It is at the top."
            : "";
      return {
        text: joinResult(
          `Scrolled ${direction}.${edge}`,
          await snapshotBlock(session, snapshotChars),
        ),
      };
    }
    case "browser_read": {
      const offset = Math.max(0, Math.floor(numberOr(Number(args.offset), 0)));
      setLabel("Reading the page");
      await settle(session, signal);
      const reply = await sizedPageReply(
        (maxChars) => session.agent("read", { offset, maxChars }),
        pageSize(READ_CHARS, request),
      );
      if (!reply.ok) return failure(session, reply, snapshotChars);
      const total = numberOr(reply.total, 0);
      const next =
        typeof reply.nextOffset === "number" ? reply.nextOffset : null;
      const shown = Math.min(total, offset + String(reply.text ?? "").length);
      // the pane's own record of the page, not the runtime's, for the lines outside the block.
      const current = await session.refresh().catch(() => nav);
      const where = untrusted(
        current?.title || current?.url || site || "the page",
        80,
      );
      return {
        text: joinResult(
          `Read ${where} (characters ${offset}–${shown} of ${total}).`,
          `URL: ${untrusted(current?.url ?? "", 300)}`,
          pageBlock("text", String(reply.text ?? "")),
          next !== null
            ? `More text follows: call browser_read with offset ${next}.`
            : "That is the end of the page.",
        ),
      };
    }
    case "browser_find": {
      const query = String(args.text ?? args.query ?? "").trim();
      if (!query)
        return { text: "Error: browser_find needs text to look for." };
      setLabel(`Finding "${quoteForSummary(query, 32)}"`);
      const reply = await session.agent("find", { query, limit: 20 });
      if (!reply.ok) return failure(session, reply, snapshotChars);
      noteRefs(reply);
      const count = numberOr(reply.count, 0);
      if (count === 0) {
        return {
          text: `No matches for "${quoteForSummary(query)}" on this page.`,
        };
      }
      return {
        text: joinResult(
          `Found ${count} match${count === 1 ? "" : "es"} for "${quoteForSummary(query)}".`,
          pageBlock("find", String(reply.text ?? "")),
        ),
      };
    }
    case "browser_screenshot": {
      setLabel("Taking a screenshot");
      await settle(session, signal);
      const reply = await session.agent("screenshot", { marks: true });
      noteRefs(reply);
      if (!reply.ok || typeof reply.data !== "string")
        return failure(session, reply, snapshotChars);
      const image = await shrinkImage(
        reply.data,
        typeof reply.mime === "string" ? reply.mime : "image/png",
      );
      return {
        text: `Screenshot of ${untrusted(nav?.url ?? "the page", 300)}.\nThe labels on the page are element refs you can use with the other browser tools.`,
        images: [image],
      };
    }
    default:
      return { text: `Error: unknown browser tool ${request.tool}.` };
  }
}

/** re-encodes to a JPEG no larger than a vision model reads anyway (the backend caps at 1024). */
async function shrinkImage(
  data: string,
  mime: string,
): Promise<ClientToolImage> {
  try {
    const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: mime }));
    const scale = Math.min(
      1,
      SCREENSHOT_EDGE / Math.max(bitmap.width, bitmap.height),
    );
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    canvas
      .getContext("2d")
      ?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.82),
    );
    if (!blob) return { data, mimeType: mime };
    const buffer = new Uint8Array(await blob.arrayBuffer());
    let binary = "";
    for (let i = 0; i < buffer.length; i += 0x8000) {
      binary += String.fromCharCode(...buffer.subarray(i, i + 0x8000));
    }
    return { data: btoa(binary), mimeType: "image/jpeg" };
  } catch {
    return { data, mimeType: mime };
  }
}
