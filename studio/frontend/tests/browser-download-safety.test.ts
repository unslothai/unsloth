// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import { test } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const data = new Map<string, string>();
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => void data.set(key, value),
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, { localStorage, window: { localStorage, location: { protocol: "http:" }, addEventListener() {} } });

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const safety = await import("../src/features/browser/download-safety.ts");
const source = await import("../src/features/browser/download-source.ts");
const { handleNativeDownload } = await import("../src/features/browser/native-download-prompts.ts");

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/dangerous-download-names.json", import.meta.url), "utf8"),
) as { cases: { name: string; dangerous: boolean }[] };

test("every name in the shared fixture is judged as the desktop app judges it", () => {
  for (const { name, dangerous } of fixture.cases) assert.equal(safety.isDangerousDownload(name), dangerous, name);
  assert.equal(safety.isDangerousDownload("C:\\Users\\a\\setup.exe"), true);
  assert.equal(safety.isDangerousDownload("dir.exe/readme.txt"), false);
});

test("a download's source keeps where it came from, not what can carry a secret", () => {
  const cases: [string | null, string | null][] = [
    ["https://user:pass@example.com:8443/f/a.zip?sig=secret#part", "https://example.com:8443/f/a.zip"],
    ["http://example.com", "http://example.com/"],
    ["ftp://example.com/a.zip", null],
    ["blob:https://example.com/123", null],
    ["not a url", null],
    [null, null],
  ];
  for (const [url, expected] of cases) assert.equal(source.sanitizedSource(url), expected, String(url));
});

type Call = { command: string; args: Record<string, unknown> };
type Shown = { message: string; options?: Record<string, unknown>; level: string };

function harness(callResult: (command: string) => Promise<unknown> = async () => ({ name: "setup.exe", downloadId: "k0" })) {
  const calls: Call[] = [];
  const shown: Shown[] = [];
  const recorded: { name: string; url: string | null; nativeId?: string }[] = [];
  const notify = Object.assign(
    (message: string, options?: Record<string, unknown>) => shown.push({ message, options, level: "info" }),
    {
      success: (message: string, options?: Record<string, unknown>) => shown.push({ message, options, level: "success" }),
      error: (message: string, options?: Record<string, unknown>) => shown.push({ message, options, level: "error" }),
      warning: (message: string, options?: Record<string, unknown>) => shown.push({ message, options, level: "warning" }),
    },
  );
  const deps = {
    call: (<T,>(command: string, args: Record<string, unknown>) => {
      calls.push({ command, args });
      return callResult(command) as Promise<T>;
    }) as <T>(command: string, args: Record<string, unknown>) => Promise<T>,
    toast: notify,
    record: (item: { name: string; url: string | null; nativeId?: string }) => void recorded.push(item),
    t: (key: string, values?: Record<string, unknown>) => `${key}:${values?.name ?? ""}`,
  };
  return { calls, shown, recorded, deps };
}

const finished = { kind: "download" as const, tabId: "gone", url: "https://example.com/setup.exe", path: null, size: 10, done: true, success: true };

test("a staged download asks once; Keep keeps it under the name the app gives back", async () => {
  const h = harness(async () => ({ name: "setup (1).exe", downloadId: "k1" }));
  handleNativeDownload({ ...finished, name: "setup.exe", id: "s1", needsApproval: true, marked: true }, h.deps as never);
  const prompt = h.shown.at(-1);
  assert.match(String(prompt?.options?.id), /^browser-download-s1-\d+$/);
  assert.equal(prompt?.options?.duration, Number.POSITIVE_INFINITY);
  const action = prompt?.options?.action as { onClick: () => void };
  action.onClick();
  action.onClick();
  (prompt?.options?.onDismiss as () => void)();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(h.calls, [{ command: "browser_download_keep", args: { id: "s1" } }]);
  // Under the name it got, with its handle so Download history can show it in the folder.
  assert.deepEqual(h.recorded.map((item) => [item.name, item.nativeId]), [["setup (1).exe", "k1"]]);
  assert.equal(h.shown.at(-1)?.level, "success");
});

test("Discard, or closing the prompt, discards once and records nothing", () => {
  const h = harness();
  handleNativeDownload({ ...finished, name: "setup.exe", id: "s2", needsApproval: true, marked: true }, h.deps as never);
  const prompt = h.shown.at(-1);
  (prompt?.options?.cancel as { onClick: () => void }).onClick();
  (prompt?.options?.onDismiss as () => void)();
  assert.deepEqual(h.calls, [{ command: "browser_download_discard", args: { id: "s2" } }]);
  assert.equal(h.recorded.length, 0);
});

test("a refused Keep leaves the file staged with Discard and the reason", async () => {
  const h = harness(async (command) => {
    if (command === "browser_download_keep") throw "couldn't mark it";
  });
  handleNativeDownload({ ...finished, name: "setup.exe", id: "s3", needsApproval: true, marked: false }, h.deps as never);
  const first = h.shown.at(-1);
  (first?.options?.action as { onClick: () => void }).onClick();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const retry = h.shown.at(-1);
  // A new toast id: sonner removes the clicked toast's id, which would take this prompt with it.
  assert.notEqual(retry?.options?.id, first?.options?.id);
  assert.equal(retry?.message, "browser.downloadSafety.keepFailed:setup.exe");
  assert.equal(retry?.options?.description, "couldn't mark it");
  assert.equal(retry?.options?.action, undefined);
  // The clicked prompt closing late discards nothing.
  (first?.options?.onDismiss as () => void)();
  assert.deepEqual(h.calls.map((call) => call.command), ["browser_download_keep"]);
  (retry?.options?.cancel as { onClick: () => void }).onClick();
  assert.deepEqual(h.calls.map((call) => call.command), ["browser_download_keep", "browser_download_discard"]);
  assert.equal(h.recorded.length, 0);
});

test("a failed Discard asks again, so the staged file is not left behind", async () => {
  let failures = 1;
  const h = harness(async (command) => {
    if (command === "browser_download_discard" && failures-- > 0) throw new Error("in use");
  });
  handleNativeDownload({ ...finished, name: "setup.exe", id: "s5", needsApproval: true, marked: true }, h.deps as never);
  (h.shown.at(-1)?.options?.cancel as { onClick: () => void }).onClick();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const retry = h.shown.at(-1);
  assert.equal(retry?.message, "browser.downloadSafety.discardFailed:setup.exe");
  assert.equal(retry?.options?.description, "in use");
  assert.ok(retry?.options?.action);
  (retry?.options?.onDismiss as () => void)();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(h.calls.map((call) => call.command), ["browser_download_discard", "browser_download_discard"]);
  assert.equal(h.shown.length, 2);
});

test("an ordinary download from a closed tab is recorded; an unmarked one warns", () => {
  const h = harness();
  handleNativeDownload({ ...finished, name: "notes.pdf", marked: true, downloadId: "n1" }, h.deps as never);
  handleNativeDownload({ ...finished, name: "data.zip", marked: false }, h.deps as never);
  assert.deepEqual(h.recorded.map((item) => [item.name, item.nativeId]), [["notes.pdf", "n1"], ["data.zip", undefined]]);
  assert.deepEqual(h.shown.map((item) => item.level), ["success", "warning"]);
  assert.equal(h.calls.length, 0);
});

test("a staged download says it will ask; an ordinary one says it is downloading", () => {
  const h = harness();
  handleNativeDownload({ ...finished, name: "setup.exe", id: "s4", done: false, success: false }, h.deps as never);
  handleNativeDownload({ ...finished, name: "notes.pdf", done: false, success: false }, h.deps as never);
  assert.deepEqual(h.shown.map((item) => item.message), [
    "browser.downloadSafety.stagedDownloading:setup.exe",
    "browser.native.downloading:notes.pdf",
  ]);
});

test("native-view hands download events over before its tab guard", () => {
  const source = readFileSync(new URL("../src/features/browser/native-view.ts", import.meta.url), "utf8");
  const handler = source.slice(source.indexOf("function onNativeEvent"));
  assert.ok(handler.indexOf('event.kind === "download"') < handler.indexOf("store.tabs.find"));
});

type Downloads = { saveBrowserDownload: (download: { blob: Blob; name: string; contentType: string; url: string | null }) => Promise<void> };

function loadDownloads(answer: "save" | "cancel" | "dismiss", marked: boolean | null = null) {
  const saved: string[] = [];
  const sources: (string | null | undefined)[] = [];
  const recorded: string[] = [];
  const prompts: string[] = [];
  const warnings: string[] = [];
  const toast = Object.assign(
    (message: string, options: { action: { onClick: () => void }; cancel: { onClick: () => void }; onDismiss: () => void }) => {
      prompts.push(message);
      if (answer === "save") options.action.onClick();
      else if (answer === "cancel") options.cancel.onClick();
      else options.onDismiss();
    },
    { error: () => undefined, warning: (message: string) => void warnings.push(message) },
  );
  const module = loadWithStubs<Downloads>(new URL("../src/features/browser/downloads.ts", import.meta.url), {
    "@/i18n": { getLocale: () => "en", translate: (key: string) => key },
    // The desktop app: saves go through its dialog, which marks them.
    "@/lib/api-base": { isTauri: true },
    "@/lib/native-files": { DownloadCancelledError: Error, downloadFile: async () => undefined, isDownloadCancelled: () => false },
    "@/lib/toast": { toast },
    "./address": { fileNameFromUrl: () => "x", withBaseUrl: (html: string) => html },
    "./api": { fetchBrowserPage: async () => undefined },
    "./download-safety": safety,
    "./native-downloads": {
      saveNativeDownload: async (_: Blob, name: string, source?: string | null) => {
        saved.push(name);
        sources.push(source);
        return { id: "n1", name, marked };
      },
    },
    "./prefs-store": { useBrowserPrefsStore: { getState: () => ({ askWhereToSave: false }) } },
    "./history-store": { useBrowserHistoryStore: { getState: () => ({ recordDownload: (item: { name: string }) => void recorded.push(item.name) }) } },
  });
  return { module, saved, sources, recorded, prompts, warnings };
}

test("a panel save of a file that runs code waits for Save anyway", async () => {
  const blob = new Blob(["x"]);
  const saving = loadDownloads("save");
  await saving.module.saveBrowserDownload({ blob, name: "setup.exe", contentType: "", url: null });
  assert.deepEqual([saving.prompts.length, saving.saved, saving.recorded], [1, ["setup.exe"], ["setup.exe"]]);
  for (const answer of ["cancel", "dismiss"] as const) {
    const refused = loadDownloads(answer);
    await refused.module.saveBrowserDownload({ blob, name: "setup.exe", contentType: "", url: null });
    assert.deepEqual([refused.saved, refused.recorded], [[], []], answer);
  }
  const plain = loadDownloads("cancel");
  await plain.module.saveBrowserDownload({ blob, name: "notes.pdf", contentType: "", url: null });
  assert.deepEqual([plain.prompts.length, plain.saved], [0, ["notes.pdf"]]);
});

test("a panel save the system couldn't mark warns; a marked one does not", async () => {
  const blob = new Blob(["x"]);
  const unmarked = loadDownloads("save", false);
  await unmarked.module.saveBrowserDownload({ blob, name: "notes.pdf", contentType: "", url: "https://example.com/notes.pdf" });
  assert.deepEqual([unmarked.recorded, unmarked.warnings], [["notes.pdf"], ["browser.downloadSafety.notMarked"]]);
  // The page it came from goes to the app, which marks the file with it.
  assert.deepEqual(unmarked.sources, ["https://example.com/notes.pdf"]);
  for (const marked of [true, null]) {
    const fine = loadDownloads("save", marked);
    await fine.module.saveBrowserDownload({ blob, name: "notes.pdf", contentType: "", url: null });
    assert.deepEqual([fine.recorded, fine.warnings], [["notes.pdf"], []], String(marked));
  }
});

test("clearing browsing data never drops downloads; only an account switch does", () => {
  const clear = readFileSync(new URL("../src/lib/native-browser-clear.ts", import.meta.url), "utf8");
  const clearing = clear.slice(clear.indexOf("export async function clearNativeBrowsingData"), clear.indexOf("export async function forgetNativeAccountDownloads"));
  assert.doesNotMatch(clearing, /browser_downloads_account_switched/);
  const dialog = readFileSync(new URL("../src/features/browser/clear-data-dialog.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(dialog, /forgetNativeAccountDownloads/);
});

test("Keep and Discard leave no id behind once done", async () => {
  const h = harness(async () => "setup.exe");
  for (const id of ["d1", "d2"]) handleNativeDownload({ ...finished, name: "setup.exe", id, needsApproval: true, marked: true }, h.deps as never);
  (h.shown[0]?.options?.action as { onClick: () => void }).onClick();
  (h.shown[1]?.options?.cancel as { onClick: () => void }).onClick();
  await new Promise((resolve) => setTimeout(resolve, 0));
  // Neither id is held once done, so the set never grows with the session.
  handleNativeDownload({ ...finished, name: "setup.exe", id: "d1", needsApproval: true, marked: true }, h.deps as never);
  (h.shown.at(-1)?.options?.cancel as { onClick: () => void }).onClick();
  assert.deepEqual(h.calls.map((call) => call.command), ["browser_download_keep", "browser_download_discard", "browser_download_discard"]);
});

test("download history keeps sanitized sources, and older history is sanitized once", async () => {
  data.set(
    "unsloth_browser_history",
    JSON.stringify({
      version: 1,
      state: {
        history: [{ id: "h", url: "https://example.com/?q=kept", title: "t", visitedAt: 1 }],
        downloads: [{ id: "d", name: "a.zip", url: "https://u:p@example.com/a.zip?sig=secret", size: 1, contentType: "", downloadedAt: 1 }],
        icons: {},
      },
    }),
  );
  const { useBrowserHistoryStore } = await import("../src/features/browser/history-store.ts");
  await useBrowserHistoryStore.persist.rehydrate();
  const state = useBrowserHistoryStore.getState();
  assert.equal(state.downloads[0]?.url, "https://example.com/a.zip");
  assert.equal(state.history[0]?.url, "https://example.com/?q=kept");
  state.recordDownload({ name: "b.zip", url: "https://example.com/b.zip?token=1#x", size: 1, contentType: "" });
  assert.equal(useBrowserHistoryStore.getState().downloads[0]?.url, "https://example.com/b.zip");
});
