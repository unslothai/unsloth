// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Logs: "Download all logs (.zip)" and "Open logs folder".
//
// Both branches are driven in one process, which a plain import cannot do:
// `isTauri` is decided once when lib/api-base evaluates. loadWithStubs re-runs
// the real source per test with the api-base the test wants, and is also what
// keeps "@tauri-apps/api/core" -- a package that resolves only inside a Tauri
// webview -- out of the runner.
//
// The tab is rendered with react-dom/server. Its effects (the poll loop) never
// run there, and a setState after the render is a no-op on the server, so a
// handler can be called straight off the recorded button props: what is asserted
// is which request went out and which toast came back, not the disabled repaint.

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";

import type * as DebugLogsApi from "../src/features/settings/api/debug-logs.ts";
import * as debugLogBuffer from "../src/features/settings/lib/debug-log-buffer.ts";
import * as debugLogError from "../src/features/settings/lib/debug-log-error.ts";
import type * as DebuggingTabModule from "../src/features/settings/tabs/debugging-tab.tsx";
import { en } from "../src/i18n/locales/en.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const API_URL = new URL(
  "../src/features/settings/api/debug-logs.ts",
  import.meta.url,
);
const TAB_URL = new URL(
  "../src/features/settings/tabs/debugging-tab.tsx",
  import.meta.url,
);
const API_BASE = "http://127.0.0.1:7860";
const EXPORT_PATH = "/api/settings/debug/logs/export";
const SAVED_PATH = "/home/tester/Downloads/unsloth-logs-20260910-101112.zip";
const ARCHIVE_NAME = /^unsloth-logs-\d{8}-\d{6}\.zip$/;

/** The shipped English message, so a renamed key fails here rather than silently. */
function t(key: string, values?: Record<string, string>): string {
  const message = key
    .split(".")
    .reduce<unknown>(
      (node, part) => (node as Record<string, unknown> | undefined)?.[part],
      en as unknown,
    );
  assert.equal(typeof message, "string", `no English message for "${key}"`);
  return String(message).replace(/\{(\w+)\}/g, (match, name: string) =>
    values && name in values ? values[name] : match,
  );
}

type InvokeCall = { command: string; args: unknown };
type ToastCall = {
  kind: string;
  title: string;
  options: Record<string, unknown> | undefined;
};
type ButtonProps = Record<string, unknown>;

/** Let every already-resolved promise in the handler chain settle. */
async function flush(): Promise<void> {
  for (let index = 0; index < 5; index += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

function makeWorld(options: {
  isTauri: boolean;
  invoke?: (command: string, args: unknown) => Promise<unknown>;
  respond?: () => Response;
}) {
  const invokes: InvokeCall[] = [];
  const requests: string[] = [];
  const downloads: { filename: string; bytes: number }[] = [];
  const toasts: ToastCall[] = [];
  const buttons: ButtonProps[] = [];
  let tauriImports = 0;

  const tauriCore = {
    invoke: async (command: string, args: unknown) => {
      invokes.push({ command, args });
      return options.invoke ? await options.invoke(command, args) : undefined;
    },
  };

  const apiStubs: Record<string, unknown> = {
    "@/features/auth": {
      authFetch: async (input: string) => {
        requests.push(input);
        return options.respond?.() ?? new Response(new Blob(["PK"]));
      },
    },
    "@/lib/api-base": {
      isTauri: options.isTauri,
      apiUrl: (path: string) => `${API_BASE}${path}`,
    },
    "@/lib/format-fastapi-error": formatFastApiError,
    "@/lib/native-files": {
      browserDownload: async (blob: Blob, filename: string) => {
        downloads.push({ filename, bytes: blob.size });
      },
    },
    "../lib/debug-log-error": debugLogError,
  };
  // A getter, so "did the web build reach for Tauri at all?" is answerable:
  // loadWithStubs reads the entry only when the module actually requires it.
  Object.defineProperty(apiStubs, "@tauri-apps/api/core", {
    enumerable: true,
    get: () => {
      tauriImports += 1;
      return tauriCore;
    },
  });

  const api = loadWithStubs<typeof DebugLogsApi>(API_URL, apiStubs);

  const toast = {
    success: (title: string, opts?: Record<string, unknown>) =>
      toasts.push({ kind: "success", title, options: opts }),
    error: (title: string, opts?: Record<string, unknown>) =>
      toasts.push({ kind: "error", title, options: opts }),
  };

  const { DebuggingTab } = loadWithStubs<typeof DebuggingTabModule>(TAB_URL, {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@/components/ui/button": {
      Button: (props: ButtonProps) => {
        buttons.push(props);
        return React.createElement(
          "button",
          {
            type: "button",
            disabled: props.disabled,
            "aria-busy": props["aria-busy"],
            "data-testid": props["data-testid"],
          },
          props.children as React.ReactNode,
        );
      },
    },
    "@/features/hub/hooks/use-copy-feedback": {
      useCopyFeedback: () => ({ copied: false, copy: async () => {} }),
    },
    "@/i18n": { useT: () => t },
    "@/lib/api-base": apiStubs["@/lib/api-base"],
    "@/lib/strip-ansi": { stripAnsi: (value: string) => value },
    "@/lib/toast": { toast },
    "@hugeicons/core-free-icons": { Tick02Icon: {} },
    "@hugeicons/react": { HugeiconsIcon: () => null },
    "../api/debug-logs": api,
    "../components/settings-row": {
      SettingsRow: (props: { label: string; children?: React.ReactNode }) =>
        React.createElement("div", null, props.label, props.children),
    },
    "../components/settings-section": {
      SettingsSection: (props: { title: string; children?: React.ReactNode }) =>
        React.createElement("section", null, props.title, props.children),
    },
    "../lib/debug-log-buffer": debugLogBuffer,
    "../lib/debug-log-error": debugLogError,
  });

  function render(): string {
    buttons.length = 0;
    return renderToStaticMarkup(React.createElement(DebuggingTab));
  }

  function button(testId: string): ButtonProps {
    const found = buttons.find((props) => props["data-testid"] === testId);
    assert.ok(found, `no button rendered with data-testid="${testId}"`);
    return found;
  }

  return {
    api,
    buttons,
    button,
    downloads,
    invokes,
    render,
    requests,
    toasts,
    tauriImports: () => tauriImports,
  };
}

function jsonResponse(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

test("the desktop export streams through the Tauri command", async () => {
  const world = makeWorld({
    isTauri: true,
    invoke: async () => SAVED_PATH,
  });

  assert.equal(await world.api.exportAllLogs(), SAVED_PATH);
  assert.equal(world.invokes.length, 1);
  assert.equal(world.invokes[0].command, "download_logs_to_downloads");
  const args = world.invokes[0].args as { url: string; filename: string };
  assert.equal(args.url, `${API_BASE}${EXPORT_PATH}`);
  assert.match(args.filename, ARCHIVE_NAME);
  // The whole point of the desktop branch: the response never crosses into JS.
  assert.deepEqual(world.requests, []);
});

test("the browser export fetches the route and never reaches for Tauri", async () => {
  const world = makeWorld({ isTauri: false });

  // Null, not a path: only the browser knows where its downloads land.
  assert.equal(await world.api.exportAllLogs(), null);
  assert.deepEqual(world.requests, [EXPORT_PATH]);
  assert.equal(world.downloads.length, 1);
  assert.match(world.downloads[0].filename, ARCHIVE_NAME);
  assert.equal(world.invokes.length, 0);
  assert.equal(world.tauriImports(), 0);
});

test("openLogsFolder invokes open_logs_dir with no arguments", async () => {
  const world = makeWorld({ isTauri: true });

  await world.api.openLogsFolder();
  assert.deepEqual(world.invokes, [
    { command: "open_logs_dir", args: undefined },
  ]);
});

test("a missing export route is reported as an outdated backend", async () => {
  const world = makeWorld({
    isTauri: false,
    respond: () => jsonResponse(404, "Not Found"),
  });

  const error = await world.api.exportAllLogs().then(
    () => null,
    (reason: unknown) => reason,
  );
  assert.ok(error instanceof world.api.LogExportError);
  assert.equal(error.failure, "outdated");
});

test("a rejected caller is told the session is the problem, not the backend", async () => {
  const world = makeWorld({
    isTauri: false,
    respond: () => jsonResponse(403, "Forbidden"),
  });

  const error = await world.api.exportAllLogs().then(
    () => null,
    (reason: unknown) => reason,
  );
  assert.ok(error instanceof world.api.LogExportError);
  assert.equal(error.failure, "forbidden");
});

test("the desktop stream failure carries the HTTP status back out of Rust", async () => {
  const world = makeWorld({
    isTauri: true,
    // What `stream_url_to_path` rejects with; Rust hands back a string, not a status.
    invoke: async () => {
      throw "Download failed with status 404.";
    },
  });

  const error = await world.api.exportAllLogs().then(
    () => null,
    (reason: unknown) => reason,
  );
  assert.ok(error instanceof world.api.LogExportError);
  assert.equal(error.failure, "outdated");
});

test("the download button is offered everywhere, the folder button only on desktop", () => {
  const web = makeWorld({ isTauri: false });
  const webMarkup = web.render();
  assert.ok(webMarkup.includes(t("settings.debugging.downloadAllLogs")));
  assert.ok(!webMarkup.includes(t("settings.debugging.openLogsFolder")));
  assert.equal(
    web.buttons.filter(
      (props) => props["data-testid"] === "debug-log-open-folder",
    ).length,
    0,
  );
  // The masking of credentials in the archive is stated where it is downloaded.
  assert.ok(webMarkup.includes(t("settings.debugging.exportMaskedNote")));

  const desktop = makeWorld({ isTauri: true });
  const desktopMarkup = desktop.render();
  assert.ok(desktopMarkup.includes(t("settings.debugging.openLogsFolder")));
  assert.equal(desktop.button("debug-log-open-folder")["aria-busy"], false);
  assert.equal(desktop.button("debug-log-download-all")["aria-busy"], false);
});

test("a 404 from the tab surfaces the backend-too-old message", async () => {
  const world = makeWorld({
    isTauri: false,
    respond: () => jsonResponse(404, "Not Found"),
  });
  world.render();

  (world.button("debug-log-download-all").onClick as () => void)();
  await flush();

  assert.deepEqual(
    world.toasts.map((call) => [call.kind, call.title]),
    [["error", t("settings.debugging.exportTooOld")]],
  );
});

test("a desktop export toasts the saved path and reveals that folder", async () => {
  const world = makeWorld({
    isTauri: true,
    invoke: async (command) =>
      command === "download_logs_to_downloads" ? SAVED_PATH : undefined,
  });
  world.render();

  (world.button("debug-log-download-all").onClick as () => void)();
  await flush();

  assert.equal(world.toasts.length, 1);
  assert.equal(world.toasts[0].kind, "success");
  assert.ok(
    world.toasts[0].title.includes(SAVED_PATH),
    `toast "${world.toasts[0].title}" does not name the saved file`,
  );

  // The reveal action must open the folder the toast just named -- Downloads --
  // not ~/.unsloth/studio, where the logs came from.
  const action = world.toasts[0].options?.action as {
    label: string;
    onClick: () => void;
  };
  assert.equal(action.label, t("settings.debugging.showInFolder"));
  action.onClick();
  await flush();
  assert.deepEqual(
    world.invokes.map((call) => call.command),
    ["download_logs_to_downloads", "open_models_dir"],
  );
  const revealed = world.invokes[1].args as { path: string };
  assert.ok(
    SAVED_PATH.startsWith(`${revealed.path}/`),
    `revealed "${revealed.path}" does not contain the saved file`,
  );
  assert.notEqual(revealed.path, SAVED_PATH);
});

test("the open-logs-folder button still reaches the command", async () => {
  const world = makeWorld({ isTauri: true, invoke: async () => undefined });
  world.render();

  (world.button("debug-log-open-folder").onClick as () => void)();
  await flush();

  assert.deepEqual(
    world.invokes.map((call) => call.command),
    ["open_logs_dir"],
  );
});

test("a browser export reports the download without naming a folder", async () => {
  const world = makeWorld({ isTauri: false });
  world.render();

  (world.button("debug-log-download-all").onClick as () => void)();
  await flush();

  assert.deepEqual(
    world.toasts.map((call) => [call.kind, call.title]),
    [["success", t("settings.debugging.downloadedToBrowser")]],
  );
});
