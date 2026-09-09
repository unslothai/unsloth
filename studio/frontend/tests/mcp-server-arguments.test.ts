// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  createMcpStdioSnapshot,
  resolveMcpStdioUrl,
} from "../src/features/chat/mcp-server-form.ts";
import {
  readAfterPendingMcpServerMutations,
  readMcpServerMutationSnapshot,
  subscribeToMcpServerMutationSettlements,
  trackMcpServerMutation,
  waitForPendingMcpServerMutations,
} from "../src/features/chat/api/mcp-server-mutation-tracker.ts";

import { readSrc } from "./helpers/kit.ts";

const MCP_SERVERS_API = readSrc("features/chat/api/mcp-servers-api.ts");
const CHAT_MCP_SERVERS_DIALOG = readSrc(
  "features/chat/chat-mcp-servers-dialog.tsx",
);
const MCP_COMPOSER_BUTTON = readSrc("features/chat/mcp-composer-button.tsx");

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function sourceBetween(source: string, start: string, end: string): string {
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex + start.length);
  assert.notEqual(startIndex, -1, `missing source marker: ${start}`);
  assert.notEqual(endIndex, -1, `missing source marker: ${end}`);
  return source.slice(startIndex, endIndex);
}

function typescriptFilesUnder(directory: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...typescriptFilesUnder(path));
    } else if (/\.[cm]?tsx?$/.test(entry.name)) {
      files.push(path);
    }
  }
  return files;
}

test("an unchanged stdio form reuses the exact original URL", () => {
  const originalUrl = `python  -m mod --name "a b" ''`;
  const snapshot = createMcpStdioSnapshot(originalUrl, "python", [
    "-m",
    "mod",
    "--name",
    "a b",
    "",
  ]);

  assert.deepEqual(
    resolveMcpStdioUrl("python", ["-m", "mod", "--name", "a b", ""], snapshot),
    { kind: "reuse", url: originalUrl },
  );
});

test("missing legacy arguments default to an empty ordered list", () => {
  const snapshot = createMcpStdioSnapshot("python", "python");
  assert.deepEqual(snapshot.arguments, []);
  assert.deepEqual(resolveMcpStdioUrl("python", [], snapshot), {
    kind: "reuse",
    url: "python",
  });
});

test("command, order, value, and intentional empty argument changes require encoding", () => {
  const snapshot = createMcpStdioSnapshot("python -m mod", "python", [
    "-m",
    "mod",
  ]);

  for (const [command, arguments_] of [
    ["python3", ["-m", "mod"]],
    ["python", ["mod", "-m"]],
    ["python", ["-m", "other"]],
    ["python", ["-m", "mod", ""]],
  ] as const) {
    assert.deepEqual(resolveMcpStdioUrl(command, arguments_, snapshot), {
      kind: "encode",
      command,
      arguments: [...arguments_],
    });
  }
});

test("the helper never parses, splits, joins, trims, or quotes commands", () => {
  const helper = readSrc("features/chat/mcp-server-form.ts");
  assert.doesNotMatch(helper, /\.(?:split|join|trim)\s*\(/);
  assert.doesNotMatch(helper, /JSON\.stringify|replace\s*\(/);
});

test("the dialog wires backend codec calls, stale guards, and a stdio-only editor", () => {
  assert.match(MCP_SERVERS_API, /mcpRequest\("\/stdio\/decode"/);
  assert.match(MCP_SERVERS_API, /mcpRequest\("\/stdio\/encode"/);
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /await decodeMcpStdioCommand\(server\.url\)/,
  );
  assert.match(CHAT_MCP_SERVERS_DIALOG, /await encodeMcpStdioCommand\(\{/);
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /formGenerationRef\.current !== generation/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /activeEditIdRef\.current !== server\.id/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /function handleOpenChange[\s\S]*formGenerationRef\.current \+= 1;[\s\S]*onOpenChange\(next\)/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /addressIsCommand && \(\s*<ArgumentsEditor/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /const addressIsCommand = form\.transport === "stdio"/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /function formWithAddress[\s\S]*transportFromAddress\([\s\S]*preservePartialHttp \? form\.credentialTransport : null[\s\S]*headers: transportChanged \? \[\] : form\.headers/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /queueMicrotask\(\(\) => \{\s*if \(cancelled\) return;[\s\S]*setView\(\{ kind: "list" \}\)/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /function ArgumentsEditor[\s\S]*\{ id: newRowId\(\), value: "" \}/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /form\.transport === "http" && \([\s\S]*Use OAuth sign-in/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /const decision = resolveMcpStdioUrl\(/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /decision\.kind === "reuse"[\s\S]*url = view\.kind === "edit" \? undefined : decision\.url/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /const url = stdio\s*\? await encodeStdioForGeneration\([\s\S]*testMcpServer\(\{\s*url,/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /return rows\.map\(\(row\) => row\.value\)/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /function ArgumentsEditor[\s\S]*data-reload-snapshot-sensitive/,
  );
  assert.match(
    MCP_SERVERS_API,
    /export function testMcpServer[\s\S]*body: \{\s*url:/,
  );
  assert.doesNotMatch(
    CHAT_MCP_SERVERS_DIALOG,
    /npx -y @modelcontextprotocol\/server-filesystem \/tmp/,
  );
  assert.match(CHAT_MCP_SERVERS_DIALOG, /URL or executable/);
  assert.match(CHAT_MCP_SERVERS_DIALOG, /https:\/\/example\.com\/mcp or npx/);
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /Add local arguments in the Arguments rows/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /setForm\(\(prev\) => formWithAddress\(prev, url, true\)\)/,
  );
  assert.doesNotMatch(
    CHAT_MCP_SERVERS_DIALOG,
    /form\.url\.(?:split|join)\s*\(/,
  );
  assert.doesNotMatch(
    CHAT_MCP_SERVERS_DIALOG,
    /form\.arguments[^;\n]*\.join\s*\(/,
  );
});

test("every mutable MCP form editor is locked for the full pending interval", () => {
  const argumentsEditor = sourceBetween(
    CHAT_MCP_SERVERS_DIALOG,
    "function ArgumentsEditor",
    "function HeadersEditor",
  );
  const headersEditor = sourceBetween(
    CHAT_MCP_SERVERS_DIALOG,
    "function HeadersEditor",
    "export interface ChatMcpServersDialogProps",
  );

  assert.equal(
    argumentsEditor.match(/disabled=\{disabled\}/g)?.length,
    3,
    "argument add, input, and remove must all be locked",
  );
  assert.equal(
    headersEditor.match(/disabled=\{disabled\}/g)?.length,
    4,
    "header/env add, key, value, and remove must all be locked",
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /const formPending = importing \|\| codecPending \|\| testing \|\| saving/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /id="mcp-display-name"[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /id="mcp-url"[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /<ArgumentsEditor[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /id="mcp-oauth"[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /<HeadersEditor[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
});

test("a decode error is announced and executable edits unlock manual recovery", () => {
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /id="mcp-url"[\s\S]*?onChange=\{\(e\) => \{[\s\S]*?setCodecError\(null\)[\s\S]*?formWithAddress/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /role="alert"\s*aria-live="assertive"[\s\S]*?\{codecError\}/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /aria-busy=\{decodingCommand\}[\s\S]*role="status"\s*aria-live="polite"[\s\S]*Reading local command…/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /codecError && \([\s\S]*view\.kind === "edit"[\s\S]*void startEdit\(server\)[\s\S]*Retry/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /disabled=\{\s*formPending \|\|\s*codecError !== null \|\|\s*form\.transport === "unknown" \|\|\s*!form\.url\.trim\(\)/,
  );
});

test("dialog actions and reconciliation stop when the dialog closes", () => {
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /useEffect\(\(\) => \{\s*if \(!open\) \{[\s\S]*subscribeToMcpServerMutationSettlements/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /actionGenerationRef\.current !== generation \|\| !openRef\.current/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /open=\{open && confirmingDelete !== null\}/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /<Button size="sm" onClick=\{startCreate\} disabled=\{importing \|\| blenderBusy\}>/,
  );
});

test("pending MCP actions remain keyed to their own server", () => {
  assert.match(
    MCP_COMPOSER_BUTTON,
    /pendingUrlsRef = useRef\(new Set<string>\(\)\)/,
  );
  assert.match(
    MCP_COMPOSER_BUTTON,
    /pendingUrlsRef\.current\.delete\(norm\);\s*setPendingUrls\(new Set\(pendingUrlsRef\.current\)\)/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /refreshingIdsRef = useRef\(new Set<string>\(\)\)/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /togglingIdsRef = useRef\(new Set<string>\(\)\)/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /if \(!togglingIdsRef\.current\.has\(row\.id\)\) return row;[\s\S]*is_enabled: optimistic\.is_enabled/,
  );
});

test("dialog closure is blocked while a CRUD mutation is in flight", () => {
  const closeHandler = sourceBetween(
    CHAT_MCP_SERVERS_DIALOG,
    "function handleOpenChange",
    "async function encodeStdioForGeneration",
  );

  assert.match(
    closeHandler,
    /if \(!next && \(blenderBusy \|\| \(saving && !codecPending\) \|\| busyIdsRef\.current\.size > 0\)\)[\s\S]*return;[\s\S]*if \(!next\) \{[\s\S]*formGenerationRef\.current \+= 1;/,
    "the in-flight mutation guard must run before close invalidates the form generation",
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /<DialogContent[\s\S]*?showCloseButton=\{!blenderBusy && !\(saving && !codecPending\) && busyIds\.size === 0\}[\s\S]*?>/,
    "the built-in close control must disappear during the same mutation window",
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /onClick=\{cancelForm\}[\s\S]*?disabled=\{saving && !codecPending\}/,
    "the visible Cancel action must use the same mutation boundary",
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /async function encodeStdioForGeneration[\s\S]*setCodecPending\(true\);[\s\S]*await encodeMcpStdioCommand/,
    "codec encoding remains distinguishable so it can still be cancelled before CRUD starts",
  );
});

test("composer applies mutation responses before releasing each preset", () => {
  assert.match(
    MCP_COMPOSER_BUTTON,
    /const applyServer = useCallback[\s\S]*setServers\(\(current\)[\s\S]*candidate\.id === server\.id/,
  );
  assert.match(
    MCP_COMPOSER_BUTTON,
    /applyServer\(\s*await createMcpServer\([\s\S]*pendingUrlsRef\.current\.delete\(norm\)/,
  );
  assert.match(
    MCP_COMPOSER_BUTTON,
    /applyServer\(\s*await updateMcpServer\([\s\S]*pendingUrlsRef\.current\.delete\(norm\)/,
  );
  assert.match(
    MCP_COMPOSER_BUTTON,
    /const \[serversLoaded, setServersLoaded\] = useState\(false\)[\s\S]*const hasLoadedServerSnapshotRef = useRef\(false\);[\s\S]*setServersLoaded\(false\);[\s\S]*setServers\(rows\);\s*hasLoadedServerSnapshotRef\.current = true;\s*setServersLoaded\(true\)/,
  );
  assert.match(
    MCP_COMPOSER_BUTTON,
    /catch \{\s*if \(\s*listRefreshGenerationRef\.current === generation &&\s*hasLoadedServerSnapshotRef\.current\s*\) \{\s*setServersLoaded\(true\);/,
  );
  assert.match(
    MCP_COMPOSER_BUTTON,
    /disabled=\{!serversLoaded \|\| pendingUrls\.has\(normalizeMcpUrl\(opts\.url\)\)\}/,
  );
});

test("MCP configuration remains reachable when the loaded model lacks tools", () => {
  assert.doesNotMatch(MCP_COMPOSER_BUTTON, /aria-disabled=\{true\}/);
  assert.match(MCP_COMPOSER_BUTTON, /The loaded model cannot use MCP tools/);
  assert.doesNotMatch(MCP_COMPOSER_BUTTON, /disabled=\{[^}]*!usable/);
  assert.match(
    MCP_COMPOSER_BUTTON,
    /<DropdownMenuItem\s+onSelect=\{\(\) => \{\s*setMenuOpen\(false\);\s*setDialogOpen\(true\);/,
  );
});

test("full unmount invalidates cancellable stdio encode continuations", async () => {
  const refsAndCleanup = sourceBetween(
    CHAT_MCP_SERVERS_DIALOG,
    "const formGenerationRef = useRef(0)",
    "const refresh = useCallback",
  );
  const openLifecycle = sourceBetween(
    CHAT_MCP_SERVERS_DIALOG,
    "useEffect(() => {\n    formGenerationRef.current += 1;",
    "function startCreate",
  );
  const encode = sourceBetween(
    CHAT_MCP_SERVERS_DIALOG,
    "async function encodeStdioForGeneration",
    "async function testConnection",
  );
  const testContinuation = sourceBetween(
    CHAT_MCP_SERVERS_DIALOG,
    "async function testConnection",
    "async function submitForm",
  );
  const crudContinuation = sourceBetween(
    CHAT_MCP_SERVERS_DIALOG,
    "async function submitForm",
    "async function onImportFile",
  );

  assert.match(
    refsAndCleanup,
    /useEffect\(\(\) => \{\s*return \(\) => \{\s*formGenerationRef\.current \+= 1;\s*actionGenerationRef\.current \+= 1;\s*activeEditIdRef\.current = null;[\s\S]*\};\s*\}, \[\]\)/,
    "the component mount lifetime must invalidate the form identity on teardown",
  );
  assert.match(
    openLifecycle,
    /queueMicrotask\(\(\) => \{\s*if \(cancelled\) return;[\s\S]*setCodecPending\(false\);\s*setDecodingCommand\(false\);[\s\S]*setConfirmingDelete\(null\);[\s\S]*if \(!open\) return;/,
    "route teardown must clear transient form state before a later reopen",
  );
  assert.match(
    encode,
    /await encodeMcpStdioCommand\([\s\S]*formGenerationRef\.current !== generation\) return null/,
  );
  assert.match(
    testContinuation,
    /await encodeStdioForGeneration\([\s\S]*if \(url === null \|\| formGenerationRef\.current !== generation\) return;[\s\S]*testMcpServer\(/,
  );
  assert.match(
    crudContinuation,
    /await encodeStdioForGeneration\([\s\S]*if \(encodedUrl === null\) return;[\s\S]*if \(formGenerationRef\.current !== generation\) return;[\s\S]*(?:updateMcpServer|createMcpServer)\(/,
  );
  assert.match(
    crudContinuation,
    /await updateMcpServer\([\s\S]*if \(formGenerationRef\.current !== generation\) return;\s*toast\.success\("MCP server updated"\)/,
    "an unmounted edit must not emit a stale success toast",
  );
  assert.match(
    crudContinuation,
    /await createMcpServer\([\s\S]*if \(formGenerationRef\.current !== generation\) return;\s*toast\.success\("MCP server added"\)/,
    "an unmounted create must not emit a stale success toast",
  );

  const generation = { current: 7 };
  const activeEditId = { current: "server-1" as string | null };
  const encoded = deferred<string>();
  let testCalls = 0;
  let createCalls = 0;
  let updateCalls = 0;
  let staleUiEffects = 0;

  async function continueAfterEncode(kind: "test" | "create" | "update") {
    const capturedGeneration = generation.current;
    await encoded.promise;
    if (generation.current !== capturedGeneration) return;
    if (kind === "test") testCalls += 1;
    if (kind === "create") createCalls += 1;
    if (kind === "update") updateCalls += 1;
    staleUiEffects += 1;
  }

  const continuations = [
    continueAfterEncode("test"),
    continueAfterEncode("create"),
    continueAfterEncode("update"),
  ];
  generation.current += 1;
  activeEditId.current = null;
  encoded.resolve("encoded command");
  await Promise.all(continuations);

  assert.equal(activeEditId.current, null);
  assert.equal(testCalls, 0);
  assert.equal(createCalls, 0);
  assert.equal(updateCalls, 0);
  assert.equal(staleUiEffects, 0);
});

test("open-time reconciliation waits for every mutation across component lifetimes", async () => {
  const first = deferred<void>();
  const second = deferred<void>();
  const batchNotified = deferred<void>();
  let notificationCount = 0;
  const unsubscribe = subscribeToMcpServerMutationSettlements(() => {
    notificationCount += 1;
    batchNotified.resolve();
  });
  trackMcpServerMutation(first.promise);

  let reconciled = false;
  const reconciliation = waitForPendingMcpServerMutations().then(() => {
    reconciled = true;
  });
  await Promise.resolve();
  assert.equal(
    reconciled,
    false,
    "the first pending mutation must hold refresh",
  );

  // Simulate another API mutation starting after the opening component has
  // already captured and begun waiting on the first batch.
  trackMcpServerMutation(second.promise);
  first.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(
    reconciled,
    false,
    "a mutation added during settlement must also hold refresh",
  );

  second.resolve();
  await reconciliation;
  await batchNotified.promise;
  unsubscribe();
  assert.equal(reconciled, true);
  assert.equal(
    notificationCount,
    2,
    "each settled mutation publishes its epoch",
  );
});

test("a settled successor can reconcile while an older mutation remains pending", async () => {
  const first = deferred<void>();
  const second = deferred<void>();
  let authoritativeRows = ["old"];
  const snapshots: string[][] = [];
  const successorVisible = deferred<void>();
  const predecessorVisible = deferred<void>();

  const unsubscribe = subscribeToMcpServerMutationSettlements(() => {
    void readMcpServerMutationSnapshot(async () => [...authoritativeRows]).then(
      (rows) => {
        snapshots.push(rows);
        if (rows.includes("successor")) successorVisible.resolve();
        if (rows.includes("predecessor")) predecessorVisible.resolve();
      },
    );
  });

  trackMcpServerMutation(first.promise);
  trackMcpServerMutation(second.promise);
  authoritativeRows = ["old", "successor"];
  second.resolve();
  await successorVisible.promise;
  assert.deepEqual(snapshots.at(-1), ["old", "successor"]);

  authoritativeRows = ["old", "successor", "predecessor"];
  first.resolve();
  await predecessorVisible.promise;
  await waitForPendingMcpServerMutations();
  unsubscribe();
  assert.deepEqual(snapshots.at(-1), ["old", "successor", "predecessor"]);
});

test("failed mutations settle waiters without leaking a tracker rejection", async () => {
  const mutation = deferred<void>();
  const tracked = trackMcpServerMutation(mutation.promise);
  assert.equal(tracked, mutation.promise);
  const expectedFailure = assert.rejects(tracked, /save failed/);
  const reconciliation = waitForPendingMcpServerMutations();

  mutation.reject(new Error("save failed"));
  await expectedFailure;
  await reconciliation;
  await waitForPendingMcpServerMutations();
});

test("settlement refreshes a mounted background consumer after another consumer mutates", async () => {
  let authoritativeRows = ["old"];
  let backgroundRows = await readAfterPendingMcpServerMutations(async () => [
    ...authoritativeRows,
  ]);
  let foregroundRows = [...backgroundRows];
  const backgroundUpdated = deferred<void>();
  let notificationCount = 0;
  let notifiedEpoch = 0;

  const unsubscribe = subscribeToMcpServerMutationSettlements((epoch) => {
    notificationCount += 1;
    notifiedEpoch = epoch;
    void readAfterPendingMcpServerMutations(async () => [
      ...authoritativeRows,
    ]).then((rows) => {
      backgroundRows = rows;
      backgroundUpdated.resolve();
    });
  });

  const mutation = deferred<string>();
  const tracked = trackMcpServerMutation(mutation.promise);
  assert.equal(tracked, mutation.promise);
  authoritativeRows = ["new"];
  mutation.resolve("saved");

  assert.equal(await tracked, "saved");
  foregroundRows = await readAfterPendingMcpServerMutations(async () => [
    ...authoritativeRows,
  ]);
  await backgroundUpdated.promise;
  unsubscribe();

  assert.deepEqual(foregroundRows, ["new"]);
  assert.deepEqual(backgroundRows, ["new"]);
  assert.equal(notificationCount, 1);
  assert.ok(
    notifiedEpoch >= 2,
    "registration and settlement both advance epoch",
  );
});

test("a list read retries when a mutation starts after its pre-read drain", async () => {
  let authoritativeRows = ["old"];
  let readCount = 0;
  const firstReadStarted = deferred<void>();
  const firstReadResult = deferred<string[]>();

  const stableRead = readAfterPendingMcpServerMutations(async () => {
    readCount += 1;
    if (readCount === 1) {
      firstReadStarted.resolve();
      return firstReadResult.promise;
    }
    return [...authoritativeRows];
  });

  await firstReadStarted.promise;
  const mutation = deferred<void>();
  const tracked = trackMcpServerMutation(mutation.promise);
  firstReadResult.resolve(["old"]);
  authoritativeRows = ["new"];
  mutation.resolve();

  await tracked;
  assert.deepEqual(await stableRead, ["new"]);
  assert.equal(readCount, 2, "the overlapping old read must be retried once");
});

test("a rejected overlapping list read retries after settlement subscribers join it", async () => {
  let authoritativeRows = ["old"];
  let readCount = 0;
  let listRequest: Promise<string[]> | null = null;
  const firstReadStarted = deferred<void>();
  const firstReadResult = deferred<string[]>();

  function listRows(): Promise<string[]> {
    if (listRequest) return listRequest;
    const request = readAfterPendingMcpServerMutations(async () => {
      readCount += 1;
      if (readCount === 1) {
        firstReadStarted.resolve();
        return firstReadResult.promise;
      }
      return [...authoritativeRows];
    });
    listRequest = request;
    void request.then(
      () => {
        if (listRequest === request) listRequest = null;
      },
      () => {
        if (listRequest === request) listRequest = null;
      },
    );
    return request;
  }

  const originalRequest = listRows();
  await firstReadStarted.promise;

  const subscriberJoined = deferred<void>();
  let subscriberRequest: Promise<string[]> | null = null;
  let notificationCount = 0;
  const unsubscribe = subscribeToMcpServerMutationSettlements(() => {
    notificationCount += 1;
    subscriberRequest = listRows();
    subscriberJoined.resolve();
  });

  const mutation = deferred<void>();
  const tracked = trackMcpServerMutation(mutation.promise);
  authoritativeRows = ["new"];
  mutation.resolve();
  await tracked;
  await subscriberJoined.promise;

  assert.equal(
    subscriberRequest,
    originalRequest,
    "the settlement subscriber must initially join the in-flight GET",
  );
  firstReadResult.reject(new Error("transient list failure"));

  assert.deepEqual(await originalRequest, ["new"]);
  assert.deepEqual(await subscriberRequest, ["new"]);
  unsubscribe();
  assert.equal(notificationCount, 1);
  assert.equal(readCount, 2, "the failed overlapping GET gets one fresh retry");
});

test("a list rejection without an epoch change preserves the original error", async () => {
  const expected = new Error("unrelated list failure");
  let readCount = 0;
  await assert.rejects(
    readAfterPendingMcpServerMutations(async () => {
      readCount += 1;
      throw expected;
    }),
    (error) => error === expected,
  );
  assert.equal(readCount, 1, "an unrelated failure must not be retried");
});

test("every list consumer uses the shared pending-mutation read barrier", () => {
  const chatRoot = fileURLToPath(
    new URL("../src/features/chat/", import.meta.url),
  );
  const listApi = sourceBetween(
    MCP_SERVERS_API,
    "export function listMcpServers",
    "export function createMcpServer",
  );
  const dialogRefresh = sourceBetween(
    CHAT_MCP_SERVERS_DIALOG,
    "const refresh = useCallback",
    "useEffect(() =>",
  );
  const composerRefresh = sourceBetween(
    MCP_COMPOSER_BUTTON,
    "const refresh = useCallback",
    "// Load the server list on mount",
  );

  const listOccurrences = typescriptFilesUnder(chatRoot)
    .flatMap((file) => {
      const count =
        readFileSync(file, "utf8").match(/\blistMcpServers\s*\(/g)?.length ?? 0;
      return Array.from({ length: count }, () =>
        relative(chatRoot, file).replaceAll("\\", "/"),
      );
    })
    .sort();

  assert.deepEqual(listOccurrences, [
    "api/mcp-servers-api.ts",
    "chat-mcp-servers-dialog.tsx",
    "mcp-composer-button.tsx",
  ]);

  assert.equal(
    MCP_SERVERS_API.match(/return trackMcpServerMutation\(/g)?.length,
    5,
    "create, update, delete, import, and managed Blender updates must register at the API boundary",
  );
  assert.match(
    listApi,
    /readAfterPendingMcpServerMutations\(\(\) =>[\s\S]*mcpRequest<McpServerConfig\[\]>\("\/"\)/,
    "the shared query boundary must retry reads that overlap tracked mutations",
  );
  assert.match(
    listApi,
    /readMcpServerMutationSnapshot\(\(\) =>[\s\S]*mcpRequest<McpServerConfig\[\]>\("\/"\)/,
    "settlement refreshes must not wait on unrelated older mutations",
  );
  assert.match(
    listApi,
    /minimumMutationEpoch \?\? getMcpServerMutationEpoch\(\)[\s\S]*mcpServerSettlementListRequest\.minimumEpoch >= requestedEpoch/,
    "a newer settlement must replace a snapshot cached for an older epoch",
  );
  assert.match(
    listApi,
    /const slot = \{ minimumEpoch: requestedEpoch, promise: request \}[\s\S]*mcpServerSettlementListRequest === slot/,
    "an older completion must not clear the successor epoch slot",
  );
  assert.match(
    listApi,
    /if \(mcpServerListRequest\) return mcpServerListRequest/,
  );
  assert.match(
    dialogRefresh,
    /await listMcpServers\(\{\s*waitForPendingMutations,\s*minimumMutationEpoch,\s*\}\)/,
  );
  assert.match(
    composerRefresh,
    /await listMcpServers\(\{\s*waitForPendingMutations,\s*minimumMutationEpoch,\s*\}\)/,
  );
  assert.match(
    CHAT_MCP_SERVERS_DIALOG,
    /subscribeToMcpServerMutationSettlements\(\(epoch\) => \{\s*void refresh\(false, epoch\)/,
  );
  assert.match(
    MCP_COMPOSER_BUTTON,
    /subscribeToMcpServerMutationSettlements\(\(epoch\) => \{\s*void refresh\(false, epoch\)/,
  );
  assert.match(
    dialogRefresh,
    /listRefreshGenerationRef\.current !== generation/,
  );
  assert.match(
    composerRefresh,
    /listRefreshGenerationRef\.current !== generation/,
  );
  assert.doesNotMatch(
    CHAT_MCP_SERVERS_DIALOG,
    /waitForPendingMcpServerMutations/,
  );
  assert.doesNotMatch(MCP_COMPOSER_BUTTON, /waitForPendingMcpServerMutations/);
  assert.doesNotMatch(CHAT_MCP_SERVERS_DIALOG, /await refresh\(\)/);
  assert.doesNotMatch(MCP_COMPOSER_BUTTON, /await refresh\(\)/);
});
