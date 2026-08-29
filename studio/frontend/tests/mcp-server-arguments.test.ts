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
  const helper = readFileSync(
    new URL("../src/features/chat/mcp-server-form.ts", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(helper, /\.(?:split|join|trim)\s*\(/);
  assert.doesNotMatch(helper, /JSON\.stringify|replace\s*\(/);
});

test("the dialog wires backend codec calls, stale guards, and a stdio-only editor", () => {
  const dialog = readFileSync(
    new URL(
      "../src/features/chat/chat-mcp-servers-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const api = readFileSync(
    new URL("../src/features/chat/api/mcp-servers-api.ts", import.meta.url),
    "utf8",
  );

  assert.match(api, /mcpRequest\("\/stdio\/decode"/);
  assert.match(api, /mcpRequest\("\/stdio\/encode"/);
  assert.match(dialog, /await decodeMcpStdioCommand\(server\.url\)/);
  assert.match(dialog, /await encodeMcpStdioCommand\(\{/);
  assert.match(dialog, /formGenerationRef\.current !== generation/);
  assert.match(dialog, /activeEditIdRef\.current !== server\.id/);
  assert.match(
    dialog,
    /function handleOpenChange[\s\S]*formGenerationRef\.current \+= 1;[\s\S]*onOpenChange\(next\)/,
  );
  assert.match(dialog, /addressIsCommand && \(\s*<ArgumentsEditor/);
  assert.match(dialog, /const addressIsCommand = form\.transport === "stdio"/);
  assert.match(
    dialog,
    /function formWithAddress[\s\S]*transportFromAddress\([\s\S]*preservePartialHttp \? form\.credentialTransport : null[\s\S]*headers: transportChanged \? \[\] : form\.headers/,
  );
  assert.match(
    dialog,
    /queueMicrotask\(\(\) => \{\s*if \(cancelled\) return;[\s\S]*setView\(\{ kind: "list" \}\)/,
  );
  assert.match(
    dialog,
    /function ArgumentsEditor[\s\S]*\{ id: newRowId\(\), value: "" \}/,
  );
  assert.match(
    dialog,
    /form\.transport === "http" && \([\s\S]*Use OAuth sign-in/,
  );
  assert.match(dialog, /const decision = resolveMcpStdioUrl\(/);
  assert.match(
    dialog,
    /decision\.kind === "reuse"[\s\S]*url = view\.kind === "edit" \? undefined : decision\.url/,
  );
  assert.match(
    dialog,
    /const url = stdio\s*\? await encodeStdioForGeneration\([\s\S]*testMcpServer\(\{\s*url,/,
  );
  assert.match(dialog, /return rows\.map\(\(row\) => row\.value\)/);
  assert.match(
    dialog,
    /function ArgumentsEditor[\s\S]*data-reload-snapshot-sensitive/,
  );
  assert.match(api, /export function testMcpServer[\s\S]*body: \{\s*url:/);
  assert.doesNotMatch(
    dialog,
    /npx -y @modelcontextprotocol\/server-filesystem \/tmp/,
  );
  assert.match(dialog, /URL or executable/);
  assert.match(dialog, /https:\/\/example\.com\/mcp or npx/);
  assert.match(dialog, /Add local arguments in the Arguments rows/);
  assert.match(
    dialog,
    /setForm\(\(prev\) => formWithAddress\(prev, url, true\)\)/,
  );
  assert.doesNotMatch(dialog, /form\.url\.(?:split|join)\s*\(/);
  assert.doesNotMatch(dialog, /form\.arguments[^;\n]*\.join\s*\(/);
});

test("every mutable MCP form editor is locked for the full pending interval", () => {
  const dialog = readFileSync(
    new URL(
      "../src/features/chat/chat-mcp-servers-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const argumentsEditor = sourceBetween(
    dialog,
    "function ArgumentsEditor",
    "function HeadersEditor",
  );
  const headersEditor = sourceBetween(
    dialog,
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
    dialog,
    /const formPending = importing \|\| codecPending \|\| testing \|\| saving/,
  );
  assert.match(
    dialog,
    /id="mcp-display-name"[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
  assert.match(
    dialog,
    /id="mcp-url"[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
  assert.match(
    dialog,
    /<ArgumentsEditor[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
  assert.match(
    dialog,
    /id="mcp-oauth"[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
  assert.match(
    dialog,
    /<HeadersEditor[\s\S]*?disabled=\{formPending\}[\s\S]*?\/>/,
  );
});

test("a decode error is announced and executable edits unlock manual recovery", () => {
  const dialog = readFileSync(
    new URL(
      "../src/features/chat/chat-mcp-servers-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );

  assert.match(
    dialog,
    /id="mcp-url"[\s\S]*?onChange=\{\(e\) => \{[\s\S]*?setCodecError\(null\)[\s\S]*?formWithAddress/,
  );
  assert.match(
    dialog,
    /role="alert"\s*aria-live="assertive"[\s\S]*?\{codecError\}/,
  );
  assert.match(
    dialog,
    /aria-busy=\{decodingCommand\}[\s\S]*role="status"\s*aria-live="polite"[\s\S]*Reading local command…/,
  );
  assert.match(
    dialog,
    /codecError && \([\s\S]*view\.kind === "edit"[\s\S]*void startEdit\(server\)[\s\S]*Retry/,
  );
  assert.match(
    dialog,
    /disabled=\{\s*formPending \|\|\s*codecError !== null \|\|\s*form\.transport === "unknown" \|\|\s*!form\.url\.trim\(\)/,
  );
});

test("dialog actions and reconciliation stop when the dialog closes", () => {
  const dialog = readFileSync(
    new URL(
      "../src/features/chat/chat-mcp-servers-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );

  assert.match(
    dialog,
    /useEffect\(\(\) => \{\s*if \(!open\) \{[\s\S]*subscribeToMcpServerMutationSettlements/,
  );
  assert.match(
    dialog,
    /actionGenerationRef\.current !== generation \|\| !openRef\.current/,
  );
  assert.match(dialog, /open=\{open && confirmingDelete !== null\}/);
  assert.match(
    dialog,
    /<Button size="sm" onClick=\{startCreate\} disabled=\{importing\}>/,
  );
});

test("pending MCP actions remain keyed to their own server", () => {
  const dialog = readFileSync(
    new URL(
      "../src/features/chat/chat-mcp-servers-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const composer = readFileSync(
    new URL("../src/features/chat/mcp-composer-button.tsx", import.meta.url),
    "utf8",
  );

  assert.match(composer, /pendingUrlsRef = useRef\(new Set<string>\(\)\)/);
  assert.match(
    composer,
    /pendingUrlsRef\.current\.delete\(norm\);\s*setPendingUrls\(new Set\(pendingUrlsRef\.current\)\)/,
  );
  assert.match(dialog, /refreshingIdsRef = useRef\(new Set<string>\(\)\)/);
  assert.match(dialog, /togglingIdsRef = useRef\(new Set<string>\(\)\)/);
  assert.match(
    dialog,
    /if \(!togglingIdsRef\.current\.has\(row\.id\)\) return row;[\s\S]*is_enabled: optimistic\.is_enabled/,
  );
});

test("dialog closure is blocked while a CRUD mutation is in flight", () => {
  const dialog = readFileSync(
    new URL(
      "../src/features/chat/chat-mcp-servers-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const closeHandler = sourceBetween(
    dialog,
    "function handleOpenChange",
    "async function encodeStdioForGeneration",
  );

  assert.match(
    closeHandler,
    /if \(!next && \(\(saving && !codecPending\) \|\| busyIdsRef\.current\.size > 0\)\)[\s\S]*return;[\s\S]*if \(!next\) \{[\s\S]*formGenerationRef\.current \+= 1;/,
    "the in-flight mutation guard must run before close invalidates the form generation",
  );
  assert.match(
    dialog,
    /<DialogContent[\s\S]*?showCloseButton=\{!\(saving && !codecPending\) && busyIds\.size === 0\}[\s\S]*?>/,
    "the built-in close control must disappear during the same mutation window",
  );
  assert.match(
    dialog,
    /onClick=\{cancelForm\}[\s\S]*?disabled=\{saving && !codecPending\}/,
    "the visible Cancel action must use the same mutation boundary",
  );
  assert.match(
    dialog,
    /async function encodeStdioForGeneration[\s\S]*setCodecPending\(true\);[\s\S]*await encodeMcpStdioCommand/,
    "codec encoding remains distinguishable so it can still be cancelled before CRUD starts",
  );
});

test("composer applies mutation responses before releasing each preset", () => {
  const composer = readFileSync(
    new URL("../src/features/chat/mcp-composer-button.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    composer,
    /const applyServer = useCallback[\s\S]*setServers\(\(current\)[\s\S]*candidate\.id === server\.id/,
  );
  assert.match(
    composer,
    /applyServer\(\s*await createMcpServer\([\s\S]*pendingUrlsRef\.current\.delete\(norm\)/,
  );
  assert.match(
    composer,
    /applyServer\(\s*await updateMcpServer\([\s\S]*pendingUrlsRef\.current\.delete\(norm\)/,
  );
  assert.match(
    composer,
    /const \[serversLoaded, setServersLoaded\] = useState\(false\)[\s\S]*const hasLoadedServerSnapshotRef = useRef\(false\);[\s\S]*setServersLoaded\(false\);[\s\S]*setServers\(rows\);\s*hasLoadedServerSnapshotRef\.current = true;\s*setServersLoaded\(true\)/,
  );
  assert.match(
    composer,
    /catch \{\s*if \(\s*listRefreshGenerationRef\.current === generation &&\s*hasLoadedServerSnapshotRef\.current\s*\) \{\s*setServersLoaded\(true\);/,
  );
  assert.match(
    composer,
    /disabled=\{\s*!serversLoaded \|\| pendingUrls\.has\(normalizeMcpUrl\(opts\.url\)\)/,
  );
});

test("full unmount invalidates cancellable stdio encode continuations", async () => {
  const dialog = readFileSync(
    new URL(
      "../src/features/chat/chat-mcp-servers-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const refsAndCleanup = sourceBetween(
    dialog,
    "const formGenerationRef = useRef(0)",
    "const refresh = useCallback",
  );
  const openLifecycle = sourceBetween(
    dialog,
    "useEffect(() => {\n    formGenerationRef.current += 1;",
    "function startCreate",
  );
  const encode = sourceBetween(
    dialog,
    "async function encodeStdioForGeneration",
    "async function testConnection",
  );
  const testContinuation = sourceBetween(
    dialog,
    "async function testConnection",
    "async function submitForm",
  );
  const crudContinuation = sourceBetween(
    dialog,
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
  const dialog = readFileSync(
    new URL(
      "../src/features/chat/chat-mcp-servers-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const api = readFileSync(
    new URL("../src/features/chat/api/mcp-servers-api.ts", import.meta.url),
    "utf8",
  );
  const composer = readFileSync(
    new URL("../src/features/chat/mcp-composer-button.tsx", import.meta.url),
    "utf8",
  );
  const listApi = sourceBetween(
    api,
    "export function listMcpServers",
    "export function createMcpServer",
  );
  const dialogRefresh = sourceBetween(
    dialog,
    "const refresh = useCallback",
    "useEffect(() =>",
  );
  const composerRefresh = sourceBetween(
    composer,
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
    api.match(/return trackMcpServerMutation\(/g)?.length,
    4,
    "create, update, delete, and import must register at the API boundary",
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
    dialog,
    /subscribeToMcpServerMutationSettlements\(\(epoch\) => \{\s*void refresh\(false, epoch\)/,
  );
  assert.match(
    composer,
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
  assert.doesNotMatch(dialog, /waitForPendingMcpServerMutations/);
  assert.doesNotMatch(composer, /waitForPendingMcpServerMutations/);
  assert.doesNotMatch(dialog, /await refresh\(\)/);
  assert.doesNotMatch(composer, /await refresh\(\)/);
});
