// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";
import {
  elementsOf,
  loadWithStubs,
  stubEveryImport,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

registerBundlerResolver();

const { pendingItems, webhookUrl } = await import(
  "../src/features/flows/capability-items.ts"
);

const none = {
  python: [],
  hosts: [],
  mcp: false,
  providers: [],
  triggers: false,
};

test("nothing pending shows no approval banner", () => {
  assert.deepEqual(pendingItems(none), []);
  assert.deepEqual(pendingItems(null), []);
});

test("each pending capability is one readable line", () => {
  const items = pendingItems({
    python: ["sha256:a", "sha256:b"],
    hosts: ["api.example.com"],
    mcp: true,
    providers: ["openai"],
    triggers: true,
  });
  assert.deepEqual(
    items.map((i) => i.label),
    [
      "Runs Python code (2 blocks)",
      "Calls api.example.com",
      "Uses your MCP servers' tools",
      "Uses your openai connection",
      "Starts on its own (schedule or webhook) while Studio is open",
    ],
  );
  assert.equal(new Set(items.map((i) => i.key)).size, items.length);
});

test("the webhook URL joins the page origin and the route path", () => {
  assert.equal(
    webhookUrl("http://127.0.0.1:8888/", "/api/flows/hooks/abc"),
    "http://127.0.0.1:8888/api/flows/hooks/abc",
  );
});

// The two components below guard races inside React effects. They run here with a tiny hook
// runtime: state and refs persist across renders by call order, effects run after a render
// when their deps change (cleanup first), and the test controls every API promise.

type Deferred<T> = { promise: Promise<T>; resolve: (v: T) => void; reject: (e: unknown) => void };
function deferred<T>(): Deferred<T> {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((a, b) => {
    resolve = a;
    reject = b;
  });
  return { promise, resolve, reject };
}

const flush = async () => {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
};

function hookRuntime() {
  const slots: unknown[] = [];
  const effects: { deps?: unknown[]; cleanup?: unknown; run?: () => unknown }[] = [];
  let i = 0;
  const changed = (a: unknown[] | undefined, b: unknown[] | undefined) =>
    !a || !b || a.length !== b.length || a.some((v, k) => !Object.is(v, b[k]));
  const react = {
    __esModule: true,
    useState<T>(init: T) {
      const k = i++;
      if (!(k in slots)) slots[k] = init;
      return [slots[k], (v: T) => void (slots[k] = v)];
    },
    useRef<T>(init: T) {
      const k = i++;
      if (!(k in slots)) slots[k] = { current: init };
      return slots[k];
    },
    useCallback<T>(fn: T, deps: unknown[]) {
      const k = i++;
      const prev = slots[k] as { fn: T; deps: unknown[] } | undefined;
      if (!prev || changed(prev.deps, deps)) slots[k] = { fn, deps };
      return (slots[k] as { fn: T }).fn;
    },
    useEffect(run: () => unknown, deps?: unknown[]) {
      const k = i++;
      const prev = effects[k];
      if (!prev || changed(prev.deps, deps)) effects[k] = { deps, cleanup: prev?.cleanup, run };
    },
  };
  return {
    react,
    /** One render plus its effects, as React commits it. */
    render<P>(component: (props: P) => unknown, props: P) {
      i = 0;
      const tree = component(props);
      for (const e of effects) {
        if (!e?.run) continue;
        if (typeof e.cleanup === "function") e.cleanup();
        e.cleanup = e.run();
        e.run = undefined;
      }
      return tree;
    },
  };
}

const textOf = (tree: unknown) =>
  elementsOf(tree)
    .flatMap((e) => [e.props.children].flat())
    .filter((c) => typeof c === "string")
    .join(" ");

function loadTriggerPanel(apiBase: string) {
  const hooks = hookRuntime();
  const lists: { flowId: string; d: Deferred<unknown> }[] = [];
  const mod = loadWithStubs<{ TriggerPanel: (p: { nodeId: string; kind: string }) => unknown }>(
    new URL("../src/features/flows/editor/trigger-panel.tsx", import.meta.url),
    stubEveryImport({
      react: hooks.react,
      "react/jsx-runtime": stubJsxRuntime(),
      "@/lib/api-base": { getApiBase: () => apiBase },
      "../capability-items": { webhookUrl },
      "../stores/doc-store": { useDocStore: (sel: (s: unknown) => unknown) => sel({ flowId: "f1" }) },
      "../stores/ui-store": { useUiStore: (sel: (s: unknown) => unknown) => sel({ saveState: "saved" }) },
      "../api": {
        listTriggers: (flowId: string) => {
          const d = deferred<unknown>();
          lists.push({ flowId, d });
          return d.promise;
        },
      },
    }),
  );
  const render = (nodeId: string) =>
    hooks.render(mod.TriggerPanel, { nodeId, kind: "trigger.webhook" });
  return { render, lists };
}

const hooksFor = (nodeId: string, secret: string) => ({
  owner_only: false,
  triggers: [{ node_id: nodeId, path: `/api/flows/hooks/${secret}` }],
});

test("the webhook URL uses the backend origin", async () => {
  const panel = loadTriggerPanel("http://127.0.0.1:8888");
  panel.render("a");
  panel.lists[0].d.resolve(hooksFor("a", "secretA"));
  await flush();
  assert.match(textOf(panel.render("a")), /http:\/\/127\.0\.0\.1:8888\/api\/flows\/hooks\/secretA/);
});

test("switching to another trigger block never shows the last block's secret", async () => {
  const panel = loadTriggerPanel("http://backend");
  panel.render("a");
  panel.lists[0].d.resolve(hooksFor("a", "secretA"));
  await flush();
  assert.match(textOf(panel.render("a")), /secretA/);
  panel.render("b"); // commit: the identity-change effect clears the old trigger
  assert.doesNotMatch(textOf(panel.render("b")), /secretA/);
});

test("a trigger list that lands after the block changed is dropped", async () => {
  const panel = loadTriggerPanel("http://backend");
  panel.render("a");
  panel.render("b");
  panel.lists[0].d.resolve(hooksFor("a", "secretA"));
  await flush();
  assert.doesNotMatch(textOf(panel.render("b")), /secretA/);
  panel.lists[1].d.resolve(hooksFor("b", "secretB"));
  await flush();
  assert.match(textOf(panel.render("b")), /secretB/);
});

function loadBanner() {
  const hooks = hookRuntime();
  const gets: { flowId: string; d: Deferred<unknown> }[] = [];
  const approvals: { flowId: string; d: Deferred<unknown> }[] = [];
  const mod = loadWithStubs<{ CapabilityBanner: (p: { flowId: string }) => unknown }>(
    new URL("../src/features/flows/editor/capability-banner.tsx", import.meta.url),
    stubEveryImport({
      react: hooks.react,
      "react/jsx-runtime": stubJsxRuntime(),
      "../capability-items": { pendingItems },
      "../stores/ui-store": { useUiStore: (sel: (s: unknown) => unknown) => sel({ saveState: "saved" }) },
      "../api": {
        getCapabilities: (flowId: string) => {
          const d = deferred<unknown>();
          gets.push({ flowId, d });
          return d.promise;
        },
        approveCapabilities: (flowId: string) => {
          const d = deferred<unknown>();
          approvals.push({ flowId, d });
          return d.promise;
        },
      },
    }),
  );
  return { render: (flowId: string) => hooks.render(mod.CapabilityBanner, { flowId }), gets, approvals };
}

const report = (host: string, pending = true) => ({
  can_approve: true,
  pending: { python: [], hosts: pending ? [host] : [], mcp: false, providers: [], triggers: false },
});

test("the approval banner never shows another flow's report", async () => {
  const banner = loadBanner();
  banner.render("f1");
  banner.render("f2");
  banner.gets[0].d.resolve(report("one.example"));
  await flush();
  assert.equal(banner.render("f2"), null, "f1's late report is dropped");
  const f2 = banner.gets.find((g) => g.flowId === "f2");
  assert.ok(f2);
  f2.d.resolve(report("two.example"));
  await flush();
  assert.match(textOf(banner.render("f2")), /Calls two\.example/);
});

test("approving never lands on another flow, and a flow switch clears the banner", async () => {
  const banner = loadBanner();
  banner.render("f1");
  banner.gets[0].d.resolve(report("one.example"));
  await flush();
  const tree = banner.render("f1");
  const approve = elementsOf(tree).find((e) => e.props["data-testid"] === "flows-capabilities-approve");
  assert.ok(approve);
  (approve.props.onClick as () => void)();
  assert.equal(banner.approvals[0].flowId, "f1");
  banner.render("f2");
  assert.equal(banner.render("f2"), null, "the switch resets the report");
  // f1's approval answers after the switch, still with pending items it would show.
  banner.approvals[0].d.resolve(report("one.example"));
  await flush();
  assert.equal(banner.render("f2"), null);
});
