// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Slot = {
  value?: unknown;
  set?: (v: unknown) => void;
  current?: unknown;
  deps?: unknown[];
  memo?: unknown;
  cleanup?: (() => void) | void;
};

function createReact() {
  let current: Instance | null = null;
  type Instance = {
    hooks: Slot[];
    i: number;
    pending: (() => void)[];
    mounted: boolean;
    queued: boolean;
    rerender: () => void;
  };
  const changed = (prev: unknown[] | undefined, next: unknown[] | undefined) =>
    !prev || !next || prev.length !== next.length || prev.some((d, i) => !Object.is(d, next[i]));
  const slot = (): [Instance, Slot, boolean] => {
    const inst = current!;
    const i = inst.i++;
    const fresh = !(i in inst.hooks);
    if (fresh) inst.hooks[i] = {};
    return [inst, inst.hooks[i], fresh];
  };
  const hooks = {
    useState(init: unknown) {
      const [inst, s, fresh] = slot();
      if (fresh) {
        s.value = typeof init === "function" ? (init as () => unknown)() : init;
        s.set = (v: unknown) => {
          const next = typeof v === "function" ? (v as (p: unknown) => unknown)(s.value) : v;
          if (Object.is(next, s.value)) return;
          s.value = next;
          if (inst.mounted && !inst.queued) {
            inst.queued = true;
            queueMicrotask(inst.rerender);
          }
        };
      }
      return [s.value, s.set];
    },
    useRef(init: unknown) {
      const [, s, fresh] = slot();
      if (fresh) s.current = init;
      return s;
    },
    useMemo(fn: () => unknown, deps: unknown[]) {
      const [, s] = slot();
      if (changed(s.deps, deps)) {
        s.memo = fn();
        s.deps = deps;
      }
      return s.memo;
    },
    useCallback(fn: unknown, deps: unknown[]) {
      return hooks.useMemo(() => fn, deps);
    },
    useEffect(fn: () => (() => void) | void, deps?: unknown[]) {
      const [inst, s] = slot();
      if (changed(s.deps, deps)) {
        s.deps = deps;
        inst.pending.push(() => {
          s.cleanup?.();
          s.cleanup = fn();
        });
      }
    },
  };
  function mount<T>(render: () => T) {
    let out!: T;
    const inst: Instance = {
      hooks: [],
      i: 0,
      pending: [],
      mounted: true,
      queued: false,
      rerender: () => {
        inst.queued = false;
        if (!inst.mounted) return;
        inst.i = 0;
        current = inst;
        out = render();
        current = null;
        const effects = inst.pending.splice(0);
        for (const effect of effects) effect();
      },
    };
    inst.rerender();
    const cleanupAll = () => {
      for (const s of inst.hooks) {
        const cleanup = s.cleanup;
        s.cleanup = undefined;
        cleanup?.();
      }
    };
    return {
      get value() {
        return out;
      },
      rerender: () => inst.rerender(),
      unmount: () => {
        inst.mounted = false;
        cleanupAll();
      },
      strictRemount: () => {
        cleanupAll();
        for (const s of inst.hooks) s.deps = undefined;
        inst.rerender();
      },
    };
  }
  return { hooks, mount };
}

class RecipeApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

type Stored = { id: string; name: string; payload: unknown; createdAt: number; updatedAt: number };
type Put = {
  body: Stored & { baseUpdatedAt?: number; alsoBaseUpdatedAt?: number[] };
  keepalive: boolean | undefined;
  outcome?: number;
};

// Same rule as data_recipes_db.upsert_recipe: a write lands only on a version it names.
function fakeServer(initial: Stored) {
  let stored = { ...initial };
  const puts: Put[] = [];
  const queue: (() => void)[] = [];
  const cancels: ((landed: boolean) => void)[] = [];
  // Applies a PUT now and returns the transport error to deliver later.
  let gets = 0;
  let failStatus: number | null = null;
  const lands: (() => { fail: () => void; answer: () => void })[] = [];
  const recipeRequest = (_path: string, init: RequestInit = {}) => {
    if ((init.method ?? "GET") === "GET") {
      gets += 1;
      return Promise.resolve({ ...stored });
    }
    const put: Put = { body: JSON.parse(String(init.body)), keepalive: init.keepalive };
    puts.push(put);
    return new Promise((resolve, reject) => {
      cancels.push((landed: boolean) => {
        if (landed) apply();
        reject(new TypeError("Failed to fetch"));
      });
      lands.push(() => {
        const ok = apply();
        const answered = { ...stored };
        return {
          fail: () => reject(new TypeError("Failed to fetch")),
          answer: () =>
            ok ? resolve(answered) : reject(new RecipeApiError("conflict", 409)),
        };
      });
      const apply = () => {
        const { baseUpdatedAt, alsoBaseUpdatedAt = [], ...record } = put.body;
        if (
          baseUpdatedAt !== undefined &&
          ![baseUpdatedAt, ...alsoBaseUpdatedAt].includes(stored.updatedAt)
        ) {
          put.outcome = 409;
          return false;
        }
        put.outcome = 200;
        stored = record;
        return true;
      };
      queue.push(() => {
        if (failStatus === 200) {
          failStatus = null;
          put.outcome = 200;
          const { baseUpdatedAt: _b, alsoBaseUpdatedAt: _a, ...landed } = put.body;
          stored = landed;
          resolve(null);
          return;
        }
        if (failStatus !== null) {
          put.outcome = failStatus;
          reject(new RecipeApiError("Service unavailable", failStatus));
          failStatus = null;
          return;
        }
        const { baseUpdatedAt, alsoBaseUpdatedAt = [], ...record } = put.body;
        if (
          baseUpdatedAt !== undefined &&
          ![baseUpdatedAt, ...alsoBaseUpdatedAt].includes(stored.updatedAt)
        ) {
          put.outcome = 409;
          reject(new RecipeApiError("This recipe was changed in another window.", 409));
          return;
        }
        put.outcome = 200;
        stored = record;
        resolve({ ...stored });
      });
    });
  };
  return {
    puts,
    recipeRequest,
    get gets() {
      return gets;
    },
    failNext(status: number) {
      failStatus = status;
    },
    get stored() {
      return stored;
    },
    otherWindowSaves: () => {
      stored = { ...stored, name: "Other window", updatedAt: stored.updatedAt + 5 };
    },
    landOldestWithoutReply() {
      queue.shift();
      cancels.shift();
      const { fail, answer } = lands.shift()!();
      const later = (fn: () => void) => async () => {
        fn();
        await settle();
      };
      return { fail: later(fail), answer: later(answer) };
    },
    async releaseOldest() {
      lands.shift();
      cancels.shift();
      queue.shift()?.();
      await settle();
    },
    async cancelOldest(landed: boolean) {
      lands.shift();
      queue.shift();
      cancels.shift()?.(landed);
      await settle();
    },
    async release(order: "fifo" | "lifo" = "fifo") {
      while (queue.length) {
        const run = order === "fifo" ? queue.shift() : queue.pop();
        if (order === "fifo") {
          cancels.shift();
          lands.shift();
        } else {
          cancels.pop();
          lands.pop();
        }
        run?.();
        await settle();
      }
    },
  };
}

// Each test gets its own budget: a PUT a test leaves unreleased must not starve the next one.
type KeepaliveBudget = typeof import("../src/lib/keepalive-budget.ts");
const freshBudget = () =>
  loadWithStubs<KeepaliveBudget>(new URL("../src/lib/keepalive-budget.ts", import.meta.url), {});

const settle = async () => {
  for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));
};

function setup({ payload = { columns: ["a"] } as unknown, updatedAt = 2000 } = {}) {
  const timers = new Map<number, () => void>();
  let nextTimer = 1;
  const listeners = new Map<string, Set<(event?: unknown) => void>>();
  const on = (type: string, fn: (event?: unknown) => void) => {
    if (!listeners.has(type)) listeners.set(type, new Set());
    listeners.get(type)!.add(fn);
  };
  const off = (type: string, fn: (event?: unknown) => void) => listeners.get(type)?.delete(fn);
  const doc = { visibilityState: "visible", addEventListener: on, removeEventListener: off };
  const account = { marker: "alice" as string | null };
  Object.assign(globalThis, {
    window: {
      setTimeout: (fn: () => void) => {
        timers.set(nextTimer, fn);
        return nextTimer++;
      },
      clearTimeout: (id: number) => timers.delete(id),
      addEventListener: on,
      removeEventListener: off,
      localStorage: { getItem: () => account.marker },
    },
    document: doc,
  });

  const { hooks, mount } = createReact();
  const server = fakeServer({
    id: "r1",
    name: "My recipe",
    payload,
    createdAt: 1000,
    updatedAt,
  });
  const errors: string[] = [];
  const uploads = { pending: [] as string[], removed: [] as string[] };
  const db = loadWithStubs<Record<string, (...args: unknown[]) => unknown>>(
    new URL("../src/features/data-recipes/data/recipes-db.ts", import.meta.url),
    {
      react: hooks,
      "@/features/recipe-studio": { createEmptyRecipePayload: () => ({}) },
      "@/utils": { normalizeNonEmptyName: (n: string) => n.trim() || "Unnamed" },
      "@/lib/keepalive-budget": freshBudget(),
      "./legacy-import": { importLegacyRecipes: async () => undefined },
      "./recipes-api": { recipeRequest: server.recipeRequest, RecipeApiError },
    },
  );
  db.primeRecipeCache({ ...server.stored });

  const { EditRecipePage } = loadWithStubs<{
    EditRecipePage: (props: { recipeId: string }) => {
      type: unknown;
      props: Record<string, unknown>;
    };
  }>(new URL("../src/features/data-recipes/pages/edit-recipe-page.tsx", import.meta.url), {
    react: hooks,
    "react/jsx-runtime": {
      jsx: (type: unknown, props: Record<string, unknown>) => ({ type, props }),
      jsxs: (type: unknown, props: Record<string, unknown>) => ({ type, props }),
    },
    "@/components/app-readiness": { useAppShellReadySignal: () => () => undefined },
    "@/components/ui/button": { Button: "button" },
    "@/features/recipe-studio": { RecipeStudioPage: "RecipeStudioPage" },
    "@tanstack/react-router": { useNavigate: () => () => undefined },
    "../data/recipes-db": db,
    "../data/recipes-api": { RecipeApiError },
  });

  const { useRecipePersistence } = loadWithStubs<{
    useRecipePersistence: (params: Record<string, unknown>) => {
      setWorkflowName: (v: string) => void;
      persistRecipe: () => Promise<void>;
      saveLoading: boolean;
    };
  }>(new URL("../src/features/recipe-studio/hooks/use-recipe-persistence.ts", import.meta.url), {
    react: hooks,
    "@/lib/time-format": { useTimeFormat: () => undefined },
    "@/lib/account-transition": {
      BROWSER_ACCOUNT_KEY: "account",
      accountTransitionPending: () => false,
      sameBrowserAccount: (since: string | null) => since === account.marker,
    },
    "@/lib/copy-to-clipboard": { copyToClipboard: async () => true },
    "@/shared/toast": {
      toastError: (title: string) => void errors.push(title),
      toastSuccess: () => undefined,
    },
    "@/utils": { normalizeNonEmptyName: (n: string, f: string) => n.trim() || f },
    "../api": {
      removeUnstructuredBlock: async (uid: string) => void uploads.removed.push(uid),
    },
    "../executions/execution-helpers": {
      buildSignature: (name: string, p: unknown) => JSON.stringify([name, p]),
      formatSavedLabel: () => "",
    },
    "../stores/recipe-studio": {
      useRecipeStudioStore: {
        getState: () => ({ pendingUploadCleanups: uploads.pending }),
        setState: () => undefined,
      },
    },
    "../utils/import": {
      importRecipePayload: (json: string) => ({
        snapshot: { payload: JSON.parse(json) },
        errors: [],
      }),
    },
  });

  let editorPayload: unknown = null;
  const render = () => {
    const element = EditRecipePage({ recipeId: "r1" });
    // Loading renders RecipeLoadState, not the editor.
    if (element.type !== "RecipeStudioPage") return null as never;
    const page = element.props;
    return useRecipePersistence({
      recipeId: page.recipeId,
      initialRecipeName: page.initialRecipeName,
      initialPayload: page.initialPayload,
      initialSavedAt: page.initialSavedAt,
      onPersistRecipe: page.onPersistRecipe,
      payloadResult: hooks.useMemo(() => ({ payload: editorPayload }), [editorPayload]),
      resetRecipe: hooks.useCallback(() => (editorPayload = null), []),
      loadRecipe: hooks.useCallback((s: { payload: unknown }) => (editorPayload = s.payload), []),
      getCurrentPayloadFromStore: hooks.useCallback(() => editorPayload, []),
    });
  };
  let latest: ReturnType<typeof mount<ReturnType<typeof render>>>;
  const open = () => {
    latest = mount(render);
    latest.rerender();
    return latest;
  };
  const editor = open();

  const consoleError = console.error;
  console.error = () => undefined;
  test.after(() => {
    console.error = consoleError;
  });

  return {
    db,
    open,
    editor,
    server,
    errors,
    uploads,
    account,
    edit: async (next: unknown) => {
      editorPayload = next;
      latest.rerender();
      await settle();
    },
    rename: async (name: string) => {
      editor.value.setWorkflowName(name);
      await settle();
    },
    fireDebounce: async () => {
      const due = [...timers.values()];
      timers.clear();
      for (const fn of due) fn();
      await settle();
    },
    fire: async (type: string, event: unknown = { persisted: false }) => {
      for (const fn of [...(listeners.get(type) ?? [])]) fn(event);
      await settle();
    },
    hide: async () => {
      doc.visibilityState = "hidden";
      for (const fn of [...(listeners.get("visibilitychange") ?? [])]) fn();
      await settle();
    },
  };
}

const payloads = (s: ReturnType<typeof setup>) => s.server.puts.map((p) => p.body.payload);

test("leaving the editor inside the autosave delay saves the edit once", async () => {
  const s = setup();
  await s.edit({ columns: ["a", "b"] });
  s.editor.unmount();
  await settle();
  assert.deepEqual(payloads(s), [{ columns: ["a", "b"] }]);
  await s.server.release();
  await s.fireDebounce();
  assert.equal(s.server.puts.length, 1);
  assert.deepEqual(s.server.stored.payload, { columns: ["a", "b"] });
});

test("a rename right before leaving is saved", async () => {
  const s = setup();
  await s.rename("Renamed");
  s.editor.unmount();
  await s.server.release();
  assert.equal(s.server.stored.name, "Renamed");
});

test("closing or reloading the page sends the pending edit with keepalive", async () => {
  const s = setup();
  await s.edit({ columns: ["x"] });
  await s.fire("pagehide");
  assert.equal(s.server.puts.length, 1);
  assert.equal(s.server.puts[0].keepalive, true);
  await s.fire("pagehide");
  s.editor.unmount();
  await s.server.release();
  assert.equal(s.server.puts.length, 1);
  assert.deepEqual(s.server.stored.payload, { columns: ["x"] });
});

test("a page that goes hidden saves the pending edit", async () => {
  const s = setup();
  await s.edit({ columns: ["hidden"] });
  await s.hide();
  assert.equal(s.server.puts.length, 1);
  assert.equal(s.server.puts[0].keepalive, true);
});

test("an edit too big for keepalive is still sent", async () => {
  const s = setup();
  await s.edit({ prompt: "x".repeat(70 * 1024) });
  await s.fire("pagehide");
  assert.equal(s.server.puts.length, 1);
  assert.equal(s.server.puts[0].keepalive, false);
});

test("an edit that already autosaved is not sent again on leave", async () => {
  const s = setup();
  await s.edit({ columns: ["saved"] });
  await s.fireDebounce();
  await s.server.release();
  s.editor.unmount();
  await s.fire("pagehide");
  assert.equal(s.server.puts.length, 1);
});

test("leaving while the same content is saving sends nothing more", async () => {
  const s = setup();
  await s.edit({ columns: ["same"] });
  await s.fireDebounce();
  await s.fire("pagehide");
  s.editor.unmount();
  await s.server.release();
  assert.equal(s.server.puts.length, 1);
});

test("leaving during a save sends the newer edit after it lands", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  s.editor.unmount();
  await settle();
  assert.equal(s.server.puts.length, 1);
  await s.server.release();
  assert.deepEqual(payloads(s), [{ columns: ["first"] }, { columns: ["second"] }]);
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
});

test("closing the page during a save sends the newer edit on top of it at once", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.fire("pagehide");
  const [first, exit] = s.server.puts;
  assert.equal(s.server.puts.length, 2);
  assert.equal(exit.body.baseUpdatedAt, first.body.updatedAt);
  assert.deepEqual(exit.body.alsoBaseUpdatedAt, [first.body.baseUpdatedAt]);
  assert.ok(exit.keepalive === true && exit.body.updatedAt > first.body.updatedAt);
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
  assert.deepEqual(s.errors, []);
});

test("a newer edit that reaches the server first cannot overwrite anything", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.fire("pagehide");
  await s.server.release("lifo");
  // The exit write lands on the version this tab last confirmed; the older autosave then conflicts.
  assert.deepEqual(
    s.server.puts.map((p) => p.outcome),
    [409, 200],
  );
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
});

test("content another window rejected is not sent again on leave", async () => {
  const s = setup();
  s.server.otherWindowSaves();
  await s.edit({ columns: ["mine"] });
  await s.fireDebounce();
  await s.server.release();
  assert.deepEqual(s.errors, ["Save failed"]);
  await s.fire("pagehide");
  s.editor.unmount();
  await settle();
  assert.equal(s.server.puts.length, 1);
  assert.equal(s.server.stored.name, "Other window");
});

test("no edit, a dev double mount, or leaving sends nothing", async () => {
  const s = setup();
  s.editor.strictRemount();
  await settle();
  await s.fire("pagehide");
  s.editor.unmount();
  await settle();
  assert.equal(s.server.puts.length, 0);
});

test("the Save button saves once and ignores a second press while saving", async () => {
  const s = setup();
  await s.edit({ columns: ["manual"] });
  void s.editor.value.persistRecipe();
  void s.editor.value.persistRecipe();
  await settle();
  assert.equal(s.server.puts.length, 1);
  await s.server.release();
  await s.fireDebounce();
  s.editor.unmount();
  await settle();
  assert.equal(s.server.puts.length, 1);
  assert.deepEqual(s.server.stored.payload, { columns: ["manual"] });
});

for (const landed of [true, false]) {
  test(`a reload that cancels the save in flight still keeps the newer edit (save ${landed ? "landed" : "lost"})`, async () => {
    const s = setup();
    await s.edit({ columns: ["first"] });
    await s.fireDebounce();
    await s.edit({ columns: ["second"] });
    await s.server.cancelOldest(landed);
    await s.fire("pagehide");
    assert.equal(s.server.puts.length, 2);
    assert.equal(s.server.puts[1].keepalive, true);
    await s.server.release();
    assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
    assert.equal(s.server.puts[1].outcome, 200);
  });
}

test("a save lost to the network does not make the next ordinary save guess", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.server.cancelOldest(false);
  await s.edit({ columns: ["second"] });
  await s.fireDebounce();
  assert.equal(s.server.puts.length, 2);
  assert.equal(s.server.puts[1].body.baseUpdatedAt, 2000);
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
});

test("undoing an edit while it saves still saves the undo on leave", async () => {
  for (const leave of ["unmount", "pagehide"] as const) {
    const s = setup();
    await s.edit({ columns: ["b"] });
    await s.fireDebounce();
    await s.edit({ columns: ["a"] });
    if (leave === "pagehide") await s.fire("pagehide");
    else s.editor.unmount();
    await s.server.release();
    assert.deepEqual(payloads(s).slice(0, 2), [{ columns: ["b"] }, { columns: ["a"] }], leave);
    assert.deepEqual(s.server.stored.payload, { columns: ["a"] }, leave);
  }
});

test("a save finishing after its editor closed only cleans uploads queued before it", async () => {
  const s = setup();
  s.uploads.pending.push("mine");
  await s.edit({ columns: ["b"] });
  s.editor.unmount();
  s.uploads.pending.push("next-editor");
  await s.server.release();
  assert.deepEqual(s.uploads.removed, ["mine"]);
});

test("a recovered exit save leaves the saved record cached", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.server.cancelOldest(true);
  await s.fire("pagehide");
  await s.server.release();
  assert.deepEqual(s.server.puts.slice(1).map((p) => p.outcome), [200]);
  assert.equal(
    (s.db.getCachedRecipe("r1") as { updatedAt: number } | null)?.updatedAt,
    s.server.stored.updatedAt,
  );
});

test("a recovery save is one keepalive request however many bases it names", async () => {
  const s = setup();
  await s.edit({ text: "x".repeat(40 * 1024) });
  await s.fireDebounce();
  await s.edit({ text: "y".repeat(40 * 1024) });
  await s.server.cancelOldest(false);
  await s.fire("pagehide");
  assert.deepEqual(s.server.puts.slice(1).map((p) => p.keepalive), [true]);
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { text: "y".repeat(40 * 1024) });
});

test("hiding the page during a save sends the newer edit only after it lands", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.hide();
  assert.equal(s.server.puts.length, 1);
  await s.server.release();
  assert.deepEqual(s.server.puts.map((p) => p.outcome), [200, 200]);
  assert.equal(s.server.puts[1].keepalive, true);
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
  assert.deepEqual(s.errors, []);
});

test("an exit save deferred behind a save in flight only cleans uploads queued at exit", async () => {
  const s = setup();
  s.uploads.pending.push("mine");
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  s.editor.unmount();
  await settle();
  s.uploads.pending.push("next-editor");
  await s.server.release();
  assert.equal(s.server.puts.length, 2);
  assert.ok(!s.uploads.removed.includes("next-editor"));
});

test("a keepalive save still in flight from an earlier call counts against the budget", async () => {
  const s = setup();
  await s.edit({ text: "x".repeat(40 * 1024) });
  await s.hide();
  await s.edit({ text: "y".repeat(40 * 1024) });
  await s.fire("pagehide");
  assert.deepEqual(s.server.puts.map((p) => p.keepalive), [true, false]);
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { text: "y".repeat(40 * 1024) });
});

test("the editor route remounts per recipe so pending versions never cross recipes", () => {
  const route = readFileSync(
    new URL("../src/app/routes/data-recipes.$recipeId.tsx", import.meta.url),
    "utf8",
  );
  assert.match(route, /<EditRecipePage key=\{recipeId\} recipeId=\{recipeId\} \/>/);
});

test("reopening a recipe while its exit save is in flight starts from the saved version", async () => {
  const s = setup();
  await s.edit({ columns: ["left"] });
  s.editor.unmount();
  await settle();
  const reopened = s.open();
  await s.server.release();
  await new Promise((r) => setTimeout(r, 5));
  reopened.rerender();
  await settle();
  await s.edit({ columns: ["again"] });
  await s.fireDebounce();
  await s.server.release();
  assert.deepEqual(s.server.puts.map((p) => p.outcome), [200, 200]);
  assert.deepEqual(s.server.stored.payload, { columns: ["again"] });
});

test("a recovery copy that landed without an answer is still tried as the base next time", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.server.cancelOldest(false);
  await s.edit({ columns: ["second"] });
  await s.fire("pagehide");
  await s.server.releaseOldest();
  await s.server.cancelOldest(true);
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
  await s.edit({ columns: ["third"] });
  await s.fire("pagehide");
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["third"] });
});

test("a chat settings keepalive body shares the budget with recipe saves", async () => {
  const keepaliveBudget = freshBudget();
  const sent: (boolean | undefined)[] = [];
  const api = loadWithStubs<{
    saveChatSettingsPatch: (p: object, o: { keepalive?: boolean }) => Promise<unknown>;
  }>(new URL("../src/features/chat/api/chat-settings-api.ts", import.meta.url), {
    "@/features/auth": {
      authFetch: async (_u: string, init: RequestInit) => {
        sent.push(init.keepalive);
        return new Response(JSON.stringify({ settings: {} }), { status: 200 });
      },
    },
    "@/lib/keepalive-budget": keepaliveBudget,
    "@/lib/format-fastapi-error": { formatApiErrorBody: () => null },
    "../utils/settings-retry": {
      ChatSettingsRequestError: Error,
      asBodyTransportFailure: (e: unknown) => e,
      isAbortOrTimeout: () => false,
    },
  });
  const held = keepaliveBudget.reserveKeepaliveBody("x".repeat(40 * 1024));
  assert.ok(held);
  await api.saveChatSettingsPatch({ note: "y".repeat(30 * 1024) }, { keepalive: true });
  held();
  await api.saveChatSettingsPatch({ note: "y".repeat(30 * 1024) }, { keepalive: true });
  assert.deepEqual(sent, [false, true]);
});

test("an older save's late failure after a newer save landed does not block the content", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.fire("pagehide");
  const first = s.server.landOldestWithoutReply();
  await s.server.release();
  await first.fail();
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
  assert.deepEqual(s.errors, []);
});

test("an older save succeeding keeps a newer unanswered version as a candidate", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.fire("pagehide");
  const second = s.server.puts[1].body.updatedAt;
  // The older save lands, the newer copy lands with its answer lost, then the older answers.
  const first = s.server.landOldestWithoutReply();
  await s.server.cancelOldest(true);
  await first.answer();
  assert.equal(s.server.stored.updatedAt, second);
  await s.edit({ columns: ["third"] });
  await s.fire("pagehide");
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["third"] });
});

test("chat settings hold their keepalive reservation until the response body is read", async () => {
  const keepaliveBudget = freshBudget();
  let readBody: () => void = () => undefined;
  const api = loadWithStubs<{
    saveChatSettingsPatch: (p: object, o: { keepalive?: boolean }) => Promise<unknown>;
  }>(new URL("../src/features/chat/api/chat-settings-api.ts", import.meta.url), {
    "@/features/auth": {
      authFetch: async () =>
        ({
          ok: true,
          status: 200,
          json: () => new Promise((r) => (readBody = () => r({ settings: {} }))),
        }) as unknown as Response,
    },
    "@/lib/keepalive-budget": keepaliveBudget,
    "@/lib/format-fastapi-error": { formatApiErrorBody: () => null },
    "../utils/settings-retry": {
      ChatSettingsRequestError: Error,
      asBodyTransportFailure: (e: unknown) => e,
      isAbortOrTimeout: () => false,
    },
  });
  const saving = api.saveChatSettingsPatch({ note: "y".repeat(40 * 1024) }, { keepalive: true });
  await settle();
  assert.equal(keepaliveBudget.reserveKeepaliveBody("x".repeat(30 * 1024)), null);
  readBody();
  await saving;
  const after = keepaliveBudget.reserveKeepaliveBody("x".repeat(30 * 1024));
  assert.ok(after);
  after();
});

test("reopening waits for an exit save deferred behind a save in flight", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["left"] });
  s.editor.unmount();
  await settle();
  const reopened = s.open();
  await s.server.releaseOldest();
  await new Promise((r) => setTimeout(r, 5));
  await s.server.release();
  await new Promise((r) => setTimeout(r, 5));
  reopened.rerender();
  await settle();
  await s.edit({ columns: ["again"] });
  await s.fireDebounce();
  await s.server.release();
  assert.deepEqual(s.server.puts.map((p) => p.outcome), [200, 200, 200]);
  assert.deepEqual(s.server.stored.payload, { columns: ["again"] });
});

test("an edit left after another tab switched accounts is not sent with the new account", async () => {
  for (const leave of ["unmount", "pagehide"] as const) {
    const s = setup();
    await s.edit({ columns: ["alice"] });
    s.account.marker = "bob";
    if (leave === "pagehide") await s.fire("pagehide");
    else s.editor.unmount();
    await s.server.release();
    assert.equal(s.server.puts.length, 0, leave);
  }
});

test("an upload removed during a save is cleaned when it lands after the editor closed", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  s.uploads.pending.push("removed-mid-save");
  s.editor.unmount();
  s.uploads.pending.push("next-editor");
  await s.server.release();
  assert.deepEqual(s.uploads.removed, ["removed-mid-save"]);
});

test("an in-app exit after a save whose answer was lost still lands the newer edit", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  s.editor.unmount();
  await s.server.cancelOldest(true);
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
});

test("reopening after an exit save whose answer was lost reads the recipe again", async () => {
  const s = setup();
  await s.edit({ columns: ["left"] });
  s.editor.unmount();
  await settle();
  const reopened = s.open();
  await s.server.cancelOldest(true);
  await new Promise((r) => setTimeout(r, 5));
  reopened.rerender();
  await settle();
  await s.edit({ columns: ["again"] });
  await s.fireDebounce();
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["again"] });
});

test("an older save landing does not clean uploads while a newer save is unsettled", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.fire("pagehide");
  s.uploads.pending.push("in-newer");
  await s.server.releaseOldest();
  assert.equal(s.uploads.removed.length, 0);
  await s.server.release();
  assert.ok(s.uploads.removed.includes("in-newer"));
});

test("an older save answering last keeps a newer lost save's uncertainty", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.fire("pagehide");
  const first = s.server.landOldestWithoutReply();
  await s.server.cancelOldest(true);
  await first.answer();
  assert.equal(s.db.getCachedRecipe("r1"), null);
});

test("an upload a pending save still writes is not cleaned when a later save fails first", async () => {
  const s = setup();
  await s.edit({ columns: ["a"], ui: { unstructured_upload_uid: "u1" } });
  await s.fireDebounce();
  await s.edit({ columns: ["a"] });
  s.uploads.pending.push("u1");
  await s.fire("pagehide");
  s.server.failNext(503);
  await s.server.release("lifo");
  assert.deepEqual(s.server.puts.map((p) => p.outcome), [200, 503]);
  assert.ok(!s.uploads.removed.includes("u1"));
});

test("a pagehide into the back/forward cache waits for the save in flight", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.fire("pagehide", { persisted: true });
  assert.equal(s.server.puts.length, 1);
  await s.server.release("lifo");
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
  assert.deepEqual(s.errors, []);
});

test("recovery keeps every plausible unanswered version, not just the newest", async () => {
  const s = setup();
  await s.edit({ columns: ["x"] });
  await s.fireDebounce();
  await s.server.cancelOldest(true);
  await s.edit({ columns: ["y"] });
  await s.fireDebounce();
  await s.server.cancelOldest(false);
  await s.edit({ columns: ["z"] });
  await s.fire("pagehide");
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["z"] });
});

test("an unload save after the cache was evicted is sent without reading first", async () => {
  const s = setup();
  s.server.otherWindowSaves();
  await s.edit({ columns: ["mine"] });
  await s.fireDebounce();
  await s.server.release();
  const gets = s.server.gets;
  await s.edit({ columns: ["again"] });
  await s.fire("pagehide");
  assert.equal(s.server.gets, gets);
  assert.equal(s.server.puts.length, 2);
});

test("a transient HTTP failure keeps the recovery candidates", async () => {
  const s = setup();
  await s.edit({ columns: ["x"] });
  await s.fireDebounce();
  await s.server.cancelOldest(true);
  await s.edit({ columns: ["y"] });
  s.server.failNext(503);
  await s.hide();
  await s.server.release();
  await s.edit({ columns: ["z"] });
  await s.fire("pagehide");
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["z"] });
});

test("a committed save whose answer body was unreadable is not trusted from the cache", async () => {
  const s = setup();
  await s.edit({ columns: ["x"] });
  s.server.failNext(200);
  await s.fireDebounce();
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["x"] });
  assert.equal(s.db.getCachedRecipe("r1"), null);
});

test("an unload save also tries the confirmed base when the save in flight is cancelled unsent", async () => {
  const s = setup();
  await s.edit({ columns: ["first"] });
  await s.fireDebounce();
  await s.edit({ columns: ["second"] });
  await s.fire("pagehide");
  await s.server.cancelOldest(false);
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["second"] });
});

test("recovery keeps a landed version however many lost saves follow it", async () => {
  const s = setup();
  await s.edit({ columns: ["p"] });
  await s.fireDebounce();
  await s.server.cancelOldest(true);
  for (const v of ["q", "r", "t"]) {
    // Real edits are 800 ms apart; distinct milliseconds keep the lost versions distinct.
    await new Promise((r) => setTimeout(r, 3));
    await s.edit({ columns: [v] });
    await s.fireDebounce();
    await s.server.cancelOldest(false);
  }
  await s.edit({ columns: ["u"] });
  s.editor.unmount();
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["u"] });
});

test("returning to the saved content after a lost hidden-page save still sends it on exit", async () => {
  const s = setup();
  await s.edit({ columns: ["b"] });
  await s.hide();
  await s.server.cancelOldest(true);
  assert.deepEqual(s.server.stored.payload, { columns: ["b"] });
  await s.edit({ columns: ["a"] });
  s.editor.unmount();
  await s.server.release();
  assert.deepEqual(s.server.stored.payload, { columns: ["a"] });
});

test("a later save never reuses the version of an unanswered one", async () => {
  const s = setup({ updatedAt: Date.now() + 60_000 });
  await s.edit({ columns: ["x"] });
  await s.fireDebounce();
  await s.server.cancelOldest(true);
  await s.edit({ columns: ["y"] });
  await s.fireDebounce();
  const [lost, next] = s.server.puts;
  assert.ok(next.body.updatedAt > lost.body.updatedAt);
});
