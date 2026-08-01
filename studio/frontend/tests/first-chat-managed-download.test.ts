import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";

type Outcome = "started" | "cancelling" | "conflict" | "busy" | "error";
type Terminal = {
  onComplete(v: string | null): void;
  onCancelled(v: string | null): void;
  onError(v: string | null): void;
};
type State = { modelLoading: boolean; checkpoint: string };
type Shared = {
  state: State;
  waiters: Array<() => void>;
  loads: number;
  publishes: number;
  load: () => Promise<object>;
  publish: (loaded: object) => Promise<void>;
};

function loadHelper(): (deps: object) => Promise<boolean> {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  const start = source.indexOf(
    "export async function runFirstChatManagedDownload",
  );
  const body = source.indexOf("{", start);
  let depth = 0;
  let end = body;
  for (; end < source.length; end += 1) {
    if (source[end] === "{") depth += 1;
    if (source[end] === "}" && --depth === 0) break;
  }
  const fn = source.slice(start, end + 1).replace("export ", "");
  return new Function(
    `${stripTypeScriptTypes(fn, { mode: "transform" })}; return runFirstChatManagedDownload;`,
  )();
}

const runManaged = loadHelper();
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((ok, fail) => {
    resolve = ok;
    reject = fail;
  });
  return { promise, resolve, reject };
}
function shared(): Shared {
  const value: Shared = {
    state: { modelLoading: false, checkpoint: "" },
    waiters: [],
    loads: 0,
    publishes: 0,
    load: async () => ({ id: "loaded" }),
    publish: async () => {
      value.publishes += 1;
      value.state.checkpoint = "loaded";
    },
  };
  return value;
}
function caller(
  common: Shared,
  start: () => Promise<Outcome>,
  options: { tryAdoptServerActiveModel?: () => Promise<boolean> } = {},
) {
  let handlers: Terminal | undefined;
  let unsubscribed = 0;
  let dismissed = 0;
  const deps = {
    subscribe: (next: Terminal) => {
      handlers = next;
      return () => {
        unsubscribed += 1;
      };
    },
    isRequestedVariant: (variant: string | null) => variant === "q4",
    requestStart: start,
    readActivationState: () => common.state,
    tryAdoptServerActiveModel:
      options.tryAdoptServerActiveModel ?? (async () => false),
    waitForModelReady: () =>
      common.state.modelLoading
        ? new Promise<void>((resolve) => common.waiters.push(resolve))
        : Promise.resolve(),
    reserveActivation: () => {
      if (common.state.modelLoading) return false;
      common.state.modelLoading = true;
      return true;
    },
    loadModel: async () => {
      common.loads += 1;
      return common.load();
    },
    isExternalCheckpoint: (checkpoint: string) =>
      checkpoint.startsWith("external:"),
    publishLoadedModel: (loaded: object) => common.publish(loaded),
    releaseActivation: () => {
      common.state.modelLoading = false;
      common.waiters.splice(0).forEach((resolve) => resolve());
    },
    dismissToast: () => {
      dismissed += 1;
    },
  };
  return {
    deps,
    fire: () => handlers!,
    stats: () => ({ unsubscribed, dismissed }),
  };
}

test("terminal handoff accepts fast matches, ignores mismatches, and cleans non-starts", async () => {
  const fastShared = shared();
  let fast!: ReturnType<typeof caller>;
  fast = caller(fastShared, async () => {
    fast.fire().onComplete("q4");
    return "busy";
  });
  assert.equal(await runManaged(fast.deps), true);
  assert.deepEqual(fast.stats(), { unsubscribed: 1, dismissed: 0 });

  const gatedShared = shared();
  const start = deferred<Outcome>();
  const gated = caller(gatedShared, () => start.promise);
  const result = runManaged(gated.deps);
  gated.fire().onComplete("q8");
  start.resolve("started");
  await tick();
  assert.equal(gatedShared.loads, 0);
  gated.fire().onComplete("q4");
  assert.equal(await result, true);

  const stopped = caller(shared(), async () => "busy");
  assert.equal(await runManaged(stopped.deps), false);
  assert.deepEqual(stopped.stats(), { unsubscribed: 1, dismissed: 1 });
});

test("a successor waits out an existing cancellation and starts again", async () => {
  const owner = caller(shared(), async () => "started");
  const ownerResult = runManaged(owner.deps);
  await tick();
  owner.fire().onCancelled("q4");
  assert.equal(await ownerResult, false);

  const common = shared();
  let starts = 0;
  const successor = caller(common, async () => {
    starts += 1;
    return starts === 1 ? "cancelling" : "started";
  });
  const result = runManaged(successor.deps);
  await tick();
  successor.fire().onCancelled("q4");
  await tick();
  assert.equal(starts, 2);
  successor.fire().onComplete("q4");
  assert.equal(await result, true);
  assert.equal(common.loads, 1);
  assert.equal(successor.stats().unsubscribed, 2);
});

test("concurrent callers activate and publish once while loading stays reserved", async () => {
  const common = shared();
  const publishGate = deferred<void>();
  common.publish = async () => {
    common.publishes += 1;
    common.state.checkpoint = "loaded";
    await publishGate.promise;
  };
  const first = caller(common, async () => "started");
  const second = caller(common, async () => "started");
  let settled = 0;
  const a = runManaged(first.deps).finally(() => {
    settled += 1;
  });
  const b = runManaged(second.deps).finally(() => {
    settled += 1;
  });
  first.fire().onComplete("q4");
  second.fire().onComplete("q4");
  await tick();
  assert.equal(common.loads, 1);
  assert.equal(common.publishes, 1);
  assert.equal(common.state.modelLoading, true);
  assert.equal(settled, 0);
  publishGate.resolve();
  assert.deepEqual(await Promise.all([a, b]), [true, true]);
  assert.equal(common.loads, 1);
  assert.equal(first.stats().dismissed + second.stats().dismissed, 1);
});

test("a follower retries after the first activation fails", async () => {
  const common = shared();
  const firstLoad = deferred<object>();
  common.load = () =>
    common.loads === 1 ? firstLoad.promise : Promise.resolve({ id: "retry" });
  const first = caller(common, async () => "started");
  const second = caller(common, async () => "started");
  const a = runManaged(first.deps);
  first.fire().onComplete("q4");
  await tick();
  const b = runManaged(second.deps);
  second.fire().onComplete("q4");
  firstLoad.reject(new Error("load failed"));
  assert.deepEqual(await Promise.all([a, b]), [false, true]);
  assert.equal(common.loads, 2);
  assert.equal(common.state.modelLoading, false);
});

test("an external checkpoint wins without stale publication", async () => {
  const common = shared();
  const loadGate = deferred<object>();
  common.load = () => loadGate.promise;
  const current = caller(common, async () => "started");
  const result = runManaged(current.deps);
  current.fire().onComplete("q4");
  await tick();
  common.state.checkpoint = "external:user-model";
  loadGate.resolve({ id: "stale" });
  assert.equal(await result, true);
  assert.equal(common.state.checkpoint, "external:user-model");
  assert.equal(common.publishes, 0);
  assert.equal(common.state.modelLoading, false);
  assert.equal(current.stats().dismissed, 1);
});

test("a server model loaded during the download is adopted before activation", async () => {
  const common = shared();
  let adoptionChecks = 0;
  const current = caller(common, async () => "started", {
    tryAdoptServerActiveModel: async () => {
      adoptionChecks += 1;
      common.state.checkpoint = "server:model";
      return true;
    },
  });
  const result = runManaged(current.deps);
  current.fire().onComplete("q4");
  assert.equal(await result, true);
  assert.equal(adoptionChecks, 1);
  assert.equal(common.loads, 0);
  assert.equal(common.publishes, 0);
});
