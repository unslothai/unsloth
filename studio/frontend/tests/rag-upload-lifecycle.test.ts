// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { terminalJobStatus } from "../src/features/rag/types/rag.ts";
import type {
  IndexJob,
  JobEvent,
  RagDocument,
} from "../src/features/rag/types/rag.ts";
import type { RagDocumentScope } from "../src/features/rag/components/use-rag-documents.ts";

type ScopeOverride =
  | RagDocumentScope
  | Promise<RagDocumentScope | null>
  | (() => Promise<RagDocumentScope | null>);
type Hook = {
  documents: RagDocument[];
  uploading: boolean;
  hasIndexing: boolean;
  upload: (files: File[], scope?: ScopeOverride) => Promise<void>;
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

/** Drive renders of the shipped hook with controlled network responses. */
function harness(
  options: {
    filename?: string;
    events?: () => AsyncGenerator<JobEvent>;
    getJob?: () => Promise<IndexJob>;
  } = {},
) {
  const slots: unknown[] = [];
  const effects: Array<() => void> = [];
  let cursor = 0;
  const errors: string[] = [];
  const infos: string[] = [];
  const uploads: string[] = [];
  const uploaded = {
    documentId: "doc",
    jobId: "job",
    filename: options.filename ?? "report.pdf",
  };
  const lister = async () => [];
  let scope: RagDocumentScope | null = { type: "thread", threadId: "thread" };
  const react = {
    useRef(value: unknown) {
      const index = cursor++;
      slots[index] ??= { current: value };
      return slots[index];
    },
    useState(value: unknown) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = value;
      return [
        slots[index],
        (next: unknown) => {
          slots[index] = typeof next === "function" ? next(slots[index]) : next;
        },
      ];
    },
    useCallback(fn: unknown) {
      return fn;
    },
    useEffect(effect: () => void | (() => void), deps: unknown[]) {
      const index = cursor++;
      const previous = slots[index] as
        { deps: unknown[]; cleanup?: () => void } | undefined;
      if (previous && deps.every((dep, i) => Object.is(dep, previous.deps[i])))
        return;
      effects.push(() => {
        previous?.cleanup?.();
        slots[index] = { deps, cleanup: effect() };
      });
    },
  };
  const { useRagDocuments: runHook } = loadWithStubs<{
    useRagDocuments: (
      scope: RagDocumentScope | null,
      lister: () => Promise<RagDocument[]>,
    ) => Hook;
  }>(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    {
      react,
      "@/features/native-intents": {},
      "@/lib/toast": {
        toast: {
          error: (message: string) => errors.push(message),
          info: (message: string) => infos.push(message),
        },
      },
      "../types/rag": { terminalJobStatus },
      "../api/rag-availability": {
        useRagAvailabilityStore: {
          getState: () => ({ isUnavailable: () => false }),
        },
      },
      "./vision-overrides": { resolveVisionOverrides: async () => ({}) },
      "../api/rag-api": {
        uploadThreadDocument: async (threadId: string) => {
          uploads.push(threadId);
          return uploaded;
        },
        streamJobEvents:
          options.events ??
          async function* () {
            yield { type: "complete", num_chunks: 10 } as JobEvent;
          },
        getJob:
          options.getJob ??
          (async () => ({ status: "completed", numChunks: 10 })),
      },
    },
  );
  return {
    errors,
    infos,
    uploads,
    setScope(next: RagDocumentScope | null) {
      scope = next;
    },
    render() {
      cursor = 0;
      const result = runHook(scope, lister);
      for (const effect of effects.splice(0)) effect();
      return result;
    },
    dispose() {
      for (const slot of slots) (slot as { cleanup?: () => void })?.cleanup?.();
    },
  };
}

function report(filename = "report.pdf") {
  return new File(["report"], filename, { lastModified: 1 });
}

for (const extension of [
  "pdf",
  "txt",
  "md",
  "markdown",
  "docx",
  "html",
  "htm",
]) {
  const filename = `report.${extension}`;
  test(`reselecting an attached ${extension} skips chat creation and duplicate toasts`, async () => {
    const app = harness({ filename });
    try {
      let hook = app.render();
      await flush();
      await hook.upload([report(filename)]);
      await flush();
      hook = app.render();
      assert.equal(hook.documents[0]?.status, "completed");
      let initialized = 0;
      await hook.upload([report(filename)], async () => {
        initialized += 1;
        throw new Error("Thread was not persisted");
      });
      await flush();
      assert.equal(initialized, 0);
      assert.equal(app.uploads.length, 1);
      assert.deepEqual(app.errors, []);
      assert.deepEqual(app.infos, []);
      assert.equal(app.render().documents.length, 1);
    } finally {
      app.dispose();
    }
  });

  test(`chat initialization failure removes the ${extension} chip and reports once`, async () => {
    const app = harness({ filename });
    try {
      const hook = app.render();
      await flush();
      await hook.upload([report(filename)], async () => {
        throw new Error("Thread deleted");
      });
      assert.deepEqual(app.errors, ["Couldn't attach documents"]);
      assert.deepEqual(app.uploads, []);
      assert.deepEqual(app.render().documents, []);
      assert.equal(app.render().uploading, false);
    } finally {
      app.dispose();
    }
  });
}

test("failed scope resolution cannot upload into the previous chat", async () => {
  const app = harness();
  try {
    const hook = app.render();
    await flush();
    await hook.upload([report()], Promise.resolve(null));
    assert.deepEqual(app.uploads, []);
    assert.deepEqual(app.errors, ["Couldn't attach documents"]);
  } finally {
    app.dispose();
  }
});

test("an ended event stream keeps the file indexing until the job completes", async () => {
  const completion = deferred<IndexJob>();
  let reads = 0;
  const app = harness({
    events: async function* () {
      yield { type: "progress", progress: 0.4, stage: "captioning" };
    },
    getJob: async () => {
      reads += 1;
      return reads === 1
        ? { id: "job", documentId: "doc", status: "running", progress: 0.4 }
        : completion.promise;
    },
  });
  try {
    const hook = app.render();
    await flush();
    await hook.upload([report()]);
    await flush();
    assert.equal(app.render().documents[0]?.status, "running");
    assert.equal(app.render().hasIndexing, true);
    assert.equal(reads, 2);
    completion.resolve({
      id: "job",
      documentId: "doc",
      status: "completed",
      numChunks: 12,
    });
    await flush();
    assert.equal(app.render().documents[0]?.status, "completed");
    assert.equal(app.render().documents[0]?.numChunks, 12);
    assert.equal(app.render().hasIndexing, false);
  } finally {
    app.dispose();
  }
});

test("an original job failure is preserved after its event stream ends", async () => {
  let reads = 0;
  const app = harness({
    events: async function* () {},
    getJob: async () => ({
      id: "job",
      documentId: "doc",
      status: ++reads === 1 ? "running" : "failed",
      error: "Invalid PDF",
    }),
  });
  try {
    const hook = app.render();
    await flush();
    await hook.upload([report()]);
    await flush();
    assert.deepEqual(app.render().documents, []);
    assert.deepEqual(app.errors, ["Couldn't index report.pdf"]);
  } finally {
    app.dispose();
  }
});

test("an upload begun without a scope stops at the chat the user left", async () => {
  const app = harness();
  try {
    app.setScope(null);
    let hook = app.render();
    await flush();
    const materialized = deferred<RagDocumentScope>();
    const pending = hook.upload([report()], async () => materialized.promise);
    await flush();
    // The user picks an existing chat while the new one is still materializing.
    app.setScope({ type: "thread", threadId: "other" });
    hook = app.render();
    await flush();
    materialized.resolve({ type: "thread", threadId: "abandoned" });
    await pending;
    await flush();
    assert.deepEqual(app.uploads, [], "posted into a chat the user had left");
    assert.deepEqual(app.render().documents, []);
  } finally {
    app.dispose();
  }
});
