// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type FolderSyncJob,
  type LinkedFolder,
  jobChangedSources,
  linkedFolderSourcesChanged,
} from "../src/features/rag/types/rag.ts";
import {
  MAX_FOLDER_FILES,
  filesFromFolderInput,
} from "../src/lib/dropped-folders.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const folder = (overrides: Partial<LinkedFolder> = {}): LinkedFolder => ({
  id: "f1",
  displayName: "Docs",
  scopeType: "project",
  scopeId: "p1",
  status: "idle",
  documentCount: 3,
  lastSyncedAt: "2026-10-10T00:00:00Z",
  lastChangedAt: "2026-10-09T00:00:00Z",
  ...overrides,
});

// Every 30 s pass bumps lastSyncedAt; reading that as a change refetched every document list.
test("a periodic pass that changed nothing is not a sources change", () => {
  const before = [folder()];
  const after = [folder({ lastSyncedAt: "2026-10-10T00:00:30Z" })];
  assert.equal(linkedFolderSourcesChanged(before, after), false);
  assert.equal(
    linkedFolderSourcesChanged(before, [
      folder({ lastChangedAt: "2026-10-10T00:00:30Z" }),
    ]),
    true,
  );
  assert.equal(
    linkedFolderSourcesChanged(before, [folder({ documentCount: 4 })]),
    true,
  );
});

test("an older backend without lastChangedAt still reports syncs as changes", () => {
  const legacy = (at: string) =>
    folder({ lastChangedAt: undefined, lastSyncedAt: at });
  assert.equal(linkedFolderSourcesChanged([legacy("a")], [legacy("b")]), true);
});

test("a finished job changed sources only if it added, removed or renamed something", () => {
  const job = (counts: Partial<FolderSyncJob>): FolderSyncJob => ({
    id: "j",
    linkedFolderId: "f1",
    mode: "sync",
    status: "completed",
    ...counts,
  });
  assert.equal(
    jobChangedSources(
      job({ indexedFiles: 0, removedFiles: 0, renamedFiles: 0 }),
    ),
    false,
  );
  assert.equal(
    jobChangedSources(job({ indexedFiles: 2, removedFiles: 0 })),
    true,
  );
  assert.equal(
    jobChangedSources(job({ indexedFiles: 0, renamedFiles: 1 })),
    true,
  );
  assert.equal(jobChangedSources(job({})), true);
});

function inFolder(path: string, content = "x"): File {
  const file = new File([content], path.split("/").pop() ?? path);
  Object.defineProperty(file, "webkitRelativePath", { value: path });
  return file;
}

test("a picked folder leaves out dependency, hidden and secret files", () => {
  const { files, truncated } = filesFromFolderInput([
    inFolder("repo/README.md"),
    inFolder("repo/src/app.ts"),
    inFolder("repo/node_modules/dep/index.js"),
    inFolder("repo/.git/config"),
    inFolder("repo/.venv/lib/site.py"),
    inFolder("repo/prod.env"),
    inFolder("repo/package-lock.json"),
    inFolder("repo/src/__init__.py", ""),
  ]);
  assert.deepEqual(
    files.map((file) => file.webkitRelativePath),
    ["repo/README.md", "repo/src/app.ts"],
  );
  assert.equal(truncated, 0);
});

// Unsupported files must not use up the cap, or a folder of images hides the documents after them.
test("only files the caller takes count toward the folder cap", () => {
  const images = Array.from({ length: MAX_FOLDER_FILES }, (_, i) =>
    inFolder(`mixed/${i}.png`),
  );
  const { files, truncated } = filesFromFolderInput(
    [...images, inFolder("mixed/notes.md")],
    (name) => name.endsWith(".md"),
  );
  assert.deepEqual(
    files.map((file) => file.name),
    ["notes.md"],
  );
  assert.equal(truncated, 0);
});

test("a picked folder past the cap keeps the first files and counts the rest", () => {
  const many = Array.from({ length: MAX_FOLDER_FILES + 5 }, (_, i) =>
    inFolder(`big/${i}.md`),
  );
  const { files, truncated } = filesFromFolderInput(many);
  assert.equal(files.length, MAX_FOLDER_FILES);
  assert.equal(truncated, 5);
});

/** Drives useUploadQueue renders with a stub React whose effects run after each render. */
function queueHarness() {
  const slots: unknown[] = [];
  const effects: Array<() => void> = [];
  const toasts: string[] = [];
  let cursor = 0;
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
          slots[index] = next;
        },
      ];
    },
    useCallback: (fn: unknown) => fn,
    useEffect(effect: () => void, deps: unknown[]) {
      const index = cursor++;
      const previous = slots[index] as unknown[] | undefined;
      if (previous && deps.every((dep, i) => Object.is(dep, previous[i])))
        return;
      slots[index] = deps;
      effects.push(effect);
    },
  };
  const { useUploadQueue } = loadWithStubs<{
    useUploadQueue: (
      run: (items: string[]) => void,
      busy: boolean,
      key: string | null,
    ) => { enqueue: (items: string[]) => void; queued: number };
  }>(
    new URL(
      "../src/features/rag/components/use-upload-queue.ts",
      import.meta.url,
    ),
    {
      react,
      "@/lib/toast": {
        toast: { info: (message: string) => toasts.push(message) },
      },
    },
  );
  const ran: string[][] = [];
  return {
    ran,
    toasts,
    render(busy: boolean, key: string | null) {
      cursor = 0;
      const result = useUploadQueue((items) => ran.push(items), busy, key);
      for (const effect of effects.splice(0)) effect();
      return result;
    },
  };
}

test("files added during an upload wait for it, then go in one batch", () => {
  const app = queueHarness();
  app.render(false, "p1").enqueue(["a"]);
  assert.deepEqual(app.ran, [["a"]]);

  let hook = app.render(true, "p1");
  hook.enqueue(["b"]);
  hook = app.render(true, "p1");
  hook.enqueue(["c"]);
  assert.deepEqual(
    app.ran,
    [["a"]],
    "nothing starts while the first upload runs",
  );
  assert.equal(app.render(true, "p1").queued, 2);

  app.render(false, "p1");
  assert.deepEqual(app.ran, [["a"], ["b", "c"]]);
  assert.equal(app.render(false, "p1").queued, 0);
});

test("a drop that resolves after a switch is not added to the new destination", () => {
  const app = queueHarness();
  const stale = app.render(false, "p1").enqueue;
  app.render(false, "p2");
  stale(["a"]);
  assert.deepEqual(app.ran, []);
  assert.ok(app.toasts.some((message) => message.includes("not added")));
  app.render(false, "p2").enqueue(["b"]);
  assert.deepEqual(app.ran, [["b"]]);
});

test("files queued for one project are not added to the next", () => {
  const app = queueHarness();
  app.render(true, "p1").enqueue(["b"]);
  app.render(true, "p2");
  app.render(false, "p2");
  assert.deepEqual(app.ran, []);
  assert.ok(app.toasts.some((message) => message.includes("not added")));
});
