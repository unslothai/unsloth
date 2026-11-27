// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  CHANGE_STATUS_LABELS,
  type DirState,
  addToChatText,
  changeStatusTone,
  closeSubtree,
  flattenFileTree,
  joinProjectPath,
  pathsToLoad,
  splitProjectPath,
  toggleOpenPath,
} from "../src/features/chat/utils/project-files.ts";
import { readSrc } from "./helpers/kit.ts";

test("paths split into a dimmed folder and an emphasised name", () => {
  assert.deepEqual(splitProjectPath("src/app/main.py"), {
    dir: "src/app",
    name: "main.py",
  });
  assert.deepEqual(splitProjectPath("main.py"), { dir: "", name: "main.py" });
  assert.deepEqual(splitProjectPath("docs/"), { dir: "", name: "docs" });
  assert.equal(joinProjectPath("", "a"), "a");
  assert.equal(joinProjectPath("a/b", "c"), "a/b/c");
  assert.equal(addToChatText("src/app/main.py"), "File: src/app/main.py");
});

test("every change status has a label and a tone", () => {
  assert.deepEqual(Object.values(CHANGE_STATUS_LABELS), [
    "Modified",
    "Added",
    "Deleted",
    "Renamed",
    "Untracked",
    "Conflicted",
  ]);
  assert.equal(changeStatusTone("modified"), "amber");
  assert.equal(changeStatusTone("added"), "green");
  assert.equal(changeStatusTone("untracked"), "green");
  assert.equal(changeStatusTone("deleted"), "red");
  assert.equal(changeStatusTone("conflicted"), "red");
  assert.equal(changeStatusTone("renamed"), "neutral");
});

function ready(
  entries: { name: string; kind: "dir" | "file"; size?: number }[],
  truncated = false,
): DirState {
  return {
    status: "ready",
    listing: {
      root: "~/code/app",
      path: "",
      chosen: true,
      truncated,
      entries: entries.map((e) => ({ hidden: false, ...e })),
    },
  };
}

test("opening and closing folders, with a closed folder's descendants closed too", () => {
  let open = toggleOpenPath(new Set(), "src");
  open = toggleOpenPath(open, "src/app");
  open = toggleOpenPath(open, "docs");
  assert.deepEqual([...open], ["src", "src/app", "docs"]);
  assert.deepEqual([...closeSubtree(open, "src")], ["docs"]);
  assert.deepEqual([...toggleOpenPath(open, "docs")], ["src", "src/app"]);
  assert.deepEqual(
    [...closeSubtree(new Set(["src", "src2"]), "src")],
    ["src2"],
  );
});

test("only folders never requested are loaded, the root first", () => {
  assert.deepEqual(pathsToLoad({}, new Set()), [""]);
  const dirs = { "": ready([]), src: { status: "loading" } as DirState };
  assert.deepEqual(pathsToLoad(dirs, new Set(["src", "docs"])), ["docs"]);
});

test("folders named like Object.prototype keys load and render", () => {
  const names = ["constructor", "toString", "__proto__"];
  const dirs: Record<string, DirState> = {
    "": ready(names.map((name) => ({ name, kind: "dir" as const }))),
  };
  const open = new Set(names);
  assert.deepEqual(pathsToLoad(dirs, open), names);
  assert.deepEqual(
    flattenFileTree(dirs, open).map((r) => [r.type, r.path]),
    names.flatMap((name) => [
      ["entry", name],
      ["loading", name],
    ]),
  );
});

test("the tree flattens depth first with status rows under open folders", () => {
  const dirs: Record<string, DirState> = {
    "": ready(
      [
        { name: "src", kind: "dir" },
        { name: "empty", kind: "dir" },
        { name: "broken", kind: "dir" },
        { name: "slow", kind: "dir" },
        { name: "README.md", kind: "file", size: 10 },
      ],
      true,
    ),
    src: ready([{ name: "main.py", kind: "file", size: 5 }]),
    empty: ready([]),
    broken: { status: "error", message: "Folder not found" },
    slow: { status: "loading" },
  };
  const rows = flattenFileTree(
    dirs,
    new Set(["src", "empty", "broken", "slow"]),
  );
  assert.deepEqual(
    rows.map((r) => [r.type, r.path, r.depth]),
    [
      ["entry", "src", 0],
      ["entry", "src/main.py", 1],
      ["entry", "empty", 0],
      ["empty", "empty", 1],
      ["entry", "broken", 0],
      ["error", "broken", 1],
      ["entry", "slow", 0],
      ["loading", "slow", 1],
      ["entry", "README.md", 0],
      ["truncated", "", 0],
    ],
  );
  const truncated = rows.at(-1);
  assert.equal(truncated?.type === "truncated" && truncated.count, 5);

  const closed = flattenFileTree(dirs, new Set());
  assert.deepEqual(
    closed.map((r) => (r.type === "entry" ? [r.path, r.open] : r.type)),
    [
      ["src", false],
      ["empty", false],
      ["broken", false],
      ["slow", false],
      ["README.md", false],
      "truncated",
    ],
  );
  assert.deepEqual(flattenFileTree({}, new Set()), [
    { type: "loading", path: "", depth: 0 },
  ]);
});

test("the Files tab is a third project tab that loads its panel lazily", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /useState<"chats" \| "sources" \| "files">/);
  assert.match(page, /onClick=\{\(\) => setProjectTab\("files"\)\}/);
  assert.match(page, /const ProjectFilesPanel = lazy\(/);
  assert.match(page, /<ProjectFilesPanel\n\s*key=\{`\$\{currentProject\?\.workingFolder \?\? \"\"\}\|\$\{currentProject\?\.workingFolderStatus \?\? \"\"\}`\}\n\s*projectId=\{projectId\}\n\s*\/>/);
});

test("Add to chat stages the path in the project composer through the staged-prompt store", () => {
  const panel = readSrc("features/chat/components/project-files-panel.tsx");
  assert.match(panel, /stageFixPrompt\(addToChatText\(path\)\)/);
  assert.match(panel, /toast\.success\("Added to the message"\)/);
  assert.match(panel, /toast\.success\("Path copied"\)/);
  assert.match(panel, /aria-expanded=\{row\.open\}/);
  assert.match(panel, /controller\.abort\(\)/);
  assert.doesNotMatch(panel, /setProjectTab/);
});

test("Open Folder sends the lease as the working folder and asks to reuse a project", () => {
  const openFolder = readSrc("features/chat/utils/open-folder-as-project.ts");
  const api = readSrc("features/chat/api/chat-api.ts");
  assert.match(
    openFolder,
    /selected\.token,\s*undefined,\s*\{ reuseExisting: true \}/,
  );
  assert.doesNotMatch(openFolder, /createLinkedFolder/);
  assert.match(
    api,
    /nativePathLease && options\.reuseExisting \? \{ reuseExisting: true \} : \{\}/,
  );
  const dialog = readSrc("features/chat/components/new-project-dialog.tsx");
  assert.match(
    dialog,
    /createChatProject\(trimmed, token \?\? undefined, projectId\.current\)/,
  );
});
