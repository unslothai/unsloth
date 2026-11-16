// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  collapseCrumbs,
  filterEntries,
  folderName,
  isPathLike,
  isWindowsPath,
  joinChild,
  readRecentFolders,
  rememberRecentFolder,
  resolveTypedPath,
  splitBreadcrumb,
  splitTypedPath,
} from "../src/features/model-picker/components/model-selector/folder-browser-path.ts";
import { readSrc } from "./helpers/kit.ts";

test("plain text narrows the list; anything naming a place is a path", () => {
  for (const text of ["mod", "my models", "a.b", ""]) {
    assert.equal(isPathLike(text), false, text);
  }
  for (const text of [
    "~",
    "~/code",
    "/srv",
    "C:\\models",
    "D:",
    "src/app",
    "..",
  ]) {
    assert.equal(isPathLike(text), true, text);
  }
});

test("a backslash separates relative paths only on a Windows server", () => {
  for (const text of ["src\\app", ".\\models", "..\\models"]) {
    assert.equal(isPathLike(text, true), true, text);
    assert.equal(isPathLike(text, false), false, text);
  }
  assert.equal(isWindowsPath("C:\\Users\\me"), true);
  assert.equal(isWindowsPath("\\\\host\\share"), true);
  assert.equal(isWindowsPath("/home/me"), false);
  assert.equal(isWindowsPath(undefined), false);
});

test("a typed path is absolute, home-relative, or read against the open folder", () => {
  assert.equal(resolveTypedPath("/home/me", "~/code"), "~/code");
  assert.equal(resolveTypedPath("/home/me", "/srv/models"), "/srv/models");
  assert.equal(resolveTypedPath("/home/me", "C:\\models"), "C:\\models");
  assert.equal(resolveTypedPath("/home/me", "code/app"), "/home/me/code/app");
  assert.equal(resolveTypedPath("/home/me", ".."), "/home/me/..");
  assert.equal(resolveTypedPath("C:\\", "models/x"), "C:\\models/x");
});

test("a half-typed last name splits into its parent and a filter", () => {
  assert.deepEqual(splitTypedPath("~/code/uns"), {
    parent: "~/code",
    leaf: "uns",
  });
  assert.deepEqual(splitTypedPath("/us"), { parent: "/", leaf: "us" });
  assert.deepEqual(splitTypedPath("C:/us"), { parent: "C:/", leaf: "us" });
  assert.deepEqual(splitTypedPath("C:\\Users\\m"), {
    parent: "C:\\Users",
    leaf: "m",
  });
  assert.deepEqual(splitTypedPath("~/code/"), { parent: "~", leaf: "code" });
  assert.equal(splitTypedPath("code"), null);
  assert.equal(splitTypedPath("/"), null);
});

test("filtering ranks names that start with the query ahead of names that contain it", () => {
  const entries = ["my-models", "Models", "tmp", "old models"].map((name) => ({
    name,
  }));
  assert.deepEqual(
    filterEntries(entries, "mod").map((entry) => entry.name),
    ["Models", "my-models", "old models"],
  );
  assert.equal(filterEntries(entries, "  "), entries);
  assert.deepEqual(filterEntries(entries, "zzz"), []);
});

test("children join onto either separator style without doubling it", () => {
  assert.equal(joinChild("/home/me", "code"), "/home/me/code");
  assert.equal(joinChild("/", "srv"), "/srv");
  assert.equal(joinChild("C:\\", "models"), "C:\\models");
});

test("breadcrumbs cover POSIX and drive paths", () => {
  assert.deepEqual(
    splitBreadcrumb("/home/me").map((crumb) => crumb.value),
    ["/home", "/home/me"],
  );
  assert.deepEqual(splitBreadcrumb("/"), [{ label: "/", value: "/" }]);
  assert.deepEqual(
    splitBreadcrumb("C:\\Users\\me").map((crumb) => crumb.value),
    ["C:/", "C:/Users", "C:/Users/me"],
  );
  assert.deepEqual(splitBreadcrumb(""), []);
});

test("the browser keeps the last good folder on screen when a path fails", async () => {
  const source = readSrc(
    "features/model-picker/components/model-selector/folder-browser.tsx",
  );
  assert.doesNotMatch(source, /!error && data/);
  assert.match(source, /disabled=\{!current \|\| loading\}/);
  assert.match(
    source,
    /browseFolders\(target, hidden, ctrl\.signal, showModelHints\)/,
  );
  assert.match(source, /role="listbox"/);
});

test("a deep trail folds its start behind one ellipsis", () => {
  const crumbs = splitBreadcrumb("/home/me/code/app/src");
  const { hidden, shown } = collapseCrumbs(crumbs);
  assert.deepEqual(
    shown.map((crumb) => crumb.label),
    ["code", "app", "src"],
  );
  assert.equal(hidden[hidden.length - 1].value, "/home/me");
  const short = splitBreadcrumb("/home/me/code");
  assert.deepEqual(collapseCrumbs(short), { hidden: [], shown: short });
});

test("recent folders keep the newest five without repeats", () => {
  const data = new Map<string, string>();
  const store = {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
  };
  assert.deepEqual(readRecentFolders(store), []);
  for (const name of ["a", "b", "c", "d", "e", "f", "b"]) {
    rememberRecentFolder(store, `/x/${name}`);
  }
  assert.deepEqual(readRecentFolders(store), [
    "/x/b",
    "/x/f",
    "/x/e",
    "/x/d",
    "/x/c",
  ]);
  data.set([...data.keys()][0], "not json");
  assert.deepEqual(readRecentFolders(store), []);
  assert.deepEqual(readRecentFolders(undefined), []);
});

test("a folder is named by its last segment", () => {
  assert.equal(folderName("/home/me/code/"), "code");
  assert.equal(folderName("C:\\Users\\me"), "me");
  assert.equal(folderName("/"), "/");
});

test("a navigation during New folder wins over the create response", () => {
  const source = readSrc(
    "features/model-picker/components/model-selector/folder-browser.tsx",
  );
  const create = source.slice(source.indexOf("async function createFolder()"));
  const claim = create.indexOf("abortRef.current = ctrl;");
  const request = create.indexOf("await createBrowseFolder(");
  const guard = create.indexOf("if (ctrl.signal.aborted) return;");
  assert.ok(claim > 0 && claim < request && request < guard);
  assert.ok(guard < create.indexOf("setData(created)"));
});

test("a typed path outside the browsable places can still be added where the caller registers it", () => {
  const source = readSrc(
    "features/model-picker/components/model-selector/folder-browser.tsx",
  );
  assert.match(
    source,
    /acceptUnbrowsable &&\s*opts\?\.typed &&\s*target &&\s*errorStatus\(err\) === 403/,
  );
  assert.match(source, /onSelect\(unbrowsable\);\n\s*onOpenChange\(false\);/);
  const composer = readSrc(
    "components/assistant-ui/composer-model-picker/custom-folders.tsx",
  );
  assert.match(composer, /<FolderBrowser[^>]*acceptUnbrowsable=\{true\}/);
});

test("the browser's Enter shortcuts never reach a parent dialog's save", () => {
  const source = readSrc(
    "features/model-picker/components/model-selector/folder-browser.tsx",
  );
  const keys = source.slice(source.indexOf("function onInputKeyDown("));
  const enter = keys.slice(keys.indexOf('if (event.key === "Enter") {'));
  assert.ok(enter.indexOf("event.stopPropagation();") < enter.indexOf("handleConfirm();"));
  assert.match(source, /onKeyDown=\{\(event\) => \{\n\s*if \(event\.key === "Enter"\) event\.stopPropagation\(\);/);
});
