// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The folder browser's one text box both narrows the open folder and takes a typed path.

import assert from "node:assert/strict";
import test from "node:test";

import {
  filterEntries,
  isPathLike,
  joinChild,
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
    ["/", "/home", "/home/me"],
  );
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
  // The listing is not gated on the error, and the confirm button no longer is either.
  assert.doesNotMatch(source, /!error && data/);
  assert.match(source, /disabled=\{!current \|\| loading\}/);
  assert.match(
    source,
    /browseFolders\(target, hidden, ctrl\.signal, showModelHints\)/,
  );
  assert.match(source, /role="listbox"/);
});
