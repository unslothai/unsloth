// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  markdownSandboxLinkHref,
  sandboxFileForHref,
  sandboxFileForSrc,
} from "../src/components/assistant-ui/sandbox-files.ts";

const scope = { threadId: "thread1", projectId: null };

test("a bare file link resolves to the chat's sandbox route", () => {
  assert.equal(
    markdownSandboxLinkHref("outputs/report.csv", scope),
    "/api/inference/sandbox/thread1/outputs/report.csv",
  );
  assert.equal(
    markdownSandboxLinkHref("page.html", { threadId: "t", projectId: "p1" }),
    "/api/inference/sandbox/project-p1/page.html",
  );
});

test("a one-segment link with a file type, or a path under a folder, is a file", () => {
  for (const href of ["report.md", "data.json", "outputs/run.v2/table.parquet", "v1.2/report.pdf"]) {
    assert.equal(sandboxFileForHref(href), href, href);
  }
});

test("a route link keeps the session it records", () => {
  assert.equal(sandboxFileForHref("/api/inference/sandbox/other/plot.csv"), "plot.csv");
  assert.equal(
    markdownSandboxLinkHref("/api/inference/sandbox/other/plot.csv", scope),
    "/api/inference/sandbox/other/plot.csv",
  );
});

test("web links, anchors, words, app routes and escapes are not files", () => {
  for (const href of [
    "https://docs.unsloth.ai/page.html",
    "mailto:a@b.co",
    "#results.csv",
    "notes",
    "/assets/app.js",
    "../other/secret.txt",
    "outputs/%2e%2e/x.txt",
    "//evil.example/x.txt",
    "www.example.com",
    "docs.unsloth.ai/get-started",
    "example.tech",
    "docs.museum/report.pdf",
    "192.0.2.1/report.pdf",
    "пример.рф/report.pdf",
  ]) {
    assert.equal(sandboxFileForHref(href), null, href);
  }
});

test("image srcs still only resolve for inline image types", () => {
  assert.equal(sandboxFileForSrc("outputs/plot.png"), "outputs/plot.png");
  assert.equal(sandboxFileForSrc("outputs/report.csv"), null);
});
