// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  SANDBOX_INLINE_IMAGE_EXTS,
  markdownSandboxImageSrc,
  sandboxFileForSrc,
} from "../src/components/assistant-ui/sandbox-files.ts";
import { safeMarkdownUrl } from "../src/lib/safe-markdown-url.ts";

/*
 * THREE PIECES ONLY WORK TOGETHER, and nothing in the type system joins them:
 *
 *   1. the renderer has to resolve a scheme-less sandbox `src` through `markdownSandboxImageSrc`
 *      before it reaches the DOM (the sanitizer keeps what carries no scheme -- see
 *      `safe-markdown-url.ts` -- but keeping it was never the same as being able to fetch it);
 *   2. the fetch has to carry the Authorization header, which is the only way the route authenticates
 *      a header-auth client (`_authenticate_header_or_query`), and revoke what it created;
 *   3. the extension set has to agree with the backend's `_SANDBOX_MEDIA_TYPES`, which decides
 *      inline-image from attachment in a different language, two directories away.
 *
 * Each is checked against the real source, and where a check is a string search it says what it is
 * defending, so a rename that breaks the join fails here by name rather than by measuring as a
 * change that does nothing.
 */

const read = (relative: string): string =>
  readFileSync(new URL(relative, import.meta.url), "utf8");

const MARKDOWN_TEXT = read("../src/components/assistant-ui/markdown-text.tsx");
const HOOK = read("../src/components/assistant-ui/use-sandbox-image.ts");
const TOOL_UI = read("../src/components/assistant-ui/tool-ui-python.tsx");
const BACKEND = read(
  "../../backend/routes/inference.py",
);

test("a scheme-less sandbox src is rewritten before it reaches the DOM, and rendered from a blob:", () => {
  // PRECONDITION: the renderer still relies on the sanitizer's deny rule rather than repeating it.
  assert.ok(
    MARKDOWN_TEXT.includes("urlTransform={safeMarkdownUrl}"),
    "PRECONDITION: the chat renderer still passes the url transform",
  );
  assert.ok(
    /img: MarkdownImage,/.test(MARKDOWN_TEXT),
    "registering `img` replaces Streamdown's own renderer wholesale, so the swap has to live in the " +
      "component it replaces -- an unregistered <img src> is the bug this file exists for",
  );
  assert.ok(
    /markdownSandboxImageSrc\(src,\s*\{/.test(MARKDOWN_TEXT),
    "a src that RECORDS a session keeps it -- the folder its files were written to is where they " +
      "still are after a move, and it is what the tool card above the prose already resolves from; " +
      "only one that records nothing falls back to this chat's scope",
  );
  assert.ok(
    MARKDOWN_TEXT.includes("useSandboxImage(file)"),
    "and it is fetched rather than rendered raw",
  );

  // ANTI-VACUITY: the rewrite really is what produces the src, and a data: URI really is untouched.
  const imgNode = { tagName: "img" } as Parameters<typeof safeMarkdownUrl>[2];
  const written = "/api/inference/sandbox/__LOCALID_Y3VK67e/plot.png";
  assert.equal(safeMarkdownUrl(written, "src", imgNode), written);
  // The recorded session wins (it is where the file was WRITTEN); a bare path records nothing and
  // only then falls back to this chat's scope.
  assert.equal(
    markdownSandboxImageSrc(written, { threadId: "t-1", projectId: null }),
    "/api/inference/sandbox/__LOCALID_Y3VK67e/plot.png",
  );
  assert.equal(
    markdownSandboxImageSrc("plot.png", { threadId: "t-1", projectId: null }),
    "/api/inference/sandbox/t-1/plot.png",
  );
  assert.equal(markdownSandboxImageSrc("data:image/png;base64,AAAA", { threadId: "t-1", projectId: null }), null);
});

test("the fetch carries the header and gives the object URL back on cleanup", () => {
  // The route answers on the Authorization header; a bare <img src> gets a 401 and the renderer's
  // "Image not available" placeholder, which is what this whole file defends.
  assert.ok(/authFetch\(url,\s*\{ signal: controller\.signal \}\)/.test(HOOK), "authenticated fetch");
  assert.ok(HOOK.includes("URL.createObjectURL(blob)"), "into an object URL");
  // Two assertions, because a comment sits between the two statements in the hook's cleanup.
  assert.ok(HOOK.includes("controller.abort();"), "the in-flight fetch is cancelled");
  assert.ok(
    HOOK.includes("if (objectUrl) URL.revokeObjectURL(objectUrl);"),
    "and an object URL a cancelled fetch had already built is revoked, or those bytes stay pinned " +
      "for the rest of the session",
  );

  // ONE hook, not two. `tool-ui-python.tsx` was the only place this fetch existed; duplicating it is
  // how the markdown path ended up without one in the first place.
  assert.ok(TOOL_UI.includes("useSandboxImage(pythonToolImagePath(sessionId, filename))"));
  assert.equal(
    TOOL_UI.includes("createObjectURL"),
    false,
    "the Python card now goes through the shared hook instead of keeping its own copy",
  );
});

test("the inline-image set matches what the backend actually serves inline", () => {
  // Two lists in two languages. The frontend list decides what becomes an <img>; the backend map
  // decides what the route serves inline at all, and everything else comes back as a download card.
  const served = BACKEND.slice(
    BACKEND.indexOf("_SANDBOX_MEDIA_TYPES = {"),
    BACKEND.indexOf("_SANDBOX_MEDIA_TYPES = {") >= 0
      ? BACKEND.indexOf("}", BACKEND.indexOf("_SANDBOX_MEDIA_TYPES = {"))
      : 0,
  );
  assert.ok(served.length > 50, "PRECONDITION: the backend map was actually read");
  for (const ext of SANDBOX_INLINE_IMAGE_EXTS) {
    assert.ok(
      served.includes(`"${ext}"`),
      `${ext} is rendered as an <img> but the route would serve it as application/octet-stream`,
    );
  }
  assert.ok(
    !served.includes('".svg"'),
    "SVG stays out on purpose: the filename is model-chosen, so inline SVG would be same-origin " +
      "script execution",
  );
  assert.ok(
    /X-Content-Type-Options"\s*=\s*"nosniff/.test(BACKEND) ||
      BACKEND.includes('"X-Content-Type-Options": "nosniff"'),
    "the pin that makes a raster codec safe to add",
  );
});

test("a non-image stays a download card, and a scheme-carrying src is left alone", () => {
  assert.equal(sandboxFileForSrc("report.csv"), null);
  assert.equal(sandboxFileForSrc("diagram.svg"), null);
  assert.equal(sandboxFileForSrc("/assets/logo.png"), null);
  assert.equal(sandboxFileForSrc("//img.example.com/x.png"), null);
  // The sid segment and the `?session=` form both go in the bin: the caller re-derives the session.
  assert.equal(
    sandboxFileForSrc("/api/inference/sandbox/_/loss%20curve%20%231.png?session=session%2Fid"),
    "loss curve #1.png",
  );
});

test("the img restatement keeps what the wholesale replacement silently dropped", () => {
  // Registering `img` replaces Streamdown's renderer WHOLESALE, so some of its lines live here by name.
  // The LOADED/SIZED gating machinery deliberately does NOT: a sandbox image is hidden until the authed
  // fetch lands and fetch state already drives visibility, so nothing here needs a decode gate. What IS
  // kept is what carries no machinery -- string checks, because that renderer is minified.
  assert.ok(
    MARKDOWN_TEXT.includes('data-streamdown="image"'),
    "the <img> itself carries the attribute (wrapper and fallback had it; img silently did not)",
  );
  assert.ok(
    MARKDOWN_TEXT.includes("bg-black/10") &&
      MARKDOWN_TEXT.includes("pointer-events-none absolute inset-0"),
    "the hover overlay div, dropped entirely by the wholesale replacement",
  );
  assert.ok(
    MARKDOWN_TEXT.includes('.replace(/\\.[^/.]+$/'),
    'download names: a real extension on the path wins whole; otherwise alt\'s extension is stripped ' +
      'and one inferred from blob.type is appended -- alt="plot" downloads "plot.png", not "plot"',
  );
  assert.ok(
    MARKDOWN_TEXT.includes("decodeSegment((file ?? src"),
    "the tail is cut with raw delimiters split off FIRST, then decoded: `loss curve #1.png` must " +
      "save under its real name, not as loss%20curve%20%231.png",
  );
});
