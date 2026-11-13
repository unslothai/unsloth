// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildConversationHtmlBody,
  buildConversationHtmlDocument,
  markdownToHtml,
} from "../src/features/chat/utils/conversation-html.ts";

test("renders markdown with role sections", () => {
  const body = buildConversationHtmlBody([
    {
      title: "Chat",
      messages: [
        { role: "user", content: "What is 2+2?" },
        { role: "assistant", content: "**4**\n\n```python\nprint(4)\n```" },
        { role: "assistant", content: "   " },
      ],
    },
  ]);
  assert.match(body, /<section class="message user">/);
  assert.match(body, /<h3 class="role">User<\/h3>/);
  assert.match(body, /<strong>4<\/strong>/);
  assert.match(body, /<pre><code class="language-python">print\(4\)/);
  assert.equal(body.match(/<section/g)?.length, 2);
  assert.doesNotMatch(body, /conversation-title/);
});

test("user messages keep their text as typed", () => {
  const body = buildConversationHtmlBody([
    {
      title: "Chat",
      messages: [
        {
          role: "user",
          content:
            'class A:\n    def __init__(self, *args):\n        self.x = args[0] * 2\n# TODO remove this line\n<b>a,b</b>\n1,2',
        },
      ],
    },
  ]);
  assert.match(
    body,
    /<div class="content">class A:\n {4}def __init__\(self, \*args\):\n {8}self\.x = args\[0\] \* 2\n# TODO remove this line\n&lt;b&gt;a,b&lt;\/b&gt;\n1,2<\/div>/,
  );
  assert.doesNotMatch(body, /<strong>|<em>|<h1>|<p>|<b>/);
});

test("raw HTML from a model is shown as text, except the thinking wrapper", () => {
  const html = markdownToHtml(
    '<script>alert(1)</script>\n\n<img src=x onerror="alert(1)">\n\n<details>\n<summary>thinking</summary>\n\nhmm\n\n</details>',
  );
  assert.doesNotMatch(html, /<script|<img/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /<details>/);
  assert.match(html, /<summary>thinking<\/summary>/);
});

test("only safe links and inline data images survive", () => {
  const html = markdownToHtml(
    "[ok](https://example.com/a?b=1&c=2) [bad](javascript:alert(1)) ![remote](https://example.com/x.png) ![inline](data:image/png;base64,AAAA)",
  );
  assert.match(
    html,
    /<a href="https:\/\/example.com\/a\?b=1&amp;c=2" rel="noopener noreferrer">ok<\/a>/,
  );
  assert.doesNotMatch(html, /javascript:/);
  assert.match(html, /bad/);
  assert.doesNotMatch(html, /<img src="https/);
  assert.match(html, /\[remote\]/);
  assert.match(html, /<img src="data:image\/png;base64,AAAA" alt="inline">/);
});

test("several conversations get their own titles; empty ones are dropped", () => {
  const body = buildConversationHtmlBody([
    { title: "Model A <b>", messages: [{ role: "user", content: "hi" }] },
    { title: "Model B", messages: [{ role: "user", content: "hello" }] },
    { title: "Empty", messages: [] },
  ]);
  assert.match(body, /<h2 class="conversation-title">Model A &lt;b&gt;<\/h2>/);
  assert.match(body, /Model B/);
  assert.doesNotMatch(body, /Empty/);
  assert.equal(buildConversationHtmlBody([{ title: "x", messages: [] }]), "");
});

test("the document is standalone and locks out scripts and remote loads", () => {
  const doc = buildConversationHtmlDocument("A </title><script>", "<p>x</p>");
  assert.match(doc, /^<!doctype html>/);
  assert.match(
    doc,
    /Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"/,
  );
  assert.match(doc, /<title>A &lt;\/title&gt;&lt;script&gt;<\/title>/);
  assert.match(doc, /<p>x<\/p>/);
});
