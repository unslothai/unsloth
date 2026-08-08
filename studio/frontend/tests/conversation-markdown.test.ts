// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildConversationMarkdown,
  contentBlocksToMarkdownBlocks,
  renderConversationBlocks,
} from "../src/features/chat/utils/conversation-markdown.ts";

test("exports a readable markdown transcript in conversation order", () => {
  assert.equal(
    buildConversationMarkdown([
      { role: "system", content: "Be concise." },
      { role: "user", content: "Explain `RED → GREEN`." },
      { role: "assistant", content: "1. Write a failing test.\n2. Fix it." },
    ]),
    [
      "## System",
      "",
      "Be concise.",
      "",
      "## User",
      "",
      "Explain `RED → GREEN`.",
      "",
      "## Assistant",
      "",
      "1. Write a failing test.\n2. Fix it.",
      "",
    ].join("\n"),
  );
});

test("omits empty messages without rewriting markdown content", () => {
  assert.equal(
    buildConversationMarkdown([
      { role: "user", content: "  " },
      { role: "assistant", content: "# Existing heading\n\n> quote" },
    ]),
    "## Assistant\n\n# Existing heading\n\n> quote\n",
  );
});

test("keeps an unknown role label and returns empty output for empty content", () => {
  assert.equal(
    buildConversationMarkdown([{ role: "tool", content: "result" }]),
    "## Tool\n\nresult\n",
  );
  assert.equal(
    buildConversationMarkdown([{ role: "user", content: "\n\t" }]),
    "",
  );
});

test("labels a missing role as a generic message", () => {
  assert.equal(
    buildConversationMarkdown([{ role: "", content: "orphaned content" }]),
    "## Message\n\norphaned content\n",
  );
});

test("renders a multi-line arg as code instead of an escaped json string", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "tool-call",
        name: "render_html",
        args: {
          code: "<!DOCTYPE html>\n<html>\n  <body>hi</body>\n</html>",
          title: "Canvas",
        },
        result: "Rendered HTML canvas: Canvas.",
      },
    ]),
    [
      "**tool call:** `render_html`",
      "",
      "**code:**",
      "",
      "```",
      "<!DOCTYPE html>",
      "<html>",
      "  <body>hi</body>",
      "</html>",
      "```",
      "",
      "**title:** `Canvas`",
      "",
      "**result:** `Rendered HTML canvas: Canvas.`",
    ].join("\n"),
  );
});

test("keeps single line markup inert without fencing it", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "tool-call",
        name: "render_html",
        args: { code: "<script>alert(1)</script>" },
        result: "ok",
      },
    ]),
    [
      "**tool call:** `render_html`",
      "",
      "**code:** `<script>alert(1)</script>`",
      "",
      "**result:** `ok`",
    ].join("\n"),
  );
});

test("renders markdown syntax in a tool value as its own text", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "tool-call",
        name: "search",
        args: { top_hit: "[login](https://phish.test)" },
        result: "![x](https://evil.test/pixel.png) **bold**",
      },
    ]),
    [
      "**tool call:** `search`",
      "",
      "**top_hit:** `[login](https://phish.test)`",
      "",
      "**result:** `![x](https://evil.test/pixel.png) **bold**`",
    ].join("\n"),
  );
});

test("keeps a tool name with a backtick inside its code span", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "tool-call", name: "foo`bar" }]),
    "**tool call:** ``foo`bar``",
  );
});

test("escapes emphasis characters in an arg key", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "tool-call", name: "run", args: { "a*b": "1" } },
    ]),
    ["**tool call:** `run`", "", "**a\\*b:** `1`"].join("\n"),
  );
});

test("reasoning that quotes a closing details tag stays inside the block", () => {
  const markdown = renderConversationBlocks([
    { kind: "thinking", text: "the tag is </details> here" },
  ]);
  assert.equal(
    markdown,
    [
      "<details>",
      "<summary>thinking</summary>",
      "",
      "the tag is &lt;/details> here",
      "",
      "</details>",
    ].join("\n"),
  );
});

test("widens code spans and fences past backticks in the payload", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "tool-call", name: "run", args: { cmd: "echo ```x```" } },
    ]),
    ["**tool call:** `run`", "", "**cmd:** ```` echo ```x``` ````"].join("\n"),
  );

  const multiline = renderConversationBlocks([
    { kind: "tool-call", name: "run", args: { script: "```\nx\n```" } },
  ]);
  assert.ok(multiline.includes("````\n```\nx\n```\n````"));
});

test("keeps whitespace that markdown depends on", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "    indented_code_block()\n" },
      { kind: "thinking", text: "  padded reasoning  " },
      { kind: "text", text: "   " },
    ]),
    [
      "    indented_code_block()\n",
      "",
      "<details>",
      "<summary>thinking</summary>",
      "",
      "  padded reasoning  ",
      "",
      "</details>",
    ].join("\n"),
  );
});

test("collapses thinking and leaves prose untouched", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "thinking", text: "weighing options" },
      { kind: "text", text: "Here is the answer." },
      { kind: "attachment", label: "[image attachment]" },
    ]),
    [
      "<details>",
      "<summary>thinking</summary>",
      "",
      "weighing options",
      "",
      "</details>",
      "",
      "Here is the answer.",
      "",
      "\\[image attachment\\]",
    ].join("\n"),
  );
});

test("omits generated image bytes while retaining useful result metadata", () => {
  const blocks = contentBlocksToMarkdownBlocks([
    {
      type: "tool-call",
      toolName: "image_generation",
      result: {
        image_b64: "very-large-base64-payload",
        image_mime: "image/png",
        size: "1024x1024",
      },
    },
  ]);

  const markdown = renderConversationBlocks(blocks);
  assert.doesNotMatch(markdown, /very-large-base64-payload/);
  assert.match(markdown, /generated image omitted/);
  assert.match(markdown, /image\/png/);
  assert.match(markdown, /1024x1024/);
});

test("keeps the placeholder when a result carries its own image key", () => {
  const markdown = renderConversationBlocks(
    contentBlocksToMarkdownBlocks([
      {
        type: "tool-call",
        toolName: "image_generation",
        result: {
          image_b64: "very-large-base64-payload",
          image: "thumbnail-that-should-not-win",
        },
      },
    ]),
  );
  assert.match(markdown, /generated image omitted/);
  assert.doesNotMatch(markdown, /thumbnail-that-should-not-win/);
});

test("omits Gemini inline data bytes while keeping the part metadata", () => {
  const markdown = renderConversationBlocks(
    contentBlocksToMarkdownBlocks([
      {
        type: "tool-call",
        toolName: "code_execution",
        args: {
          google: {
            native_part: {
              parts: [
                { executableCode: { code: "print(1)", language: "PYTHON" } },
                {
                  inlineData: {
                    mimeType: "image/png",
                    data: "very-large-base64-payload",
                  },
                },
              ],
            },
          },
        },
      },
    ]),
  );
  assert.doesNotMatch(markdown, /very-large-base64-payload/);
  assert.match(markdown, /inline data omitted/);
  assert.match(markdown, /image\/png/);
  assert.match(markdown, /print\(1\)/);
});

test("omits Gemini inline data bytes from a legacy single-object part", () => {
  const markdown = renderConversationBlocks(
    contentBlocksToMarkdownBlocks([
      {
        type: "tool-call",
        toolName: "code_execution",
        args: {
          google: {
            native_part: {
              inlineData: { mimeType: "image/png", data: "legacy-payload" },
            },
          },
        },
      },
    ]),
  );
  assert.doesNotMatch(markdown, /legacy-payload/);
  assert.match(markdown, /inline data omitted/);
});

test("omits generated audio bytes from a text part", () => {
  const markdown = renderConversationBlocks(
    contentBlocksToMarkdownBlocks([
      {
        type: "text",
        text: 'Here it is: <audio-player src="data:audio/wav;base64,QUJD" />',
      },
    ]),
  );
  assert.equal(markdown, "Here it is: [generated audio omitted]");
});

test("returns no blocks for a message stored without content", () => {
  assert.deepEqual(contentBlocksToMarkdownBlocks(undefined), []);
  assert.deepEqual(contentBlocksToMarkdownBlocks(null), []);
  assert.equal(
    renderConversationBlocks(contentBlocksToMarkdownBlocks(undefined)),
    "",
  );
});

test("preserves assistant citation sources in markdown exports", () => {
  assert.equal(
    renderConversationBlocks(
      contentBlocksToMarkdownBlocks([
        {
          type: "source",
          title: "Unsloth documentation",
          url: "https://docs.unsloth.ai/",
        },
      ]),
    ),
    "**source:** [Unsloth documentation](<https://docs.unsloth.ai/>)",
  );
});

test("does not turn unsafe citation schemes into markdown links", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "source", title: "Untrusted source", url: "javascript:alert(1)" },
    ]),
    "**source:** `Untrusted source`",
  );
  assert.equal(
    renderConversationBlocks([
      {
        kind: "source",
        title: "Injected source",
        url: "https://safe.test/\n[evil](https://evil.test)",
      },
    ]),
    "**source:** `Injected source`",
  );
  assert.equal(
    renderConversationBlocks([
      {
        kind: "source",
        title: "Parenthesized source",
        url: "https://en.wikipedia.org/wiki/Foo_(bar)",
      },
      {
        kind: "source",
        title: "Attempted link injection",
        url: "https://safe.test/) [evil](https://evil.test",
      },
    ]),
    [
      "**source:** [Parenthesized source](<https://en.wikipedia.org/wiki/Foo_(bar)>)",
      "",
      "**source:** [Attempted link injection](<https://safe.test/)%20[evil](https://evil.test>)",
    ].join("\n"),
  );
});

test("an unclosed fence cannot swallow the message that follows it", () => {
  assert.equal(
    buildConversationMarkdown([
      {
        role: "user",
        content: renderConversationBlocks(
          contentBlocksToMarkdownBlocks([
            { type: "text", text: "look:\n```js\nvar a = 1;" },
          ]),
        ),
      },
      { role: "assistant", content: "Done." },
    ]),
    [
      "## User",
      "",
      "look:",
      "```js",
      "var a = 1;",
      "```",
      "",
      "## Assistant",
      "",
      "Done.",
      "",
    ].join("\n"),
  );
});

test("an unterminated html comment cannot hide the message that follows it", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<!-- note" }]),
    "<!-- note-->",
  );
});

test("reasoning that leaves a fence open still closes its details block", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "thinking", text: "~~~\nsketch" }]),
    [
      "<details>",
      "<summary>thinking</summary>",
      "",
      "~~~",
      "sketch",
      "~~~",
      "",
      "</details>",
    ].join("\n"),
  );
});

test("renders scalar args inline instead of spending a fence on each", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "tool-call",
        name: "web_search",
        args: { query: "lora", limit: 10, recursive: true, cursor: null },
      },
    ]),
    [
      "**tool call:** `web_search`",
      "",
      "**query:** `lora`",
      "",
      "**limit:** `10`",
      "",
      "**recursive:** `true`",
      "",
      "**cursor:** `null`",
    ].join("\n"),
  );
});

test("a line break in an arg key cannot end the bold label", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "tool-call",
        name: "run",
        args: { "a:**\n\n<img src=x onerror=alert(1)>": "1" },
      },
    ]),
    [
      "**tool call:** `run`",
      "",
      "**a:\\*\\* \\<img src=x onerror=alert(1)>:** `1`",
    ].join("\n"),
  );
});

test("a line break in a citation title cannot end the link label", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "source",
        title: "ok\n\n<img src=x onerror=alert(1)>",
        url: "https://good.test/",
      },
    ]),
    "**source:** [ok \\<img src=x onerror=alert(1)>](<https://good.test/>)",
  );
});

test("a rejected destination leaves a bare url title unlinkable", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "source", title: "https://evil.test/track", url: "javascript:alert(1)" },
    ]),
    "**source:** `https://evil.test/track`",
  );
});

test("keeps an attachment label from resolving as a link reference", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "attachment", label: "[audio attachment]" }]),
    "\\[audio attachment\\]",
  );
});

test("renders an empty tool value as a code span, not two bare backticks", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "tool-call", name: "run", args: { k: "" } }]),
    ["**tool call:** `run`", "", "**k:** `  `"].join("\n"),
  );
});

test("fences a value whose only line break is a carriage return", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "tool-call", name: "run", args: { k: "left\rright" } },
    ]),
    ["**tool call:** `run`", "", "**k:**", "", "```", "left\rright", "```"].join(
      "\n",
    ),
  );
});

test("drops a fragment url that cannot be encoded instead of throwing", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "source", title: "Broken", url: "#\ud800" },
    ]),
    "**source:** `Broken`",
  );
});

test("closes a fence opened in a body that uses bare carriage returns", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "look:\r```js\rvar a = 1;" }]),
    "look:\r```js\rvar a = 1;\r```",
  );
});

test("leaves a paragraph that only looks like a fence alone", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "```a`b" }]),
    "```a`b",
  );
});

test("does not treat a fence closer carrying text as a closer", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "```\nx\n``` trailing" }]),
    "```\nx\n``` trailing\n```",
  );
});

test("leaves a comment inside a fence literal", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "```\n<!-- literal\n```" }]),
    "```\n<!-- literal\n```",
  );
});

test("closes a list-nested fence inside the list, not at column zero", () => {
  // A closer at column zero ends the list first, so it opens a new top-level
  // fence that swallows every turn after this one.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "Steps:\n\n- run the install\n\n  ```sh\n  npm ci" },
    ]),
    "Steps:\n\n- run the install\n\n  ```sh\n  npm ci\n  ```",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "- run\n\n  ~~~\n  x" }]),
    "- run\n\n  ~~~\n  x\n  ~~~",
  );
});

test("closes a raw html block a blank line cannot end", () => {
  // CommonMark start condition 1: only its own end tag ends the block, so the
  // next role heading is raw html without this.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<pre>\nhello" }]),
    "<pre>\nhello\n</pre>",
  );
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: '<SCRIPT src="a.js">\nvar x = 1;' },
    ]),
    '<SCRIPT src="a.js">\nvar x = 1;\n</script>',
  );
  // Conditions 3, 4 and 5 run to the end of the document just the same.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<?php\necho 1;" }]),
    "<?php\necho 1;\n?>",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<![CDATA[\nraw" }]),
    "<![CDATA[\nraw\n]]>",
  );
});

test("leaves a raw html block that closes itself alone", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<pre>\nhello\n</pre>" }]),
    "<pre>\nhello\n</pre>",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<pre>hello</pre>" }]),
    "<pre>hello</pre>",
  );
  // A blank line ends start conditions 6 and 7, and every turn is followed by
  // one, so repairing them would only add noise.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<div>\nhello" }]),
    "<div>\nhello",
  );
  // Indented four spaces it is a code block, not a raw html block.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "    <pre>\n    hello" }]),
    "    <pre>\n    hello",
  );
});

test("keeps a raw html start inside a fence literal", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "```\n<pre>\nhello\n```" }]),
    "```\n<pre>\nhello\n```",
  );
  // And the other way round: a fence inside a raw html block is not a fence.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<pre>\n```js\nvar a = 1;" }]),
    "<pre>\n```js\nvar a = 1;\n</pre>",
  );
});

test("collapses line breaks inside a code span so it cannot be escaped", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "source",
        title: "ok\n\n<img src=x onerror=alert(1)>",
        url: "javascript:alert(1)",
      },
    ]),
    "**source:** `ok <img src=x onerror=alert(1)>`",
  );
});

test("reasoning that quotes a whole details element stays inside the block", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "thinking", text: "<details>\n<summary>FAQ</summary>\n\nbody\n\n</details>" },
    ]),
    [
      "<details>",
      "<summary>thinking</summary>",
      "",
      "&lt;details>",
      "<summary>FAQ</summary>",
      "",
      "body",
      "",
      "&lt;/details>",
      "",
      "</details>",
    ].join("\n"),
  );
});

test("escapes a details opener carrying attributes", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "thinking", text: '<details open class="x">' }]),
    [
      "<details>",
      "<summary>thinking</summary>",
      "",
      '&lt;details open class="x">',
      "",
      "</details>",
    ].join("\n"),
  );
});

test("a comment delimiter inside a code span opens nothing", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "Prefix it with `<!--` to comment it out." },
    ]),
    "Prefix it with `<!--` to comment it out.",
  );
});

test("still closes a comment opened mid-line outside a code span", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "see <div><!-- note" }]),
    "see <div><!-- note-->",
  );
});

test("closes an unmatched details element a later turn would fall inside", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "<details>\n<summary>Steps</summary>\n\nfirst" },
    ]),
    "<details>\n<summary>Steps</summary>\n\nfirst\n\n</details>",
  );
  // Mid-line, and in the case shape the html tokenizer accepts but a
  // <details ...> pattern does not.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "hello <DETAILS/>" }]),
    "hello <DETAILS/>\n\n</details>",
  );
});

test("leaves a details element the message already matched alone", () => {
  const matched = "<details>\n<summary>FAQ</summary>\n\nbody\n\n</details>";
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: matched }]),
    matched,
  );
  const nested = `<details>\n<summary>outer</summary>\n\n${matched}\n\n</details>`;
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: nested }]),
    nested,
  );
});

test("counts details tags in order so a stray closer licenses no opener", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "</details>\n\n<details>" },
    ]),
    "</details>\n\n<details>\n\n</details>",
  );
});

test("keeps a details tag inside a fence, a code span or a comment literal", () => {
  const fenced = "```html\n<details>\n```";
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: fenced }]),
    fenced,
  );
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "use `<details>` for this" },
    ]),
    "use `<details>` for this",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<!-- <details> -->" }]),
    "<!-- <details> -->",
  );
});

test("keeps a citation destination from decoding into another host", () => {
  // &commat; is an entity reference inside a markdown destination, so a viewer
  // resolves this to docs.unsloth.ai@evil.test: credentials on evil.test.
  assert.equal(
    renderConversationBlocks([
      {
        kind: "source",
        title: "Docs",
        url: "https://docs.unsloth.ai&commat;evil.test/",
      },
    ]),
    "**source:** [Docs](<https://docs.unsloth.ai&amp;commat;evil.test/>)",
  );
  assert.equal(
    renderConversationBlocks([
      {
        kind: "source",
        title: "Numeric",
        url: "https://x.test/?a=&#64;evil.test/",
      },
    ]),
    "**source:** [Numeric](<https://x.test/?a=&amp;#64;evil.test/>)",
  );
});

test("keeps a backslash in a citation destination from being eaten", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "source", title: "Query", url: "https://x.test/?q=\\*" },
    ]),
    "**source:** [Query](<https://x.test/?q=%5C*>)",
  );
});

test("leaves an ordinary query separator in a citation readable", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "source",
        title: "Search",
        url: "https://x.test/search?q=lora&page=2&sort=new",
      },
    ]),
    "**source:** [Search](<https://x.test/search?q=lora&page=2&sort=new>)",
  );
});

test("neutralises an opener the message never finished writing", () => {
  // A synthesized </script> written after a bare <script> is read as that
  // tag's attributes, so the element stays open and eats the next turn;
  // escaping the < opens nothing at all.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<script" }]),
    "&lt;script",
  );
  // Any tag, not only a raw text one: an unterminated attribute value runs to
  // the first > anywhere in the document.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: '<div class="x' }]),
    '&lt;div class="x',
  );
  // Handing back what the unfinished tag had swallowed can reveal an opener
  // that was hiding inside it, so the repair repeats until nothing is left.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<details\n<script" }]),
    "&lt;details\n&lt;script",
  );
  // plaintext has no end tag in any parser, so it can only be neutralised.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<plaintext>" }]),
    "&lt;plaintext>",
  );
});

test("closes the element that was opened, not the one markdown ended on", () => {
  // CommonMark 4.6 condition 1 ends the block at any of the four end tags,
  // "it need not match the start tag", but the html tokenizer needs this
  // element's own end tag (WHATWG 13.2.5, appropriate end tag token).
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "<script>\nvar x = 1;\n</pre>" },
    ]),
    "<script>\nvar x = 1;\n</pre>\n</script>",
  );
  // A </script> inside a code span is rendered as an escaped <code>, so the
  // browser never sees a closer there.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "hello <script> world `</script>`" },
    ]),
    "hello <script> world `</script>`\n</script>",
  );
});

test("closes a persistent element opened part way through a line", () => {
  // CommonMark emits this as inline html and starts no block at all, but the
  // browser is in script data from here to the end of the document.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "hello <script>" }]),
    "hello <script>\n</script>",
  );
  // The tokenizer's raw text set is wider than condition 1: an unmatched
  // <iframe> or <xmp> renders nothing of what follows it either.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<iframe>" }]),
    "<iframe>\n</iframe>",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "see <xmp>" }]),
    "see <xmp>\n</xmp>",
  );
});

test("leaves an indented code block exactly as the message wrote it", () => {
  // Four spaces of indentation is code, so the delimiter in it opens nothing
  // and a repair glued to the line would change the sample being exported.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "Template:\n\n    <!-- TODO fill this in" },
    ]),
    "Template:\n\n    <!-- TODO fill this in",
  );
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "Example:\n\n    <details>" },
    ]),
    "Example:\n\n    <details>",
  );
  // It cannot interrupt a paragraph, so this one really is prose.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "note\n    <!-- x" }]),
    "note\n    <!-- x-->",
  );
});

test("reads a fence opener whose info string carries a line separator", () => {
  // U+2028 is an ordinary character to a markdown parser but a line terminator
  // to a JavaScript dot, and the fence it opens absorbs every later turn.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "```js x\nvar a = 1;" }]),
    "```js x\nvar a = 1;\n```",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "~~~a b\nsketch" }]),
    "~~~a b\nsketch\n~~~",
  );
});

test("does not read a block quote marker as the end of a tag", () => {
  // The renderer strips the marker, so its > is not one the tokenizer sees:
  // the start tag runs on and swallows the message's own </script> as its
  // attributes, which leaves the element open and needing a closer.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "> <script\n> </script>" },
    ]),
    "> <script\n> </script>\n</script>",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "> <script>\n> body" }]),
    "> <script>\n> body\n</script>",
  );
});

test("waits for the terminator the raw block is actually waiting for", () => {
  // The tokenizer ends a bogus comment at the first >, but CommonMark start
  // conditions 3 and 5 end their blocks at ?> and ]]>. Stopping at the > in
  // the php comparison leaves every later turn inside the block as raw text.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "<?php\nif ($a > $b) { echo 1; }" },
    ]),
    "<?php\nif ($a > $b) { echo 1; }\n?>",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<![CDATA[\na > b" }]),
    "<![CDATA[\na > b\n]]>",
  );
  // Condition 4 really does end at a >, and a block that closed itself needs
  // nothing.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<!DOCTYPE html>\nhi" }]),
    "<!DOCTYPE html>\nhi",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<?php echo 1; ?>\nhi" }]),
    "<?php echo 1; ?>\nhi",
  );
});

test("leaves a backslash-escaped tag out of the repair scan", () => {
  // CommonMark 2.4: \< is a literal <, so it opens nothing and the closer the
  // scan would append is text the message never wrote.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "\\<script>" }]),
    "\\<script>",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "\\<!-- note" }]),
    "\\<!-- note",
  );
  // And the element it was hiding is found again: reading the escape as live
  // markup swallowed the details opener as raw text, which left the next turn
  // inside it.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "\\<script>\n\n<details>\n\nhi" },
    ]),
    "\\<script>\n\n<details>\n\nhi\n\n</details>",
  );
  // A doubled backslash escapes itself, so the < after it is live.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "\\\\<script>" }]),
    "\\\\<script>\n</script>",
  );
  // The escape is markdown's, so it does not apply where markdown is not
  // reading inlines.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "```\n\\<script>" }]),
    "```\n\\<script>\n```",
  );
});

test("closes a template element before emitting the next turn", () => {
  // Its children are parsed and then put in a DocumentFragment that is never
  // rendered, so an unmatched opener takes every later turn with it.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<template>" }]),
    "<template>\n</template>",
  );
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "SFC:\n\n<template>\n  <div>{{ msg }}</div>" },
    ]),
    "SFC:\n\n<template>\n  <div>{{ msg }}</div>\n</template>",
  );
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "<template>\n<td>a</td>\n</template>" },
    ]),
    "<template>\n<td>a</td>\n</template>",
  );
});

test("keeps scanning the line a literal block ended part way through", () => {
  // The terminator only ends the block, not the line: a details opened after
  // it is live, and skipping the rest of the line loses its closer.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<!--\n--> <details>" }]),
    "<!--\n--> <details>\n\n</details>",
  );
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<?\n?> <script>" }]),
    "<?\n?> <script>\n</script>",
  );
});

test("reduces an imported role to one line of plain text", () => {
  // An imported transcript keeps its own role strings, and this one lands in a
  // heading that closeOpenBlocks never sees.
  assert.equal(
    buildConversationMarkdown([{ role: "user\n\n<details>", content: "hi" }]),
    "## User details\n\nhi\n",
  );
  assert.equal(
    buildConversationMarkdown([{ role: "   ", content: "hi" }]),
    "## Message\n\nhi\n",
  );
  // The ordinary roles are untouched.
  assert.equal(
    buildConversationMarkdown([{ role: "reviewer", content: "hi" }]),
    "## Reviewer\n\nhi\n",
  );
});

test("keeps every line of an indented code block literal", () => {
  // Only the first line follows a blank one, so without the state the rest of
  // the sample is scanned as live html and rewritten.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "look:\n\n    first\n    <script>" },
    ]),
    "look:\n\n    first\n    <script>",
  );
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "look:\n\n    first\n    <div class=\"x" },
    ]),
    "look:\n\n    first\n    <div class=\"x",
  );
  // A closer in there must not spend the closer of a real open element.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "<details>\nreal\n\ntext\n\n    a\n    </details>" },
    ]),
    "<details>\nreal\n\ntext\n\n    a\n    </details>\n\n</details>",
  );
});

test("reads a fence against its block quote rather than the raw line", () => {
  // The marker keeps FENCE_LINE_PATTERN from matching, so quoted code was
  // scanned as live html.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "> ```\n> <script>\n> ```" }]),
    "> ```\n> <script>\n> ```",
  );
  // The quote ending ends the fence, so nothing is appended for it.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "> ```js\n> var a = 1;" }]),
    "> ```js\n> var a = 1;",
  );
});

test("treats a backtick run as one delimiter at both ends", () => {
  // ```x`` has no matching run, so it is live text: masking it as a span would
  // hide the opener it carries.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "```<script>``" }]),
    "```<script>``\n</script>",
  );
  // A real span still masks.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "`<script>`" }]),
    "`<script>`",
  );
});

test("reads an angle bracket link destination as a url", () => {
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "[x](<details>)" }]),
    "[x](<details>)",
  );
  // And it must not spend the closer of a real open element either.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "<details>\nreal\n\n[x](</details>)" },
    ]),
    "<details>\nreal\n\n[x](</details>)\n\n</details>",
  );
});

test("reads a closing tag inside a raw text element as text", () => {
  // The tokenizer takes </details> there as script data, so it must not spend
  // the closer of a details that is genuinely open around it.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<details>\n<script>\n</details>" }]),
    "<details>\n<script>\n</details>\n</script>\n\n</details>",
  );
});

test("keeps the edge spaces of a code span value", () => {
  // CommonMark 6.1 strips one space from each end of a span that has one at
  // both, so a padded tool argument would render with its whitespace missing.
  assert.equal(
    renderConversationBlocks([
      { kind: "tool-call", name: "t", args: { k: " padded " } },
    ]),
    "**tool call:** `t`\n\n**k:** `  padded  `",
  );
  // All spaces is exempt from the rule, and needs no extra pair.
  assert.equal(
    renderConversationBlocks([{ kind: "tool-call", name: "t", args: { k: "  " } }]),
    "**tool call:** `t`\n\n**k:** `  `",
  );
});

test("follows a code span across a soft line break", () => {
  // The span is one inline, so the tag inside it is literal on both lines.
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "<details>\nreal\n\nuse `foo\nbar </details>` here" },
    ]),
    "<details>\nreal\n\nuse `foo\nbar </details>` here\n\n</details>",
  );
  // A run with no match is live text, not an unterminated span.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "`foo\nbar <details> here" }]),
    "`foo\nbar <details> here\n\n</details>",
  );
});

test("reads an image description as alt text", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "<details>\nreal\n\n![caption </details>](image.png)" },
    ]),
    "<details>\nreal\n\n![caption </details>](image.png)\n\n</details>",
  );
});

test("closes a select the message left open", () => {
  // The browser stays in select insertion mode and folds the next role heading
  // into the control instead of rendering it.
  assert.equal(
    renderConversationBlocks([{ kind: "text", text: "<select>\n<option>one" }]),
    "<select>\n<option>one\n\n</select>",
  );
});
