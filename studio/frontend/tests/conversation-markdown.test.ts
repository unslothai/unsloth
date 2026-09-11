// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildConversationMarkdown,
  contentBlocksToMarkdownBlocks,
  renderConversationBlocks,
} from "../src/features/chat/utils/conversation-markdown.ts";

const renderText = (text: string): string =>
  renderConversationBlocks([{ kind: "text", text }]);

const renderSource = (title: string, url: string): string =>
  renderConversationBlocks([{ kind: "source", title, url }]);

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
    renderSource("Untrusted source", "javascript:alert(1)"),
    "**source:** `Untrusted source`",
  );
  assert.equal(
    renderSource("Injected source", "https://safe.test/\n[evil](https://evil.test)"),
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
  assert.equal(renderText("<!-- note"), "<!-- note-->");
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
    renderSource("ok\n\n<img src=x onerror=alert(1)>", "https://good.test/"),
    "**source:** [ok \\<img src=x onerror=alert(1)>](<https://good.test/>)",
  );
});

test("a rejected destination leaves a bare url title unlinkable", () => {
  assert.equal(
    renderSource("https://evil.test/track", "javascript:alert(1)"),
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
  assert.equal(renderSource("Broken", "#\ud800"), "**source:** `Broken`");
});

test("closes a fence opened in a body that uses bare carriage returns", () => {
  assert.equal(
    renderText("look:\r```js\rvar a = 1;"),
    "look:\r```js\rvar a = 1;\r```",
  );
});

test("leaves a paragraph that only looks like a fence alone", () => {
  assert.equal(renderText("```a`b"), "```a`b");
});

test("does not treat a fence closer carrying text as a closer", () => {
  assert.equal(renderText("```\nx\n``` trailing"), "```\nx\n``` trailing\n```");
});

test("leaves a comment inside a fence literal", () => {
  assert.equal(renderText("```\n<!-- literal\n```"), "```\n<!-- literal\n```");
});

test("closes a list-nested fence inside the list, not at column zero", () => {
  // A closer at column zero ends the list first, opening a top-level fence instead.
  assert.equal(
    renderText("Steps:\n\n- run the install\n\n  ```sh\n  npm ci"),
    "Steps:\n\n- run the install\n\n  ```sh\n  npm ci\n  ```",
  );
  assert.equal(renderText("- run\n\n  ~~~\n  x"), "- run\n\n  ~~~\n  x\n  ~~~");
});

test("closes a raw html block a blank line cannot end", () => {
  // CommonMark start condition 1: only its own end tag ends the block.
  assert.equal(renderText("<pre>\nhello"), "<pre>\nhello\n</pre>");
  assert.equal(
    renderText('<SCRIPT src="a.js">\nvar x = 1;'),
    '<SCRIPT src="a.js">\nvar x = 1;\n</script>',
  );
  // Conditions 3, 4 and 5 run to the end of the document just the same.
  assert.equal(renderText("<?php\necho 1;"), "<?php\necho 1;\n?>");
  assert.equal(renderText("<![CDATA[\nraw"), "<![CDATA[\nraw\n]]>");
});

test("leaves a raw html block that closes itself alone", () => {
  assert.equal(renderText("<pre>\nhello\n</pre>"), "<pre>\nhello\n</pre>");
  assert.equal(renderText("<pre>hello</pre>"), "<pre>hello</pre>");
  // A blank line ends start conditions 6 and 7, and every turn is followed by one.
  assert.equal(renderText("<div>\nhello"), "<div>\nhello");
  // Indented four spaces it is a code block, not a raw html block.
  assert.equal(renderText("    <pre>\n    hello"), "    <pre>\n    hello");
});

test("keeps a raw html start inside a fence literal", () => {
  assert.equal(renderText("```\n<pre>\nhello\n```"), "```\n<pre>\nhello\n```");
  // And the other way round: a fence inside a raw html block is not a fence.
  assert.equal(
    renderText("<pre>\n```js\nvar a = 1;"),
    "<pre>\n```js\nvar a = 1;\n</pre>",
  );
});

test("collapses line breaks inside a code span so it cannot be escaped", () => {
  assert.equal(
    renderSource("ok\n\n<img src=x onerror=alert(1)>", "javascript:alert(1)"),
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
    renderText("Prefix it with `<!--` to comment it out."),
    "Prefix it with `<!--` to comment it out.",
  );
});

test("still closes a comment opened mid-line outside a code span", () => {
  assert.equal(renderText("see <div><!-- note"), "see <div><!-- note-->");
});

test("closes an unmatched details element a later turn would fall inside", () => {
  assert.equal(
    renderText("<details>\n<summary>Steps</summary>\n\nfirst"),
    "<details>\n<summary>Steps</summary>\n\nfirst\n\n</details>",
  );
  // Mid-line, in the case shape the tokenizer accepts but a <details ...> does not.
  assert.equal(
    renderText("hello <DETAILS/>"),
    "hello <DETAILS/>\n\n</details>",
  );
});

test("leaves a details element the message already matched alone", () => {
  const matched = "<details>\n<summary>FAQ</summary>\n\nbody\n\n</details>";
  assert.equal(renderText(matched), matched);
  const nested = `<details>\n<summary>outer</summary>\n\n${matched}\n\n</details>`;
  assert.equal(renderText(nested), nested);
});

test("counts details tags in order so a stray closer licenses no opener", () => {
  assert.equal(
    renderText("</details>\n\n<details>"),
    "</details>\n\n<details>\n\n</details>",
  );
});

test("keeps a details tag inside a fence, a code span or a comment literal", () => {
  const fenced = "```html\n<details>\n```";
  assert.equal(renderText(fenced), fenced);
  assert.equal(
    renderText("use `<details>` for this"),
    "use `<details>` for this",
  );
  assert.equal(renderText("<!-- <details> -->"), "<!-- <details> -->");
});

test("keeps a citation destination from decoding into another host", () => {
  // &commat; is an entity reference in a destination: a viewer resolves this to
  // docs.unsloth.ai@evil.test, which is credentials on evil.test.
  assert.equal(
    renderSource("Docs", "https://docs.unsloth.ai&commat;evil.test/"),
    "**source:** [Docs](<https://docs.unsloth.ai&amp;commat;evil.test/>)",
  );
  assert.equal(
    renderSource("Numeric", "https://x.test/?a=&#64;evil.test/"),
    "**source:** [Numeric](<https://x.test/?a=&amp;#64;evil.test/>)",
  );
});

test("keeps a backslash in a citation destination from being eaten", () => {
  assert.equal(
    renderSource("Query", "https://x.test/?q=\\*"),
    "**source:** [Query](<https://x.test/?q=%5C*>)",
  );
});

test("leaves an ordinary query separator in a citation readable", () => {
  assert.equal(
    renderSource("Search", "https://x.test/search?q=lora&page=2&sort=new"),
    "**source:** [Search](<https://x.test/search?q=lora&page=2&sort=new>)",
  );
});

test("neutralises an opener the message never finished writing", () => {
  // A synthesized </script> after a bare <script> is read as that tag's attributes,
  // so the element stays open; escaping the < opens nothing at all.
  assert.equal(renderText("<script"), "&lt;script");
  // Any tag: an unterminated attribute value runs to the first > in the document.
  assert.equal(renderText('<div class="x'), '&lt;div class="x');
  // Handing back what an unfinished tag swallowed can reveal an opener, so repeat.
  assert.equal(renderText("<details\n<script"), "&lt;details\n&lt;script");
  // plaintext has no end tag in any parser, so it can only be neutralised.
  assert.equal(renderText("<plaintext>"), "&lt;plaintext>");
});

test("closes the element that was opened, not the one markdown ended on", () => {
  // CommonMark 4.6 condition 1 ends the block at any of the four end tags, "it need
  // not match the start tag"; the tokenizer needs this element's own (WHATWG 13.2.5).
  assert.equal(
    renderText("<script>\nvar x = 1;\n</pre>"),
    "<script>\nvar x = 1;\n</pre>\n</script>",
  );
  // A </script> in a code span renders as escaped <code>: never a closer.
  assert.equal(
    renderText("hello <script> world `</script>`"),
    "hello <script> world `</script>`\n</script>",
  );
});

test("closes a persistent element opened part way through a line", () => {
  // CommonMark starts no block here, but the browser is in script data to EOF.
  assert.equal(renderText("hello <script>"), "hello <script>\n</script>");
  // The tokenizer's raw text set is wider than condition 1: iframe and xmp too.
  assert.equal(renderText("<iframe>"), "<iframe>\n</iframe>");
  assert.equal(renderText("see <xmp>"), "see <xmp>\n</xmp>");
});

test("leaves an indented code block exactly as the message wrote it", () => {
  // Four spaces is code, so the delimiter opens nothing and a repair would alter it.
  assert.equal(
    renderText("Template:\n\n    <!-- TODO fill this in"),
    "Template:\n\n    <!-- TODO fill this in",
  );
  assert.equal(
    renderText("Example:\n\n    <details>"),
    "Example:\n\n    <details>",
  );
  // It cannot interrupt a paragraph, so this one really is prose.
  assert.equal(renderText("note\n    <!-- x"), "note\n    <!-- x-->");
});

test("reads a fence opener whose info string carries a line separator", () => {
  // U+2028 is ordinary to markdown but a line terminator to a JavaScript dot.
  assert.equal(renderText("```js x\nvar a = 1;"), "```js x\nvar a = 1;\n```");
  assert.equal(renderText("~~~a b\nsketch"), "~~~a b\nsketch\n~~~");
});

test("does not read a block quote marker as the end of a tag", () => {
  // The renderer strips the marker, so its > is not one the tokenizer sees: the start
  // tag runs on and swallows the message's own </script> as attributes.
  assert.equal(
    renderText("> <script\n> </script>"),
    "> <script\n> </script>\n</script>",
  );
  assert.equal(
    renderText("> <script>\n> body"),
    "> <script>\n> body\n</script>",
  );
});

test("waits for the terminator the raw block is actually waiting for", () => {
  // The tokenizer ends a bogus comment at the first >, but CommonMark conditions 3
  // and 5 end at ?> and ]]>: stopping at the > in the php comparison loses the rest.
  assert.equal(
    renderText("<?php\nif ($a > $b) { echo 1; }"),
    "<?php\nif ($a > $b) { echo 1; }\n?>",
  );
  assert.equal(renderText("<![CDATA[\na > b"), "<![CDATA[\na > b\n]]>");
  // Condition 4 really does end at a >, and a self-closed block needs nothing.
  assert.equal(renderText("<!DOCTYPE html>\nhi"), "<!DOCTYPE html>\nhi");
  assert.equal(renderText("<?php echo 1; ?>\nhi"), "<?php echo 1; ?>\nhi");
});

test("leaves a backslash-escaped tag out of the repair scan", () => {
  // CommonMark 2.4: \< is a literal <, so it opens nothing and needs no closer.
  assert.equal(renderText("\\<script>"), "\\<script>");
  assert.equal(renderText("\\<!-- note"), "\\<!-- note");
  // And the element it hid is found again: reading the escape as live markup
  // swallowed the details opener as raw text.
  assert.equal(
    renderText("\\<script>\n\n<details>\n\nhi"),
    "\\<script>\n\n<details>\n\nhi\n\n</details>",
  );
  // A doubled backslash escapes itself, so the < after it is live.
  assert.equal(renderText("\\\\<script>"), "\\\\<script>\n</script>");
  // The escape is markdown's, so it does not apply where inlines are not read.
  assert.equal(renderText("```\n\\<script>"), "```\n\\<script>\n```");
});

test("closes a template element before emitting the next turn", () => {
  // Children land in a DocumentFragment that is never rendered, so later turns go too.
  assert.equal(renderText("<template>"), "<template>\n</template>");
  assert.equal(
    renderText("SFC:\n\n<template>\n  <div>{{ msg }}</div>"),
    "SFC:\n\n<template>\n  <div>{{ msg }}</div>\n</template>",
  );
  assert.equal(
    renderText("<template>\n<td>a</td>\n</template>"),
    "<template>\n<td>a</td>\n</template>",
  );
});

test("keeps scanning the line a literal block ended part way through", () => {
  // The terminator ends the block, not the line: a details opened after it is live.
  assert.equal(
    renderText("<!--\n--> <details>"),
    "<!--\n--> <details>\n\n</details>",
  );
  assert.equal(renderText("<?\n?> <script>"), "<?\n?> <script>\n</script>");
});

test("reduces an imported role to one line of plain text", () => {
  // Imported role strings land in a heading closeOpenBlocks never sees.
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
  // Only the first line follows a blank one, so without state the rest is scanned live.
  assert.equal(
    renderText("look:\n\n    first\n    <script>"),
    "look:\n\n    first\n    <script>",
  );
  assert.equal(
    renderText("look:\n\n    first\n    <div class=\"x"),
    "look:\n\n    first\n    <div class=\"x",
  );
  // A closer in there must not spend the closer of a real open element.
  assert.equal(
    renderText("<details>\nreal\n\ntext\n\n    a\n    </details>"),
    "<details>\nreal\n\ntext\n\n    a\n    </details>\n\n</details>",
  );
});

test("reads a fence against its block quote rather than the raw line", () => {
  // The marker keeps FENCE_LINE_PATTERN from matching, so quoted code was scanned live.
  assert.equal(
    renderText("> ```\n> <script>\n> ```"),
    "> ```\n> <script>\n> ```",
  );
  // The quote ending ends the fence, so nothing is appended for it.
  assert.equal(renderText("> ```js\n> var a = 1;"), "> ```js\n> var a = 1;");
});

test("treats a backtick run as one delimiter at both ends", () => {
  // ```x`` has no matching run, so it is live text carrying an opener.
  assert.equal(renderText("```<script>``"), "```<script>``\n</script>");
  // A real span still masks.
  assert.equal(renderText("`<script>`"), "`<script>`");
});

test("reads an angle bracket link destination as a url", () => {
  assert.equal(renderText("[x](<details>)"), "[x](<details>)");
  // And it must not spend the closer of a real open element either.
  assert.equal(
    renderText("<details>\nreal\n\n[x](</details>)"),
    "<details>\nreal\n\n[x](</details>)\n\n</details>",
  );
});

test("reads a closing tag inside a raw text element as text", () => {
  // </details> there is script data, so it must not spend a real details closer.
  assert.equal(
    renderText("<details>\n<script>\n</details>"),
    "<details>\n<script>\n</details>\n</script>\n\n</details>",
  );
});

test("keeps the edge spaces of a code span value", () => {
  // CommonMark 6.1 strips one space from each end of a span padded at both.
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
    renderText("<details>\nreal\n\nuse `foo\nbar </details>` here"),
    "<details>\nreal\n\nuse `foo\nbar </details>` here\n\n</details>",
  );
  // A run with no match is live text, not an unterminated span.
  assert.equal(
    renderText("`foo\nbar <details> here"),
    "`foo\nbar <details> here\n\n</details>",
  );
});

test("reads an image description as alt text", () => {
  assert.equal(
    renderText("<details>\nreal\n\n![caption </details>](image.png)"),
    "<details>\nreal\n\n![caption </details>](image.png)\n\n</details>",
  );
});

test("closes a select the message left open", () => {
  // The browser stays in select insertion mode and folds the next heading in.
  assert.equal(
    renderText("<select>\n<option>one"),
    "<select>\n<option>one\n\n</select>",
  );
});
