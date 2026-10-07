// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { markdownToSpeechText } from "../src/features/chat/utils/speech-text.ts";

test("read-aloud does not speak emphasis markers (#12547)", () => {
  assert.equal(
    markdownToSpeechText(
      "This is **bold**, *italic*, __also bold__ and ~~gone~~.",
    ),
    "This is bold, italic, also bold and gone.",
  );
});

test("headings, bullets and numbered items lose their symbols but keep a pause", () => {
  const spoken = markdownToSpeechText(
    "## Steps\n\n- Install the package\n- Run `unsloth studio`\n\n1. First\n2. Second!",
  );
  assert.equal(
    spoken,
    "Steps.\nInstall the package.\nRun unsloth studio.\nFirst.\nSecond!",
  );
  assert.doesNotMatch(spoken, /[#*`]|^\s*[-\d]/m);
});

test("links and images read as their text, not their URLs", () => {
  assert.equal(
    markdownToSpeechText(
      "See [the docs](https://unsloth.ai/docs) and ![a cat](cat.png).",
    ),
    "See the docs and a cat.",
  );
});

test("code keeps its content but drops the fences", () => {
  assert.equal(
    markdownToSpeechText("Run this:\n\n```bash\npip install unsloth\n```"),
    "Run this:\npip install unsloth",
  );
});

test("blockquotes, rules, tables and html are read as plain text", () => {
  assert.equal(
    markdownToSpeechText(
      "> quoted **line**\n\n---\n\n| Model | Size |\n| --- | --- |\n| Qwen | 7B |\n\n<br>",
    ),
    "quoted line\nModel, Size.\nQwen, 7B.",
  );
});

test("math is read from its source and plain text passes through unchanged", () => {
  assert.equal(
    markdownToSpeechText("Euler: $e^{i\\pi}+1=0$"),
    "Euler: e^{i\\pi}+1=0",
  );
  assert.equal(markdownToSpeechText("Just a sentence."), "Just a sentence.");
  assert.equal(
    markdownToSpeechText("It costs $5 and $10 today."),
    "It costs $5 and $10 today.",
  );
  assert.equal(markdownToSpeechText("Inline \\(x^2\\)"), "Inline x^2");
  assert.equal(markdownToSpeechText(""), "");
});

test("text inside raw HTML and footnotes is read, as the page shows it", () => {
  assert.equal(
    markdownToSpeechText(
      "<details><summary>More</summary>\n\nHidden <b>text</b>\n\n</details>\n\nSee note[^1].\n\n[^1]: The note.\n\nEnd.",
    ),
    "More\nHidden text\nSee note.\nEnd.\nThe note.",
  );
  assert.equal(
    markdownToSpeechText("<div>Visible answer</div>\n<script>x()</script>"),
    "Visible answer",
  );
});

test("tags the page shows as text are read, and entities read as their characters", () => {
  assert.equal(
    markdownToSpeechText("Use <placeholder> here, a Vec<T>, and <b>bold</b>."),
    "Use <placeholder> here, a Vec<T>, and bold.",
  );
  assert.equal(
    markdownToSpeechText("<div>AT&amp;T &lt; 5 &#33; &#x41;</div>"),
    "AT&T < 5 ! A",
  );
});
