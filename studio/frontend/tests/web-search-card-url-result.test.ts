// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const CARD = fileURLToPath(
  new URL(
    "../src/components/assistant-ui/tool-ui-web-search.tsx",
    import.meta.url,
  ),
);

function urlBranch(): string {
  const source = readFileSync(CARD, "utf8");
  const start = source.indexOf("        ) : safeUrl || resultText ? (");
  assert.notEqual(start, -1, "the url/result branch changed shape");
  return source.slice(start, source.indexOf("</ToolFallbackContent>", start));
}

test("a url does not hide the tool result", () => {
  // url mode returns the fetched page or why it failed; a link INSTEAD of it drops both.
  const branch = urlBranch();
  const link = branch.indexOf("href={safeUrl}");
  const body = branch.indexOf("{resultText}");
  assert.ok(link !== -1, "the link is gone");
  assert.ok(body !== -1, "the result body is gone");
  assert.ok(
    link < body,
    "the result must render below the link, not instead of it",
  );
});

test("only OpenAI's synthesized card label is suppressed", () => {
  assert.match(urlBranch(), /resultText && !resultIsCardLabel/);
  // Scoped to safeUrl so a query-only search keeps its result and the box is never empty.
  assert.match(
    readFileSync(CARD, "utf8"),
    /const resultIsCardLabel = !!safeUrl && !!actionType;/,
  );
});

test("a finished find_in_page does not claim it found the pattern", () => {
  // The action carries the requested pattern and nothing about whether it matched,
  // and this file already holds that "a call that found nothing must not claim it did".
  const card = readFileSync(CARD, "utf8");
  assert.doesNotMatch(card, /`Found "\$\{pattern\}"/);
  assert.match(
    card,
    /`Searched for "\$\{pattern\}" in \$\{displayDomain \|\| "page"\}`/,
  );
});

test("a URL embedded mid-line cannot become the source pill's href", () => {
  // Title and snippet carry page-controlled text. The parser is line anchored so
  // only a real `URL:` line can point the pill; see the backend's matching
  // one-line normalization in _web_search_source_records.
  const card = readFileSync(CARD, "utf8");
  const re = (name: string): RegExp => {
    const m = card.match(new RegExp(`const ${name} = (/.*?/[a-z]*);`, "s"));
    assert.ok(m, `${name} not found`);
    return new Function(`return ${m[1]}`)() as RegExp;
  };
  const block =
    "Title: ok URL: https://phish.example\nURL: https://real.example/a\nSnippet: s";
  assert.equal(block.match(re("RE_URL"))?.[1], "https://real.example/a");
  assert.equal(
    block.match(re("RE_TITLE"))?.[1],
    "ok URL: https://phish.example",
  );

  // A well-formed multi-entry payload still parses unchanged.
  const good = "Title: A\nURL: https://a.example\nSnippet: sa";
  assert.equal(good.match(re("RE_URL"))?.[1], "https://a.example");
  assert.equal(good.match(re("RE_TITLE"))?.[1], "A");
});

test("the Sources group cannot receive the same url twice", () => {
  // Every card carries its own sources and the last also merges the run's
  // aggregate, so a url found by two searches arrives twice; `id` is the url,
  // so an undeduped list means duplicate React keys.
  const adapter = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    ),
    "utf8",
  );
  assert.match(adapter, /sourcesByUrl\.set\(part\.url,/);
  assert.match(
    adapter,
    /const dedupedSourceParts = \[\.\.\.sourcesByUrl\.values\(\)\];/,
  );
  assert.match(adapter, /\.\.\.dedupedSourceParts,/);
  assert.doesNotMatch(adapter, /\.\.\.sourceParts,/);
  // and the adapter's own copy of the block parser is anchored like the card's
  assert.match(adapter, /block\.match\(\/\^Title:\\s\*\(\.\+\)\$\/m\)/);
  assert.match(adapter, /block\.match\(\/\^URL:\\s\*\(\.\+\)\$\/m\)/);
});

test("the no-output message is only claimed on a completed call", () => {
  const card = readFileSync(
    fileURLToPath(
      new URL(
        "../src/components/assistant-ui/tool-ui-code-execution.tsx",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  // A cancelled call leaves status incomplete with no result; claiming
  // completion there contradicts the card's own trigger.
  assert.match(
    card,
    /status\?\.type === "complete" \? \(\s*<p[^>]*>\s*Command completed with no output\./,
  );
});

test("deduplicating sources keeps the richer entry, not the first", () => {
  // A source recovered from `action.sources` alone titles itself with its own
  // URL and carries no description; a first-seen filter would keep that stub
  // over the citation for the same page that actually names it.
  const adapter = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    ),
    "utf8",
  );
  assert.match(
    adapter,
    /const sourcesByUrl = new Map<string, \(typeof sourceParts\)\[number\]>\(\);/,
  );
  assert.match(
    adapter,
    /seen\.title === seen\.url && part\.title !== part\.url/,
  );
  assert.match(
    adapter,
    /seen\.metadata\?\.description\s*\?\s*seen\.metadata\s*:\s*part\.metadata/,
  );
  assert.doesNotMatch(adapter, /const seenSourceUrls = new Set<string>\(\);/);
  assert.match(adapter, /\.\.\.dedupedSourceParts,/);
});

test("a second tool_end reaches the card the first one ended", () => {
  // The citation backfill sends one for the last web_search card. The live
  // mapping is dropped at the first tool_end so a later tool_start opens a
  // fresh card, and without a fallback the backfill minted a new part id,
  // matched nothing, and its merged sources were silently discarded.
  const adapter = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    ),
    "utf8",
  );
  assert.match(
    adapter,
    /const endedToolPartIdByBackendId = new Map<string, string>\(\);/,
  );
  assert.match(
    adapter,
    /endedToolPartIdByBackendId\.set\(backendToolCallId, id\);/,
  );
  assert.match(
    adapter,
    /alreadyEndedId &&\s*!toolPartIdByBackendId\.has\(backendToolCallId\)\s*\?\s*alreadyEndedId/,
  );
});

test("copying a reply does not carry citation markers to the clipboard", () => {
  const thread = readFileSync(
    fileURLToPath(
      new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    ),
    "utf8",
  );
  assert.match(
    thread,
    /scrubOpenAICitationMarkers\(\s*stripSearchImageTokens\(aui\.message\(\)\.getCopyText\(\)\),\s*\)/,
  );
});

test("every presentation path scrubs citation markers, not just the screen", () => {
  // MarkdownText cleans the rendered derivative only, so each consumer that
  // turns a reply back into prose has to strip them too, next to the
  // search-image tokens it already strips for the same reason.
  const sites: Array<[string, RegExp]> = [
    [
      "../src/features/chat/adapters/studio-speech-synthesis-adapter.ts",
      /scrubOpenAICitationMarkers\(stripSearchImageTokens\(spokenText\)\)/,
    ],
    [
      "../src/features/chat/utils/conversation-markdown-export.ts",
      /scrubOpenAICitationMarkers\(\s*stripSearchImageTokens\(renderMessage\(message\)\),\s*\)/,
    ],
    [
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      /scrubOpenAICitationMarkers\(\s*stripSearchImageTokens\(messageToMarkdown\(msg\)\),\s*\)/,
    ],
    [
      "../src/components/assistant-ui/thread.tsx",
      /scrubOpenAICitationMarkers\(\s*stripSearchImageTokens\(\s*replySourceMarkdown\(/,
    ],
    [
      "../src/components/assistant-ui/thread.tsx",
      /scrubOpenAICitationMarkers\(stripSearchImageTokens\(content\)\)/,
    ],
  ];
  for (const [relative, pattern] of sites) {
    const source = readFileSync(
      fileURLToPath(new URL(relative, import.meta.url)),
      "utf8",
    );
    assert.match(source, pattern, relative);
  }
});
