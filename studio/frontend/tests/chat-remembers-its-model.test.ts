// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every thread row has carried modelId since long before this notice, so the model a chat
// was started on is already known for chats that already exist. What was missing was
// showing it and offering it back. The rules below are the ones that keep the offer from
// becoming a nuisance: it never loads anything on its own, and it stays quiet whenever it
// could not be honoured.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function read(path: string): string {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

const notice = read("../src/features/chat/components/chat-model-notice.tsx");
const page = read("../src/features/chat/chat-page.tsx");
const sidebar = read("../src/components/app-sidebar.tsx");
const items = read("../src/features/chat/hooks/use-chat-sidebar-items.ts");

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  assert.ok(start !== -1, `not found: ${from}`);
  const end = source.indexOf(to, start + from.length);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

// Source assertions: use-chat-sidebar-items reaches chat-api, and the label helper reaches
// external-providers, neither of which resolves in a bare node test. The sibling
// thread-scoped suites do the same for the same reason.

test("a single chat row carries the model it was started on", () => {
  const single = slice(items, 'type: "single",', "});");
  assert.match(single, /modelId: t\.modelId \|\| undefined,/);
});

test("a legacy row with no model reads as unknown rather than empty string", () => {
  // db.ts backfills "" onto old records, and "" would render an empty label.
  assert.match(items, /modelId: t\.modelId \|\| undefined,/);
  assert.doesNotMatch(items, /modelId: t\.modelId,/);
});

test("a compare row carries no single model", () => {
  // Two panes, two models: there is no honest one to show, so the field is absent.
  const compare = slice(items, 'type: "compare",', "};");
  assert.doesNotMatch(compare, /modelId/);
});

test("the notice never switches a model on its own", () => {
  // Opening a chat must not evict what is resident: a local load is multi-gigabyte.
  assert.doesNotMatch(notice, /loadModel|setCheckpoint/);
  // The only way out of it is the button.
  assert.match(notice, /onClick=\{\(\) => onSwitch\(createdModelId\)\}/);
});

test("the notice stays quiet when it has nothing to offer", () => {
  const body = notice.slice(notice.indexOf("export function ChatModelNotice"));
  // no stamp, already on it, or a model that has since gone away
  assert.match(
    body,
    /if \(!createdModelId \|\| createdModelId === checkpoint\) return null;/,
  );
  assert.match(
    body,
    /if \(!selectableModelIds\.has\(createdModelId\)\) return null;/,
  );
});

test("switching chats does not show the outgoing chat's model", () => {
  // The read is async, so a stale value would sit over the incoming chat until it lands.
  const hook = notice.slice(
    notice.indexOf("export function useChatCreatedModel"),
    notice.indexOf("type ChatModelNoticeProps"),
  );
  assert.match(hook, /setModelId\(null\);\s*\n\s*void getStoredChatThread/);
  // and a read that resolves after the chat changed is dropped
  assert.match(hook, /if \(!cancelled\) setModelId/);
  assert.match(hook, /cancelled = true;/);
});

test("the notice is wired to the picker's own handler, not a private path", () => {
  // handleCheckpointChange carries the confirmations, VRAM checks and external
  // handling; a second switch path would drift from it.
  assert.match(page, /onSwitch=\{handleCheckpointChange\}/);
  assert.match(page, /selectableModelIds=\{selectableModelIds\}/);
  // Offered only for a real saved chat, in the single-chat view.
  assert.match(page, /view\.mode === "single" && \(\s*<ChatModelNotice/);
});

test("only models that can actually be selected are offered", () => {
  const set = page.slice(page.indexOf("const selectableModelIds = useMemo"));
  for (const source of ["models", "loraModels", "externalModels"]) {
    assert.ok(
      set.slice(0, 400).includes(`...${source}.map`),
      `${source} is missing from the selectable set`,
    );
  }
});

test("the notice clears the chat header instead of rendering underneath it", () => {
  // The bug this pins: the notice was an in-flow sibling of the chat header, and the
  // header is `absolute ... top-[--studio-content-top-inset] z-40` with an OPAQUE
  // bg-background. Per CSS painting order a positioned element is painted above every
  // non-positioned one whatever the source order, so the whole bar sat underneath the
  // header and only the 10px the header's `right-[10px]` leaves uncovered was visible.
  // Measured on a built Studio: the notice's rect was {x:280,y:0,w:1220,h:37}, exactly
  // the header's own band, and the before/after screenshots differed by a 10x37 sliver.
  const header = slice(page, "chat-header-fade", "</div>");
  assert.match(header, /z-40/, "the header is still the z-40 absolute overlay");

  const body = notice.slice(notice.indexOf("export function ChatModelNotice"));
  const div = slice(body, '<div className="', '">');
  // Positioned, so it is not painted under the header.
  assert.match(div, /\babsolute\b/);
  // Offset by the SAME header height the header, its fade and the drop overlay use,
  // so a change to either variable moves all four together.
  assert.match(div, /top-\[calc\(var\(--studio-content-top-inset,0px\)\+var\(--studio-chat-header-height,48px\)\)\]/);
  // Between the header fade (z-20) and the header itself (z-40): over the gradient,
  // under the model picker, whose menu must stay clickable.
  const z = /\bz-(\d+)\b/.exec(div);
  assert.ok(z, "the notice needs an explicit z-index");
  assert.ok(
    Number(z[1]) > 20 && Number(z[1]) < 40,
    `notice z-${z[1]} must sit above the z-20 fade and below the z-40 header`,
  );
  // An overlay over the scrolling conversation must be opaque, or messages read
  // through it. bg-muted/40 was fine only while the bar took its own row.
  assert.doesNotMatch(div, /bg-muted\//, "a translucent overlay lets messages show through");
});

test("the sidebar label cannot collide with the spinner or the unread dot", () => {
  // Both take the same right-hand slot, and the dot is positioned over it.
  assert.match(
    sidebar,
    /\{item\.modelId && !showWorkSpinner && !hasUnreadActivity && \(/,
  );
  // The title truncates first, so a long model name never squeezes it out.
  const label = sidebar.slice(
    sidebar.indexOf("{item.modelId && !showWorkSpinner"),
  );
  assert.match(label.slice(0, 400), /max-w-\[45%\]/);
  assert.match(label.slice(0, 400), /compareModelDisplayName\(item\.modelId\)/);
});
