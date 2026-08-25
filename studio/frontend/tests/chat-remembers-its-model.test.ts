// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every thread row has carried modelId since long before this notice, so the model a chat
// was started on is already known for chats that already exist. What was missing was
// offering it back. The rules below are the ones that keep the offer from becoming a
// nuisance: it never loads anything on its own, and it stays quiet whenever it could not
// be honoured.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { chatModelSwitchMeta } from "../src/features/chat/components/chat-model-notice-switch.ts";
import { chatLocalModelOptions } from "../src/features/chat/local-model-options.ts";

function read(path: string): string {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

const notice = read("../src/features/chat/components/chat-model-notice.tsx");
const page = read("../src/features/chat/chat-page.tsx");
const thread = read("../src/components/assistant-ui/thread.tsx");
const researchPanel = read(
  "../src/features/chat/components/research-activity-panel.tsx",
);
const artifact = read("../src/features/chat/artifacts/artifact-surface.tsx");

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  assert.ok(start !== -1, `not found: ${from}`);
  const end = source.indexOf(to, start + from.length);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

// Source assertions: the notice reaches external-providers, which does not resolve in a
// bare node test. The sibling thread-scoped suites do the same for the same reason.

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
  // Clearing it inside the effect is not enough: the effect is passive, so the first
  // render for the incoming chat commits with the outgoing chat's model already on
  // screen. The answer is keyed by the chat it was read for, so a foreign value cannot
  // be returned at all rather than being returned briefly.
  const hook = notice.slice(
    notice.indexOf("export function useChatCreatedModel"),
    notice.indexOf("type ChatModelNoticeProps"),
  );
  assert.match(hook, /setRead\(\{ threadId, modelId:/);
  assert.match(hook, /return read && read\.threadId === threadId \? read\.modelId : null;/);
  // and a read that resolves after the chat changed is dropped
  assert.match(hook, /if \(!cancelled\) setRead/);
  assert.match(hook, /cancelled = true;/);
});

test("the notice is wired to the picker's own handler, not a private path", () => {
  // handleCheckpointChange carries the confirmations, VRAM checks and external
  // handling; a second switch path would drift from it. The wrapper only
  // resolves the row metadata the picker also supplies, then calls it.
  assert.match(page, /onSwitch=\{handleSwitchBackToChatModel\}/);
  const wrapper = slice(
    page,
    "const handleSwitchBackToChatModel",
    "const inventoryRefreshStartedRef",
  );
  assert.match(wrapper, /handleCheckpointChange\(/);
  assert.match(wrapper, /chatModelSwitchMeta\(modelId, loraModels\)/);
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
  // Anchored on the marker attribute, not on `<div className="`: the bar carries
  // other attributes now, and the shape of the opening tag is not the contract.
  const tag = slice(body, "data-chat-model-notice", "\n    >");
  const div = slice(tag, 'className="', '"');
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

test("the conversation reserves the space the notice overlay takes", () => {
  // Measured on a built Studio before this: the viewport reserved exactly the
  // header's 48px, the notice sat opaque at y 48..85, and the first message's own
  // rect started at y=48 -- elementFromPoint returned the notice for every sample
  // across its band, in a 2-turn chat AND in a 40-turn chat scrolled to the top.
  // An overlay that reserves nothing hides content; the header gets away with it
  // only because chat-header-fade dissolves what slides under it.

  // One declaration of the height, on the nearest ancestor of both, so the bar and
  // the padding cannot drift apart.
  //
  // `has-[>...]`, not `has-[...]`: the descendant form made every DOM change anywhere in the
  // thread re-check this `:has()` on an ancestor of every message, and answering it walks the
  // whole thread. Measured at the 500K rung on a 357,843-element thread, appending one empty span
  // inside a message cost 17.5 / 18.6 ms with the descendant form and 0.10 ms once this rule and
  // the sidebar wrapper's were both put in their child form. The notice is a direct child of the
  // declaring element, so the two selectors match the same elements; that is asserted separately
  // in `tests/thread-ancestor-has-scope.test.ts`, which is what keeps the child form honest.
  assert.match(
    page,
    /has-\[>\[data-chat-model-notice\]\]:\[--studio-chat-notice-height:2\.25rem\]/,
  );
  // The notice claims the same variable rather than a padding of its own.
  assert.match(notice, /data-chat-model-notice=""/);
  assert.match(notice, /h-\[var\(--studio-chat-notice-height,2\.25rem\)\]/);
  assert.doesNotMatch(
    notice.slice(notice.indexOf("data-chat-model-notice")),
    /py-1\.5/,
    "a fixed height and a vertical padding would disagree about the bar's size",
  );
  // The viewport adds it to what it already reserved for the header, defaulting to
  // 0px so every surface without a notice keeps exactly the padding it had.
  assert.match(
    thread,
    /pt-\[calc\(var\(--studio-content-top-inset,0px\)\+48px\+var\(--studio-chat-notice-height,0px\)\)\]/,
  );
  // And the fade moves down with it, or it would dissolve behind the opaque bar.
  const fade = slice(page, "chat-header-fade", '"');
  assert.match(fade, /\+var\(--studio-chat-notice-height,0px\)\)\]/);
});

test("the research panel reserves the notice's height too", () => {
  // The thread viewport is not the only surface the bar covers. The notice is an
  // absolute child of the chat content container (chat-page.tsx), so its
  // containing block is that whole container -- the deep-research column
  // included, not just the thread pane. ResearchActivityPanel offsets itself by
  // the header height for exactly the same reason, which lands its header at the
  // notice's top edge.
  //
  // Measured on a Studio built from this tree, saved chat on claude-opus-4-5 with
  // the composer on gpt-5-mini and the research panel open: notice rect
  // {top:48,bottom:84,left:280,right:1590}, aside {top:48,left:1099.41,right:1600}
  // -> 36px x 490.59px of the aside covered, which is its entire header band.
  // elementFromPoint at the centre of the "Deep research" h2 (1212,73) and at the
  // centre of the close button (1568,78) BOTH returned the notice, so it was
  // swallowing the clicks, not merely painting over them. With the checkpoint set
  // to the chat's own model the notice self-suppresses and both return the h2 and
  // the button. The aside's margin-top was 48px in every case: nothing
  // compensated for it.
  const style = slice(researchPanel, 'variant === "panel"', ": undefined");
  // Both edges move, or the panel keeps its old height and overflows the bottom.
  assert.match(
    style,
    /marginTop:\s*\n?\s*"calc\(var\(--studio-content-top-inset, 0px\) \+ var\(--studio-chat-header-height, 48px\) \+ var\(--studio-chat-notice-height, 0px\)\)"/,
  );
  assert.match(
    style,
    /height:\s*\n?\s*"calc\(100% - var\(--studio-content-top-inset, 0px\) - var\(--studio-chat-header-height, 48px\) - var\(--studio-chat-notice-height, 0px\)\)"/,
  );
  // 0px is the only safe fallback: the sheet variant and every chat without a
  // notice must keep the exact geometry they had before this notice existed.
  assert.doesNotMatch(
    style,
    /--studio-chat-notice-height,\s*2\.25rem/,
    "the panel must not assume a notice is present",
  );
});

test("the canvas panel reserves the notice's height too", () => {
  // The other column the notice spans. The canvas panel's top offset was a fixed
  // 90px, which clears the 48px header on top of a 34px content inset by 8px and
  // nothing more, so a 36px notice covers the top 28px of the panel: the
  // preview/source tabs and the close control sit in exactly that band, and the
  // notice is opaque and takes pointer events at z-30. Same fix as the research
  // panel, and 0px whenever no notice is on screen.
  const panel = slice(artifact, 'variant === "panel"', "aria-label=");
  assert.match(
    panel,
    /marginTop:\s*\n?\s*"calc\(90px \+ var\(--studio-chat-notice-height, 0px\)\)"/,
  );
  // Both edges move, or the panel keeps its height and overflows the bottom.
  assert.match(
    panel,
    /height:\s*\n?\s*"calc\(100% - 122px - var\(--studio-chat-notice-height, 0px\)\)"/,
  );
  // The class list must not still carry the fixed geometry the style replaces.
  assert.doesNotMatch(panel, /mt-\[90px\]/);
  assert.doesNotMatch(panel, /h-\[calc\(100%_-_122px\)\]/);
  // 0px is the only safe fallback: the overlay variant and every chat without a
  // notice keep the exact geometry they had before this notice existed.
  assert.doesNotMatch(
    panel,
    /--studio-chat-notice-height,\s*2\.25rem/,
    "the panel must not assume a notice is present",
  );
});

test("a chat started as New Chat gets the notice once its row exists", () => {
  // ?new=<nonce> carries no thread in the URL and keeps none after the first send, so
  // the notice saw nothing until the chat was reopened. The store's id is only this
  // chat's after ThreadNewChatSwitch has blanked the previous one, hence the latch.
  const gate = slice(page, "const newChatBlankedRef", "const newChatThreadId =");
  assert.match(gate, /activeThreadId === null \|\| isAssistantLocalThreadId\(activeThreadId\)/);
  assert.match(gate, /newChatBlankedRef\.current = search\.new;/);

  const derived = slice(page, "const newChatThreadId =", "\n  const");
  assert.match(derived, /newChatBlankedRef\.current === search\.new/);
  // The latched id is only ever the persisted one, so an unsent chat still offers nothing.
  assert.match(derived, /\? persistedActiveThreadId/);
  assert.match(derived, /: null/);

  const notice = slice(page, "<ChatModelNotice", "/>");
  assert.match(notice, /threadId=\{view\.threadId \?\? newChatThreadId \?\? undefined\}/);
});

// The switch back is the picker's handler, so it has to arrive carrying what the picker
// itself would have put on it. A local or fine-tuned row is in neither /api/models/list
// nor the external ids, so with the bare id selectModel resolves isGguf false and the
// /load request loses n_parallel, n_batch, n_ubatch and llama_extra_args and sizes the
// context down the transformers path.

test("switching back to a single-file local GGUF still loads as a GGUF", () => {
  const meta = chatModelSwitchMeta("/models/qwen3-4b-q4.gguf", [
    {
      id: "/models/qwen3-4b-q4.gguf",
      name: "qwen3-4b-q4",
      source: "local",
      isGguf: true,
    },
  ]);
  assert.deepEqual(meta, {
    source: "local",
    isLora: false,
    isDownloaded: true,
    isGguf: true,
  });
});

test("a single .gguf file carries its format out of the local inventory", () => {
  // chatLocalModelOptions is where the format is either kept or lost; the resolver
  // above has nothing else to read it from.
  const [option] = chatLocalModelOptions([
    {
      id: "/models/qwen3-4b-q4.gguf",
      display_name: "qwen3-4b-q4",
      path: "/models/qwen3-4b-q4.GGUF",
      source: "custom",
      model_format: "gguf",
    },
  ]);
  assert.equal(option.isGguf, true);
});

test("a GGUF directory is not claimed as a GGUF, since its quant is unknown", () => {
  // That row expands in the picker to pick a variant, and a chat row records only
  // modelId. Sending llama-server flags for a quant nobody chose is worse than the
  // plain switch.
  const [option] = chatLocalModelOptions([
    {
      id: "/models/Qwen3-4B-GGUF",
      display_name: "Qwen3-4B-GGUF",
      path: "/models/Qwen3-4B-GGUF",
      source: "models_dir",
      model_format: "gguf",
    },
  ]);
  assert.equal(option.isGguf, undefined);
  assert.equal(
    chatModelSwitchMeta("/models/Qwen3-4B-GGUF", [option])?.isGguf,
    false,
  );
});

test("switching back to a fine-tuned row mirrors the picker's own metadata", () => {
  // Same fields the fine-tuned list's selectionMeta sets, so the two doors agree.
  const adapters = [
    { id: "run-1", name: "run-1", source: "training" as const },
    {
      id: "run-2",
      name: "run-2",
      source: "exported" as const,
      exportType: "merged" as const,
    },
  ];
  assert.deepEqual(chatModelSwitchMeta("run-1", adapters), {
    source: "lora",
    isLora: true,
    isDownloaded: true,
    isGguf: false,
  });
  assert.deepEqual(chatModelSwitchMeta("run-2", adapters), {
    source: "exported",
    isLora: false,
    isDownloaded: true,
    isGguf: false,
  });
});

test("a Hub or external id is still switched on the id alone", () => {
  // Those two resolve without help: /api/models/list carries isGguf, and
  // isExternalModelId routes the rest. Inventing a "local" source for them would
  // send them down the wrong branch of handleCheckpointChange.
  assert.equal(chatModelSwitchMeta("unsloth/Qwen3-4B-GGUF", []), undefined);
  assert.equal(chatModelSwitchMeta("openai:gpt-5-mini", []), undefined);
});
