// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
import * as guards from "../src/features/chat/utils/composer-send-guard.ts";

const source = readSrc("components/assistant-ui/thread.tsx");
const start = source.lastIndexOf(
  "  const incognito =",
  source.indexOf("  const draftThreadId ="),
);
const end = source.indexOf("  // react-textarea-autosize re-measures", start);
assert.ok(start >= 0 && end > start);
const inputStart = source.indexOf("      const composer = aui.composer();", source.indexOf("  const setComposerText = useCallback("));
const inputEnd = source.indexOf("    },", inputStart);
assert.ok(inputStart >= 0 && inputEnd > inputStart);
const code = ts.transpileModule(
  `return function render(props) {
  const referenceThreadId = props.threadId;
  const threadListItemId = props.runtimeId;
  const threadListItemRemoteId = props.remoteId;
  const composerText = props.text;
  const composerAttachmentSignature = props.attachments.map(a => a.id).join(',');
  const useChatRuntimeStore = select => select({incognito: props.incognito});
  const useChatPreferencesStore = select => select({shareComposerDraft: props.share});
  const NewChatDraftIdContext = null;
  const useContext = () => props.nonce;
  const justSentRef = useRef(null), draftKeyRef = useRef(null), imeSessionOpenRef = useRef(false);
  const setIsWritingExpanded = () => {};
  ${source.slice(start, end)}
  return {send() { armJustSent(props.text); clearStoredDraft(); aui.composer().setText(''); }, input(value) { const nativeEvent = undefined; ${source.slice(inputStart, inputEnd)} }};
}`,
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
).outputText;

type Attachment = { id: string; file: { paste: string } };
type Props = {
  threadId: string | null;
  runtimeId: string;
  remoteId: string | undefined;
  nonce: string | undefined;
  incognito: boolean;
  share: boolean;
  text: string;
  attachments: Attachment[];
};
type Effect = { deps: unknown[]; cleanup?: () => void };

async function harness() {
  installLocalStorageFake();
  const drafts = await import(
    `../src/features/chat/utils/composer-draft.ts?lifecycle=${Math.random()}`
  );
  const props: Props = {
    threadId: null,
    runtimeId: "local-new",
    remoteId: undefined,
    nonce: "nonce-1",
    incognito: false,
    share: false,
    text: "",
    attachments: [],
  };
  const slots: unknown[] = [];
  const timers = new Map<number, () => void>();
  let cursor = 0,
    nextTimer = 0,
    attachmentId = 0;
  let dirty = false;
  let effects: Array<{ slot: Effect; setup: () => void | (() => void) }> = [];
  const composer = {
    getState: () => ({
      isEditing: true,
      text: props.text,
      attachments: props.attachments,
    }),
    setText(text: string) {
      dirty ||= props.text !== text;
      props.text = text;
    },
    async addAttachment(file: Attachment["file"]) {
      props.attachments = [
        ...props.attachments,
        { id: String(++attachmentId), file },
      ];
      dirty = true;
    },
    attachment: ({ id }: { id: string }) => ({
      async remove() {
        props.attachments = props.attachments.filter((a) => a.id !== id);
        dirty = true;
      },
    }),
  };
  const deps = {
    ...drafts,
    ...guards,
    isTextReplacement: () => false,
    isDeliberateWrite: () => false,
    isCompositionWrite: () => false,
    flushResourcesSync: (fn: () => void) => fn(),
    INCOGNITO_DRAFT_SCOPE: "\0incognito",
    pastedTextOf: (file?: Attachment["file"]) => file?.paste,
    createPastedTextFile: (paste: string) => ({ paste }),
    aui: {
      composer: () => composer,
      threadListItem: () => ({
        getState: () => ({ remoteId: props.remoteId }),
      }),
    },
    useRef(value: unknown) {
      const i = cursor++;
      slots[i] ??= { current: value };
      return slots[i];
    },
    useState(value: unknown) {
      const i = cursor++;
      if (!(i in slots))
        slots[i] = typeof value === "function" ? value() : value;
      return [
        slots[i],
        (next: unknown) => {
          slots[i] = next;
          dirty = true;
        },
      ];
    },
    useCallback(fn: unknown) {
      cursor++;
      return fn;
    },
    useEffect(setup: () => void | (() => void), next: unknown[]) {
      const i = cursor++;
      const previous = slots[i] as Effect | undefined;
      if (
        previous &&
        next.every((value, j) => Object.is(value, previous.deps[j]))
      )
        return;
      const slot = { deps: next, cleanup: previous?.cleanup };
      slots[i] = slot;
      effects.push({ slot, setup });
    },
    setTimeout(fn: () => void) {
      timers.set(++nextTimer, fn);
      return nextTimer;
    },
    clearTimeout(id: number) {
      timers.delete(id);
    },
  };
  const render = new Function(...Object.keys(deps), code)(
    ...Object.values(deps),
  );
  let actions: { send: () => void; input: (value: string) => boolean };
  async function update(changes: Partial<Props> = {}) {
    if (changes.share && !props.share) drafts.adoptCurrentDraft();
    Object.assign(props, changes);
    for (let pass = 0; pass < 30; pass++) {
      dirty = false;
      cursor = 0;
      effects = [];
      actions = render({ ...props });
      // Commit cleanups before setups, including the pending debounce flush.
      for (const { slot } of effects) slot.cleanup?.();
      for (const { slot, setup } of effects)
        slot.cleanup = setup() || undefined;
      await Promise.resolve();
      await Promise.resolve();
      if (!dirty) return;
    }
    assert.fail("draft effects did not settle");
  }
  await update();
  return {
    props,
    drafts,
    update,
    async type(text: string) {
      await update({ text });
    },
    async paste(text: string) {
      await composer.addAttachment({ paste: text });
      await update();
    },
    async send() {
      actions.send();
      await update({ attachments: [] });
    },
    async delayedInput(value: string) {
      const accepted = actions.input(value);
      await update();
      return accepted;
    },
    flush() {
      for (const fn of [...timers.values()]) fn();
      timers.clear();
    },
    close() {
      for (const slot of slots) (slot as Effect | undefined)?.cleanup?.();
    },
  };
}

for (const flushed of [false, true]) {
  test(`first save preserves text and pastes, debounce flushed=${flushed}`, async (t) => {
    const h = await harness();
    t.after(h.close);
    await h.type("first message");
    await h.send();
    await h.type("follow-up");
    await h.paste("follow-up paste");
    if (flushed) h.flush();
    await h.update({ remoteId: "saved" });
    await h.update({ threadId: "saved" });
    assert.equal(h.props.text, "follow-up");
    assert.equal(h.props.attachments[0]?.file.paste, "follow-up paste");
    await h.update({
      threadId: "other",
      runtimeId: "other",
      remoteId: "other",
      nonce: undefined,
    });
    assert.equal(h.props.text, "");
    assert.equal(h.props.attachments.length, 0);
    await h.update({
      threadId: "saved",
      runtimeId: "local-new",
      remoteId: "saved",
    });
    assert.equal(h.props.text, "follow-up");
    assert.equal(h.props.attachments[0]?.file.paste, "follow-up paste");
    assert.equal(
      h.drafts.latestDraft(
        h.drafts.composerDraftKey(h.drafts.NEW_CHAT_DRAFT_ID),
      ),
      "",
    );
  });
}

test("enabling sharing in New chat adopts text and pastes before the debounce", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.type("new prompt");
  await h.paste("new paste");
  await h.update({ share: true });
  assert.equal(h.props.text, "new prompt");
  assert.equal(h.props.attachments[0]?.file.paste, "new paste");
  await h.update({
    threadId: "other",
    runtimeId: "other",
    remoteId: "other",
    nonce: undefined,
  });
  assert.equal(h.props.text, "new prompt");
  await h.send();
  await h.update({
    threadId: null,
    runtimeId: "local-new",
    remoteId: undefined,
    nonce: "nonce-1",
    share: false,
  });
  assert.equal(h.props.text, "");
  assert.equal(h.props.attachments.length, 0);
});

test("navigation before the runtime switch does not promote the outgoing New chat draft", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.update({ nonce: undefined });
  await h.type("unsent new chat");
  await h.paste("private to new chat");
  h.drafts.writeComposerDraft(
    h.drafts.composerDraftKey("existing"),
    "existing draft",
  );
  await h.update({ threadId: "existing" });
  assert.equal(h.props.text, "unsent new chat");
  await h.update({
    runtimeId: "existing",
    remoteId: "existing",
    text: "",
    attachments: [],
  });
  assert.equal(h.props.text, "existing draft");
  assert.equal(h.props.attachments.length, 0);
  assert.equal(
    h.drafts.latestDraft(h.drafts.composerDraftKey(h.drafts.NEW_CHAT_DRAFT_ID)),
    "unsent new chat",
  );
});

test("unsaved New chat navigation retains the common new-chat slot", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.type("old new chat");
  await h.paste("old paste");
  await h.update({ nonce: "nonce-2" });
  assert.equal(h.props.text, "old new chat");
  assert.equal(h.props.attachments[0]?.file.paste, "old paste");
});

test("a temporary chat never promotes its draft into a saved chat", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.update({ incognito: true });
  await h.type("temporary");
  await h.paste("temporary paste");
  await h.update({ incognito: false, threadId: "saved", remoteId: "saved" });
  assert.equal(h.props.text, "");
  assert.equal(h.props.attachments.length, 0);
});

test("updating navigation metadata for the same saved chat preserves pending edits", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.type("follow-up");
  await h.update({ threadId: "saved", remoteId: "saved" });
  await h.type("newer follow-up");
  await h.update({ nonce: undefined });
  assert.equal(h.props.text, "newer follow-up");
});

test("temporary text and pastes survive sharing rehydration from another window", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.update({ incognito: true });
  await h.type("temporary draft");
  await h.paste("temporary paste");
  await h.update({ share: true });
  assert.equal(h.props.text, "temporary draft");
  assert.equal(h.props.attachments[0]?.file.paste, "temporary paste");
  await h.update({ share: false });
  assert.equal(h.props.text, "temporary draft");
  assert.equal(h.props.attachments[0]?.file.paste, "temporary paste");
});

test("sharing after first save adopts the promoted text-only draft", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.type("follow-up");
  await h.update({ threadId: "saved", remoteId: "saved" });
  await h.update({ share: true });
  assert.equal(h.props.text, "follow-up");
  assert.equal(h.drafts.latestDraft(h.drafts.composerDraftKey("saved")), "");
});

test("switching temporary conversations still clears the old temporary draft", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.update({ incognito: true });
  await h.type("temporary draft");
  await h.update({ runtimeId: "another-temporary-thread" });
  assert.equal(h.props.text, "");
});

test("a saved New chat keeps its follow-up through Compare backgrounding", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.type("first message");
  await h.send();
  await h.update({ threadId: "saved", remoteId: "saved" });
  await h.type("follow-up");
  await h.paste("follow-up paste");
  await h.update({ threadId: null });
  await h.update({ threadId: "saved" });
  assert.equal(h.props.text, "follow-up");
  assert.equal(h.props.attachments[0]?.file.paste, "follow-up paste");
  assert.equal(
    h.drafts.latestDraft(h.drafts.composerDraftKey("saved")),
    "follow-up",
  );
});

test("a bare chat's uninitialized active ID still uses the common New chat slot", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.update({ threadId: "__LOCALID_first", runtimeId: "__LOCALID_first" });
  await h.type("unfinished bare chat");
  await h.paste("bare chat paste");
  h.flush();
  assert.equal(
    h.drafts.readComposerDraft(
      h.drafts.composerDraftKey(h.drafts.NEW_CHAT_DRAFT_ID),
    ),
    "unfinished bare chat",
  );
  assert.deepEqual(
    h.drafts.readPasteDraft(
      h.drafts.composerPasteDraftKey(h.drafts.NEW_CHAT_DRAFT_ID),
    ),
    ["bare chat paste"],
  );
  await h.update({
    threadId: null,
    runtimeId: "__LOCALID_second",
    text: "",
    attachments: [],
  });
  assert.equal(h.props.text, "unfinished bare chat");
  assert.equal(h.props.attachments[0]?.file.paste, "bare chat paste");
});

test("a route changing before its runtime cannot erase the destination's pasted draft", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.update({ threadId: "A", runtimeId: "A", remoteId: "A" });
  await h.type("draft A");
  await h.paste("paste A");
  h.drafts.writeComposerDraft(h.drafts.composerDraftKey("B"), "draft B");
  h.drafts.writePasteDraft(h.drafts.composerPasteDraftKey("B"), ["paste B"]);
  await h.update({ threadId: "B" });
  await h.update({ runtimeId: "B", remoteId: "B", text: "", attachments: [] });
  assert.equal(h.props.text, "draft B");
  assert.equal(h.props.attachments[0]?.file.paste, "paste B");
  assert.deepEqual(
    h.drafts.latestPasteDraft(h.drafts.composerPasteDraftKey("B")),
    ["paste B"],
  );
});

test("a fresh runtime restores pastes from the same common New chat slot", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.type("unfinished");
  await h.paste("unsaved paste");
  await h.update({
    runtimeId: "another-new-runtime",
    text: "",
    attachments: [],
  });
  assert.equal(h.props.text, "unfinished");
  assert.equal(h.props.attachments[0]?.file.paste, "unsaved paste");
});

test("first persistence keeps the guard against a delayed submitted-text write", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.type("submitted prompt");
  await h.send();
  await h.update({ threadId: "saved", remoteId: "saved" });
  assert.equal(await h.delayedInput("submitted prompt"), false);
  h.flush();
  assert.equal(h.props.text, "");
  assert.equal(h.drafts.latestDraft(h.drafts.composerDraftKey("saved")), "");
  await h.update({ threadId: "other", runtimeId: "other", remoteId: "other" });
  assert.equal(await h.delayedInput("submitted prompt"), true);
  assert.equal(h.props.text, "submitted prompt");
});
