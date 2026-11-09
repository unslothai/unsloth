// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
import * as guards from "../src/features/chat/utils/composer-send-guard.ts";

const source = readSrc("components/assistant-ui/thread.tsx");
const start = source.indexOf("  // A fresh New chat gets");
const end = source.indexOf("  // react-textarea-autosize re-measures", start);
assert.ok(start >= 0 && end > start);
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
  return {send() { armJustSent(props.text); clearStoredDraft(); aui.composer().setText(''); }};
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
  let actions: { send: () => void };
  async function update(changes: Partial<Props> = {}) {
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
      h.drafts.latestDraft(h.drafts.composerDraftKey("nonce-1")),
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
  assert.equal(h.props.text, "existing draft");
  assert.equal(h.props.attachments.length, 0);
  assert.equal(
    h.drafts.latestDraft(h.drafts.composerDraftKey("local-new")),
    "unsent new chat",
  );
});

test("a fresh nonce with a reused unsaved runtime still starts empty", async (t) => {
  const h = await harness();
  t.after(h.close);
  await h.type("old new chat");
  await h.paste("old paste");
  await h.update({ nonce: "nonce-2" });
  assert.equal(h.props.text, "");
  assert.equal(h.props.attachments.length, 0);
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
