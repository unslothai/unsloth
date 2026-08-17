// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `deleteThreadMessage` is the only code path that destroys a user's messages, and a delete
// cannot be undone. It had no test.
//
// What makes it worth pinning is that the interesting behaviour is not "the message is gone".
// It is which OTHER messages go with it, and what happens to the ones that stay. Deleting a
// prompt also deletes the replies hanging off it, and `MessageRepository.deleteMessage` relinks
// every surviving child onto the deleted node's parent. Get either half wrong and the thread
// still renders as a perfectly ordinary conversation, just not the user's one, which is why
// eyeballing a screenshot cannot catch it.
//
// `remoteId` is left undefined throughout: that is the branch where nothing is written to the
// backend, so these cases exercise the repository surgery on its own. The two backend modules
// are stubbed to throw rather than mocked to succeed, so a delete path that started calling
// them without a remote id would fail here instead of quietly taking a different route.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type { ExportedMessageRepository } from "@assistant-ui/react";

// delete-thread-message reaches the chat api and history storage through extensionless
// relative imports that vite resolves and bare node does not, and those two drag the auth
// flow in behind them. The resolver adds vite's rules and redirects those two to a stub.
register("./helpers/delete-thread-message-resolver.mjs", import.meta.url);

const { deleteThreadMessage } = await import(
  "../src/features/chat/utils/delete-thread-message.ts"
);

type Role = "user" | "assistant";

function message(id: string, role: Role) {
  return {
    id,
    role,
    content: [{ type: "text" as const, text: `text-${id}` }],
    createdAt: new Date(0),
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {},
    },
    ...(role === "user"
      ? { attachments: [] }
      : { status: { type: "complete" as const, reason: "stop" as const } }),
  };
}

/** A linear thread u1 -> a1 -> u2 -> a2 -> ... with `pairs` turns. */
function linear(pairs: number): ExportedMessageRepository {
  const messages: {
    parentId: string | null;
    message: ReturnType<typeof message>;
  }[] = [];
  let parentId: string | null = null;
  for (let i = 1; i <= pairs; i++) {
    messages.push({ parentId, message: message(`u${i}`, "user") });
    parentId = `u${i}`;
    messages.push({ parentId, message: message(`a${i}`, "assistant") });
    parentId = `a${i}`;
  }
  return { headId: parentId, messages } as ExportedMessageRepository;
}

/** Collect what a delete produced: the imported repository, or null if none was imported. */
function threadOver(exported: ExportedMessageRepository) {
  let imported: ExportedMessageRepository | null = null;
  return {
    thread: {
      export: () => exported,
      import: (data: ExportedMessageRepository) => {
        imported = data;
      },
    },
    result: () => imported,
  };
}

function idsOf(repo: ExportedMessageRepository | null): string[] {
  return (repo?.messages ?? []).map(({ message: m }) => m.id);
}

function parentOf(
  repo: ExportedMessageRepository | null,
  id: string,
): string | null | undefined {
  return repo?.messages.find(({ message: m }) => m.id === id)?.parentId;
}

test("deleting the only message empties the thread", async () => {
  const t = threadOver({
    headId: "u1",
    messages: [{ parentId: null, message: message("u1", "user") }],
  } as ExportedMessageRepository);
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "u1",
    remoteId: undefined,
  });
  assert.deepEqual(idsOf(t.result()), []);
});

test("deleting a prompt takes its reply with it and leaves the rest in order", async () => {
  const t = threadOver(linear(4));
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "u3",
    remoteId: undefined,
  });
  // u3 and a3 go; the tail must survive AND stay after the head, not be reordered.
  assert.deepEqual(idsOf(t.result()), ["u1", "a1", "u2", "a2", "u4", "a4"]);
});

test("the survivors are relinked onto the deleted prompt's parent", async () => {
  const t = threadOver(linear(4));
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "u3",
    remoteId: undefined,
  });
  // u3 hung off a2, so what followed a3 has to hang off a2 now. A broken relink leaves an
  // orphan whose parent id names a message that no longer exists.
  assert.equal(parentOf(t.result(), "u4"), "a2");
  for (const { parentId, message: m } of t.result()?.messages ?? []) {
    if (parentId !== null) {
      assert.ok(
        idsOf(t.result()).includes(parentId),
        `${m.id} points at missing parent ${parentId}`,
      );
    }
  }
});

test("deleting the first prompt cascades and leaves no dangling root", async () => {
  const t = threadOver(linear(3));
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "u1",
    remoteId: undefined,
  });
  assert.deepEqual(idsOf(t.result()), ["u2", "a2", "u3", "a3"]);
  assert.equal(parentOf(t.result(), "u2"), null);
});

test("deleting the last message removes exactly that one", async () => {
  const t = threadOver(linear(3));
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "a3",
    remoteId: undefined,
  });
  assert.deepEqual(idsOf(t.result()), ["u1", "a1", "u2", "a2", "u3"]);
});

test("deleting an assistant reply does NOT cascade to its prompt", async () => {
  // The cascade is deliberately one-directional: a reply is owned by its prompt, not the
  // other way round. Deleting a2 must not take u2 with it.
  const t = threadOver(linear(3));
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "a2",
    remoteId: undefined,
  });
  assert.deepEqual(idsOf(t.result()), ["u1", "a1", "u2", "u3", "a3"]);
});

test("every reply on a branched prompt is cascaded, not just the visible one", async () => {
  // A regenerate leaves several assistant replies under one prompt. Cascading only the first
  // would leave the others parented onto a message that is gone.
  const exported = {
    headId: "a1b",
    messages: [
      { parentId: null, message: message("u1", "user") },
      { parentId: "u1", message: message("a1a", "assistant") },
      { parentId: "u1", message: message("a1b", "assistant") },
    ],
  } as ExportedMessageRepository;
  const t = threadOver(exported);
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "u1",
    remoteId: undefined,
  });
  assert.deepEqual(idsOf(t.result()), []);
});

test("nothing is imported when the id is not in the thread", async () => {
  // Better to fail than to import a repository built from a surgery that did not happen:
  // `import` replaces the whole thread, so a silent no-op here would be a silent wipe.
  const t = threadOver(linear(2));
  await assert.rejects(
    deleteThreadMessage({
      thread: t.thread,
      messageId: "nope",
      remoteId: undefined,
    }),
  );
  assert.equal(t.result(), null);
});
