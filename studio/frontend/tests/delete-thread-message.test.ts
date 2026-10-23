// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Backend modules are stubbed to throw so a delete without remoteId cannot hit them silently.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type { ExportedMessageRepository } from "@assistant-ui/react";

// The resolver adds vite's extensionless resolution and stubs the chat api and history storage.
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

function headOf(
  repo: ExportedMessageRepository | null,
): string | null | undefined {
  return repo?.headId;
}

function regenerated(): ExportedMessageRepository {
  return {
    headId: "a1b",
    messages: [
      { parentId: null, message: message("u1", "user") },
      { parentId: "u1", message: message("a1a", "assistant") },
      { parentId: "u1", message: message("a1b", "assistant") },
    ],
  } as ExportedMessageRepository;
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
  assert.deepEqual(idsOf(t.result()), ["u1", "a1", "u2", "a2", "u4", "a4"]);
});

test("the survivors are relinked onto the deleted prompt's parent", async () => {
  const t = threadOver(linear(4));
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "u3",
    remoteId: undefined,
  });
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
  const t = threadOver(linear(3));
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "a2",
    remoteId: undefined,
  });
  assert.deepEqual(idsOf(t.result()), ["u1", "a1", "u2", "u3", "a3"]);
});

test("every reply on a branched prompt is cascaded, not just the visible one", async () => {
  const t = threadOver(regenerated());
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "u1",
    remoteId: undefined,
  });
  assert.deepEqual(idsOf(t.result()), []);
});

test("nothing is imported when the id is not in the thread", async () => {
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

test("what is left is still pointed at by a head that exists", async () => {
  // import() passes headId to resetHead, which throws on an unknown id after the backend prune ran.
  for (const [exported, messageId] of [
    [linear(3), "a3"],
    [linear(3), "u1"],
    [linear(4), "u3"],
    [regenerated(), "a1b"],
  ] as const) {
    const t = threadOver(exported);
    await deleteThreadMessage({
      thread: t.thread,
      messageId,
      remoteId: undefined,
    });
    const head = headOf(t.result());
    assert.ok(
      idsOf(t.result()).some((id) => id === head),
      `head ${String(head)} is not a message in the thread after deleting ${messageId}`,
    );
  }

  const empty = threadOver({
    headId: "u1",
    messages: [{ parentId: null, message: message("u1", "user") }],
  } as ExportedMessageRepository);
  await deleteThreadMessage({
    thread: empty.thread,
    messageId: "u1",
    remoteId: undefined,
  });
  assert.equal(headOf(empty.result()), null);
});

test("deleting the shown reply falls back to the sibling it was regenerated from", async () => {
  const t = threadOver(regenerated());
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "a1b",
    remoteId: undefined,
  });
  assert.deepEqual(idsOf(t.result()), ["u1", "a1a"]);
  assert.equal(headOf(t.result()), "a1a");
});

test("deleting a hidden branch leaves the shown message shown", async () => {
  const t = threadOver({
    headId: "a2",
    messages: [
      { parentId: null, message: message("u1", "user") },
      { parentId: "u1", message: message("a1a", "assistant") },
      { parentId: "u1", message: message("a1b", "assistant") },
      { parentId: "a1a", message: message("u2", "user") },
      { parentId: "u2", message: message("a2", "assistant") },
    ],
  } as ExportedMessageRepository);
  await deleteThreadMessage({
    thread: t.thread,
    messageId: "a1b",
    remoteId: undefined,
  });
  assert.deepEqual(idsOf(t.result()), ["u1", "a1a", "u2", "a2"]);
  assert.equal(headOf(t.result()), "a2");
});
