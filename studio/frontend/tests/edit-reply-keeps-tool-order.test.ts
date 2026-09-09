// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Exported = {
  headId: string | null;
  messages: { parentId: string | null; message: Record<string, unknown> }[];
};

type Part = Record<string, unknown>;

type Module = {
  extractTaggedText: (content: unknown) => string;
  updateThreadMessage: (args: {
    thread: { export: () => Exported; import: (data: Exported) => void };
    messageId: string;
    remoteId: string | undefined;
    newText: string;
    isIncognito: boolean;
  }) => Promise<Part[]>;
};

const SEARCH: Part = {
  type: "tool-call",
  toolCallId: "call-1",
  toolName: "web_search",
  args: { query: "unsloth" },
  result: { ok: true },
};
const PYTHON: Part = {
  type: "tool-call",
  toolCallId: "call-2",
  toolName: "python",
  args: { code: "print(1)" },
};

function harness() {
  const saved: Record<string, unknown>[] = [];
  const module = loadWithStubs<Module>(
    new URL(
      "../src/features/chat/utils/update-thread-message.ts",
      import.meta.url,
    ),
    {
      "../api/chat-api": {
        saveChatMessage: async (record: Record<string, unknown>) => {
          saved.push(record);
          return record;
        },
        // delete-thread-message loads for real under the passthrough below, and it
        // reads the thread back; nothing here drives that path.
        listChatMessages: async () => [],
      },
      "@assistant-ui/core/internal": { MessageRepository: class {} },
      // The persistence boundary. Cut here so the passthrough stops before db.ts and
      // its IndexedDB driver, which no node test can load: this suite is about the
      // record's shape and part order, not about storing it.
      "./chat-history-storage": {
        ensureStoredChatThread: async () => undefined,
        syncStoredChatMessages: async () => undefined,
      },
    },
    // The record this test asserts on is built by the real exportedItemToRecord, and
    // the metadata it strips comes from the real RESEARCH_METADATA_KEYS, so those
    // siblings load rather than being faked.
    { relativePassthrough: true },
  );
  return { module, saved };
}

function thread(content: Part[]): Exported {
  return {
    headId: "a0",
    messages: [
      {
        parentId: null,
        message: {
          id: "u0",
          role: "user",
          content: [{ type: "text", text: "search for it" }],
          createdAt: new Date(1000),
        },
      },
      {
        parentId: "u0",
        message: {
          id: "a0",
          role: "assistant",
          content,
          createdAt: new Date(2000),
        },
      },
    ],
  };
}

async function save(content: Part[], newText?: string) {
  const h = harness();
  const exported = thread(content);
  const text = newText ?? h.module.extractTaggedText(content);
  const result = await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText: text,
    isIncognito: false,
  });
  return { result, saved: h.saved[0], text };
}

test("a no-op Save leaves the tool card between the two sentences", async () => {
  const content = [
    { type: "text", text: "Let me look that up." },
    SEARCH,
    { type: "text", text: "Here is what I found." },
  ];

  const { result } = await save(content);

  assert.deepEqual(
    result.map((part) => part.type),
    ["text", "tool-call", "text"],
  );
  assert.equal(result[0].text, "Let me look that up.");
  assert.equal(result[1], SEARCH);
  assert.equal(result[2].text, "Here is what I found.");
});

test("what is written to the server is what the thread shows", async () => {
  const content = [
    { type: "text", text: "Let me look that up." },
    SEARCH,
    { type: "text", text: "Here is what I found." },
  ];

  const { result, saved } = await save(content);

  assert.deepEqual(saved.content, result);
});

test("several cards keep their order and the prose between them", async () => {
  const content = [
    { type: "text", text: "First I search." },
    SEARCH,
    { type: "text", text: "Then I compute." },
    PYTHON,
    { type: "text", text: "Done." },
  ];

  const { result } = await save(content);

  assert.deepEqual(
    result.map((part) => part.type),
    ["text", "tool-call", "text", "tool-call", "text"],
  );
  assert.equal(result[1], SEARCH);
  assert.equal(result[3], PYTHON);
});

test("reasoning still round-trips alongside a card", async () => {
  const content = [
    { type: "reasoning", text: "The user wants a lookup." },
    SEARCH,
    { type: "text", text: "Here is what I found." },
  ];

  const { result } = await save(content);

  assert.deepEqual(
    result.map((part) => part.type),
    ["reasoning", "tool-call", "text"],
  );
});

test("an edit that rewrites the prose still keeps the card where it was", async () => {
  const content = [
    { type: "text", text: "Let me look that up." },
    SEARCH,
    { type: "text", text: "Here is what I found." },
  ];
  const { text } = await save(content);

  const { result } = await save(
    content,
    text.replace("Here is what I found.", "Rewritten answer."),
  );

  assert.deepEqual(
    result.map((part) => part.type),
    ["text", "tool-call", "text"],
  );
  assert.equal(result[2].text, "Rewritten answer.");
});

test("deleting a marker keeps the call rather than dropping it", async () => {
  const content = [
    { type: "text", text: "Let me look that up." },
    SEARCH,
    { type: "text", text: "Here is what I found." },
  ];

  const { result } = await save(
    content,
    "Let me look that up.\n\nHere is what I found.",
  );

  assert.deepEqual(result, [
    { type: "text", text: "Let me look that up.\n\nHere is what I found." },
    SEARCH,
  ]);
});

test("deleting one marker leaves the other card where its own marker is", async () => {
  const content = [
    { type: "text", text: "First I search." },
    SEARCH,
    { type: "text", text: "Then I compute." },
    PYTHON,
    { type: "text", text: "Done." },
  ];
  const shown = await save(content).then((r) => r.text);

  const { result } = await save(
    content,
    shown.replace(/<TOOL 1:[^>]*>\n\n/, ""),
  );

  // The python card is still the one between "Then I compute." and "Done."; only the
  // card whose marker went away is moved, and it is moved to the end rather than lost.
  assert.deepEqual(result, [
    { type: "text", text: "First I search.\n\nThen I compute." },
    PYTHON,
    { type: "text", text: "Done." },
    SEARCH,
  ]);
});

test("moving a marker moves that card and no other", async () => {
  const content = [
    { type: "text", text: "First I search." },
    SEARCH,
    { type: "text", text: "Then I compute." },
    PYTHON,
    { type: "text", text: "Done." },
  ];

  const { result } = await save(
    content,
    "First I search.\n\n<TOOL 2: python>\n\nThen I compute.\n\n<TOOL 1: web_search>\n\nDone.",
  );

  assert.deepEqual(result, [
    { type: "text", text: "First I search." },
    PYTHON,
    { type: "text", text: "Then I compute." },
    SEARCH,
    { type: "text", text: "Done." },
  ]);
});

test("a half-deleted marker keeps the prose that follows it", async () => {
  const head = "Here is the plan.";
  const tail = "I searched and found three papers.";
  const content = [
    { type: "text", text: head },
    SEARCH,
    { type: "text", text: tail },
  ];
  const shown = await save(content).then((r) => r.text);
  // Whatever the marker looks like, it is what sits between the two sentences.
  const marker = shown.slice(head.length + 2, shown.length - tail.length - 2);

  // Backspacing into the marker used to turn every following sentence into a part the
  // save silently dropped, and the loss was written to the server.
  const broken = [
    marker.slice(0, -1),
    marker.slice(1),
    marker.slice(0, Math.ceil(marker.length / 2)),
  ];
  for (const half of broken) {
    const { result } = await save(content, `${head}\n\n${half}\n\n${tail}`);
    const text = result
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("\n\n");
    assert.ok(
      text.includes(head) && text.includes(tail),
      `prose lost for ${JSON.stringify(half)}`,
    );
    assert.equal(
      result.filter((part) => part.type === "tool-call").length,
      1,
      `card lost for ${JSON.stringify(half)}`,
    );
  }
});

test("a reply that is only a tool card keeps it", async () => {
  const { result } = await save([SEARCH]);

  assert.deepEqual(result, [SEARCH]);
});

function tool(id: string, name = "web_search"): Part {
  return { type: "tool-call", toolCallId: id, toolName: name, args: {}, result: {} };
}

async function roundTrip(content: Part[]): Promise<Part[]> {
  const { result } = await save(content);
  return result;
}

const shapes: any[][] = [
  [{ type: "text", text: "A" }, tool("1"), { type: "text", text: "B" }],
  [tool("1"), { type: "text", text: "A" }],
  [{ type: "text", text: "A" }, tool("1")],
  [tool("1"), tool("2")],
  [{ type: "text", text: "A" }, tool("1"), { type: "text", text: "B" }, tool("2"),
   { type: "text", text: "C" }],
  [{ type: "reasoning", text: "R" }, tool("1"), { type: "text", text: "A" }],
  [{ type: "text", text: "A" }, { type: "source", title: "S", url: "u" },
   { type: "text", text: "B" }],
  [{ type: "text", text: "A" }, tool("1"), tool("2"), { type: "text", text: "B" }],
  [{ type: "reasoning", text: "R" }],
  [tool("1")],
];

test("a no-op save preserves the part sequence for every shape", async () => {
  for (const shape of shapes) {
    const out = await roundTrip(shape);
    assert.deepEqual(out.map((p: any) => p.type), shape.map((p: any) => p.type),
      `shape ${JSON.stringify(shape.map((p: any) => p.type))}`);
    assert.deepEqual(
      out.filter((p: any) => p.type === "tool-call").map((p: any) => p.toolCallId),
      shape.filter((p: any) => p.type === "tool-call").map((p: any) => p.toolCallId));
  }
});

test("a second no-op save changes nothing further (idempotent)", async () => {
  for (const shape of shapes) {
    const once = await roundTrip(shape);
    const twice = await roundTrip(once);
    assert.deepEqual(twice.map((p: any) => p.type), once.map((p: any) => p.type));
  }
});

test("prose that mentions the marker tag is not mistaken for one", async () => {
  const content = [
    { type: "text", text: "Write <TOOL>web_search</TOOL> to call it." },
    tool("1"),
    { type: "text", text: "Done." },
  ];
  const out = await roundTrip(content);
  // Not just the count: the sentence has to come back whole, with the card still
  // between it and "Done." rather than wedged into the middle of it.
  assert.deepEqual(out, content);
});

test("prose that spells a real marker is not mistaken for that card's marker", async () => {
  // A reply explaining the marker syntax, that itself used the tool it names: the
  // literal occurrence sits BEFORE the card's own marker and reads exactly like it.
  const content = [
    { type: "text", text: "Your edit box shows <TOOL 1: web_search> where the card sits." },
    SEARCH,
    { type: "text", text: "That is all it means." },
  ];

  assert.deepEqual(await roundTrip(content), content);
});

test("text that already contains an escaped marker round-trips too", async () => {
  const content = [
    { type: "text", text: "Write <\\TOOL 1: web_search> to show the escape." },
    SEARCH,
    { type: "text", text: "Done." },
  ];

  assert.deepEqual(await roundTrip(content), content);
});

test("an indented code block after a card keeps its indentation", async () => {
  // Four leading spaces are a Markdown code block, not padding: trimming them at the
  // marker boundary would render the reply differently after a no-op Save.
  const content = [
    { type: "text", text: "Here is the fix:" },
    SEARCH,
    { type: "text", text: "    const x = 1;\n    return x;" },
  ];

  assert.deepEqual(await roundTrip(content), content);
});

test("a raw string part is prose, and does not consume a card's slot", async () => {
  // Legacy and imported rows can hold a bare string. extractTaggedText emits it as
  // prose without numbering it, so restoration must not count it as a card either.
  const legacy = ["intro", SEARCH, { type: "text", text: "answer" }] as unknown as Part[];
  const { result } = await save(legacy);

  assert.deepEqual(result, [
    { type: "text", text: "intro" },
    SEARCH,
    { type: "text", text: "answer" },
  ]);
});

test("a reply stored as one plain string round-trips, markers and all", async () => {
  // Legacy and imported rows can hold the whole reply as a top-level string, which
  // takes its own path out of extractTaggedText and has to be escaped there too.
  const content = "Use <TOOL 1: web_search> literally" as unknown as Part[];

  const { result } = await save(content);

  assert.deepEqual(result, [
    { type: "text", text: "Use <TOOL 1: web_search> literally" },
  ]);
});

test("a code block quoting the marker syntax survives a no-op save", async () => {
  const content = [
    { type: "text", text: "```xml\n<TOOL>read_file</TOOL>\n```" },
    tool("1"),
    { type: "text", text: "Done." },
  ];

  assert.deepEqual(await roundTrip(content), content);
});

test("a no-op save returns the very same text, part for part", async () => {
  for (const shape of shapes) {
    assert.deepEqual(await roundTrip(shape), shape,
      `shape ${JSON.stringify(shape.map((p: any) => p.type))}`);
  }
});

test("no tool part is ever dropped, whatever the edit", async () => {
  for (const shape of shapes) {
    const { result } = await save(shape, "the user deleted everything");
    const before = shape.filter((p) => p.type !== "text" && p.type !== "reasoning");
    const after = result.filter((p) => p.type !== "text" && p.type !== "reasoning");
    assert.equal(
      after.length,
      before.length,
      `non-text parts lost for ${JSON.stringify(shape.map((p) => p.type))}`,
    );
  }
});
