// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

// The runner cannot resolve the providers store / encryptor specifiers alone.
registerBundlerResolver();
const { store } = installLocalStorageFake();
Object.assign((globalThis.window as { location: object }).location, {
  href: "http://localhost/",
});

const {
  answeringCheckpoint,
  buildExternalRoutingFields,
  buildTitleRequest,
  fallbackTitleFromUserText,
  isLegacyClippedTitle,
  normalizeTitle,
  planLegacyTitleRepairs,
  resolveExternalRouting,
  selectLegacyRepairPage,
  threadsAwaitingImport,
  threadsMissingMessages,
  titleCheckpoint,
  titleFromStream,
} = await import("../src/features/chat/utils/chat-title.ts");
const { useExternalProvidersStore } = await import(
  "../src/features/chat/stores/external-providers-store.ts"
);
type MessageRecord = import("../src/features/chat/types.ts").MessageRecord;
type ThreadRecord = import("../src/features/chat/types.ts").ThreadRecord;
type Chunk = import("../src/features/chat/types/api.ts").OpenAIChatChunk;

const LONG =
  "Can you plot a Mandelbrot set and explain how the escape time algorithm works";

function thread(id: string, title: string): ThreadRecord {
  return { id, title, createdAt: 1, updatedAt: 1 } as ThreadRecord;
}

function userMessage(threadId: string, text: string): MessageRecord {
  return {
    id: `${threadId}-m1`,
    threadId,
    role: "user",
    content: [{ type: "text", text }],
    createdAt: 1,
  } as MessageRecord;
}

/** A high surrogate with no low after it, or a low with no high before it. */
const UNPAIRED_SURROGATE =
  /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

test("a title the sidebar can clip keeps the whole first line", () => {
  assert.equal(fallbackTitleFromUserText(LONG), LONG);
  assert.equal(fallbackTitleFromUserText("  spaced   out  "), "spaced out");
  assert.equal(fallbackTitleFromUserText("first\nsecond"), "first");
  assert.equal(fallbackTitleFromUserText("   "), "New Chat");
});

test("only a pasted wall of text is cut, and with a real ellipsis", () => {
  const wall = "x".repeat(200);
  const title = fallbackTitleFromUserText(wall);
  // 120 UTF-16 units including the ellipsis, which is what the input accepts.
  assert.equal(title.length, 120);
  assert.ok(title.endsWith("…"));
  assert.ok(!title.includes("..."));
});

test("an emoji wall is capped by the same budget the input counts", () => {
  // maxLength counts UTF-16 units, so 120 astral code points would be 240.
  const title = fallbackTitleFromUserText("\u{1F600}".repeat(200));
  assert.ok(title.length <= 120);
  assert.equal(UNPAIRED_SURROGATE.test(title), false);
  assert.ok(title.endsWith("…"));
});

test("a line already inside the budget is stored whole", () => {
  const exact = "y".repeat(120);
  assert.equal(fallbackTitleFromUserText(exact), exact);
});

test("the cap never splits an emoji into a lone surrogate", () => {
  // A lone surrogate survives JSON.stringify but 500s the backend's SQLite bind.
  const line = "x".repeat(119) + "\u{1F600} tail";
  // A raw cut at the budget lands mid-pair.
  assert.equal(UNPAIRED_SURROGATE.test(line.slice(0, 120)), true);
  const title = fallbackTitleFromUserText(line);
  assert.equal(UNPAIRED_SURROGATE.test(title), false);
  // The emoji needs two units and only one is left, so it is left out whole.
  assert.equal(title, "x".repeat(119) + "…");
  assert.equal(title.length, 120);
});

test("a lone surrogate is dropped even when the line is under the cap", () => {
  // The cut sanitises what it walks, so under-cap lines used to be stored as
  // they came, and one unpaired surrogate 500s the backend's title write.
  const line = "x".repeat(60) + "\uD83D";
  assert.ok(line.length <= 120);
  assert.equal(UNPAIRED_SURROGATE.test(line), true);
  const title = fallbackTitleFromUserText(line);
  assert.equal(UNPAIRED_SURROGATE.test(title), false);
  assert.equal(title, "x".repeat(60));
  // A trailing low surrogate with no high before it goes the same way.
  assert.equal(fallbackTitleFromUserText("hi \uDE00"), "hi");
  // A valid pair under the cap is untouched.
  assert.equal(fallbackTitleFromUserText("hi \u{1F600}"), "hi \u{1F600}");
});

test("a legacy title is recognised only against the text it was cut from", () => {
  const legacy = LONG.slice(0, 48) + "...";
  assert.equal(isLegacyClippedTitle(legacy, LONG), true);
  assert.equal(
    isLegacyClippedTitle(legacy, "a different first message"),
    false,
  );
  // A rename that merely ends in "..." is left alone.
  assert.equal(isLegacyClippedTitle("Wait for it...", LONG), false);
  assert.equal(isLegacyClippedTitle(LONG, LONG), false);
});

test("repair rewrites legacy rows and leaves every other row untouched", () => {
  const legacy = LONG.slice(0, 48) + "...";
  const threads = [
    thread("a", legacy),
    thread("b", "Mandelbrot escape time"),
    thread("c", legacy),
  ];
  const messages = new Map<string, MessageRecord[]>([
    ["a", [userMessage("a", LONG)]],
    ["b", [userMessage("b", LONG)]],
    // No stored messages: nothing to rewrite the title from.
    ["c", []],
  ]);

  assert.deepEqual(planLegacyTitleRepairs(threads, messages), [
    { threadId: "a", previousTitle: legacy, openingMessageId: "a-m1", title: LONG },
  ]);
});

test("a drain advances even when a whole page failed and was unmarked", () => {
  // Failures get unmarked for a later refresh. Selecting the next page off the
  // same list would draw them straight back in and never reach the rest.
  const legacy = LONG.slice(0, 48) + "...";
  const threads = ["a", "b", "c", "d"].map((id) => thread(id, legacy));

  const first = selectLegacyRepairPage(threads, new Set(), 2);
  assert.deepEqual(
    first.candidates.map((t) => t.id),
    ["a", "b"],
  );
  // Every write failed, so nothing stayed marked.
  const second = selectLegacyRepairPage(first.rest, new Set(), 2);
  assert.deepEqual(
    second.candidates.map((t) => t.id),
    ["c", "d"],
  );
  assert.equal(second.hasMore, false);
  assert.deepEqual(second.rest, []);
});

test("the opening message is the earliest one, not the first row returned", () => {
  // A local read comes back in index order, so it can start on a later turn.
  const later: MessageRecord = {
    ...userMessage("a", "a later question entirely"),
    id: "a-m9",
    createdAt: 99,
  };
  const opening: MessageRecord = {
    ...userMessage("a", LONG),
    id: "a-m1",
    createdAt: 1,
  };
  const legacy = LONG.slice(0, 48) + "...";

  assert.deepEqual(
    planLegacyTitleRepairs(
      [thread("a", legacy)],
      new Map([["a", [later, opening]]]),
    ),
    // Guarded on the opening message, not the row the array happens to start on.
    [{ threadId: "a", previousTitle: legacy, openingMessageId: "a-m1", title: LONG }],
  );
});

test("two prompts sharing a timestamp break on id, as the backend does", () => {
  // The write is guarded on this id, so both orders must pick the same message.
  const legacy = LONG.slice(0, 48) + "...";
  const first: MessageRecord = { ...userMessage("a", LONG), id: "a-m1" };
  const second: MessageRecord = {
    ...userMessage("a", "a different question"),
    id: "a-m2",
  };

  for (const order of [
    [first, second],
    [second, first],
  ]) {
    assert.deepEqual(
      planLegacyTitleRepairs([thread("a", legacy)], new Map([["a", order]])),
      [
        {
          threadId: "a",
          previousTitle: legacy,
          openingMessageId: "a-m1",
          title: LONG,
        },
      ],
    );
  }
});

test("a page skips rows already tried and reports the leftovers", () => {
  const legacy = LONG.slice(0, 48) + "...";
  const threads = [
    thread("a", legacy),
    thread("b", "a plain title"),
    thread("c", legacy),
    thread("d", legacy),
  ];

  const first = selectLegacyRepairPage(threads, new Set(), 2);
  assert.deepEqual(
    first.candidates.map((t) => t.id),
    ["a", "c"],
  );
  // Without this the rest of a long history waits on an unrelated refresh.
  assert.equal(first.hasMore, true);

  const second = selectLegacyRepairPage(threads, new Set(["a", "c"]), 2);
  assert.deepEqual(
    second.candidates.map((t) => t.id),
    ["d"],
  );
  assert.equal(second.hasMore, false);

  const done = selectLegacyRepairPage(threads, new Set(["a", "c", "d"]), 2);
  assert.deepEqual(done.candidates, []);
  assert.equal(done.hasMore, false);
});

test("a thread the backend has nothing for still gets a local read", () => {
  // A not-yet-imported chat reads empty; an unknown id is missing from the map.
  const messages = new Map<string, MessageRecord[]>([
    ["a", [userMessage("a", LONG)]],
    ["b", []],
  ]);
  assert.deepEqual(threadsMissingMessages(["a", "b", "c"], messages), [
    "b",
    "c",
  ]);
});



test("a chat with nothing stored is left for a later refresh", () => {
  // Its messages may not be imported yet, so a later pass rewrites the title.
  const legacy = LONG.slice(0, 48) + "...";
  const candidates = [thread("a", legacy)];
  const messages = new Map<string, MessageRecord[]>();

  assert.deepEqual(planLegacyTitleRepairs(candidates, messages), []);
  assert.deepEqual(threadsMissingMessages(["a"], messages), ["a"]);

  messages.set("a", [userMessage("a", LONG)]);
  assert.deepEqual(threadsMissingMessages(["a"], messages), []);
  assert.deepEqual(planLegacyTitleRepairs(candidates, messages), [
    { threadId: "a", previousTitle: legacy, openingMessageId: "a-m1", title: LONG },
  ]);
});

test("a chat whose opening prompt is gone is decided, not retried forever", () => {
  // A chat that does have messages is a complete answer: the opening prompt was
  // deleted or edited, so no later pass can prove the title. Unmarking it would
  // re-select it on every refresh, since its title stays clipped.
  const legacy = LONG.slice(0, 48) + "...";
  const candidates = [thread("a", legacy)];
  const messages = new Map<string, MessageRecord[]>([
    ["a", [{ ...userMessage("a", "a different question entirely"), id: "a-m9" }]],
  ]);

  assert.deepEqual(planLegacyTitleRepairs(candidates, messages), []);
  assert.deepEqual(threadsMissingMessages(["a"], messages), []);
  // So it stays marked, and the next page passes over it.
  assert.deepEqual(
    selectLegacyRepairPage(candidates, new Set(["a"]), 100).candidates,
    [],
  );
});

test("an emptied chat is decided, one still importing is not", () => {
  // Both read back as zero messages. The ledger tells them apart: one it knows
  // was imported is simply empty, one it has never seen may still be on its way.
  const ids = ["emptied", "importing", "fine"];
  const messages = new Map<string, MessageRecord[]>([
    ["emptied", []],
    ["importing", []],
    ["fine", [userMessage("fine", LONG)]],
  ]);

  assert.deepEqual(threadsMissingMessages(ids, messages), [
    "emptied",
    "importing",
  ]);
  assert.deepEqual(
    threadsAwaitingImport(ids, messages, new Set(["emptied"])),
    ["importing"],
  );
  // An unreadable ledger decides nothing, so both stay retryable.
  assert.deepEqual(threadsAwaitingImport(ids, messages, new Set()), [
    "emptied",
    "importing",
  ]);
});


function stageConnection(overrides: Record<string, unknown> = {}): void {
  const base = {
    id: "conn-1",
    providerType: "llama_cpp",
    name: "c",
    baseUrl: "http://127.0.0.1:8080/v1",
    models: ["m"],
    createdAt: 1,
    updatedAt: 1,
  };
  store.set(
    "unsloth_chat_external_providers",
    JSON.stringify([{ ...base, ...overrides }]),
  );
}

function deltas(text: string, finishReason: string | null = "stop"): Chunk[] {
  const parts = [...text].map((ch) => ({
    choices: [{ delta: { content: ch } }],
  }));
  return [...parts, { choices: [{ delta: {}, finish_reason: finishReason }] }];
}

async function* iterate(chunks: Chunk[]): AsyncGenerator<Chunk> {
  for (const chunk of chunks) {
    yield chunk;
  }
}
const title = (chunks: Chunk[]) => titleFromStream(iterate(chunks));

// A throwaway 1024-bit RSA public key; only the encrypt path is under test.
const PUBLIC_KEY_PEM = `-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQC1n8QOqkDXkFEOC62kiqZcBCN3
l/DmD+0BGvjg8h1fFJD2Fla1ibcnmKb9Vok+PmR6jm1JX0yu8JHXPw1om01RwQWe
nehl2VzGfdEdNaRoKhW5oVsnnfmxlWJ/qWuV2rDK8DK/6UK9sC/duMkRWaRGdhyl
l+8/fuJc9JDRVzx7HwIDAQAB
-----END PUBLIC KEY-----`;

test.beforeEach(() => {
  store.clear();
  useExternalProvidersStore.getState().setConnectionsEnabled(true);
});

test("a title on a saved connection is routed to it, and streamed", async () => {
  stageConnection();
  const request = await buildTitleRequest(
    "external::conn-1::qwen3-30b",
    "User: hi",
  );

  assert.deepEqual(
    [
      request?.model,
      request?.provider_id,
      request?.provider_type,
      request?.external_model,
      request?.provider_base_url,
      request?.provider_api_type,
      request?.stream,
    ],
    [
      "external::conn-1::qwen3-30b",
      "conn-1",
      "llama_cpp",
      "qwen3-30b",
      "http://127.0.0.1:8080/v1",
      "chat_completions",
      true,
    ],
  );
  stageConnection({ apiType: "responses" });
  const responses = await buildTitleRequest("external::conn-1::m", "x");
  assert.equal(responses?.provider_api_type, "responses");
  stageConnection({ baseUrl: "", hasApiKey: true });
  store.set(
    "unsloth_chat_external_provider_keys",
    JSON.stringify({ "conn-1": "sk-stale" }),
  );
  const saved = await buildTitleRequest("external::conn-1::m", "x");
  assert.deepEqual(
    [saved?.provider_base_url, saved && "encrypted_api_key" in saved],
    [null, false],
  );
  globalThis.fetch = (async () => ({
    ok: true,
    json: async () => ({ public_key: PUBLIC_KEY_PEM }),
  })) as unknown as typeof fetch;
  stageConnection({ providerType: "openai", hasApiKey: false });
  store.set(
    "unsloth_chat_external_provider_keys",
    JSON.stringify({ "conn-1": "sk-browser-held" }),
  );
  const held = await buildTitleRequest("external::conn-1::gpt-5.4", "x");
  assert.notEqual(held?.encrypted_api_key, "sk-browser-held");
  // Envelope: version.wrappedKey.nonce.ciphertext; the RSA-1024 wrap is 128 bytes.
  assert.equal(atob((held?.encrypted_api_key ?? "").split(".")[1] ?? "").length, 128);
  const local = await buildTitleRequest(
    "unsloth/gemma-4-E2B-it-GGUF",
    "User: hi",
  );
  assert.deepEqual(
    [local?.model, local?.stream, local && "provider_id" in local],
    ["unsloth/gemma-4-E2B-it-GGUF", true, false],
  );
  assert.deepEqual(
    local?.messages?.map((m) => m.role),
    ["system", "user"],
  );
  assert.equal(local?.messages?.[1]?.content, "User: hi");
  assert.deepEqual(
    [
      local?.enable_tools,
      local?.enable_thinking,
      local?.reasoning_effort,
      local?.max_tokens,
    ],
    [false, false, "none", 24],
  );
  assert.match(
    String(local?.messages?.[0]?.content),
    /2-6 words, no quotes, no punctuation, ASCII only, do not echo input/,
  );
});

test("a connection that cannot serve the title yields no request at all", async () => {
  stageConnection({ providerType: "openai", hasApiKey: false });
  assert.equal(await buildTitleRequest("external::conn-1::gpt-5.4", "x"), null);
  assert.equal(await buildTitleRequest("external::gone::m", "x"), null);
  stageConnection({ providerType: "openai", hasApiKey: false });
  store.set(
    "unsloth_chat_external_provider_keys",
    JSON.stringify({ "conn-1": "   " }),
  );
  assert.equal(await buildTitleRequest("external::conn-1::gpt-5.4", "x"), null);
  store.set("unsloth_chat_external_provider_keys", JSON.stringify({}));
  stageConnection({
    providerType: "openai",
    hasApiKey: false,
    authKind: "chatgpt_oauth",
  });
  assert.ok(await buildTitleRequest("external::conn-1::gpt-5.4", "x"));
  stageConnection({
    providerType: "gemini",
    hasApiKey: false,
    baseUrl: "https://gw.example/v1",
  });
  assert.ok(await buildTitleRequest("external::conn-1::gemini-2.5-pro", "x"));

  stageConnection();
  useExternalProvidersStore.getState().setConnectionsEnabled(false);
  assert.equal(
    await buildTitleRequest("external::conn-1::qwen3-30b", "x"),
    null,
  );
  assert.ok(await buildTitleRequest("unsloth/gemma-4-E2B-it-GGUF", "x"));
});

test("reasoning is asked off only where the connection's model allows it", async () => {
  async function effortFor(providerType: string, model: string) {
    stageConnection({ providerType, hasApiKey: true, models: [model] });
    return (await buildTitleRequest(`external::conn-1::${model}`, "x"))
      ?.reasoning_effort;
  }

  for (const [providerType, model, expected] of [
    ["openai", "gpt-5", "minimal"],
    ["openai", "o3", "low"],
    ["openai_codex", "gpt-5", "minimal"],
    // No reasoning at all: either field would reach OpenAI as reasoning.effort, a 400.
    ["openai", "gpt-4o", undefined],
    ["openai", "gpt-5-chat-latest", undefined],
    ["gemini", "gemini-2.5-pro", "none"],
    ["llama_cpp", "qwen3-30b", "none"],
    // Its own levels would floor "none" at "medium" if clamped like OpenAI.
    ["mistral", "magistral-medium-latest", "none"],
  ] as const) {
    assert.equal(
      await effortFor(providerType, model),
      expected,
      `${providerType}/${model}`,
    );
  }
});

test("what answered is read from the reply, whichever kind of turn it was", () => {
  const external = {
    providerId: "conn-a",
    providerType: "openai",
    externalModel: "gpt-5.5",
  };
  const research = (
    inferenceRequest: Record<string, unknown>,
    status = "completed",
  ) =>
    answeringCheckpoint({
      researchRun: { status, config: { inferenceRequest } },
    });
  const withRun = (modelId: string, status = "completed") =>
    answeringCheckpoint({
      responseDetails: { modelId },
      researchRun: { status, config: { inferenceRequest: external } },
    });

  assert.equal(
    answeringCheckpoint({
      responseDetails: { modelId: "external::conn-a::m" },
    }),
    "external::conn-a::m",
  );
  assert.equal(
    research({ model: "gpt-5.5", ...external }),
    "external::conn-a::gpt-5.5",
  );
  assert.equal(
    research({ model: "unsloth/gemma-4-E2B-it-GGUF" }),
    "unsloth/gemma-4-E2B-it-GGUF",
  );
  assert.equal(research({ model: "external::conn-a::m" }), "");
  for (const missing of [
    "providerId",
    "providerType",
    "externalModel",
  ] as const) {
    const partial: Record<string, unknown> = { model: "m", ...external };
    delete partial[missing];
    assert.equal(research(partial), "m", `missing ${missing}`);
    assert.equal(
      research({ ...partial, model: "external::conn-a::m" }),
      "",
      `missing ${missing}, external id`,
    );
    assert.equal(
      research({ model: "m", ...external, [missing]: "" }),
      "m",
      `blank ${missing}`,
    );
  }
  for (const status of ["running", "cancelled", "failed"]) {
    assert.equal(research({ model: "m", ...external }, status), "", status);
  }
  assert.equal(
    withRun("unsloth/gemma-4-E2B-it-GGUF"),
    "unsloth/gemma-4-E2B-it-GGUF",
  );
  assert.equal(withRun("external::conn-b::m"), "external::conn-b::m");
  assert.equal(withRun("external::conn-b::m", "failed"), "external::conn-b::m");
  assert.equal(withRun(""), "external::conn-a::gpt-5.5");
  assert.deepEqual(
    [
      answeringCheckpoint(undefined),
      answeringCheckpoint({ responseDetails: { modelId: 7 } }),
    ],
    ["", ""],
  );
});

test("a title is asked of the connection that answered, or of no connection", () => {
  const A = "external::conn-a::m";
  const B = "external::conn-b::m";
  const local = "unsloth/gemma-4-E2B-it-GGUF";
  for (const [answered, active, expected] of [
    [A, B, A],
    [A, local, A],
    [A, "", A],
    ["", B, ""],
    [local, B, ""],
    ["", local, local],
    [local, local, local],
    [local, "", ""],
    ["", "", ""],
  ] as const) {
    assert.equal(
      titleCheckpoint(answered, active),
      expected,
      `${answered || "-"} + ${active || "-"}`,
    );
  }
});

test("the title is assembled from the deltas, unless it was cut short or reasoned", async () => {
  const part = (text: string): Chunk => ({
    choices: [{ delta: { content: [{ type: "text", text }] } }],
  });

  assert.equal(
    await title(deltas('"Matrix Inversion" — Steps')),
    "Matrix Inversion Steps",
  );
  assert.equal(
    await title([part("Matrix Inversion"), part(" Steps"), ...deltas("")]),
    "Matrix Inversion Steps",
  );

  const cut = deltas("Inverting A Three By Three Matri", "length");
  assert.equal(await title(cut), null);
  assert.equal(
    await title([...cut, { choices: [] }, { choices: [{ delta: {} }] }]),
    null,
  );
  assert.equal(
    await title([
      {
        choices: [
          { delta: { content: [{ type: "thinking", thinking: "hmm" }] } },
        ],
      },
      ...deltas(""),
    ]),
    null,
  );
  // A closed reasoning summary (a model that cannot turn reasoning off) is dropped, the title kept.
  assert.equal(
    await title(deltas("<think>The user asks about gardens.</think>Spring Garden Plan")),
    "Spring Garden Plan",
  );
  for (const raw of [
    "<think>hmm</think>",
    "<THINK>hmm Ok",
    "hmm</ThInK> Ok",
    "",
  ]) {
    assert.equal(await title(deltas(raw)), null, raw);
  }
});

test("the answer is reduced to a short, plain, ASCII line", () => {
  for (const [raw, expected] of [
    ['  "Matrix Inversion" — Steps  ', "Matrix Inversion Steps"],
    [
      "Title: How To Invert A Three By Three Matrix Quickly",
      "How To Invert A Three By",
    ],
    ["Matrix Steps\nExplanation follows", "Matrix Steps"],
    ["`Matrix` 'Inversion' Steps", "Matrix Inversion Steps"],
    ["A.! B? C; D, E: F", "A B C D E F"],
    ["   Title: Matrix Steps", "Matrix Steps"],
    ["Assistant: what is this", null],
    ["   User: what is this", null],
    [`${"a".repeat(59)} b`, "a".repeat(59)],
    ["Base: what is this", null],
    ["Café — Résumé Matrix", "Caf R sum Matrix"],
    ["x".repeat(80), "x".repeat(60)],
    ["Inverting Title: A Matrix", "Inverting Title A Matrix"],
    ["Title : Matrix Steps", "Matrix Steps"],
    ["Asking User: what is", "Asking User what is"],
    ["   ", null],
    ["User: what is this", null],
    ["LoRA: what is this", null],
  ] as const) {
    assert.equal(normalizeTitle(raw), expected, JSON.stringify(raw));
  }
});

test("a forced refresh re-fetches the public key instead of reusing the cached one", async () => {
  let fetched = 0;
  globalThis.fetch = (async () => {
    fetched += 1;
    return { ok: true, json: async () => ({ public_key: PUBLIC_KEY_PEM }) };
  }) as unknown as typeof fetch;
  stageConnection({ providerType: "openai", hasApiKey: false });
  store.set(
    "unsloth_chat_external_provider_keys",
    JSON.stringify({ "conn-1": "sk-browser-held" }),
  );
  const target = resolveExternalRouting("external::conn-1::gpt-5.4");
  if (target.kind !== "external") throw new Error("expected external");

  await buildExternalRoutingFields(target);
  const cached = fetched;
  await buildExternalRoutingFields(target);
  assert.equal(fetched, cached, "an ordinary build reuses the cached key");
  await buildExternalRoutingFields(target, { forceRefreshPublicKey: true });
  assert.equal(fetched, cached + 1, "a forced one refetches");
});

// Pinned as source text (JSX); gaps exclude braces so they cannot match past the block.
test("the title is wired to what answered, and built inside the fallback boundary", () => {
  const provider = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  ).replace(/\s+/g, " ");
  for (const rule of [
    /const answeredWith = answeringCheckpoint\( firstAssistant\?\.metadata\?\.custom, \);/,
    /if \(!payload\.checkpoint\) return null;/,
    /checkpoint: titleCheckpoint\( answeredWith, useChatRuntimeStore\.getState\(\)\.params\.checkpoint, \),/,
    /try \{ [^{}]*?const request = await buildTitleRequest\( payload\.checkpoint, parts\.join\("\\n"\), \); if \(!request\) return null; [^{}]*?return await titleFromStream\([^{}]*?\); \} catch \{ [^{}]*?return null; \}/,
  ]) {
    assert.match(provider, rule);
  }
});

// The backend forwards these verbatim, so a strict chat-completions endpoint 400s on
// any the provider does not take, and the title is lost.
test("title sampling fields follow the provider's capabilities", async () => {
  const fields = (r: Awaited<ReturnType<typeof buildTitleRequest>>) =>
    ["temperature", "top_p", "top_k", "repetition_penalty"].filter(
      (k) => r !== null && k in r,
    );
  const local = await buildTitleRequest("unsloth/gemma-4-E2B-it-GGUF", "x");
  assert.deepEqual(fields(local), [
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
  ]);
  stageConnection({ providerType: "mistral", hasApiKey: true });
  const mistral = await buildTitleRequest("external::conn-1::mistral-small", "x");
  assert.deepEqual(fields(mistral), ["temperature", "top_p"]);
  stageConnection({ providerType: "openai", hasApiKey: true });
  const openai = await buildTitleRequest("external::conn-1::gpt-5.4", "x");
  assert.deepEqual(fields(openai), []);
});

test("a Responses custom connection still asks reasoning off", async () => {
  for (const [model, expected] of [
    ["gpt-5.4", "none"],
    ["gpt-5", "minimal"],
  ] as const) {
    stageConnection({ providerType: "custom", apiType: "responses", models: [model] });
    const request = await buildTitleRequest(`external::conn-1::${model}`, "x");
    assert.equal(request?.reasoning_effort, expected, model);
  }
});

test("the title budget is floored at the provider's minimum output", async () => {
  const local = await buildTitleRequest("unsloth/gemma-4-E2B-it-GGUF", "x");
  assert.equal(local?.max_tokens, 24);
  stageConnection({ providerType: "kimi", hasApiKey: true });
  const kimi = await buildTitleRequest("external::conn-1::kimi-k2-thinking", "x");
  assert.equal(kimi?.max_tokens, 16000);
  // Thinking that cannot be turned off counts toward the cap, so it gets headroom.
  for (const [providerType, model, expected] of [
    ["gemini", "gemini-2.5-pro", 1024],
    ["gemini", "gemini-2.5-flash", 64],
    ["openai", "o3", 1024],
    ["openai", "gpt-5.4", 64],
  ] as const) {
    stageConnection({ providerType, baseUrl: "", hasApiKey: true, models: [model] });
    const request = await buildTitleRequest(`external::conn-1::${model}`, "x");
    assert.equal(request?.max_tokens, expected, model);
  }
});
