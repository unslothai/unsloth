// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { CarriedPart } from "../src/features/chat/utils/chat-generation-recovery.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  generationRawContent,
  restoreCarriedParts,
  restoreCarriedPartsFromRaw,
  recoveredContentToImport,
} = await import("../src/features/chat/utils/chat-generation-recovery.ts");
const { createGenerationToolRecovery } = await import(
  "../src/features/chat/utils/generation-tool-recovery.ts"
);

type Part = Record<string, unknown>;

/** xorshift, so a failing seed replays exactly. */
function rng(seed: number) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

// Astral pairs, since offsets are UTF-16 code units. No angle brackets: a think tag is a parser question.
const ALPHABET = [
  "a",
  "b",
  " ",
  "\n",
  "\t",
  "é",
  "👍🏽",
  "😀",
  "中",
  "​",
  "'",
  '"',
  "\\",
  "&",
  "0",
];

function randomText(next: () => number): string {
  const length = Math.floor(next() * 12);
  let out = "";
  for (let i = 0; i < length; i++) {
    out += ALPHABET[Math.floor(next() * ALPHABET.length)];
  }
  return out;
}

const CARRIED_KINDS = ["tool-call", "source", "file", "image"] as const;

function randomContent(next: () => number, seed: number): Part[] {
  const length = Math.floor(next() * 12);
  const parts: Part[] = [];
  for (let i = 0; i < length; i++) {
    const roll = next();
    if (roll < 0.38) {
      parts.push({ type: "text", text: randomText(next) });
    } else if (roll < 0.68) {
      parts.push({ type: "reasoning", text: randomText(next) });
    } else {
      const kind = CARRIED_KINDS[Math.floor(next() * CARRIED_KINDS.length)];
      parts.push(
        kind === "tool-call"
          ? {
              type: "tool-call",
              toolCallId: `c${seed}_${i}`,
              toolName: "edit_file",
              args: {},
              argsText: "{}",
              ...(next() < 0.5 ? { result: "ok" } : {}),
            }
          : { type: kind, id: `p${seed}_${i}`, url: "https://example.com" },
      );
    }
  }
  return parts;
}

const isSpoken = (part: Part) =>
  part.type === "text" || part.type === "reasoning";
const spokenText = (parts: Part[], type: string) =>
  parts
    .filter((part) => part.type === type)
    .map((part) => String(part.text ?? ""))
    .join("");
const identify = (part: Part) => String(part.toolCallId ?? part.id ?? "?");
const carriedIds = (parts: Part[]) =>
  parts.filter((part) => !isSpoken(part)).map(identify);

function prefixLengths(parts: Part[]): [string, number][] {
  const out: [string, number][] = [];
  let seen = 0;
  for (const part of parts) {
    if (isSpoken(part)) seen += String(part.text ?? "").length;
    else out.push([identify(part), seen]);
  }
  return out.sort();
}

const roundTrip = (content: Part[]) => {
  const { raw, reasoningOpen, carried } = generationRawContent(content);
  return restoreCarriedPartsFromRaw(
    reasoningOpen ? `${raw}</think>` : raw,
    carried,
  ) as unknown as Part[];
};

test("a raw round trip keeps every card, its order, its position and the words", () => {
  for (let seed = 1; seed <= 4000; seed++) {
    const content = randomContent(rng(seed * 2654435761), seed);
    const where = () => `seed ${seed}: ${JSON.stringify(content)}`;
    const out = roundTrip(content);

    assert.deepEqual(carriedIds(out), carriedIds(content), where());
    assert.equal(
      spokenText(out, "text"),
      spokenText(content, "text"),
      `text ${where()}`,
    );
    assert.equal(
      spokenText(out, "reasoning"),
      spokenText(content, "reasoning"),
      `reasoning ${where()}`,
    );
    assert.deepEqual(prefixLengths(out), prefixLengths(content), where());

    const shape = (parts: Part[]) =>
      parts.map((part) => [part.type, identify(part), part.text]);
    assert.deepEqual(shape(roundTrip(out)), shape(out), `twice ${where()}`);
  }
});

test("restoring onto parsed parts keeps the same properties", () => {
  for (let seed = 1; seed <= 3000; seed++) {
    const content = randomContent(rng(seed * 40503), seed);
    const where = () => `seed ${seed}: ${JSON.stringify(content)}`;
    const { carried } = generationRawContent(content);
    const out = restoreCarriedParts(
      content.filter(isSpoken),
      carried,
    ) as Part[];

    assert.deepEqual(carriedIds(out), carriedIds(content), where());
    assert.equal(
      spokenText(out, "text"),
      spokenText(content, "text"),
      `text ${where()}`,
    );
    assert.equal(
      spokenText(out, "reasoning"),
      spokenText(content, "reasoning"),
      `reasoning ${where()}`,
    );
    assert.deepEqual(prefixLengths(out), prefixLengths(content), where());
  }
});

type Event = Record<string, unknown>;

/** `cards` counts what the adapter would draw: a start on an already-open id reuses its card. */
function randomEvents(next: () => number): { events: Event[]; cards: number } {
  const rounds = 1 + Math.floor(next() * 5);
  const events: Event[] = [];
  const open: string[] = [];
  let cards = 0;
  for (let i = 0; i < rounds; i++) {
    const roll = next();
    const id = roll < 0.25 ? "" : roll < 0.5 ? "call_0" : `call_${i}`;
    events.push({
      type: "tool_start",
      tool_call_id: id,
      tool_name: "edit_file",
      arguments: { path: `f${i}.ts` },
    });
    const reused = Boolean(id) && open.includes(id);
    if (!reused) cards += 1;
    if (id && !reused) open.push(id);
    if (next() < 0.75) {
      events.push({ type: "tool_end", tool_call_id: id, result: `r${i}` });
      const at = open.indexOf(id);
      if (at !== -1) open.splice(at, 1);
    }
  }
  return { events, cards };
}

test("replay opens one card per call and lands each result once", () => {
  for (let seed = 1; seed <= 4000; seed++) {
    const { events, cards } = randomEvents(rng(seed * 2246822519));
    const where = () => `seed ${seed}: ${JSON.stringify(events)}`;

    const carried: CarriedPart[] = [];
    const replay = createGenerationToolRecovery(carried, "run", 0).apply;
    events.forEach((event, i) => replay(event, i, i + 1));
    assert.equal(carried.length, cards, where());

    const results = carried
      .map(({ part }) => (part as Part).result)
      .filter((value) => value !== undefined);
    assert.equal(new Set(results).size, results.length, `twice ${where()}`);

    // The follow stream can deliver an update more than once.
    const doubled: CarriedPart[] = [];
    const again = createGenerationToolRecovery(doubled, "run", 0).apply;
    events.forEach((event, i) => {
      again(event, i, i + 1);
      again(event, i, i + 1);
    });
    assert.deepEqual(
      doubled.map(({ part }) => part),
      carried.map(({ part }) => part),
      `repeated ${where()}`,
    );

    const seeded: CarriedPart[] = [
      { at: 0, part: { type: "source", id: "kept", url: "https://e.com" } },
    ];
    const third = createGenerationToolRecovery(seeded, "run", 0).apply;
    events.forEach((event, i) => third(event, i, i + 1));
    assert.ok(
      seeded.some(({ part }) => (part as Part).id === "kept"),
      `kept ${where()}`,
    );
  }
});

test("an import that is not the view keeps the recovered body whole", () => {
  for (let seed = 1; seed <= 3000; seed++) {
    const next = rng(seed * 22695477);
    const view = randomContent(next, seed);
    const recovered =
      next() < 0.5
        ? randomContent(next, seed + 100000)
        : [...view, { type: "text", text: randomText(next) }];
    const out = recoveredContentToImport(view, recovered) as Part[];
    if (out === view) continue;
    const where = () => `seed ${seed}`;
    for (const id of carriedIds(recovered)) {
      assert.ok(carriedIds(out).includes(id), `dropped ${id} ${where()}`);
    }
    assert.equal(
      spokenText(out, "text"),
      spokenText(recovered, "text"),
      `text ${where()}`,
    );
  }
});

// A think tag can arrive split across two chunks, so a card offset -- the raw length at a
// chunk boundary -- can land inside one. The generator above keeps angle brackets out, so
// these two are spelled out.
test("a card offset inside a split think tag lands past the tag, not in it", () => {
  const raw = "<think>\n\nreasoned";
  const card: Part = {
    type: "tool-call",
    toolCallId: "c1",
    toolName: "edit_file",
    args: {},
  };
  const out = restoreCarriedPartsFromRaw(raw, [
    { at: "<thi".length, part: card },
  ]) as unknown as Part[];

  assert.deepEqual(
    out.map((part) => part.type),
    ["tool-call", "reasoning"],
  );
  assert.equal(out[1].text, "\n\nreasoned");
  assert.equal(generationRawContent(out).raw, raw);
});

test("cards on both sides of a split think tag do not duplicate it", () => {
  const raw = "<think>\n\nreasoned";
  const carried: CarriedPart[] = [
    { at: "<thi".length, part: { type: "source", id: "s1" } },
    {
      at: "<think>".length,
      part: {
        type: "tool-call",
        toolCallId: "c1",
        toolName: "edit_file",
        args: {},
      },
    },
  ];
  const out = restoreCarriedPartsFromRaw(raw, carried) as unknown as Part[];

  assert.deepEqual(
    out.map((part) => part.type),
    ["source", "tool-call", "reasoning"],
  );
  // The tag survives exactly once: projecting the reply back has to give the raw it came from,
  // or the next publish compares the reply against a body it never streamed.
  assert.equal(generationRawContent(out).raw, raw);
});
