// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What the gate does to the plugin, as opposed to what the gate decides.
 *
 * Streamdown calls `highlight()` from an effect keyed on the code string, so a block that has
 * stopped streaming calls in exactly ONCE and never again. Everything here therefore turns on the
 * retained callback: it is the only way a settled block's rendering can change afterwards, and it
 * is the only way this mechanism can be wrong at runtime after passing the decision tests.
 *
 * The claim these tests exist to protect is that PLAIN and HIGHLIGHTED are the same text laid out
 * the same way. The line count and the per-line text are asserted equal on every swap, because
 * that equality is what makes the swap layout-neutral, and a block that changed height under a
 * scrolling user would be a worse bug than the slow DOM this removes.
 */

import assert from "node:assert/strict";
import test from "node:test";
import type {
  HighlightOptions,
  HighlightResult,
  ThemeInput,
} from "@streamdown/code";

import {
  type CodeHighlightGate,
  createCodeHighlightGate,
} from "../src/components/assistant-ui/code-highlight-gate.ts";
import {
  CODE_FENCE_ATTRIBUTE,
  MIN_INCREMENTAL_CHARS,
  createCodePlugin,
} from "../src/components/assistant-ui/code-plugin.ts";

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];
const LANGUAGE = "typescript" as HighlightOptions["language"];

const SOURCE = `${Array.from({ length: 120 }, (index_) => index_)
  .map(
    (_, index) =>
      `export const value_${index} = { id: ${index}, label: "row ${index}" };`,
  )
  .join("\n")}\n`.trimEnd();

// Long enough to take the incremental path, which is the one a stream depends on.
assert.ok(SOURCE.length > MIN_INCREMENTAL_CHARS);

type Plugin = ReturnType<typeof createCodePlugin>;

/** One block, rendering into a slot the way `HighlightedCodeBlockBody`'s `useState` does. */
class Block {
  latest: HighlightResult | null = null;
  updates = 0;
  readonly receive = (result: HighlightResult): void => {
    this.latest = result;
    this.updates += 1;
  };
}

/** Drive one `highlight()` call and settle whatever it produces, sync or async. */
const render = async (
  plugin: Plugin,
  block: Block,
  code: string,
): Promise<void> => {
  const sync = plugin.highlight(
    { code, language: LANGUAGE, themes: THEMES } as HighlightOptions,
    block.receive,
  );
  if (sync) block.receive(sync);
  // The highlighter loads asynchronously on the first call for a key; a few turns of the
  // microtask queue plus a macrotask is enough for it to resolve and notify.
  await new Promise((resolve) => setTimeout(resolve, 0));
};

const lines = (result: HighlightResult): string[] =>
  result.tokens.map((line) => line.map((token) => token.content).join(""));

const spanCount = (result: HighlightResult): number =>
  result.tokens.reduce((total, line) => total + line.length, 0);

const fenceId = (result: HighlightResult): string => {
  for (const line of result.tokens) {
    for (const token of line) {
      const id = token.htmlAttrs?.[CODE_FENCE_ATTRIBUTE];
      if (id !== undefined) return id;
    }
  }
  throw new Error("no fence id was stamped on this result");
};

const gateOn = (): { gate: CodeHighlightGate; tick: (ms: number) => void } => {
  let clock = 1_000_000;
  const gate = createCodeHighlightGate({
    bufferPx: 100,
    viewportHeight: 900,
    now: () => clock,
  });
  return {
    gate,
    tick: (ms) => {
      clock += ms;
    },
  };
};

const FAR = { top: 90_000, bottom: 92_000 };
const NEAR = { top: 20, bottom: 400 };

test("with no gate the plugin returns exactly what it always returned", async () => {
  // The flag-off build. Not "similar output" -- the same tokens, and no attribute anywhere, so a
  // parity digest taken on a flag-off build cannot see this change at all.
  const plain = createCodePlugin({ themes: THEMES });
  const a = new Block();
  await render(plain, a, SOURCE);
  assert.ok(a.latest);
  assert.equal(
    JSON.stringify(a.latest).includes(CODE_FENCE_ATTRIBUTE),
    false,
    "an ungated result must carry no fence attribute",
  );

  const reference = createCodePlugin({ themes: THEMES });
  const b = new Block();
  await render(reference, b, SOURCE);
  assert.deepEqual(a.latest, b.latest);
});

test("a settled block that scrolls far away is served plain, one span per line", async () => {
  const { gate } = gateOn();
  const plugin = createCodePlugin({ themes: THEMES, gate });
  const block = new Block();
  await render(plugin, block, SOURCE);

  const highlighted = block.latest as HighlightResult;
  const id = fenceId(highlighted);
  assert.ok(
    spanCount(highlighted) > highlighted.tokens.length,
    "the block starts out with more spans than it has lines",
  );

  const before = block.updates;
  gate.place(id, FAR);
  assert.ok(
    block.updates > before,
    "the retained callback re-rendered the block",
  );

  const served = block.latest as HighlightResult;
  assert.equal(
    spanCount(served),
    served.tokens.length,
    "plain is exactly one token per line",
  );
  assert.deepEqual(
    lines(served),
    lines(highlighted),
    "the same lines, with the same text on each",
  );
  assert.equal(
    served.tokens.length,
    highlighted.tokens.length,
    "and the same NUMBER of lines, which is the block's height",
  );
});

test("the plain rendering has byte-identical text, so copy and find-in-page cannot notice", () => {
  // The text is what a selection, a clipboard write and a find-in-page all read. Asserting the
  // whole concatenation rather than per line catches a swap that moved a newline.
  const { gate } = gateOn();
  const plugin = createCodePlugin({ themes: THEMES, gate });
  const block = new Block();
  return render(plugin, block, SOURCE).then(() => {
    const highlighted = block.latest as HighlightResult;
    gate.place(fenceId(highlighted), FAR);
    const served = block.latest as HighlightResult;
    assert.equal(lines(served).join("\n"), lines(highlighted).join("\n"));
    assert.equal(lines(served).join("\n"), SOURCE);
  });
});

test("scrolling back re-highlights it, to the tokens it had before", async () => {
  const { gate } = gateOn();
  const plugin = createCodePlugin({ themes: THEMES, gate });
  const block = new Block();
  await render(plugin, block, SOURCE);
  const highlighted = block.latest as HighlightResult;
  const id = fenceId(highlighted);

  gate.place(id, FAR);
  assert.equal(
    spanCount(block.latest as HighlightResult),
    highlighted.tokens.length,
  );

  gate.place(id, NEAR);
  const restored = block.latest as HighlightResult;
  assert.equal(
    spanCount(restored),
    spanCount(highlighted),
    "every span came back",
  );
  assert.deepEqual(lines(restored), lines(highlighted));
  assert.deepEqual(
    restored.tokens.map((line) => line.map((token) => token.color)),
    highlighted.tokens.map((line) => line.map((token) => token.color)),
    "with the colours they had, not a re-tokenization that drifted",
  );
});

test("a block still streaming is not downgraded, however far away it is measured", async () => {
  // A streaming fence re-enters highlight() with the whole block every frame. Serving it plain
  // would both fight the incremental path and take the colours off the one block the user is
  // certainly looking at.
  const { gate, tick } = gateOn();
  const plugin = createCodePlugin({ themes: THEMES, gate });
  const block = new Block();

  await render(plugin, block, SOURCE.slice(0, 1500));
  const id = fenceId(block.latest as HighlightResult);
  // A second, longer chunk: growth from code this fence has already seen is what a stream is.
  await render(plugin, block, SOURCE.slice(0, 2500));

  gate.place(id, FAR);
  assert.ok(
    spanCount(block.latest as HighlightResult) >
      (block.latest as HighlightResult).tokens.length,
    "held highlighted while the stream is live",
  );

  // The stream ends. Nothing calls back to say so, so the hold lapses on the clock and the next
  // measurement is what downgrades it.
  tick(5000);
  gate.place(id, FAR);
  assert.equal(
    spanCount(block.latest as HighlightResult),
    (block.latest as HighlightResult).tokens.length,
    "and downgraded once it is quiet",
  );
});

test("a block that arrives complete is not mistaken for a stream", async () => {
  // Every block in a SEEDED thread calls highlight() once with its whole body, which is growth
  // from nothing. Counting that as streaming would hold all 57 blocks at the 100K rung highlighted
  // for the whole grace window on every mount -- the case this exists for, opted out of itself.
  const { gate } = gateOn();
  const plugin = createCodePlugin({ themes: THEMES, gate });
  const block = new Block();
  await render(plugin, block, SOURCE);

  gate.place(fenceId(block.latest as HighlightResult), FAR);
  const served = block.latest as HighlightResult;
  assert.equal(
    spanCount(served),
    served.tokens.length,
    "downgraded on the first measurement, with no clock to wait for",
  );
});

test("a fence that streamed while gated plain comes back with the whole block, not a stale prefix", async () => {
  // The gated path deliberately does NOT advance `fence.code`, so the plugin's cached result can
  // fall behind the code it has been shown. A re-highlight that tokenized the cached prefix would
  // restore a block missing everything that arrived while it was plain.
  const { gate, tick } = gateOn();
  const plugin = createCodePlugin({ themes: THEMES, gate });
  const block = new Block();

  const half = SOURCE.slice(0, SOURCE.lastIndexOf("\n", SOURCE.length / 2));
  await render(plugin, block, half);
  const id = fenceId(block.latest as HighlightResult);

  gate.place(id, FAR);
  tick(5000);
  assert.equal(
    spanCount(block.latest as HighlightResult),
    (block.latest as HighlightResult).tokens.length,
    "plain",
  );

  // More of the block arrives while it is off screen.
  await render(plugin, block, SOURCE);
  gate.place(id, NEAR);

  const restored = block.latest as HighlightResult;
  assert.equal(lines(restored).join("\n"), SOURCE);

  const reference = createCodePlugin({ themes: THEMES });
  const control = new Block();
  await render(reference, control, SOURCE);
  assert.deepEqual(
    lines(restored),
    lines(control.latest as HighlightResult),
    "the same text the ungated plugin produces for the whole block",
  );
});

test("the fence id rides out on the first token that is actually rendered", async () => {
  // Streamdown renders an empty line as a bare newline with no span at all, so an id put on one
  // would never reach the DOM and the block would never be located.
  const { gate } = gateOn();
  const plugin = createCodePlugin({ themes: THEMES, gate });
  const block = new Block();
  await render(plugin, block, `\n\n${SOURCE}`);

  const result = block.latest as HighlightResult;
  const stamped = result.tokens.findIndex((line) =>
    line.some((token) => token.htmlAttrs?.[CODE_FENCE_ATTRIBUTE] !== undefined),
  );
  const rendered = result.tokens.findIndex(
    (line) => line.length > 0 && !(line.length === 1 && line[0].content === ""),
  );
  assert.equal(stamped, rendered);
  assert.ok(stamped > 0, "this fixture starts with lines that render no span");
});

test("two blocks sharing one fence are both told, because the fence is keyed on content", async () => {
  // Identical code in two places is one fence, so the gate's answer is per CONTENT and not per
  // element. Both mounted blocks have to be updated together or one keeps a rendering the gate
  // has already retired.
  const { gate } = gateOn();
  const plugin = createCodePlugin({ themes: THEMES, gate });
  const first = new Block();
  const second = new Block();
  await render(plugin, first, SOURCE);
  await render(plugin, second, SOURCE);

  const id = fenceId(first.latest as HighlightResult);
  gate.place(id, FAR);

  for (const block of [first, second]) {
    const served = block.latest as HighlightResult;
    assert.equal(spanCount(served), served.tokens.length);
  }
});

test("a gate notification for a block with nothing mounted does no work", () => {
  // `forget` and eviction both leave ids the gate may still name. Publishing to an empty
  // subscriber set has to be a no-op rather than a throw.
  const { gate } = gateOn();
  createCodePlugin({ themes: THEMES, gate });
  gate.place("sf-nonexistent", FAR);
  gate.place("sf-nonexistent", NEAR);
});
