// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type {
  HighlightOptions,
  HighlightResult,
  ThemeInput,
} from "@streamdown/code";
import { createHighlighter } from "shiki";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

import { createCodePlugin } from "../src/components/assistant-ui/code-plugin.ts";

const THEMES: [ThemeInput, ThemeInput] = ["github-light", "github-dark"];

const PYTHON = `import json


def render(payload: dict) -> str:
    """Return the payload as text.

    A docstring is one grammar scope spanning several lines, so a line that
    only continues it has no meaning of its own.
    """
    if not payload:
        return ""
    return json.dumps(payload, indent=2)  # trailing comment


class Runner:
    def __init__(self, name: str = "runner"):
        self.name = name

    def run(self, *args, **kwargs):
        return f"{self.name}: {args!r} {kwargs!r}"
`;

const TYPESCRIPT = `export type Slot = {
  code: string;
  /** Tokens for the lines a newline has already terminated. */
  lines: string[][];
};

export function commit(slot: Slot, code: string): Slot {
  const boundary = code.lastIndexOf("\\n");
  if (boundary < 0) return slot;
  return { ...slot, code: code.slice(0, boundary + 1) };
}
`;

const UNBALANCED = "x = '''\nstill inside the string\nand still\n";

function highlightOnce(
  plugin: ReturnType<typeof createCodePlugin>,
  options: HighlightOptions,
): Promise<HighlightResult> {
  return new Promise((resolve) => {
    const immediate = plugin.highlight(options, resolve);
    if (immediate) resolve(immediate);
  });
}

async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  message: string,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), timeoutMs);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

const referenceHighlighters = new Map<
  HighlightOptions["language"],
  ReturnType<typeof createHighlighter>
>();

/** Tokens shiki returns for the whole string in one call. */
async function reference(code: string, language: HighlightOptions["language"]) {
  let loading = referenceHighlighters.get(language);
  if (!loading) {
    loading = createHighlighter({
      themes: THEMES,
      langs: [language],
      engine: createJavaScriptRegexEngine({ forgiving: true }),
    });
    referenceHighlighters.set(language, loading);
  }
  const highlighter = await loading;
  return highlighter.codeToTokens(code, {
    lang: language,
    themes: { light: "github-light", dark: "github-dark" },
  });
}

async function assertMatchesFullTokenization(
  source: string,
  language: HighlightOptions["language"],
  step: number,
) {
  const plugin = createCodePlugin({ themes: THEMES });
  for (let length = 1; length <= source.length; length += step) {
    const code = source.slice(0, length);
    const streamed = await highlightOnce(plugin, {
      code,
      language,
      themes: THEMES,
    });
    const full = await reference(code, language);
    assert.deepEqual(
      streamed.tokens,
      full.tokens,
      `tokens diverged after ${length} of ${source.length} characters`,
    );
    assert.equal(streamed.fg, full.fg);
    assert.equal(streamed.bg, full.bg);
    assert.equal(streamed.themeName, full.themeName);
    assert.equal(streamed.rootStyle, full.rootStyle);
  }
}

test("a streamed Python fence tokenizes exactly like a whole one", async () => {
  await assertMatchesFullTokenization(PYTHON, "python", 1);
});

test("a streamed TypeScript fence tokenizes exactly like a whole one", async () => {
  await assertMatchesFullTokenization(TYPESCRIPT, "typescript", 1);
});

test("an unterminated multi-line string keeps its scope across updates", async () => {
  await assertMatchesFullTokenization(UNBALANCED, "python", 1);
});

test("a fence arriving in line-sized chunks matches a whole one", async () => {
  await assertMatchesFullTokenization(PYTHON, "python", 17);
});

test("aliases resolve to the language shiki tokenizes with", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const aliased = await highlightOnce(plugin, {
    code: PYTHON,
    language: "py" as HighlightOptions["language"],
    themes: THEMES,
  });
  const full = await reference(PYTHON, "python");
  assert.deepEqual(aliased.tokens, full.tokens);
  assert.equal(plugin.supportsLanguage("py" as HighlightOptions["language"]), true);
  assert.equal(
    plugin.supportsLanguage("not-a-language" as HighlightOptions["language"]),
    false,
  );
});

test("an unchanged fence is answered from cache without new tokens", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const options: HighlightOptions = {
    code: PYTHON,
    language: "python" as HighlightOptions["language"],
    themes: THEMES,
  };
  const first = await highlightOnce(plugin, options);
  const second = plugin.highlight(options);
  assert.equal(second, first, "a repeat render must reuse the same result");
});

test("two fences of one language keep their own tokens", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  await highlightOnce(plugin, { code: PYTHON, language, themes: THEMES });
  const other = `${PYTHON}\n# a second fence that starts differently\nvalue = 1\n`;
  const shifted = other.slice(20);
  const streamed = await highlightOnce(plugin, {
    code: shifted,
    language,
    themes: THEMES,
  });
  const full = await reference(shifted, "python");
  assert.deepEqual(streamed.tokens, full.tokens);
});

test("all fences mounted while one grammar loads receive their result", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const blocks = [
    "value = 1\n",
    "value = 1\n",
    "value = 1\nprint(value)\n",
  ];

  const highlighted = blocks.map(
    (code) =>
      new Promise<HighlightResult>((resolve) => {
        const immediate = plugin.highlight(
          { code, language, themes: THEMES },
          resolve,
        );
        if (immediate) resolve(immediate);
      }),
  );
  const results = await withTimeout(
    Promise.all(highlighted),
    500,
    "a highlight callback was lost",
  );

  for (let index = 0; index < blocks.length; index += 1) {
    const full = await reference(blocks[index], "python");
    assert.deepEqual(results[index].tokens, full.tokens);
  }
});

test("updates queued during grammar load settle on the newest source", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const blocks = ["value = 1", "value = 10", "value = 100\n"];
  const seen: HighlightResult[] = [];
  const completed = blocks.map(
    (code) =>
      new Promise<void>((resolve) => {
        const receive = (result: HighlightResult) => {
          seen.push(result);
          resolve();
        };
        const immediate = plugin.highlight(
          { code, language, themes: THEMES },
          receive,
        );
        if (immediate) receive(immediate);
      }),
  );
  await withTimeout(
    Promise.all(completed),
    500,
    "a queued update was not highlighted",
  );

  const latest = await reference(blocks.at(-1)!, "python");
  assert.deepEqual(seen.at(-1)?.tokens, latest.tokens);
});

test("the cache limit does not discard grammar-load callbacks", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const blocks = Array.from(
    { length: 520 },
    (_, index) => `value_${index} = ${index}\n`,
  );
  const highlighted = blocks.map(
    (code) =>
      new Promise<HighlightResult>((resolve) => {
        const immediate = plugin.highlight(
          { code, language, themes: THEMES },
          resolve,
        );
        if (immediate) resolve(immediate);
      }),
  );
  const results = await withTimeout(
    Promise.all(highlighted),
    2000,
    "a pending fence was evicted",
  );

  assert.equal(results.length, blocks.length);
  for (const index of [0, blocks.length - 1]) {
    const full = await reference(blocks[index], "python");
    assert.deepEqual(results[index].tokens, full.tokens);
  }
});

test("a failed grammar load releases pending callbacks before retry", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const invalidThemes = [
    { name: "retry-light", settings: 42 } as unknown as ThemeInput,
    { name: "retry-dark", settings: 42 } as unknown as ThemeInput,
  ] as [ThemeInput, ThemeInput];
  const validThemes = [
    { name: "retry-light", settings: [] } as unknown as ThemeInput,
    { name: "retry-dark", settings: [] } as unknown as ThemeInput,
  ] as [ThemeInput, ThemeInput];
  const blocks = Array.from(
    { length: 520 },
    (_, index) => `value_${index} = ${index}\n`,
  );

  let reportFailure!: () => void;
  const failed = new Promise<void>((resolve) => {
    reportFailure = resolve;
  });
  const originalError = console.error;
  let staleCallbacks = 0;
  try {
    console.error = (message?: unknown) => {
      if (message === "[Unsloth Code] Failed to highlight code:") {
        reportFailure();
      }
    };
    for (const code of blocks) {
      plugin.highlight(
        { code, language, themes: invalidThemes },
        () => {
          staleCallbacks += 1;
        },
      );
    }
    await withTimeout(failed, 500, "the invalid grammar load did not fail");
  } finally {
    console.error = originalError;
  }

  const retried = await highlightOnce(plugin, {
    code: blocks.at(-1)!,
    language,
    themes: validThemes,
  });
  assert.equal(staleCallbacks, 0, "a retry invoked callbacks from a failed load");
  assert.equal(
    retried.tokens
      .map((line) => line.map((token) => token.content).join(""))
      .join("\n"),
    blocks.at(-1),
  );
});

test("a transiently failed grammar load is retried under the same cache key", async () => {
  // The test above retries under a *different* key: swapping the theme
  // definitions builds a new `highlighters` entry whether or not the failed one
  // was cleaned up. A transient failure is the case that needs the cleanup --
  // the browser re-fetches a module whose previous fetch failed
  // (whatwg/html#10327), so the same key is asked for again and has to load. If
  // the entry that failed were left behind, every later render of that language
  // and theme pair would keep getting its null highlighter and the block would
  // never highlight again for the life of the page.
  //
  // Keeping the same theme objects keeps the key identical; mutating them in
  // place stands in for the chunk that failed once and loaded on the retry.
  const light: { name: string; settings: unknown } = {
    name: "transient-light",
    settings: 42,
  };
  const dark: { name: string; settings: unknown } = {
    name: "transient-dark",
    settings: 42,
  };
  const themes = [light, dark] as unknown as [ThemeInput, ThemeInput];
  const plugin = createCodePlugin({ themes });
  const language = "python" as HighlightOptions["language"];

  let reportFailure!: () => void;
  const failed = new Promise<void>((resolve) => {
    reportFailure = resolve;
  });
  let staleCallbacks = 0;
  const originalError = console.error;
  try {
    console.error = (message?: unknown) => {
      if (message === "[Unsloth Code] Failed to highlight code:") reportFailure();
    };
    assert.equal(
      plugin.highlight({ code: "value = 1\n", language, themes }, () => {
        staleCallbacks += 1;
      }),
      null,
      "the first render cannot have a highlighter yet",
    );
    await withTimeout(failed, 500, "the invalid grammar load did not fail");
  } finally {
    console.error = originalError;
  }

  light.settings = [];
  dark.settings = [];
  const retried = await withTimeout(
    highlightOnce(plugin, { code: "value = 1\nvalue = 2\n", language, themes }),
    2000,
    "the key that failed was never retried, so the fence stays plain forever",
  );
  assert.equal(
    retried.tokens
      .map((line) => line.map((token) => token.content).join(""))
      .join("\n"),
    "value = 1\nvalue = 2\n",
  );
  assert.equal(
    staleCallbacks,
    0,
    "a callback from the failed load fired after the retry",
  );
});

test("CRLF streaming matches whole-document tokenization", async () => {
  await assertMatchesFullTokenization(
    "def render():\r\n    value = '''first\r\nsecond'''\r\n    return value\r\n",
    "python",
    7,
  );
});

test("re-opening a thread answers every fence from cache", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const blocks = Array.from(
    { length: 60 },
    (_, index) =>
      `# block ${index}\n${PYTHON}\nvalue_${index} = ${index}\n`,
  );
  const first: HighlightResult[] = [];
  for (const code of blocks) {
    first.push(await highlightOnce(plugin, { code, language, themes: THEMES }));
  }
  // A thread mounts whole: the same object back means no tokenizer was reached.
  const reused = blocks.filter(
    (code, index) =>
      plugin.highlight({ code, language, themes: THEMES }) === first[index],
  );
  assert.equal(
    reused.length,
    blocks.length,
    "every fence should come back from cache on a second mount",
  );
});

test("a single line past the throttle keeps its text and settles exact", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "json" as HighlightOptions["language"];
  const long = `{"values": [${Array.from({ length: 400 }, (_, i) => `"item-${i}"`).join(", ")}`;
  await highlightOnce(plugin, { code: long, language, themes: THEMES });
  const grown = `${long}, "tail"`;
  const throttled = (await highlightOnce(plugin, {
    code: grown,
    language,
    themes: THEMES,
  })) as HighlightResult;
  assert.deepEqual(
    throttled.tokens,
    [[{ content: grown, offset: 0 }]],
    "the throttled frame must retain the previous plugin's plain live line",
  );
  assert.equal(
    throttled.tokens.map((line) => line.map((t) => t.content).join("")).join("\n"),
    grown,
    "every character has to be rendered even while the line is throttled",
  );

  const settled = await new Promise<HighlightResult>((resolve) => {
    const immediate = plugin.highlight(
      { code: grown, language, themes: THEMES },
      resolve,
    );
    if (immediate) {
      // Still inside the refresh interval: the trailing refresh resolves it.
      setTimeout(() => {
        const after = plugin.highlight({ code: grown, language, themes: THEMES });
        if (after) resolve(after);
      }, 400);
    }
  });
  const full = await reference(grown, "json");
  assert.deepEqual(settled.tokens, full.tokens);
});

test("a shorter prefix sibling fence keeps the longer one highlighted", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "json" as HighlightOptions["language"];
  // Single-line, so the longer fence commits nothing and stays prefix-reachable.
  const shorter = `{"values": [${Array.from({ length: 300 }, (_, i) => `"item-${i}"`).join(", ")}`;
  const longer = `${shorter}, "only-in-the-longer-fence"]}`;

  await highlightOnce(plugin, { code: longer, language, themes: THEMES });
  await highlightOnce(plugin, { code: shorter, language, themes: THEMES });

  // Document order renders the shorter fence next, within one refresh interval.
  let rendered: HighlightResult | null = null;
  for (let frame = 0; frame < 3; frame += 1) {
    rendered = plugin.highlight({ code: longer, language, themes: THEMES });
    plugin.highlight({ code: shorter, language, themes: THEMES });
  }

  assert.deepEqual(
    rendered?.tokens,
    (await reference(longer, "json")).tokens,
    "the longer fence must not be left as an unhighlighted plain tail",
  );
  assert.deepEqual(
    plugin.highlight({ code: shorter, language, themes: THEMES })?.tokens,
    (await reference(shorter, "json")).tokens,
  );
});

// None of these can close a fence: whitespace carries no marker, a closing run
// never mixes backticks with tildes, and a closer starts its own line.
for (const suffix of ["   ", "`~", "```"]) {
  test(`a sibling ending in ${JSON.stringify(suffix)} keeps its own entry`, async () => {
    const plugin = createCodePlugin({ themes: THEMES });
    const language = "json" as HighlightOptions["language"];
    const shorter = `{"values": [${Array.from({ length: 300 }, (_, i) => `"item-${i}"`).join(", ")}`;
    const longer = `${shorter}${suffix}`;

    await highlightOnce(plugin, { code: longer, language, themes: THEMES });
    await highlightOnce(plugin, { code: shorter, language, themes: THEMES });

    let rendered: HighlightResult | null = null;
    for (let frame = 0; frame < 3; frame += 1) {
      rendered = plugin.highlight({ code: longer, language, themes: THEMES });
      plugin.highlight({ code: shorter, language, themes: THEMES });
    }

    assert.deepEqual(
      rendered?.tokens,
      (await reference(longer, "json")).tokens,
      "the longer fence must not be left as an unhighlighted plain tail",
    );
  });
}

test("a changed theme definition is not answered from the old theme's cache", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const code = "value = 1\n";
  // Same names, different definitions: only the definitions tell them apart.
  const pair = (background: string): [ThemeInput, ThemeInput] =>
    (["swap-light", "swap-dark"] as const).map(
      (name) =>
        ({
          name,
          settings: [{ settings: { foreground: "#101010", background } }],
        }) as unknown as ThemeInput,
    ) as [ThemeInput, ThemeInput];

  const first = await highlightOnce(plugin, {
    code,
    language,
    themes: pair("#111111"),
  });
  const second = await highlightOnce(plugin, {
    code,
    language,
    themes: pair("#eeeeee"),
  });
  assert.notEqual(
    second.bg,
    first.bg,
    "a redefined theme must not reuse the previous definition's tokens",
  );
});

test("a shorter delimiter-like sibling cannot drop a queued refresh", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const body = `${Array.from(
    { length: 120 },
    (_, index) => `value_${index} = ${index}  # a completed Python line`,
  ).join("\n")}\n`;
  await highlightOnce(plugin, { code: body, language, themes: THEMES });

  const longer = `${body}\`\`\``;
  const arrived: HighlightResult[] = [];
  const approximate = plugin.highlight(
    { code: longer, language, themes: THEMES },
    (result) => arrived.push(result),
  );
  const shorter = plugin.highlight({ code: body, language, themes: THEMES });
  await new Promise((resolve) => setTimeout(resolve, 400));

  assert.deepEqual(
    approximate?.tokens.at(-1),
    [{ content: "```", offset: 0 }],
    "the longer fence must enter the throttled path",
  );
  assert.equal(arrived.length, 1, "the longer fence must receive its refresh");
  assert.deepEqual(arrived[0].tokens, (await reference(longer, "python")).tokens);
  assert.deepEqual(shorter?.tokens, (await reference(body, "python")).tokens);
});

test("shedding a possible closing delimiter settles before the shorter result", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const body = `${Array.from(
    { length: 120 },
    (_, index) => `value_${index} = ${index}  # a completed Python line`,
  ).join("\n")}\n`;
  // Markdown reports the closing run as body until it closes the fence.
  await highlightOnce(plugin, { code: `${body}\`\``, language, themes: THEMES });

  const settled: HighlightResult[] = [];
  plugin.highlight({ code: `${body}\`\`\``, language, themes: THEMES }, (result) =>
    settled.push(result),
  );
  const closed = plugin.highlight({ code: body, language, themes: THEMES });
  assert.equal(
    settled.length,
    1,
    "the queued refresh must settle before the shorter result returns",
  );
  await new Promise((resolve) => setTimeout(resolve, 400));

  assert.equal(settled.length, 1, "the queued refresh must settle exactly once");
  assert.deepEqual(
    settled[0].tokens,
    (await reference(`${body}\`\`\``, "python")).tokens,
  );
  assert.deepEqual(closed?.tokens, (await reference(body, "python")).tokens);
});

test("an unchanged fence does not cancel an identical sibling's refresh", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const body = `${Array.from(
    { length: 120 },
    (_, index) => `value_${index} = ${index}  # a completed Python line`,
  ).join("\n")}\n`;
  // Byte-identical fences cannot be told apart, so they share one entry.
  await highlightOnce(plugin, { code: body, language, themes: THEMES });
  await highlightOnce(plugin, { code: body, language, themes: THEMES });

  let grown = body;
  let latest: HighlightResult | null = null;
  const receive = (result: HighlightResult) => {
    latest = result;
  };
  for (let frame = 0; frame < 3; frame += 1) {
    grown = `${grown}appended_${frame} = ${frame}\n`;
    latest = plugin.highlight({ code: grown, language, themes: THEMES }, receive);
    plugin.highlight({ code: body, language, themes: THEMES });
  }
  await new Promise((resolve) => setTimeout(resolve, 500));

  assert.deepEqual(
    (latest as HighlightResult | null)?.tokens,
    (await reference(grown, "python")).tokens,
    "the growing fence must still receive its refresh",
  );
});

test("a block that keeps failing to tokenize does not fill the cache", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  // createHighlighter accepts a nameless theme, but codeToTokens cannot find it
  // again, so tokenization throws after the grammar has loaded.
  const nameless = [{ settings: [] }, { settings: [] }] as unknown as [
    ThemeInput,
    ThemeInput,
  ];
  const kept = "kept = 1\n";
  const first = await highlightOnce(plugin, {
    code: kept,
    language,
    themes: THEMES,
  });

  const originalError = console.error;
  try {
    console.error = () => {};
    for (let render = 0; render < 600; render += 1) {
      plugin.highlight({ code: "broken = 1\n", language, themes: nameless });
      // The first render only queues; let the grammar load resolve.
      if (render === 0) await new Promise((resolve) => setTimeout(resolve, 50));
    }
  } finally {
    console.error = originalError;
  }

  // A successful update runs eviction. Failed fences must not have crowded the
  // cache past MAX_FENCES in the meantime.
  await highlightOnce(plugin, { code: "later = 2\n", language, themes: THEMES });
  assert.equal(
    plugin.highlight({ code: kept, language, themes: THEMES }),
    first,
    "a failing block evicted a healthy fence",
  );
});

test("a throttled multiline tail matches the previous plugin", async () => {
  const plugin = createCodePlugin({ themes: THEMES });
  const language = "python" as HighlightOptions["language"];
  const initial = Array.from(
    { length: 90 },
    (_, index) => `value_${index} = ${index}  # a completed Python line`,
  ).join("\n");
  const first = await highlightOnce(plugin, {
    code: initial,
    language,
    themes: THEMES,
  });
  const grown = `${initial}\nnext_value = 100\nand_more = 101`;
  const throttled = await highlightOnce(plugin, {
    code: grown,
    language,
    themes: THEMES,
  });

  const keptLineCount = initial.split("\n").length - 1;
  const expectedTail = grown
    .split("\n")
    .slice(keptLineCount)
    .map((content) => [{ content, offset: 0 }]);
  assert.deepEqual(throttled.tokens, [
    ...first.tokens.slice(0, keptLineCount),
    ...expectedTail,
  ]);
});
