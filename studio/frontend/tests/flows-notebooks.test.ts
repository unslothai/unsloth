// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { downloadNotebook, notebookActions, notebookName, visibleNotebooks } =
  await import(
  "../src/features/flows/gallery/notebooks.ts"
);
type Entry = import("../src/features/flows/api.ts").NotebookEntry;

function nb(file: string, category: string, studioTask: string | null): Entry {
  return {
    id: file.toLowerCase(),
    title: file.replace(".ipynb", ""),
    notebook_file: file,
    category,
    featured: false,
    studio_task: studioTask,
    colab_url: `https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/${file}`,
    github_url: `https://github.com/unslothai/notebooks/blob/main/nb/${file}`,
  };
}

const ALPACA = nb("Llama3.1_(8B)-Alpaca.ipynb", "sft", "text");
const GEMMA = nb("Gemma3_(4B).ipynb", "sft", "text");
const GRPO = nb("Qwen3_(4B)-GRPO.ipynb", "grpo", null);
const DPO = nb("Zephyr_(7B)-DPO.ipynb", "preference", null);

test("a notebook that ships as a flow offers its template", () => {
  const a = notebookActions(ALPACA);
  assert.equal(a.template, "notebook-text-sft");
  assert.equal(a.importable, true);
  assert.equal(a.colab, ALPACA.colab_url);
});

test("a trainable notebook without a template is offered as an .ipynb import", () => {
  const a = notebookActions(GEMMA);
  assert.equal(a.template, null);
  assert.equal(a.importable, true);
  assert.match(
    a.download,
    /raw\.githubusercontent\.com\/unslothai\/notebooks\/main\/nb\/Gemma3_/,
  );
});

test("RL and preference notebooks stay Colab only", () => {
  for (const entry of [GRPO, DPO]) {
    const a = notebookActions(entry);
    assert.equal(a.template, null);
    assert.equal(a.importable, false);
  }
});

test("templates list first, Colab-only last, and search matches the file name", () => {
  const order = visibleNotebooks([GRPO, GEMMA, ALPACA, DPO], "").map(
    (e) => e.notebook_file,
  );
  assert.deepEqual(order.slice(0, 2), [
    ALPACA.notebook_file,
    GEMMA.notebook_file,
  ]);
  assert.deepEqual(
    visibleNotebooks([GRPO, GEMMA, ALPACA], "grpo").map((e) => e.notebook_file),
    [GRPO.notebook_file],
  );
});

test("a picked or dropped .ipynb file goes to the notebook importer by its own name", () => {
  const file = new File(["{}"], "Llama3.1_(8B)-Alpaca.ipynb");
  assert.equal(notebookName(file), "Llama3.1_(8B)-Alpaca.ipynb");
  assert.equal(notebookName(new File(["{}"], "story.unslothflow")), null);
  assert.equal(notebookName('{"nodes": []}'), null);
  assert.equal(notebookName(new Blob(["{}"]), "x.ipynb"), "x.ipynb");
});

test("Download .ipynb saves the file from a blob instead of navigating Studio away", async () => {
  const g = globalThis as Record<string, unknown>;
  const saved = { fetch: g.fetch, document: g.document, window: g.window };
  const clicked: { href: string; download: string }[] = [];
  const opened: string[] = [];
  g.fetch = async () => new Response("{}", { status: 200 });
  g.document = {
    createElement: () => {
      const a = { href: "", download: "", click: () => clicked.push({ href: a.href, download: a.download }) };
      return a;
    },
  };
  g.window = { open: (url: string) => opened.push(url) };
  try {
    await downloadNotebook("https://raw.githubusercontent.com/x/y.ipynb", "y.ipynb");
    assert.equal(clicked.length, 1);
    assert.equal(clicked[0].download, "y.ipynb");
    assert.ok(clicked[0].href.startsWith("blob:"));
    g.fetch = async () => new Response("", { status: 404 });
    await downloadNotebook("https://raw.githubusercontent.com/x/z.ipynb", "z.ipynb");
    assert.deepEqual(opened, ["https://raw.githubusercontent.com/x/z.ipynb"]);
  } finally {
    Object.assign(g, saved);
  }
});
