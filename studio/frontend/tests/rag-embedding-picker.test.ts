// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { embeddingMenuModels } from "../src/features/rag/lib/embedding-menu-models.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

const MENU_PICKER = read("../src/features/rag/components/embedding-model-menu-picker.tsx");
const KB_BUTTON = read("../src/features/rag/components/knowledge-base-composer-button.tsx");
const SECTION = read("../src/features/settings/components/documents-rag-section.tsx");
const PICKER = read("../src/features/settings/components/embedding-model-picker.tsx");

type Plan = {
  embeddingModel: string;
  backend: "llama" | "sentence-transformers";
  downloadRepo: string | null;
  cached: boolean;
  error: string | null;
};

function switcher(options: { plan?: Plan | Error; current?: boolean; saveError?: Error }) {
  const saves: Array<{ model: string; ggufRepo: unknown; backend: unknown }> = [];
  const store = {
    beginSave: () => 1,
    isSaveCurrent: () => options.current ?? true,
    save: async (request: () => Promise<unknown>) => {
      await request();
      return true;
    },
  };
  const mod = loadWithStubs<{
    switchEmbeddingModel: (model: string, token?: string) => Promise<unknown>;
    embeddingModelName: (model: string) => string;
    embeddingModelOwner: (model: string) => string;
  }>(new URL("../src/features/settings/lib/switch-embedding-model.ts", import.meta.url), {
    "../api/embedding-model": {
      resolveEmbeddingModel: async () => {
        if (options.plan instanceof Error) throw options.plan;
        return options.plan;
      },
      updateEmbeddingModelSettings: async (
        model: string,
        opts: { ggufRepo: unknown; backend: unknown },
      ) => {
        if (options.saveError) throw options.saveError;
        saves.push({ model, ggufRepo: opts.ggufRepo, backend: opts.backend });
        return {};
      },
    },
    "../stores/embedding-model-store": {
      useEmbeddingModelStore: { getState: () => store },
    },
  });
  return { ...mod, saves };
}

const PLAN: Plan = {
  embeddingModel: "unsloth/embeddinggemma-2",
  backend: "llama",
  downloadRepo: "unsloth/embeddinggemma-2-GGUF",
  cached: true,
  error: null,
};

test("switching saves the GGUF repo the resolve picked", async () => {
  const app = switcher({ plan: PLAN });
  assert.deepEqual(await app.switchEmbeddingModel(PLAN.embeddingModel), {
    status: "saved",
    needsDownload: false,
  });
  assert.deepEqual(app.saves, [
    { model: PLAN.embeddingModel, ggufRepo: "unsloth/embeddinggemma-2-GGUF", backend: "llama" },
  ]);
});

test("a model not on disk saves and says it needs a download", async () => {
  const app = switcher({ plan: { ...PLAN, cached: false } });
  assert.deepEqual(await app.switchEmbeddingModel(PLAN.embeddingModel), {
    status: "saved",
    needsDownload: true,
  });
});

test("a resolve error is reported and nothing is saved", async () => {
  const app = switcher({ plan: { ...PLAN, error: "Not an embedding model" } });
  assert.deepEqual(await app.switchEmbeddingModel(PLAN.embeddingModel), {
    status: "failed",
    message: "Not an embedding model",
  });
  assert.equal(app.saves.length, 0);
});

test("a newer pick from another surface wins", async () => {
  const app = switcher({ plan: PLAN, current: false });
  assert.deepEqual(await app.switchEmbeddingModel(PLAN.embeddingModel), { status: "superseded" });
  assert.equal(app.saves.length, 0);
});

test("a failed resolve still saves, and a failed save is reported", async () => {
  const resolveDown = switcher({ plan: new Error("offline") });
  assert.deepEqual(await resolveDown.switchEmbeddingModel("acme/embed"), {
    status: "saved",
    needsDownload: false,
  });
  assert.deepEqual(resolveDown.saves, [{ model: "acme/embed", ggufRepo: null, backend: null }]);
  const saveDown = switcher({ plan: PLAN, saveError: new Error("Could not verify") });
  assert.deepEqual(await saveDown.switchEmbeddingModel(PLAN.embeddingModel), {
    status: "failed",
    message: "Could not verify",
  });
});

test("model names and owners read from repo ids and local paths", () => {
  const { embeddingModelName, embeddingModelOwner } = switcher({});
  assert.equal(embeddingModelName("unsloth/bge-small-en-v1.5"), "bge-small-en-v1.5");
  assert.equal(embeddingModelName("/models/my-embedder/"), "my-embedder");
  assert.equal(embeddingModelName("bge-small"), "bge-small");
  assert.equal(embeddingModelOwner("unsloth/bge-small-en-v1.5"), "unsloth");
  assert.equal(embeddingModelOwner("/models/my-embedder"), "");
  assert.equal(embeddingModelOwner("bge-small"), "");
});

test("the menu lists current, default, then pinned models, each once", () => {
  assert.deepEqual(
    embeddingMenuModels("acme/a", "unsloth/b", ["acme/a", "acme/c", "unsloth/b", " "]),
    ["acme/a", "unsloth/b", "acme/c"],
  );
});

test("the chip shows for the owner only and swaps the menu to the model list", () => {
  assert.match(KB_BUTTON, /\{isOwner \? <EmbeddingModelMenuChip onOpen=\{\(\) => setView\("embedding"\)\} \/> : null\}/);
  assert.match(KB_BUTTON, /isOwner && view === "embedding" \? \(\s*<EmbeddingModelMenuList onBack=\{\(\) => setView\("source"\)\} \/>/);
  // Closing resets it, so the menu reopens on the source list.
  assert.match(KB_BUTTON, /else setView\("source"\);/);
  // No side submenu any more.
  assert.doesNotMatch(MENU_PICKER, /DropdownMenuPrimitive\.Sub\b|SubContent/);
});

test("Change model and the needs-download toast open the embedding row in Settings", () => {
  const opens = MENU_PICKER.match(
    /openSettings\("general", \{ scrollTarget: "general-rag-embedding" \}\)/g,
  );
  assert.equal(opens?.length, 2);
  assert.match(SECTION, /if \(scrollTarget !== "general-rag-embedding"\) return;/);
  assert.match(SECTION, /<SettingsSection ref=\{sectionRef\}/);
});

test("Settings rows pin and unpin, and pinned models stay listed", () => {
  assert.match(PICKER, /onClick=\{\(\) => onTogglePin\(item\.id\)\}/);
  assert.match(PICKER, /for \(const pin of pinnedModels \?\? \[\]\)/);
  assert.match(SECTION, /pinnedModels=\{pinnedModels\}\s*onTogglePin=\{togglePin\}/);
});

test("pins are grey with the Recents unpin glyph, and tooltips are the app's own", () => {
  for (const source of [MENU_PICKER, PICKER]) {
    assert.match(source, /icon=\{(isPinned|pinned) \? PinOffIcon : PinIcon\}/);
    assert.doesNotMatch(source, /text-primary hover:text-primary|\? "text-primary"/);
    assert.doesNotMatch(source, /title=\{/);
    assert.match(source, /<TooltipContent side="top">/);
  }
});

test("in the composer list the pin shows on row hover only, pinned or not", () => {
  assert.match(MENU_PICKER, /text-muted-foreground opacity-0 transition-colors group-hover\/row:opacity-100/);
  assert.doesNotMatch(MENU_PICKER, /!isPinned &&/);
});

test("Settings keeps the long text in the info hint and the status on the row", () => {
  assert.match(SECTION, /hint=\{`\$\{t\("settings\.general\.rag\.embeddingModelDescription"/);
  assert.match(SECTION, /settings\.general\.rag\.embeddingModelShort[\s\S]*?\{statusText \? \(/);
  // The row's `below` slot carries errors only.
  assert.match(SECTION, /below=\{\s*embeddingModelError \|\| notLoaded \?/);
});

test("Eject sits on the picker and in the RAG menu, only while a model is resident", () => {
  assert.match(SECTION, /onEject=\{embeddingModel\?\.backendLoaded \?/);
  assert.match(PICKER, /\{onEject \? \([\s\S]*?onEject\(\);[\s\S]*?group-hover\/trigger:block/);
  assert.match(MENU_PICKER, /\{settings\.backendLoaded \? \([\s\S]*?void eject\(\)/);
  assert.match(MENU_PICKER, /settings\.general\.rag\.ejectModel/);
});

test("On device explains itself and says when the model is not loaded", () => {
  assert.match(SECTION, /settings\.general\.rag\.onDeviceHint/);
  assert.match(SECTION, /const notLoaded = onDevice && !embeddingModel\?\.loaded && !downloading;/);
  // Under the picker, a filled dot, with its own hint.
  assert.match(SECTION, /below=\{\s*embeddingModelError \|\| notLoaded \?[\s\S]*?rounded-full bg-muted-foreground[\s\S]*?settings\.general\.rag\.notLoadedHint/);
});

test("ejecting frees the model through the shared residency path", async () => {
  const calls: unknown[] = [];
  const unload = async () => ({ loaded: false });
  const mod = loadWithStubs<{ ejectEmbeddingModel: () => Promise<void> }>(
    new URL("../src/features/settings/lib/switch-embedding-model.ts", import.meta.url),
    {
      "../api/embedding-model": {
        resolveEmbeddingModel: async () => null,
        unloadEmbeddingModel: unload,
        updateEmbeddingModelSettings: async () => null,
      },
      "../stores/embedding-model-store": {
        useEmbeddingModelStore: {
          getState: () => ({ applyResidency: async (request: unknown) => calls.push(request) }),
        },
      },
    },
  );
  await mod.ejectEmbeddingModel();
  assert.deepEqual(calls, [unload]);
});
