// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  PLACEHOLDER_EXAMPLE_MODEL,
  exampleModelOptions,
  followedExampleModel,
  pinQuant,
  resolveExampleModel,
  splitPinnedQuant,
} = await import("../src/features/settings/lib/example-model.ts");

const CATALOG = [
  { id: "unsloth/Llama-GGUF", loaded: false, quant: "Q8_0", quants: ["Q8_0"] },
  {
    id: "unsloth/Qwen3-GGUF",
    loaded: true,
    quant: "Q4_K_M",
    quants: ["Q4_K_M", "Q8_0"],
  },
  { id: "org/Mistral", loaded: false },
];

const base = {
  catalog: CATALOG,
  autoSwitch: false,
  keylessOnly: false,
  checkpoint: null,
  ggufVariant: null,
};

function resolve(overrides: Partial<Parameters<typeof resolveExampleModel>[0]>) {
  const catalog = overrides.catalog === undefined ? CATALOG : overrides.catalog;
  return resolveExampleModel({
    ...base,
    picked: null,
    ...overrides,
    catalog,
    options: exampleModelOptions(catalog),
  });
}

test("options list the resident model first, with every on-disk quant", () => {
  assert.deepEqual(exampleModelOptions(CATALOG), [
    {
      id: "unsloth/Qwen3-GGUF",
      loaded: true,
      residentQuant: "Q4_K_M",
      withheld: false,
      quants: ["Q4_K_M", "Q8_0"],
    },
    {
      id: "unsloth/Llama-GGUF",
      loaded: false,
      residentQuant: null,
      withheld: false,
      quants: ["Q8_0"],
    },
    {
      id: "org/Mistral",
      loaded: false,
      residentQuant: null,
      withheld: false,
      quants: [],
    },
  ]);
  // An older server sends `quant` alone; it is still the one quant to offer.
  assert.deepEqual(exampleModelOptions([{ id: "a", quant: "Q4" }])[0].quants, [
    "Q4",
  ]);
  assert.deepEqual(exampleModelOptions(null), []);
});

test("pinning and splitting a quant round-trip", () => {
  assert.equal(pinQuant("org/a", "Q8_0"), "org/a:Q8_0");
  assert.equal(pinQuant("org/a:Q4", "Q8_0"), "org/a:Q4");
  assert.equal(pinQuant("org/a", undefined), "org/a");
  assert.deepEqual(splitPinnedQuant("org/a:Q8_0"), { repo: "org/a", quant: "Q8_0" });
  assert.deepEqual(splitPinnedQuant("org/a"), { repo: "org/a", quant: null });
  assert.deepEqual(splitPinnedQuant("hf.co/org/a"), { repo: "hf.co/org/a", quant: null });
});

test("with nothing picked the snippet follows the resident model", () => {
  const r = resolve({});
  assert.equal(r.model, "unsloth/Qwen3-GGUF:Q4_K_M");
  assert.equal(r.followed, r.model);
  assert.equal(r.option?.id, "unsloth/Qwen3-GGUF");
  assert.equal(r.servable, true);
  assert.equal(r.placeholder, false);
});

test("a picked downloaded model is named, and flagged when switching is off", () => {
  const off = resolve({ picked: "unsloth/Llama-GGUF" });
  assert.equal(off.model, "unsloth/Llama-GGUF:Q8_0");
  assert.equal(off.followed, "unsloth/Qwen3-GGUF:Q4_K_M");
  assert.equal(off.servable, false);

  const on = resolve({ picked: "unsloth/Llama-GGUF", autoSwitch: true });
  assert.equal(on.servable, true);
  // A keyless caller cannot switch, so switching being on does not help it.
  const keyless = resolve({
    picked: "unsloth/Llama-GGUF",
    autoSwitch: true,
    keylessOnly: true,
  });
  assert.equal(keyless.servable, false);
});

test("a picked quant is honoured only while the repo still holds it", () => {
  assert.equal(
    resolve({ picked: "unsloth/Qwen3-GGUF:Q8_0" }).model,
    "unsloth/Qwen3-GGUF:Q8_0",
  );
  assert.equal(
    resolve({ picked: "unsloth/Qwen3-GGUF:Q2_K" }).model,
    "unsloth/Qwen3-GGUF:Q4_K_M",
  );
  assert.equal(resolve({ picked: "org/Mistral:Q8_0" }).model, "org/Mistral");
  // Case-insensitive like Hugging Face ids.
  assert.equal(
    resolve({ picked: "UNSLOTH/qwen3-gguf" }).model,
    "unsloth/Qwen3-GGUF:Q4_K_M",
  );
});

test("a pick the catalog no longer lists falls back to following", () => {
  const r = resolve({ picked: "org/deleted-GGUF:Q4" });
  assert.equal(r.model, "unsloth/Qwen3-GGUF:Q4_K_M");
  assert.equal(r.servable, true);
  // Before /v1 answers the pick is not disproved, but nothing backs it either.
  const pending = resolve({ picked: "unsloth/Llama-GGUF", catalog: null });
  assert.equal(pending.model, null);
  assert.equal(pending.placeholder, false);
});

test("an empty catalog names the placeholder, an unanswered one names nothing", () => {
  const empty = resolve({ catalog: [] });
  assert.equal(empty.model, PLACEHOLDER_EXAMPLE_MODEL);
  assert.equal(empty.placeholder, true);
  assert.equal(empty.servable, false);
  assert.equal(empty.option, null);

  const pending = resolve({ catalog: null });
  assert.equal(pending.model, null);
  assert.equal(pending.placeholder, false);
});

test("the followed model keeps the store checkpoint only while the catalog backs it", () => {
  const backed = followedExampleModel({
    ...base,
    checkpoint: "unsloth/Qwen3-GGUF",
    ggufVariant: "Q2_K",
  });
  // The catalog's quant wins over the stored one.
  assert.equal(backed, "unsloth/Qwen3-GGUF:Q4_K_M");
  const pending = followedExampleModel({
    ...base,
    catalog: null,
    checkpoint: "unsloth/Qwen3-GGUF",
    ggufVariant: "Q2_K",
  });
  assert.equal(pending, "unsloth/Qwen3-GGUF:Q2_K");
  const unloaded = followedExampleModel({
    ...base,
    checkpoint: "unsloth/Llama-GGUF",
  });
  assert.equal(unloaded, "unsloth/Qwen3-GGUF:Q4_K_M");
  assert.equal(
    followedExampleModel({ ...base, checkpoint: "unsloth/Llama-GGUF", autoSwitch: true }),
    "unsloth/Llama-GGUF:Q8_0",
  );
  assert.equal(
    followedExampleModel({ ...base, checkpoint: "/srv/models/x.gguf" }),
    "unsloth/Qwen3-GGUF:Q4_K_M",
  );
  assert.equal(
    followedExampleModel({ ...base, catalog: [], autoSwitch: true }),
    null,
  );
});

// Downloaded but nothing resident is the default state after a download, and after the
// idle unload retires a model: auto-switch is off unless the user turned it on. The
// panel used to render neither a snippet nor a reason there.
test("a downloaded but unloaded catalog still names a model, and says why it will not run", () => {
  const none = CATALOG.map((m) => ({ ...m, loaded: false }));
  const resolved = resolve({ catalog: none });
  assert.equal(resolved.model, "unsloth/Llama-GGUF:Q8_0");
  assert.equal(resolved.followed, null);
  assert.equal(resolved.option?.id, "unsloth/Llama-GGUF");
  assert.equal(resolved.servable, false);
  assert.equal(resolved.blockedBy, "autoSwitchOff");
  // Not the nothing-downloaded state: that note would say the server holds nothing.
  assert.equal(resolved.placeholder, false);
  // Switching on makes the same catalog runnable, so nothing is flagged.
  const switched = resolve({ catalog: none, autoSwitch: true });
  assert.equal(switched.model, "unsloth/Llama-GGUF:Q8_0");
  assert.equal(switched.servable, true);
  assert.equal(switched.blockedBy, null);
});

// A keyless caller is refused the switch server-side, so "turn on Switch model by
// request" is not the remedy: the message has to name a different one.
test("a keyless caller is blocked by the missing key, not by the switch", () => {
  const keyless = resolve({
    picked: "unsloth/Llama-GGUF",
    keylessOnly: true,
    autoSwitch: true,
  });
  assert.equal(keyless.servable, false);
  assert.equal(keyless.blockedBy, "keyless");
  // The same pick with a key only needs the switch, which is on.
  const keyed = resolve({ picked: "unsloth/Llama-GGUF", autoSwitch: true });
  assert.equal(keyed.servable, true);
  assert.equal(keyed.blockedBy, null);
  // Off, and it is the switch that is missing.
  assert.equal(
    resolve({ picked: "unsloth/Llama-GGUF" }).blockedBy,
    "autoSwitchOff",
  );
  // A resident pick is servable however the caller authenticates.
  assert.equal(
    resolve({ picked: "unsloth/Qwen3-GGUF", keylessOnly: true }).blockedBy,
    null,
  );
});

// Nothing downloaded keeps naming the shipped example, and it comes from the one
// module the Agents tab reads, so the two surfaces cannot drift apart.
test("the placeholder is the shipped example model", async () => {
  const empty = resolve({ catalog: [] });
  assert.equal(empty.placeholder, true);
  assert.equal(empty.model, PLACEHOLDER_EXAMPLE_MODEL);
  assert.equal(empty.blockedBy, null);
  const { EXAMPLE_MODEL_REPO, EXAMPLE_MODEL_VARIANT } = await import(
    "../src/features/settings/lib/example-model-id.ts"
  );
  assert.equal(
    PLACEHOLDER_EXAMPLE_MODEL,
    `${EXAMPLE_MODEL_REPO}:${EXAMPLE_MODEL_VARIANT}`,
  );
});

// list_local_gguf_variants qualifies a variant by its directory when the bare quant
// would name several checkpoints, so `BF16/model-BF16` is a real pin. Splitting on the
// last colon, or refusing a quant with a slash, dropped it and served another file.
test("a quant qualified by its directory survives a round-trip", () => {
  const qualified = "BF16/DeepSeek-R1-BF16";
  const pinned = pinQuant("unsloth/DeepSeek-GGUF", qualified);
  assert.equal(pinned, `unsloth/DeepSeek-GGUF:${qualified}`);
  assert.deepEqual(splitPinnedQuant(pinned), {
    repo: "unsloth/DeepSeek-GGUF",
    quant: qualified,
  });
  const catalog = [
    {
      id: "unsloth/DeepSeek-GGUF",
      loaded: false,
      quant: "Q4_K_M",
      quants: ["Q4_K_M", qualified],
    },
  ];
  // The pick holds instead of snapping back to the repo's default quant.
  const resolved = resolveExampleModel({
    ...base,
    catalog,
    autoSwitch: true,
    picked: pinned,
    options: exampleModelOptions(catalog),
  });
  assert.equal(resolved.model, pinned);
});

// `loaded` marks the repo, not the file. Q4 resident and Q8 picked is a request the
// server answers with model_not_found unless it may switch.
test("another quant of the resident repo is not itself resident", () => {
  const catalog = [
    {
      id: "unsloth/Qwen3-GGUF",
      loaded: true,
      quant: "Q4_K_M",
      quants: ["Q4_K_M", "Q8_0"],
    },
  ];
  const pick = (picked: string, over = {}) =>
    resolveExampleModel({
      ...base,
      catalog,
      picked,
      options: exampleModelOptions(catalog),
      ...over,
    });
  // The quant actually in memory needs nothing.
  assert.equal(pick("unsloth/Qwen3-GGUF:Q4_K_M").servable, true);
  assert.equal(pick("unsloth/Qwen3-GGUF:Q4_K_M").blockedBy, null);
  // A different quant of the same repo does.
  assert.equal(pick("unsloth/Qwen3-GGUF:Q8_0").servable, false);
  assert.equal(pick("unsloth/Qwen3-GGUF:Q8_0").blockedBy, "autoSwitchOff");
  assert.equal(pick("unsloth/Qwen3-GGUF:Q8_0", { autoSwitch: true }).servable, true);
  // A keyless caller is refused the switch, so the toggle is not the remedy.
  assert.equal(
    pick("unsloth/Qwen3-GGUF:Q8_0", { autoSwitch: true, keylessOnly: true }).blockedBy,
    "keyless",
  );
});

// `org/Foo` and `Org/Foo` are one repo to sameBaseModelId and to the server's alias
// index. Two rows meant two picker entries, and choosing the second selected the first.
test("case variants of one repo are a single option", () => {
  const catalog = [
    { id: "org/Foo", loaded: false, quant: "BF16", quants: ["BF16"] },
    { id: "Org/Foo", loaded: true, quant: "Q2_K", quants: ["Q2_K"] },
  ];
  const options = exampleModelOptions(catalog);
  assert.equal(options.length, 1);
  // The first spelling is the one the resolver reaches, so it keeps the id and quants.
  assert.equal(options[0].id, "org/Foo");
  assert.deepEqual(options[0].quants, ["BF16"]);
  // Residency belongs to the repo, not to one spelling of it.
  assert.equal(options[0].loaded, true);
});

// The server vouches for one spelling's quants, not necessarily the first listed.
test("merging case variants keeps whichever row carries the quants", () => {
  const catalog = [
    { id: "Org/Foo", loaded: true },
    { id: "org/Foo", loaded: false, quant: "BF16", quants: ["BF16", "Q2_K"] },
  ];
  const options = exampleModelOptions(catalog);
  assert.equal(options.length, 1);
  assert.equal(options[0].id, "Org/Foo");
  assert.equal(options[0].loaded, true);
  assert.deepEqual(options[0].quants, ["BF16", "Q2_K"]);
  // The loaded row named no quant, and the offered ones came from the other copy, so
  // nothing here proves which file is in memory.
  assert.equal(options[0].residentQuant, null);
  const mixed = resolveExampleModel({
    ...base,
    catalog,
    picked: "Org/Foo:BF16",
    options,
  });
  // Pinning one of them still needs a switch; claiming it resident would hide that.
  assert.equal(mixed.servable, false);
  assert.equal(mixed.blockedBy, "autoSwitchOff");
});

// The resident label and the scanned one can differ only in case; picking the scanned
// spelling is not a switch, so it must not warn.
test("a quant that differs only in case is the resident one", () => {
  const catalog = [
    {
      id: "unsloth/Qwen3-GGUF",
      loaded: true,
      quant: "q4_k_m",
      quants: ["q4_k_m", "Q8_0"],
    },
  ];
  const resolved = resolveExampleModel({
    ...base,
    catalog,
    picked: "unsloth/Qwen3-GGUF:Q4_K_M",
    options: exampleModelOptions(catalog),
  });
  assert.equal(resolved.servable, true);
  assert.equal(resolved.blockedBy, null);
  // The spelling the server gave is the one pinned, so the request names a real file.
  assert.equal(resolved.model, "unsloth/Qwen3-GGUF:q4_k_m");
});

// When two copies of one id disagree the server withholds `quants` and keeps `quant`.
// Merging must respect that: promoting the singular field would re-offer the pin the
// server just refused to vouch for.
test("merging never promotes a singular quant the server withheld", () => {
  const catalog = [
    { id: "Org/Foo", loaded: true, quant: "BF16" },
    { id: "org/Foo", loaded: false, quant: "Q2_K" },
  ];
  const options = exampleModelOptions(catalog);
  assert.equal(options.length, 1);
  assert.deepEqual(options[0].quants, ["BF16"]);
  // A row that does carry the plural field is still adopted.
  const vouched = exampleModelOptions([
    { id: "Org/Foo", loaded: true },
    { id: "org/Foo", loaded: false, quant: "BF16", quants: ["BF16", "Q2_K"] },
  ]);
  assert.deepEqual(vouched[0].quants, ["BF16", "Q2_K"]);
});

// An older server sends only `quant`; a current one that cannot vouch for any pin on
// an id sends `quants: []`. The picker must not read the second as the first, or it
// re-offers exactly the pin the server withheld.
test("an omitted quant list outranks a singular quant", () => {
  const withheld = exampleModelOptions([
    { id: "org/Foo", loaded: false, quant: "BF16", quants: [] },
  ]);
  assert.deepEqual(withheld[0].quants, []);
  assert.equal(withheld[0].withheld, true);
  // Nothing to pin, so the snippet names the bare repo.
  const resolved = resolveExampleModel({
    ...base,
    catalog: [{ id: "org/Foo", loaded: false, quant: "BF16", quants: [] }],
    autoSwitch: true,
    picked: "org/Foo",
    options: withheld,
  });
  assert.equal(resolved.model, "org/Foo");

  // An older server omits the field entirely: its singular quant is still the one pin.
  const legacy = exampleModelOptions([
    { id: "org/Foo", loaded: false, quant: "BF16" },
  ]);
  assert.deepEqual(legacy[0].quants, ["BF16"]);
  assert.equal(legacy[0].withheld, false);
});

// followedExampleModel reads the catalog row directly rather than the guarded option,
// so the withheld signal has to be honoured there too or the snippet pins anyway.
test("the followed model does not pin a quant the server withheld", () => {
  const withheld = [
    { id: "org/Foo", loaded: true, quant: "BF16", quants: [] },
  ];
  assert.equal(followedExampleModel({ ...base, catalog: withheld }), "org/Foo");
  // An older server, which omits the field, still pins its singular quant.
  assert.equal(
    followedExampleModel({
      ...base,
      catalog: [{ id: "org/Foo", loaded: true, quant: "BF16" }],
    }),
    "org/Foo:BF16",
  );
  // Same through the stored-checkpoint branch.
  assert.equal(
    followedExampleModel({
      ...base,
      catalog: withheld,
      checkpoint: "org/Foo",
    }),
    "org/Foo",
  );
});

// A repo the server can only name one quant for offers no choice, so the panel must not
// render a control that looks switchable and is not.
test("a single-quant model offers no quant choice", () => {
  const one = exampleModelOptions([
    { id: "org/Foo", loaded: true, quant: "Q4_K_M" },
  ]);
  assert.deepEqual(one[0].quants, ["Q4_K_M"]);
  // Several quants is a real choice, and every one of them stays selectable.
  const many = exampleModelOptions([
    { id: "org/Foo", loaded: true, quant: "Q4_K_M", quants: ["Q4_K_M", "Q8_0", "BF16"] },
  ]);
  assert.deepEqual(many[0].quants, ["Q4_K_M", "Q8_0", "BF16"]);
  const catalog = [
    { id: "org/Foo", loaded: true, quant: "Q4_K_M", quants: ["Q4_K_M", "Q8_0", "BF16"] },
  ];
  const options = exampleModelOptions(catalog);
  for (const quant of ["Q8_0", "BF16", "Q4_K_M"]) {
    const r = resolveExampleModel({
      ...base,
      catalog,
      picked: pinQuant("org/Foo", quant),
      options,
    });
    assert.equal(r.model, `org/Foo:${quant}`);
  }
});
