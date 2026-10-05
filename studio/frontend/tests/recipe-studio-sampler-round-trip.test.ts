// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildSamplerColumn } = await import(
  "../src/features/recipe-studio/utils/payload/builders-sampler.ts"
);
const { parseSampler } = await import(
  "../src/features/recipe-studio/utils/import/parsers/sampler-parser.ts"
);

type SamplerConfig = Parameters<typeof buildSamplerColumn>[0];

// Saving writes the payload, reopening re-imports that same payload
// (use-recipe-persistence re-imports the stored payload on load), so whatever
// survives this is what the user still sees after a reopen.
function roundTrip(config: SamplerConfig): SamplerConfig {
  const buildErrors: string[] = [];
  const column = buildSamplerColumn(config, buildErrors);
  assert.deepEqual(buildErrors, [], "the config should build without errors");

  const parseErrors: string[] = [];
  const parsed = parseSampler(column, config.name, config.id, parseErrors);
  assert.deepEqual(
    parseErrors,
    [],
    "the built column should parse without errors",
  );
  assert.ok(parsed, "the built column should parse back to a sampler");
  return parsed;
}

test("a gaussian sampler keeps its standard deviation across a save and reopen", () => {
  const parsed = roundTrip({
    id: "gaussian-1",
    kind: "sampler",
    sampler_type: "gaussian",
    name: "score",
    mean: "10",
    std: "2.5",
  });

  assert.equal(parsed.mean, "10");
  assert.equal(parsed.std, "2.5");
});

test("a uuid sampler keeps its format across a save and reopen", () => {
  for (const [uuidFormat, expected] of [
    ["MY-", "MY-"],
    ["short", "short"],
    ["upper", "upper"],
    ["", ""],
  ] as const) {
    const parsed = roundTrip({
      id: "uuid-1",
      kind: "sampler",
      sampler_type: "uuid",
      name: "row_id",
      uuid_format: uuidFormat,
    });

    assert.equal(
      parsed.uuid_format,
      expected,
      `uuid_format ${uuidFormat} should survive`,
    );
  }
});

test("a uuid prefix survives even when its value is a reserved word", () => {
  // buildSamplerParams reads "short" / "upper" / "uuid4" as modes, so a prefix whose
  // own value is one of those has to come back escaped or the next save turns the
  // prefix into a flag. Checked as payload stability over two save/reopen cycles,
  // which is what a user actually does.
  for (const uuidFormat of [
    "MY-",
    "short",
    "upper",
    "uuid4",
    "prefix:short",
    "prefix:upper",
    "prefix:uuid4",
    "prefix:prefix:x",
    "",
  ] as const) {
    const config: SamplerConfig = {
      id: "uuid-1",
      kind: "sampler",
      sampler_type: "uuid",
      name: "row_id",
      uuid_format: uuidFormat,
    };
    const firstSave = buildSamplerColumn(config, []);
    const reopened = roundTrip(config);
    const secondSave = buildSamplerColumn(reopened, []);

    assert.deepEqual(
      secondSave.params,
      firstSave.params,
      `uuid_format ${JSON.stringify(uuidFormat)} should re-save to the same params`,
    );
  }
});

test("a gaussian sampler still reads a hand-written std key", () => {
  const parseErrors: string[] = [];
  const parsed = parseSampler(
    {
      column_type: "sampler",
      sampler_type: "gaussian",
      name: "score",
      params: { mean: 1, std: 3 },
    },
    "score",
    "gaussian-legacy",
    parseErrors,
  );

  assert.deepEqual(parseErrors, []);
  assert.equal(parsed?.std, "3");
});
