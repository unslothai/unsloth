// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
const { loadModelAliases, saveModelAliases } = await import(
	"../src/features/settings/api/model-aliases.ts"
);

test("renaming an alias retains its API-created load configuration", async () => {
	// The settings UI must round-trip configurations while editing the existing name fields.
	const config = {
		max_seq_length: 98304,
		cache_type_kv: "q5_1",
		speculative_type: "ngram",
	};
	const stored = [{ name: "long", model: "org/model", load_config: config }];
	let written: unknown;
	globalThis.fetch = (async (_url, init) => {
		if (init?.method === "PUT") written = JSON.parse(String(init.body));
		return Response.json(written ?? { aliases: stored });
	}) as typeof fetch;
	const aliases = await loadModelAliases();
	aliases[0].name = "research";
	const saved = await saveModelAliases(aliases);
	const expected = [
		{ name: "research", model: "org/model", load_config: config },
	];
	assert.deepEqual(written, { aliases: expected });
	assert.deepEqual(saved, expected);
});

test("ordinary aliases keep their original wire representation", async () => {
	// Adding configuration support must not invent settings for legacy alias entries.
	globalThis.fetch = (async () =>
		Response.json({
			aliases: [{ name: "plain", model: "org/model" }],
		})) as typeof fetch;
	assert.deepEqual(await loadModelAliases(), [
		{ name: "plain", model: "org/model" },
	]);
});
