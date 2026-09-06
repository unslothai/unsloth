// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import type * as Session from "../src/features/auth/session.ts";

for (const [desktop, loginMode, change, expected] of [
	[true, "single", true, "/chat"],
	[true, "single", false, "/chat"],
	[true, "multi", true, "/change-password"],
	[true, "multi", false, "/chat"],
	[false, "single", true, "/change-password"],
	[false, "multi", true, "/change-password"],
] as const) {
	test(`${desktop ? "desktop" : "browser"} ${loginMode} session with change=${change} routes to ${expected}`, (t) => {
		const previousWindow = globalThis.window;
		const previousStorage = globalThis.localStorage;
		const storage = {
			getItem: () => (change ? "1" : null),
		} as unknown as Storage;
		globalThis.localStorage = storage;
		globalThis.window = {
			localStorage: storage,
			addEventListener() {},
		} as unknown as Window & typeof globalThis;
		t.after(() => {
			globalThis.window = previousWindow;
			globalThis.localStorage = previousStorage;
		});
		let listening = 0;
		const session = loadWithStubs<typeof Session>(
			new URL("../src/features/auth/session.ts", import.meta.url),
			{
				"@/lib/api-base": { isTauri: desktop },
				"@/lib/account-transition": {
					installAccountTransitionListener: () => {
						listening++;
					},
				},
				"./login-client": { getLoginMode: () => loginMode },
				"./session-events": {},
			},
		);
		assert.equal(session.getPostAuthRoute(), expected);
		assert.equal(listening, 1);
	});
}
