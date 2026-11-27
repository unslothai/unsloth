// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/settings-api-resolver.mjs", import.meta.url);

const api = await import("../src/features/chat/api/providers-api.ts");
const { getProviderCapabilities, getExternalReasoningCapabilities } =
	await import("../src/features/chat/provider-capabilities.ts");
const { supportsProviderMaxOutputTokens, providerTypeSupportsVision } =
	await import("../src/features/chat/external-providers.ts");

test("GitHub device sign-in and disconnect use the connection's authenticated backend", async () => {
	const seen: { url: string; method: string }[] = [];
	globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
		seen.push({ url: String(input), method: init?.method ?? "GET" });
		if (init?.method === "DELETE") return new Response(null, { status: 204 });
		return Response.json(
			String(input).endsWith("start")
				? {
						flow_id: "flow",
						user_code: "ABCD-EFGH",
						expires_in: 900,
						interval: 5,
						verification_uri: "https://github.com/login/device",
					}
				: { status: "connected" },
		);
	}) as typeof fetch;
	const flow = await api.startGitHubCopilot("connection");
	assert.equal(flow.user_code, "ABCD-EFGH");
	assert.equal(
		(await api.pollGitHubCopilot("connection", flow.flow_id)).status,
		"connected",
	);
	await api.cancelGitHubCopilot("connection", flow.flow_id);
	await api.disconnectGitHubCopilot("connection");
	assert.deepEqual(seen, [
		{ url: "/api/providers/connection/github/start", method: "POST" },
		{ url: "/api/providers/connection/github/flows/flow/poll", method: "POST" },
		{ url: "/api/providers/connection/github/flows/flow", method: "DELETE" },
		{ url: "/api/providers/connection/github", method: "DELETE" },
	]);
});

test("first sign-in can be cancelled without creating a provider", async () => {
	const paths: string[] = [];
	globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
		paths.push(String(input));
		if (init?.method === "DELETE") return new Response(null, { status: 204 });
		return Response.json({ flow_id: "draft" });
	}) as typeof fetch;
	const flow = await api.startGitHubCopilot(null);
	await api.cancelGitHubCopilot(null, flow.flow_id);
	assert.deepEqual(paths, [
		"/api/providers/new/github/start",
		"/api/providers/new/github/flows/draft",
	]);
});

test("Copilot errors preserve the actionable OAuth App setup message", async () => {
	globalThis.fetch = (async () =>
		Response.json(
			{
				detail: "Set UNSLOTH_GITHUB_COPILOT_CLIENT_ID and enable Device Flow.",
			},
			{ status: 503 },
		)) as typeof fetch;
	await assert.rejects(
		api.startGitHubCopilot("connection"),
		/UNSLOTH_GITHUB_COPILOT_CLIENT_ID/,
	);
});

test("text-only SDK connection hides unsupported model controls", () => {
	const capabilities = getProviderCapabilities("github_copilot");
	assert.ok(capabilities);
	assert.ok(Object.values(capabilities).every((value) => value === false));
	assert.equal(providerTypeSupportsVision("github_copilot"), false);
	assert.equal(supportsProviderMaxOutputTokens("github_copilot", null), false);
	assert.equal(
		getExternalReasoningCapabilities("github_copilot", "claude-sonnet-5")
			.supportsReasoning,
		false,
	);
});
