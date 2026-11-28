// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

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
	assert.equal(await api.cancelGitHubCopilot(null, flow.flow_id), null);
	assert.deepEqual(paths, [
		"/api/providers/new/github/start",
		"/api/providers/new/github/flows/draft",
	]);
});

test("cancelling after approval returns the connected provider", async () => {
	globalThis.fetch = (async () =>
		Response.json({ provider_id: "completed-connection" })) as typeof fetch;
	assert.deepEqual(await api.cancelGitHubCopilot(null, "draft"), {
		provider_id: "completed-connection",
	});
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

test("reconnecting a switched-off Copilot keeps its selected models", async () => {
	const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
	const callback = dialog.match(
		/<GitHubCopilotConnect\b[\s\S]*?onChanged=\{(async \(id, connected\) => \{[\s\S]*?\n\s+\})\}\s*\/>/,
	)?.[1];
	assert.ok(callback);
	const provider = {
		id: "connection",
		models: ["claude-sonnet", "gpt-fast"],
		availableModels: ["auto", "claude-sonnet", "gpt-fast"],
	};
	let disabled = true;
	let selected: string[] = [];
	let available: string[] = [];
	const updates: { models: string[]; availableModels: string[] }[] = [];
	const noop = () => {};
	const scope = {
		codexCatalogRequestRef: { current: 0 },
		formCheckRequestRef: { current: 0 },
		providersRef: { current: [] },
		setFormCheck: noop,
		useConnectionChecksStore: { getState: () => ({ clearCheck: noop }) },
		setModelsLoading: noop,
		syncExternalProvidersFromBackend: async () => (disabled ? [] : [provider]),
		onProvidersChange: noop,
		setAvailableModels: (models: string[]) => {
			available = models;
		},
		setSelectedModelIds: (models: string[]) => {
			selected = models;
		},
		setEditingProviderId: noop,
		listProviderModels: async () => {
			if (disabled) throw new Error("Provider 'GitHub Copilot' is disabled.");
			return provider.availableModels.map((id) => ({ id }));
		},
		useExternalProvidersStore: {
			getState: () => ({ disabledProviders: disabled ? [provider] : [] }),
		},
		withSavedModelIds: (catalog: string[], models: string[]) => [
			...new Set([...catalog, ...models]),
		],
		updateProviderConfig: async (_id: string, config: (typeof updates)[number]) => {
			updates.push(config);
		},
	};
	const onChanged = new Function(...Object.keys(scope), `return (${callback});`)(
		...Object.values(scope),
	) as (id: string, connected: boolean) => Promise<void>;

	await onChanged(provider.id, true);
	assert.deepEqual(selected, provider.models);
	assert.deepEqual(available, provider.availableModels);
	assert.deepEqual(updates, []);

	disabled = false;
	await onChanged(provider.id, true);
	assert.deepEqual(updates, [{
		models: provider.models,
		availableModels: provider.availableModels,
	}]);
});
