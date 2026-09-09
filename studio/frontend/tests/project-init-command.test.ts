// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
	interceptCompareProjectSlashCommand,
	parseProjectSlashCommand,
	projectInitCommandResponse,
} from "../src/features/chat/utils/slash-commands.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";
import { setAuthFetchHandler } from "./helpers/store-stubs/auth.ts";

registerStoreStubResolver();

const guidanceApi = await import(
	"../src/features/chat/api/project-guidance-api.ts"
);

function response(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "Content-Type": "application/json" },
	});
}

test.beforeEach(() => setAuthFetchHandler(null));

test("parses only the local init command", () => {
	assert.deepEqual(parseProjectSlashCommand(" /INIT "), {
		name: "init",
		action: "run",
	});
	assert.deepEqual(parseProjectSlashCommand("/init now"), {
		name: "init",
		action: "help",
	});
	assert.equal(parseProjectSlashCommand("please /init"), null);
	assert.equal(parseProjectSlashCommand("/status"), null);
});

test("compare executes init once and mirrors one result into both panes", async () => {
	let executions = 0;
	const first: string[] = [];
	const second: string[] = [];

	const intercepted = await interceptCompareProjectSlashCommand({
		input: "/init",
		userContent: "user:/init",
		panes: [
			{
				appendUserMessage: (content) => first.push(content),
				appendAssistantMessage: (content) => first.push(`assistant:${content}`),
			},
			{
				appendUserMessage: (content) => second.push(content),
				appendAssistantMessage: (content) =>
					second.push(`assistant:${content}`),
			},
		],
		execute: (command) => {
			executions += 1;
			return Promise.resolve(
				projectInitCommandResponse(command, {
					status: "created",
					created: true,
					path: "AGENTS.md",
					instructions: {
						layers: [],
						combined: "",
						truncated: false,
						issues: [],
						precedence: "root first",
						bytesRead: 0,
					},
				}),
			);
		},
	});

	assert.equal(intercepted, true);
	assert.equal(executions, 1);
	assert.deepEqual(first, [
		"user:/init",
		"assistant:Created `AGENTS.md` in this project. Future agent turns will use its instructions.",
	]);
	assert.deepEqual(second, first);
});

test("init reports the existing override path returned by the backend", () => {
	const response = projectInitCommandResponse(
		{ name: "init", action: "run" },
		{
			status: "already_exists",
			created: false,
			path: "AGENTS.override.md",
			instructions: {
				layers: [],
				combined: "",
				truncated: false,
				issues: [],
				precedence: "root first",
				bytesRead: 0,
			},
		},
	);

	assert.equal(
		response,
		"`AGENTS.override.md` already exists, so `/init` left it unchanged.",
	);
});

test("guidance panel disables create for either root instruction file", async () => {
	const panel = await readFile(
		new URL(
			"../src/features/chat/components/project-guidance-panel.tsx",
			import.meta.url,
		),
		"utf8",
	);
	const rootInstruction = panel.indexOf("const rootInstruction");
	const createButton = panel.indexOf(
		"onClick={createAgentsFile}",
		rootInstruction,
	);
	const buttonSource = panel.slice(rootInstruction, createButton);

	assert.ok(rootInstruction >= 0 && createButton > rootInstruction);
	assert.match(
		buttonSource,
		/layer\.path === "AGENTS\.override\.md" \|\| layer\.path === "AGENTS\.md"/,
	);
	assert.match(
		buttonSource,
		/disabled=\{[\s\S]*?Boolean\(rootInstruction\)[\s\S]*?\}/,
	);
	assert.match(panel, /`\$\{result\.path\} already exists`/);
	assert.match(panel, /`\$\{rootInstruction\.path\} found`/);
});

test("project guidance client uses scoped endpoints and create-only POST", async () => {
	const calls: Array<{ input: string; method: string }> = [];
	setAuthFetchHandler((input, init) => {
		calls.push({ input, method: init?.method ?? "GET" });
		if (input.endsWith("/instructions")) {
			return response({
				layers: [],
				combined: "",
				truncated: false,
				issues: [],
				precedence: "root first",
				bytesRead: 0,
			});
		}
		if (input.endsWith("/skills")) {
			return response({ skills: [], issues: [], truncated: false });
		}
		return response({
			status: "already_exists",
			created: false,
			path: "AGENTS.md",
			instructions: {
				layers: [],
				combined: "",
				truncated: false,
				issues: [],
				precedence: "root first",
				bytesRead: 0,
			},
		});
	});

	await guidanceApi.getProjectInstructions("project one");
	await guidanceApi.getProjectSkills("project one");
	const initialized = await guidanceApi.initializeProjectAgents("project one");

	assert.equal(initialized.created, false);
	assert.deepEqual(calls, [
		{
			input: "/api/agent/projects/project%20one/instructions",
			method: "GET",
		},
		{ input: "/api/agent/projects/project%20one/skills", method: "GET" },
		{ input: "/api/agent/projects/project%20one/init", method: "POST" },
	]);
});

test("adapter intercepts init before inference and token counts carry workspace identity", async () => {
	const adapter = await readFile(
		new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
		"utf8",
	);
	const wrapper = adapter.lastIndexOf("return {\n    async *run(args)");
	const parse = adapter.indexOf("parseProjectSlashCommand(", wrapper);
	const execute = adapter.indexOf(
		"await executeLocalProjectSlashCommand(slashCommand",
		parse,
	);
	const model = adapter.indexOf("yield* adapter.run(args)", wrapper);
	assert.ok(
		wrapper >= 0 && parse > wrapper && execute > parse && model > execute,
	);

	const extras = adapter.indexOf(
		"export async function buildLocalTokenCountExtras",
	);
	const session = adapter.indexOf(
		"const sessionId = sandboxSessionIdFor",
		extras,
	);
	const firstReturn = adapter.indexOf("if (!supportsTools) {", extras);
	assert.ok(extras >= 0 && session > extras && firstReturn > session);
});
