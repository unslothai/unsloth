import { spawn, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const provider = "unsloth";
const maxResultCharacters = 100_000;
const maxParallelAgents = 4;
const cancelGraceMilliseconds = 2_000;
const configPath = process.env.UNSLOTH_PI_SUBAGENT_CONFIG || "";
delete process.env.UNSLOTH_PI_SUBAGENT_CONFIG;
let config: Record<string, unknown> = {};
if (configPath) {
	try {
		const parsed = JSON.parse(fs.readFileSync(configPath, "utf8"));
		if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
			throw new Error("expected a JSON object");
		}
		config = parsed;
	} catch (error) {
		throw new Error(`Could not read Unsloth subagent configuration: ${error}`);
	}
}
const model = typeof config.model === "string" ? config.model : "";
const baseUrl = typeof config.baseUrl === "string" ? config.baseUrl : "";
const apiKey = typeof config.apiKey === "string" ? config.apiKey : "";
const contextWindow = positiveInt(config.contextWindow, 32768);
const maxTokens = positiveInt(config.maxTokens, Math.min(Math.floor(contextWindow / 4), 8192));

function positiveInt(value: unknown, fallback: number): number {
	const parsed = Number.parseInt(typeof value === "string" ? value : String(value || ""), 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function finalText(message: any): string {
	if (message?.role !== "assistant" || !Array.isArray(message.content)) return "";
	return message.content
		.filter((part: any) => part?.type === "text" && typeof part.text === "string")
		.map((part: any) => part.text)
		.join("\n")
		.trim();
}

function boundedResult(text: string): string {
	if (text.length <= maxResultCharacters) return text;
	return `${text.slice(0, maxResultCharacters)}\n\n[Local agent output truncated]`;
}

function piInvocation(args: string[]): { command: string; args: string[] } {
	const currentScript = process.argv[1];
	const bunVirtualScript = currentScript?.startsWith("/$bunfs/root/");
	if (currentScript && !bunVirtualScript && fs.existsSync(currentScript)) {
		return { command: process.execPath, args: [currentScript, ...args] };
	}
	const executable = path.basename(process.execPath).toLowerCase();
	if (!/^(node|bun)(\.exe)?$/.test(executable)) return { command: process.execPath, args };
	return { command: "pi", args };
}

function signalProcessGroup(child: ChildProcess, signal: NodeJS.Signals): void {
	if (!child.pid) return;
	try {
		process.kill(-child.pid, signal);
	} catch {
		try {
			child.kill(signal);
		} catch {
			// The process tree already exited.
		}
	}
}

async function stopChildTree(child: ChildProcess): Promise<void> {
	if (!child.pid) return;
	if (process.platform === "win32") {
		await new Promise<void>((resolve) => {
			const killer = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
				shell: false,
				stdio: "ignore",
				windowsHide: true,
			});
			killer.once("error", () => {
				try {
					child.kill("SIGKILL");
				} catch {
					// The child already exited.
				}
				resolve();
			});
			killer.once("close", (code) => {
				if (code !== 0) {
					try {
						child.kill("SIGKILL");
					} catch {
						// The child already exited.
					}
				}
				resolve();
			});
		});
		return;
	}

	signalProcessGroup(child, "SIGTERM");
	await new Promise((resolve) => setTimeout(resolve, cancelGraceMilliseconds));
	signalProcessGroup(child, "SIGKILL");
}

interface LocalAgentResult {
	task: string;
	response: string;
	transcript: any[];
	error?: string;
}

async function runLocalAgent(
	task: string,
	cwd: string,
	signal: AbortSignal | undefined,
	onProgress: (result: LocalAgentResult) => void,
): Promise<LocalAgentResult> {
	const extension = fileURLToPath(import.meta.url);
	const args = [
		"--mode",
		"json",
		"--print",
		"--no-session",
		"--provider",
		provider,
		"--model",
		model,
		"--no-extensions",
		"--extension",
		extension,
		`Task: ${task}`,
	];
	const invocation = piInvocation(args);
	let output = "";
	let stderr = "";
	let childError = "";
	let aborted = false;
	const result: LocalAgentResult = { task, response: "", transcript: [] };
	const processLine = (line: string) => {
		try {
			const event = JSON.parse(line);
			if ((event.type === "message_end" || event.type === "tool_result_end") && event.message) {
				result.transcript.push(event.message);
				onProgress(result);
			}
			if (event.type !== "message_end") return;
			const message = event.message;
			// Pi reports model/API failures as message_end events while still
			// exiting 0, so the exit status alone cannot surface them.
			if (message?.stopReason === "error" || message?.stopReason === "aborted") {
				childError =
					(typeof message.errorMessage === "string" && message.errorMessage) ||
					`The local Unsloth agent stopped: ${message.stopReason}.`;
				return;
			}
			const response = finalText(message);
			if (response) {
				result.response = boundedResult(response);
				childError = "";
			}
		} catch {
			// Ignore non-JSON diagnostic lines. The exit status still reports failures.
		}
	};

	const exitCode = await new Promise<number>((resolve, reject) => {
		const child = spawn(invocation.command, invocation.args, {
			cwd,
			detached: process.platform !== "win32",
			shell: false,
			stdio: ["ignore", "pipe", "pipe"],
			env: {
				...process.env,
				UNSLOTH_PI_SUBAGENT_CHILD: "1",
				UNSLOTH_PI_SUBAGENT_CONFIG: configPath,
			},
		});
		let cleanup: Promise<void> | undefined;
		const cancel = () => {
			if (aborted) return;
			aborted = true;
			cleanup = stopChildTree(child);
		};
		child.on("error", (error) => {
			signal?.removeEventListener("abort", cancel);
			reject(error);
		});
		child.stdout.on("data", (chunk) => {
			output += chunk.toString();
			const lines = output.split("\n");
			output = lines.pop() || "";
			for (const line of lines) processLine(line);
		});
		child.stderr.on("data", (chunk) => {
			stderr = (stderr + chunk.toString()).slice(-100_000);
		});
		child.on("close", async (code) => {
			signal?.removeEventListener("abort", cancel);
			await cleanup;
			if (output.trim()) processLine(output);
			resolve(code ?? 1);
		});
		signal?.addEventListener("abort", cancel, { once: true });
		if (signal?.aborted) cancel();
	});

	if (aborted) throw new Error("The local Unsloth agent was cancelled.");
	if (exitCode !== 0) {
		result.error = stderr.trim() || `The local Unsloth agent exited with code ${exitCode}.`;
	}
	if (childError) result.error = boundedResult(childError);
	if (!result.response && !result.error) result.response = "The local agent returned no text.";
	return result;
}

export default function unslothSubagent(pi: ExtensionAPI): void {
	if (!model || !baseUrl || !apiKey || !configPath) {
		throw new Error("Unsloth subagent configuration is incomplete.");
	}

	pi.registerProvider(provider, {
		name: "Unsloth Studio",
		baseUrl,
		apiKey,
		api: "openai-completions",
		authHeader: true,
		models: [
			{
				id: model,
				name: `${model} via Unsloth`,
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow,
				maxTokens,
			},
		],
	});

	if (process.env.UNSLOTH_PI_SUBAGENT_CHILD === "1") return;

	pi.registerTool({
		name: "unsloth_agent",
		label: "Unsloth agent",
		description:
			"Run local coding agents powered by Unsloth for debugging, implementation, and codebase research. Use task for one agent. To run multiple independent agents, use tasks; up to four run concurrently. The tool returns only after every requested agent finishes.",
		parameters: Type.Object({
			task: Type.Optional(
				Type.String({ description: "The complete task for one local Unsloth agent." }),
			),
			tasks: Type.Optional(
				Type.Array(Type.String({ description: "A complete task for one local Unsloth agent." }), {
					description: "Independent tasks to run concurrently, one local agent per task.",
					minItems: 2,
					maxItems: maxParallelAgents,
				}),
			),
		}),
		executionMode: "parallel",
		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			const singleTask = typeof params.task === "string" && params.task.trim() ? params.task.trim() : "";
			const parallelTasks = Array.isArray(params.tasks)
				? params.tasks.map((task) => task.trim()).filter(Boolean)
				: [];
			if (Boolean(singleTask) === Boolean(parallelTasks.length)) {
				throw new Error("Provide exactly one of task or tasks.");
			}
			if (parallelTasks.length > maxParallelAgents) {
				throw new Error(`At most ${maxParallelAgents} local agents can run concurrently.`);
			}
			if (parallelTasks.length === 1) {
				throw new Error("Use task for one local agent, or tasks for two to four agents.");
			}

			const tasks = singleTask ? [singleTask] : parallelTasks;
			const results: Array<LocalAgentResult | undefined> = new Array(tasks.length);
			let completed = 0;
			const details = () => ({
				provider,
				model,
				mode: tasks.length === 1 ? "single" : "parallel",
				results: results.filter((result): result is LocalAgentResult => Boolean(result)),
			});
			const emitUpdate = () => {
				onUpdate?.({
					content: [
						{
							type: "text",
							text: `Local agents: ${completed}/${tasks.length} completed`,
						},
					],
					details: details(),
				});
			};
			await Promise.all(
				tasks.map(async (task, index) => {
					try {
						results[index] = await runLocalAgent(task, ctx.cwd, signal, (partial) => {
							results[index] = partial;
							emitUpdate();
						});
					} catch (error) {
						results[index] = {
							task,
							response: "",
							transcript: results[index]?.transcript || [],
							error: String(error),
						};
					} finally {
						completed += 1;
						emitUpdate();
					}
				}),
			);
			if (signal?.aborted) throw new Error("The local Unsloth agent was cancelled.");
			const completedResults = results.filter(
				(result): result is LocalAgentResult => Boolean(result),
			);
			const succeeded = completedResults.filter((result) => !result.error).length;
			const response =
				completedResults.length === 1
					? completedResults[0].error || completedResults[0].response
					: [
							`Parallel: ${succeeded}/${tasks.length} local agents succeeded`,
							...completedResults.map(
								(result, index) =>
									`\n### Agent ${index + 1}${result.error ? " failed" : ""}\n\n${result.error || result.response}`,
							),
						].join("\n");
			if (succeeded !== completedResults.length) throw new Error(response);
			return {
				content: [{ type: "text", text: response }],
				details: details(),
			};
		},
	});
}
