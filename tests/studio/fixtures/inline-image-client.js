// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import React from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import { useChatRuntimeStore } from "/src/features/chat/index.ts";
import { MarkdownTextSource } from "/src/components/assistant-ui/markdown-text.tsx";
import { ChatProjectScopeContext } from "/src/features/chat/chat-project-scope.ts";
import { storeAuthTokens } from "/src/features/auth/session.ts";
import "/src/index.css";

document.addEventListener("securitypolicyviolation", (event) => {
	const message = `${event.effectiveDirective}: ${event.blockedURI}`;
	document
		.getElementById("errors")
		?.append(document.createTextNode(message + "\n"));
});

const config = await (await fetch("/inline-fixture/config")).json();
storeAuthTokens("inline-image-fixture", "inline-image-fixture");
const adapter = { async *run() {} };
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function until(predicate, timeout = 5000) {
	const start = performance.now();
	while (!predicate()) {
		if (performance.now() - start > timeout)
			throw new Error("Condition timed out");
		await pause(20);
	}
}
const image = () =>
	document.querySelector('#subject img[data-streamdown="image"]');
const requests = async () =>
	await (await fetch("/inline-fixture/requests")).json();
const route = (session, filename) =>
	`/api/inference/sandbox/${session}/${filename}`;

function App() {
	const runtime = useLocalRuntime(adapter);
	const [view, setView] = React.useState({
		text: "Ready",
		thread: "thread-a",
		project: null,
	});
	const [status, setStatus] = React.useState("Ready");
	const [results, setResults] = React.useState([]);
	async function run() {
		const checks = [];
		const unsupportedFormats = [];
		const show = (next) =>
			flushSync(() => {
				useChatRuntimeStore.setState({
					activeThreadId: next.thread ?? "thread-a",
				});
				setView({ thread: "thread-a", project: null, ...next });
			});
		async function check(name, action) {
			setStatus(`Running: ${name}`);
			try {
				await action();
				checks.push({ name, passed: true });
			} catch (error) {
				checks.push({ name, passed: false, error: String(error) });
			}
			setResults([...checks]);
		}
		function assert(condition, message) {
			if (!condition) throw new Error(message);
		}
		async function clear() {
			show({ text: "Reset", id: "reset" });
			await fetch("/inline-fixture/requests", { method: "DELETE" });
		}
		async function loaded(width = 32) {
			await until(() => image()?.complete && image()?.naturalWidth === width);
			assert(
				image().currentSrc.startsWith("blob:"),
				"Sandbox image did not use a blob URL",
			);
		}
		async function success(name, src, expected, scope = {}) {
			await check(name, async () => {
				await clear();
				show({ text: `![Plot](<${src}>)`, id: name, ...scope });
				await loaded(scope.width ?? 32);
				const seen = await requests();
				assert(
					seen.some((r) => r.path === expected && r.authorized),
					JSON.stringify(seen),
				);
			});
		}
		async function finish() {
			const report = {
				variant: config.variant,
				unsupportedFormats,
				userAgent: navigator.userAgent,
				checks,
				passed: checks.filter((c) => c.passed).length,
				failed: checks.filter((c) => !c.passed).length,
			};
			await fetch("/inline-fixture/report", {
				method: "POST",
				body: JSON.stringify(report),
			});
			setStatus(`Complete: ${report.passed} passed, ${report.failed} failed`);
			document.getElementById("report").textContent = JSON.stringify(report);
			document.getElementById("report").dataset.complete = "true";
		}
		if (config.variant === "before") {
			await success(
				"path: plot.png",
				"plot.png",
				route("thread-a", "plot.png"),
			);
			await success(
				"recorded scope",
				route("recorded", "plot.png"),
				route("recorded", "plot.png"),
				{ width: 80 },
			);
			await finish();
			return;
		}
		for (const name of [
			"plot.png",
			"./plot.png",
			"outputs/plot.png",
			"./outputs/plot.png",
			"outputs/./plot.png",
		]) {
			await success(
				`path: ${name}`,
				name,
				route("thread-a", name.replace(/^\.\//, "").replace("/./", "/")),
			);
		}
		for (const name of [
			"loss curve.png",
			"caf\u00e9.png",
			"\u65e5\u672c\u8a9e.png",
			"100%.png",
			"plot #1.png",
			"plot?1.png",
			"literal%2F.png",
			"a+b.png",
			"paren(1).png",
		]) {
			await success(
				`filename: ${name}`,
				encodeURIComponent(name),
				route("thread-a", encodeURIComponent(name)),
			);
		}
		for (const [ext, data] of Object.entries(config.formats)) {
			await check(`format: ${ext}`, async () => {
				const control = new Image();
				control.src = data;
				let supported = true;
				try {
					await control.decode();
				} catch {
					supported = false;
				}
				await clear();
				show({ text: `![Plot](plot.${ext})`, id: `format-${ext}` });
				if (supported) {
					assert(control.naturalWidth === 32, "Invalid codec control");
					await loaded();
				} else {
					assert(ext === "avif", `Required raster codec unavailable: ${ext}`);
					unsupportedFormats.push(ext);
					await until(() =>
						document.querySelector(
							'#subject [data-streamdown="image-fallback"]',
						),
					);
					assert(
						image().naturalWidth === 0,
						"Unsupported image decoded unexpectedly",
					);
				}
				assert(
					(await requests()).some(
						(r) => r.path === route("thread-a", `plot.${ext}`) && r.authorized,
					),
					"Missing authenticated format request",
				);
			});
		}

		await success(
			"project scope",
			"plot.png",
			route("project-p1", "plot.png"),
			{ project: "p1", width: 64 },
		);
		await success(
			"recorded scope",
			route("recorded", "plot.png"),
			route("recorded", "plot.png"),
			{ project: "p1", width: 80 },
		);
		await success(
			"query session",
			"/api/inference/sandbox/_/plot.png?session=session%2Fid",
			"/api/inference/sandbox/_/plot.png?session=session%2Fid",
			{ width: 96 },
		);
		for (const text of [
			'<img src="outputs/plot.png" alt="Plot">',
			"![Plot][chart]\n\n[chart]: outputs/plot.png",
			"[![Plot](outputs/plot.png)](https://example.invalid/)",
		]) {
			await check(`markup: ${text.slice(0, 35)}`, async () => {
				await clear();
				show({ text, id: text });
				await loaded();
				assert(
					(await requests()).every((r) => r.authorized),
					"Missing Authorization header",
				);
				assert(
					!document.querySelector("#subject p div"),
					"Image contains a block inside a paragraph",
				);
			});
		}
		await check("embedded PNG", async () => {
			await clear();
			show({ text: `![Data](${config.dataImage})`, id: "data" });
			await until(() => image()?.complete && image()?.naturalWidth === 32);
			assert(
				(await requests()).length === 0,
				"Data image fetched a sandbox URL",
			);
		});
		await check("root asset remains supported", async () => {
			await clear();
			show({ text: "![Asset](/assets/fixture.png)", id: "asset" });
			await until(() => image()?.complete && image()?.naturalWidth === 32);
		});
		const forbidden = [
			"../secret.png",
			"outputs/../../secret.png",
			"..%2Fsecret.png",
			"..%5Csecret.png",
			"/api/inference/sandbox/thread-a/../other/secret.png",
			"/api/inference/sandbox/thread-a/%2E%2E/other/secret.png",
			"./api/inference/sandbox/thread-a/../other/secret.png",
			"/api/inference/sandbox/thread-a/outputs%2F..%2F..%2Fother/secret.png",
			"/api/inference/sandbox/thread-a/%00plot.png",
			"//example.invalid/api/inference/sandbox/other/plot.png",
			"https://example.invalid/plot.png",
			"http://127.0.0.1/plot.png",
			"file:///tmp/plot.png",
			"C:\\Users\\test\\plot.png",
			"outputs\\plot.png",
			"javascript:alert%281%29",
			"data:text/html;base64,PGgxPnRlc3Q8L2gxPg==",
		];
		for (const src of forbidden)
			await check(`reject: ${src}`, async () => {
				await clear();
				show({ text: `![Rejected](<${src}>)`, id: src });
				await pause(80);
				assert(
					!image()?.getAttribute("src"),
					"Rejected source reached an image",
				);
				assert(
					(await requests()).length === 0,
					"Rejected source requested a sandbox file",
				);
			});
		for (const text of [
			"plot.png",
			"`![Plot](plot.png)`",
			"```python\n![Plot](plot.png)\n```",
			"[Documentation](https://example.invalid/page)",
		]) {
			await check(`non-image: ${text.slice(0, 30)}`, async () => {
				await clear();
				show({ text, id: text });
				await pause(80);
				assert(!image(), "Non-image content became an image");
				assert(
					(await requests()).length === 0,
					"Non-image content fetched a file",
				);
			});
		}
		await check("HTML event handlers are removed", async () => {
			await clear();
			show({
				text: '<img src="plot.png" onload="document.title=\'UNSAFE\'" onerror="document.title=\'UNSAFE\'">',
				id: "handlers",
			});
			await loaded();
			assert(document.title !== "UNSAFE", "HTML handler executed");
			assert(
				!image().hasAttribute("onload") && !image().hasAttribute("onerror"),
				"Handler survived sanitization",
			);
		});
		for (const filename of ["missing.png", "forbidden.png", "broken.png"])
			await check(`failure and recovery: ${filename}`, async () => {
				await clear();
				show({ text: `![Plot](${filename})`, id: "recover" });
				await until(() =>
					document.querySelector('#subject [data-streamdown="image-fallback"]'),
				);
				show({ text: "![Plot](plot.png)", id: "recover" });
				await loaded();
				assert(
					!document.querySelector(
						'#subject [data-streamdown="image-fallback"]',
					),
					"Fallback stayed visible",
				);
			});
		await check("unauthenticated direct request is denied", async () => {
			assert(
				(await fetch(route("thread-a", "plot.png"))).status === 401,
				"Fixture accepted no token",
			);
		});
		await check("scope switches invalidate cached Markdown", async () => {
			await clear();
			for (const scope of [
				{ thread: "thread-a", width: 32 },
				{ thread: "thread-b", width: 48 },
				{ project: "p1", width: 64 },
				{ thread: "thread-a", width: 32 },
			]) {
				show({ text: "![Plot](plot.png)", id: "scope-switch", ...scope });
				await loaded(scope.width);
			}
		});
		await check("late response cannot replace a newer scope", async () => {
			await clear();
			show({ text: "![Plot](slow.png)", id: "race", thread: "thread-a" });
			await pause(70);
			show({ text: "![Plot](plot.png)", id: "race", thread: "thread-b" });
			await loaded(48);
			await pause(650);
			assert(
				image().naturalWidth === 48,
				"Late image replaced the current scope",
			);
		});
		await check("offscreen images wait for visibility", async () => {
			await clear();
			show({ text: "![Plot](plot.png)", id: "offscreen", offscreen: true });
			window.scrollTo(0, 0);
			assert(
				image().getBoundingClientRect().top > innerHeight + 200,
				"Fixture image is inside the preload margin",
			);
			await pause(100);
			assert(
				(await requests()).length === 0,
				`Offscreen image loaded early: top=${image().getBoundingClientRect().top}, scroll=${scrollY}`,
			);
			document.getElementById("subject").scrollIntoView();
			await loaded();
		});
		await check("streaming image link completes", async () => {
			await clear();
			window.scrollTo(0, 0);
			const text = "Here is the plot:\n\n![Plot](outputs/plot.png)";
			for (let n = 1; n <= text.length; n += 3) {
				show({ text: text.slice(0, n), id: "stream", streaming: true });
				await pause(20);
			}
			show({ text, id: "stream", streaming: false });
			await loaded();
		});
		await finish();
	}
	return React.createElement(
		AssistantRuntimeProvider,
		{ runtime },
		React.createElement(
			"main",
			{ style: { padding: 24, maxWidth: 1000, margin: "auto" } },
			React.createElement("h1", null, "Inline image compatibility simulation"),
			React.createElement(
				"button",
				{ onClick: run, disabled: status.startsWith("Running") },
				"Run simulations",
			),
			...[
				["Show embedded download", `![Embedded](${config.dataImage})`],
				["Show encoded download", "![Plot](outputs/loss%20curve%20%231.png)"],
			].map(([label, text]) =>
				React.createElement(
					"button",
					{
						key: label,
						onClick: () =>
							setView({ text, id: label, thread: "thread-a", project: null }),
					},
					label,
				),
			),

			React.createElement("p", { id: "status", role: "status" }, status),
			React.createElement(
				"ul",
				null,
				...results
					.filter((r) => !r.passed)
					.map((r) =>
						React.createElement("li", { key: r.name }, `${r.name}: ${r.error}`),
					),
			),
			React.createElement("pre", { id: "report", hidden: true }),
			React.createElement("pre", { id: "errors" }),
			React.createElement(
				"div",
				{
					id: "subject",
					key: view.offscreen ? "offscreen" : "visible",
					style: { marginTop: view.offscreen ? 2500 : 24 },
				},
				React.createElement(
					ChatProjectScopeContext.Provider,
					{ value: view.project },
					React.createElement(MarkdownTextSource, {
						messageId: view.id ?? "case",
						sourceText: view.text,
						streaming: view.streaming ?? false,
						messageHasRenderableRenderHtmlTool: false,
					}),
				),
			),
		),
	);
}
createRoot(document.getElementById("root")).render(React.createElement(App));
