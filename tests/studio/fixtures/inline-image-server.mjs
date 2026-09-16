// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const repo = path.resolve(
	path.dirname(fileURLToPath(import.meta.url)),
	"../../..",
);
const frontend = path.join(repo, "studio/frontend");
const output = path.resolve(process.argv[2]);
const baseline = process.argv[3];
const { createServer } = await import(
	pathToFileURL(path.join(frontend, "node_modules/vite/dist/node/index.js"))
);
const images = JSON.parse(
	readFileSync(path.join(output, "images.json"), "utf8"),
);
const client = readFileSync(
	new URL("./inline-image-client.js", import.meta.url),
	"utf8",
);
const csp = JSON.parse(
	readFileSync(path.join(repo, "studio/src-tauri/tauri.conf.json"), "utf8"),
).app.security.csp;
const originals = new Map();
if (baseline) {
	if (!/^[a-f0-9]{40}$/.test(baseline))
		throw new Error("Expected a full baseline commit SHA");
	for (const file of [
		"components/assistant-ui/markdown-text.tsx",
		"lib/markdown-data-images.ts",
	]) {
		originals.set(
			path.join(frontend, "src", file).replaceAll("\\", "/"),
			execFileSync("git", ["show", `${baseline}:studio/frontend/src/${file}`], {
				cwd: repo,
				encoding: "utf8",
			}),
		);
	}
}
let history = [];
let reportCount = 0;
let preamble = "";
function send(res, status, type, body) {
	res.writeHead(status, { "Content-Type": type });
	res.end(body);
}
const server = await createServer({
	root: frontend,
	configFile: path.join(frontend, "vite.config.ts"),
	cacheDir: path.join(output, baseline ? "cache-before" : "cache-after"),
	server: { host: "127.0.0.1", port: 0, watch: null, hmr: false },
	plugins: [
		{
			name: "inline-image-validation",
			enforce: "pre",
			resolveId(id) {
				if (id === "virtual:inline-image-validation")
					return "\0inline-image-validation";
			},
			load(id) {
				if (id === "\0inline-image-validation") return client;
				return originals.get(id.replaceAll("\\", "/"));
			},
			configureServer(vite) {
				vite.middlewares.use(async (req, res, next) => {
					const url = new URL(req.url, "http://fixture.invalid");
					if (url.pathname === "/inline-fixture/config") {
						send(
							res,
							200,
							"application/json",
							JSON.stringify({
								variant: baseline ? "before" : "after",
								formats: Object.fromEntries(
									Object.entries(images["32"]).map(([ext, value]) => [
										ext,
										`data:${value.type};base64,${value.data}`,
									]),
								),
								dataImage: `data:image/png;base64,${images["32"].png.data}`,
							}),
						);
					} else if (url.pathname === "/inline-fixture/requests") {
						if (req.method === "DELETE") history = [];
						send(res, 200, "application/json", JSON.stringify(history));
					} else if (url.pathname === "/inline-fixture/report") {
						let body = "";
						for await (const chunk of req) body += chunk;
						const report = JSON.parse(body);
						const filename = `report-${baseline ? "before" : "after"}-${++reportCount}.json`;
						writeFileSync(
							path.join(output, filename),
							JSON.stringify(report, null, 2),
						);
						console.log(
							JSON.stringify({
								filename,
								passed: report.passed,
								failed: report.failed,
								userAgent: report.userAgent,
							}),
						);
						send(res, 200, "application/json", "{}");
					} else if (url.pathname === "/inline-fixture/preamble.js") {
						send(res, 200, "text/javascript", preamble);
					} else if (url.pathname === "/inline-images") {
						let html = await vite.transformIndexHtml(
							req.url,
							'<!doctype html><html><head><title>Inline image simulation</title></head><body><div id="root"></div><script type="module" src="/@id/__x00__inline-image-validation"></script></body></html>',
						);
						html = html.replace(
							/<script type="module">([\s\S]*?)<\/script>/,
							(_, source) => {
								preamble = source;
								return '<script type="module" src="/inline-fixture/preamble.js"></script>';
							},
						);
						res.setHeader("Content-Security-Policy", csp);
						send(res, 200, "text/html", html);
					} else if (
						url.pathname.startsWith("/api/inference/sandbox/") ||
						url.pathname === "/assets/fixture.png"
					) {
						const asset = url.pathname.startsWith("/assets/");
						const authorized =
							req.headers.authorization === "Bearer inline-image-fixture";
						if (!asset) history.push({ path: req.url, authorized });
						if (!asset && !authorized)
							return send(res, 401, "application/json", "{}");
						const filename = decodeURIComponent(url.pathname.split("/").at(-1));
						if (filename === "missing.png")
							return send(res, 404, "application/json", "{}");
						if (filename === "forbidden.png")
							return send(res, 403, "application/json", "{}");
						if (filename === "broken.png")
							return send(res, 200, "image/png", "invalid image");
						const session =
							url.searchParams.get("session") ??
							decodeURIComponent(url.pathname.split("/")[4] ?? "");
						const width =
							{
								"thread-b": 48,
								"project-p1": 64,
								recorded: 80,
								"session/id": 96,
							}[session] ?? 32;
						const ext = filename.split(".").at(-1).toLowerCase();
						const data =
							images[String(width)][ext] ?? images[String(width)].png;
						const respond = () =>
							send(res, 200, data.type, Buffer.from(data.data, "base64"));
						if (filename === "slow.png") setTimeout(respond, 500);
						else respond();
					} else if (/^\/(api|v1|seed|tools)(\/|$)/.test(url.pathname)) {
						send(res, 404, "application/json", "{}");
					} else next();
				});
			},
		},
	],
});
await server.listen();
const address = server.httpServer.address();
writeFileSync(
	path.join(output, "ready.json"),
	JSON.stringify({
		url: `http://127.0.0.1:${address.port}/inline-images`,
		pid: process.pid,
	}),
);
console.log(`Fixture ready on port ${address.port}`);
for (const signal of ["SIGINT", "SIGTERM"])
	process.on(signal, () => server.close().then(() => process.exit(0)));
