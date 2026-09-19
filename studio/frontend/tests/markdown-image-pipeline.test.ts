// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

import { markdownSandboxImageSrc } from "../src/components/assistant-ui/sandbox-files.ts";
import { rehypeSandboxImages } from "../src/components/assistant-ui/rehype-sandbox-images.ts";
import { withDataImageSupport } from "../src/lib/markdown-data-images.ts";
import { safeMarkdownUrl } from "../src/lib/safe-markdown-url.ts";

const context = { threadId: "thread-1", projectId: null as string | null };

function render(markdown: string, scope = context) {
	const sources: string[] = [];
	const html = renderToStaticMarkup(
		createElement(Streamdown, {
			mode: "static",
			children: markdown,
			rehypePlugins: withDataImageSupport({}, [[rehypeSandboxImages, scope]]),
			urlTransform: safeMarkdownUrl,
			components: {
				img: ({ src }) => {
					// No rotated workspace session in these scopes: the project cases expect the
				// `project-<id>` fallback, which is what a project that has never moved has.
				if (src)
					sources.push(
						markdownSandboxImageSrc(src, { ...scope, workspaceSessionId: null }) ?? src,
					);
					return null;
				},
			},
		}),
	);
	return { sources, html };
}

test("relative Python images survive the full Markdown pipeline", () => {
	for (const [src, file] of [
		["line_plot.png", "line_plot.png"],
		["outputs/line_plot.png", "outputs/line_plot.png"],
		["./line_plot.png", "line_plot.png"],
		["./outputs/line_plot.png", "outputs/line_plot.png"],
		["outputs/loss%20curve%20%231.png", "outputs/loss%20curve%20%231.png"],
	]) {
		const result = render(`![Plot](${src})`);
		assert.deepEqual(
			result.sources,
			[`/api/inference/sandbox/thread-1/${file}`],
			src,
		);
		assert.doesNotMatch(result.html, /Image blocked/, src);
	}
});

test("relative images use project scope and explicit URLs keep their session", () => {
	const scope = { threadId: "thread-2", projectId: "project-1" };
	assert.deepEqual(render("![Plot](line_plot.png)", scope).sources, [
		"/api/inference/sandbox/project-project-1/line_plot.png",
	]);
	const src = "/api/inference/sandbox/original/line_plot.png";
	assert.deepEqual(render(`![Plot](${src})`, scope).sources, [src]);
});

test("HTML image sources are resolved after raw HTML is parsed", () => {
	assert.deepEqual(
		render('<img src="outputs/line_plot.png" alt="Plot">').sources,
		["/api/inference/sandbox/thread-1/outputs/line_plot.png"],
	);
});

test("data images still render and remote image URLs stay blocked", () => {
	const data = "data:image/png;base64,iVBORw0KGgo=";
	assert.deepEqual(render(`![Plot](${data})`).sources, [data]);
	for (const src of [
		"https://example.com/plot.png",
		"http://127.0.0.1/plot.png",
		"//example.com/plot.png",
		"file:///tmp/plot.png",
		"javascript:alert%281%29",
		"data:text/html;base64,PGgxPkhlbGxvPC9oMT4=",
	]) {
		assert.deepEqual(render(`![Plot](${src})`).sources, [], src);
	}
});

test("URL normalization cannot turn traversal into another sandbox image", () => {
	for (const src of [
		"/api/inference/sandbox/thread-1/../other/secret.png",
		"/api/inference/sandbox/thread-1/%2e%2e/other/secret.png",
		"/api/inference/sandbox/thread-1/outputs/../../other/secret.png",
		"./api/inference/sandbox/thread-1/../other/secret.png",
		"./api/inference/sandbox/thread-1/%2e%2e/other/secret.png",
		"/api/inference/sandbox/thread-1/outputs%2F..%2F..%2Fother/secret.png",
		"/api/inference/sandbox/thread-1/..%5Cother/secret.png",
	]) {
		assert.deepEqual(render(`![Plot](${src})`).sources, [], src);
	}
});

test("encoded filenames and image formats resolve across Markdown, HTML, and scopes", () => {
	const names = [
		"plot",
		"loss curve",
		"café",
		"日本語",
		"100%",
		"plot #1",
		"plot?1",
		"literal%2F",
		"a+b",
		"paren(1)",
	];
	const scopes = [
		{
			threadId: "thread-1",
			projectId: null,
			prefix: "/api/inference/sandbox/thread-1/",
			query: "",
		},
		{
			threadId: "thread-2",
			projectId: null,
			prefix: "/api/inference/sandbox/thread-2/",
			query: "",
		},
		{
			threadId: "thread-1",
			projectId: "project-1",
			prefix: "/api/inference/sandbox/project-project-1/",
			query: "",
		},
		{
			threadId: "thread/id",
			projectId: null,
			prefix: "/api/inference/sandbox/_/",
			query: "?session=thread%2Fid",
		},
	];
	for (const ext of ["png", "jpg", "jpeg", "gif", "webp", "bmp", "avif"]) {
		for (const name of names) {
			const filename = encodeURIComponent(`${name}.${ext}`);
			for (const directory of ["", "./", "outputs/"]) {
				const src = `${directory}${filename}`;
				for (const scope of scopes) {
					const expected = `${scope.prefix}${directory === "outputs/" ? directory : ""}${filename}${scope.query}`;
					for (const markup of [
						`![Plot](<${src}>)`,
						`<img alt="Plot" src="${src}">`,
					]) {
						assert.deepEqual(
							render(markup, scope).sources,
							[expected],
							`${markup} in ${scope.threadId}/${scope.projectId}`,
						);
					}
				}
			}
		}
	}
});

test("path mutations cannot escape scope through Markdown or HTML", () => {
	const segments = [
		"..",
		"%2e%2e",
		".%2E",
		"%2e.",
		"..%2fother",
		"..%5cother",
		"other%2f..",
		"%2F",
		"%5C",
	];
	for (let code = 0; code < 32; code++)
		segments.push(`%${code.toString(16).padStart(2, "0")}`);
	segments.push("%7f");
	for (const prefix of [
		"",
		"./",
		"outputs/",
		"/api/inference/sandbox/thread-1/",
		"./api/inference/sandbox/thread-1/",
	]) {
		for (const segment of segments) {
			const src = `${prefix}${segment}/secret.png`;
			for (const markup of [
				`![Plot](<${src}>)`,
				`<img alt="Plot" src="${src}">`,
			]) {
				assert.deepEqual(render(markup).sources, [], markup);
			}
		}
	}
});
