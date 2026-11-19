// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

import {
	markdownSandboxVideoSrc,
	sandboxFileForSrc,
	sandboxFileForHref,
	sandboxVideoFileForSrc,
} from "../src/components/assistant-ui/sandbox-files.ts";
import { rehypeSandboxImages } from "../src/components/assistant-ui/rehype-sandbox-images.ts";
import { withDataImageSupport } from "../src/lib/markdown-data-images.ts";
import { safeMarkdownUrl } from "../src/lib/safe-markdown-url.ts";
import { videoMimeForName } from "../src/lib/video-utils.ts";

const scope = { threadId: "thread-1", projectId: null as string | null };

function videoSources(markdown: string) {
	const sources: (string | null)[] = [];
	renderToStaticMarkup(
		createElement(Streamdown, {
			mode: "static",
			children: markdown,
			rehypePlugins: withDataImageSupport({}, [[rehypeSandboxImages, scope]]),
			urlTransform: safeMarkdownUrl,
			components: {
				img: ({ src }) => {
					sources.push(src ? markdownSandboxVideoSrc(src, scope) : null);
					return null;
				},
			},
		}),
	);
	return sources;
}

test("a sandbox video in image syntax reaches the player as an authed route path", () => {
	assert.deepEqual(videoSources("![Spin](protein.mp4)"), [
		"/api/inference/sandbox/thread-1/protein.mp4",
	]);
	assert.deepEqual(videoSources("![Spin](outputs/clip.webm)"), [
		"/api/inference/sandbox/thread-1/outputs/clip.webm",
	]);
	assert.deepEqual(videoSources("![Plot](plot.png)"), [null]);
});

test("video files stay out of the image path and keep the sandbox guards", () => {
	assert.equal(sandboxFileForSrc("protein.mp4"), null);
	assert.equal(sandboxVideoFileForSrc("protein.mp4"), "protein.mp4");
	assert.equal(sandboxVideoFileForSrc("plot.png"), null);
	assert.equal(sandboxVideoFileForSrc("../other/clip.mp4"), null);
	assert.equal(sandboxVideoFileForSrc("https://example.com/clip.mp4"), null);
	assert.equal(sandboxVideoFileForSrc("clip.mp4/notes"), null);
	assert.equal(sandboxFileForHref("clip.m4v"), "clip.m4v");
	assert.equal(videoMimeForName("/api/inference/sandbox/t/clip.webm"), "video/webm");
	assert.equal(videoMimeForName("clip.MOV"), "video/quicktime");
});
