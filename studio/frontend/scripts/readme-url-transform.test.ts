// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";

const srcRoot = resolve(import.meta.dirname, "../src");

function resolveLocalModule(base: string, specifier: string): string | null {
  const target = resolve(base, specifier);
  const candidates = [
    target,
    `${target}.ts`,
    `${target}.tsx`,
    resolve(target, "index.ts"),
    resolve(target, "index.tsx"),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return pathToFileURL(candidate).href;
    }
  }
  return null;
}

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith("@/")) {
      const url = resolveLocalModule(srcRoot, specifier.slice(2));
      if (url) return { shortCircuit: true, url };
    }
    if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
      const url = resolveLocalModule(
        dirname(fileURLToPath(context.parentURL)),
        specifier,
      );
      if (url) return { shortCircuit: true, url };
    }
    return nextResolve(specifier, context);
  },
});

const { createReadmeUrlTransform } = await import(
  "../src/features/models/lib/hf-readme.ts"
);

const readmeAllowedTags = {
  audio: ["src", "controls", "preload", "loop", "muted"],
  video: [
    "src",
    "controls",
    "preload",
    "loop",
    "muted",
    "poster",
    "width",
    "height",
    "playsinline",
  ],
  source: ["src", "type", "media"],
  track: ["src", "kind", "srclang", "label", "default"],
};

test("README media rendering strips unsafe media urls and resolves safe relative sources", () => {
  const baseUrl = "https://huggingface.co/org/model/resolve/main/";
  const markdown = [
    '<video controls><source src="media/sample.mp4" type="video/mp4">',
    '<source src="javascript:alert(1)" type="video/mp4">',
    '<track src="javascript:alert(2)" kind="captions"></video>',
    '<audio src="data:audio/wav;base64,UklGRg=="></audio>',
    "[bad](vbscript:alert(3))",
  ].join("");

  const html = renderToStaticMarkup(
    React.createElement(
      Streamdown,
      {
        mode: "static",
        allowedTags: readmeAllowedTags,
        urlTransform: createReadmeUrlTransform(baseUrl),
      },
      markdown,
    ),
  );

  assert.match(html, /<video/);
  assert.match(html, /<audio><\/audio>/);
  assert.match(html, /<track kind="captions"\/>/);
  assert.match(
    html,
    /src="https:\/\/huggingface\.co\/org\/model\/resolve\/main\/media\/sample\.mp4"/,
  );
  assert.doesNotMatch(html, /javascript:/i);
  assert.doesNotMatch(html, /vbscript:/i);
  assert.doesNotMatch(html, /data:audio/i);
});
