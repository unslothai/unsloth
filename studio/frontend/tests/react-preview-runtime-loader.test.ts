// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type PreviewRuntimeManifest,
  createRuntimeLoader,
} from "../src/features/chat/artifacts/react-preview/runtime-loader.ts";

const BODIES: Record<string, string> = {
  "assets/preview-runtime/react.a.js": "/*react*/",
  "assets/preview-runtime/lucide-react.b.js": "/*lucide*/",
  "assets/preview-runtime/recharts.c.js": "/*recharts*/",
  "assets/preview-runtime/motion.d.js": "/*motion é*/",
  "assets/preview-runtime/tailwind.e.js": "/*tailwind*/",
};
const bytes = (text: string) => new TextEncoder().encode(text).byteLength;
const file = (url: string) => ({ url, bytes: bytes(BODIES[url]), version: "1" });

const MANIFEST: PreviewRuntimeManifest = {
  files: {
    react: file("assets/preview-runtime/react.a.js"),
    "lucide-react": file("assets/preview-runtime/lucide-react.b.js"),
    recharts: file("assets/preview-runtime/recharts.c.js"),
    motion: file("assets/preview-runtime/motion.d.js"),
    tailwind: file("assets/preview-runtime/tailwind.e.js"),
  },
  modules: {
    react: "react",
    "react/jsx-runtime": "react",
    "react-dom/client": "react",
    "lucide-react": "lucide-react",
    recharts: "recharts",
    motion: "motion",
    "motion/react": "motion",
    "framer-motion": "motion",
  },
  tailwind: "tailwind",
};

function fetcher(overrides: Record<string, string> = {}) {
  const calls: string[] = [];
  const fetchBytes = async (url: string) => {
    calls.push(url);
    const body = overrides[url] ?? BODIES[url];
    if (body === undefined) throw new Error("404");
    return new TextEncoder().encode(body);
  };
  return { calls, fetchBytes };
}

test("resolve picks files for the imports, React first, and lists unknown imports", () => {
  const loader = createRuntimeLoader(MANIFEST, fetcher().fetchBytes);
  assert.deepEqual(loader.resolve(["framer-motion", "react/jsx-runtime", "axios", "lucide-react", "axios"]), {
    names: ["react", "lucide-react", "motion"],
    unknown: ["axios"],
  });
  // React even when the code imports nothing.
  assert.deepEqual(loader.resolve([]), { names: ["react"], unknown: [] });
  // Inherited object keys are not modules.
  assert.deepEqual(loader.resolve(["constructor", "toString"]).unknown, ["constructor", "toString"]);
  assert.ok(loader.available.includes("framer-motion"));
});

test("each file is fetched once and decoded as UTF-8", async () => {
  const net = fetcher();
  const loader = createRuntimeLoader(MANIFEST, net.fetchBytes);
  const [a, b] = await Promise.all([loader.load("motion"), loader.load("motion")]);
  assert.equal(a, "/*motion é*/");
  assert.equal(b, a);
  await loader.load("motion");
  assert.deepEqual(net.calls, ["assets/preview-runtime/motion.d.js"]);
});

test("an HTML fallback or a wrong length is refused, and a failed file is retried", async () => {
  const html = "<!doctype html><p>app shell</p>";
  const net = fetcher({ "assets/preview-runtime/react.a.js": html.slice(0, bytes(BODIES["assets/preview-runtime/react.a.js"])) });
  const loader = createRuntimeLoader(MANIFEST, net.fetchBytes);
  await assert.rejects(loader.load("react"), /didn't load correctly/);
  const short = fetcher({ "assets/preview-runtime/recharts.c.js": "/*rech*/" });
  await assert.rejects(createRuntimeLoader(MANIFEST, short.fetchBytes).load("recharts"), /didn't load correctly/);
  // The failure isn't remembered: the next load fetches again.
  await assert.rejects(loader.load("react"));
  assert.equal(net.calls.length, 2);
  await assert.rejects(loader.load("nope"), /Unknown preview library/);
});
