// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  SOFTWARE_OVERRIDE_GLOBAL,
  SOFTWARE_OVERRIDE_KEY,
  browserCompositingEnv,
  detectSoftwareCompositing,
  resetSoftwareCompositing,
  softwareCompositing,
} from "../src/features/chat-background/software-compositing.ts";

const UNMASKED = 0x9246;
const RENDERER = 0x1f01;

function gl(renderer: string, unmasked = true) {
  const state = { lost: 0 };
  return {
    state,
    RENDERER,
    getExtension(name: string) {
      if (name === "WEBGL_debug_renderer_info") return unmasked ? { UNMASKED_RENDERER_WEBGL: UNMASKED } : null;
      if (name === "WEBGL_lose_context") return { loseContext: () => void state.lost++ };
      return null;
    },
    getParameter(p: number) {
      if (p === UNMASKED) return renderer;
      if (p === RENDERER) return unmasked ? "WebKit WebGL" : renderer;
      return null;
    },
  };
}
const env = (context: ReturnType<typeof gl> | null, override: boolean | null = null) => ({
  webgl: () => context as never,
  override: () => override,
});

test("software rasterizers and a missing WebGL context mean software compositing", () => {
  for (const name of [
    "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)",
    "llvmpipe (LLVM 17.0.6, 256 bits)",
    "softpipe",
    "ANGLE (Microsoft, Microsoft Basic Render Driver Direct3D11 vs_5_0 ps_5_0, D3D11)",
    "Microsoft Basic Render Driver",
  ]) {
    assert.equal(detectSoftwareCompositing(env(gl(name))), true, name);
  }
  assert.equal(detectSoftwareCompositing(env(null)), true, "no WebGL at all");
  assert.equal(
    detectSoftwareCompositing({ webgl: () => { throw new Error("blocked"); }, override: () => null }),
    true,
    "a context that throws counts as none",
  );
});

test("a GPU renderer is not software, masked or not, and the probe context is released", () => {
  for (const name of [
    "ANGLE (NVIDIA, NVIDIA GeForce RTX 4070 Direct3D11 vs_5_0 ps_5_0, D3D11)",
    "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0, D3D11)",
    "Apple GPU",
    "Apple M2",
  ]) {
    const context = gl(name);
    assert.equal(detectSoftwareCompositing(env(context)), false, name);
    assert.equal(context.state.lost, 1, `${name}: context lost after the probe`);
  }
  assert.equal(detectSoftwareCompositing(env(gl("Apple GPU", false))), false, "no debug extension: RENDERER");
  assert.equal(detectSoftwareCompositing(env(gl("llvmpipe", false))), true);
});

test("the override wins both ways", () => {
  assert.equal(detectSoftwareCompositing(env(gl("Apple GPU"), true)), true);
  assert.equal(detectSoftwareCompositing(env(null, false)), false);
});

test("the browser override reads the window flag first, then localStorage", () => {
  const g = globalThis as Record<string, unknown>;
  const store = new Map<string, string>();
  const saved = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", {
    value: { getItem: (k: string) => store.get(k) ?? null },
    configurable: true,
  });
  try {
    assert.equal(browserCompositingEnv.override(), null);
    store.set(SOFTWARE_OVERRIDE_KEY, "0");
    assert.equal(browserCompositingEnv.override(), false);
    store.set(SOFTWARE_OVERRIDE_KEY, "1");
    assert.equal(browserCompositingEnv.override(), true);
    g[SOFTWARE_OVERRIDE_GLOBAL] = false;
    assert.equal(browserCompositingEnv.override(), false, "the window flag outranks localStorage");
    store.set(SOFTWARE_OVERRIDE_KEY, "junk");
    delete g[SOFTWARE_OVERRIDE_GLOBAL];
    assert.equal(browserCompositingEnv.override(), null);
  } finally {
    delete g[SOFTWARE_OVERRIDE_GLOBAL];
    if (saved) Object.defineProperty(globalThis, "localStorage", saved);
    else delete g.localStorage;
  }
});

test("the answer is probed once and cached until reset", () => {
  resetSoftwareCompositing();
  let probes = 0;
  const counting = { webgl: () => (probes++, gl("Apple GPU") as never), override: () => null };
  assert.equal(softwareCompositing(counting), false);
  assert.equal(softwareCompositing(env(null)), false, "cached, the second environment is not asked");
  assert.equal(probes, 1);
  resetSoftwareCompositing();
  assert.equal(softwareCompositing(env(null)), true);
  resetSoftwareCompositing();
});
