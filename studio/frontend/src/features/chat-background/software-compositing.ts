// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * Whether this page is composited in software: no WebGL at all, or WebGL on a software rasterizer
 * (SwiftShader, llvmpipe, softpipe, Microsoft Basic Render Driver). There every frame redraws the whole
 * window and recomputes every backdrop-filter on screen, which is what glass-motion.ts pauses. With a GPU
 * the blur is cheap and the pause would only change the look, so glass keeps it.
 * Checked once per page and cached. For testing, `window.__UNSLOTH_SOFTWARE_COMPOSITING__ = true | false`
 * or localStorage `unsloth_software_compositing` = "1" | "0" overrides the probe.
 */

export const SOFTWARE_RENDERER = /swiftshader|llvmpipe|softpipe|basic render/i;
export const SOFTWARE_OVERRIDE_KEY = "unsloth_software_compositing";
export const SOFTWARE_OVERRIDE_GLOBAL = "__UNSLOTH_SOFTWARE_COMPOSITING__";

type GL = Pick<WebGLRenderingContext, "getExtension" | "getParameter" | "RENDERER">;

export interface CompositingEnv {
  /** A WebGL context, or null when none can be created. */
  webgl(): GL | null;
  /** The test override: true / false, or null for none. */
  override(): boolean | null;
}

/** The WebGL renderer string, unmasked when the browser allows it. */
export function rendererOf(gl: GL): string {
  const info = gl.getExtension("WEBGL_debug_renderer_info") as { UNMASKED_RENDERER_WEBGL: number } | null;
  const name = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  return typeof name === "string" ? name : "";
}

export function detectSoftwareCompositing(env: CompositingEnv): boolean {
  const forced = env.override();
  if (forced !== null) return forced;
  let gl: GL | null = null;
  try {
    gl = env.webgl();
  } catch {
    gl = null;
  }
  if (!gl) return true;
  try {
    return SOFTWARE_RENDERER.test(rendererOf(gl));
  } finally {
    // Give the context back now: browsers cap live WebGL contexts per page.
    (gl.getExtension("WEBGL_lose_context") as { loseContext(): void } | null)?.loseContext();
  }
}

function parseOverride(value: unknown): boolean | null {
  if (value === true || value === "1" || value === "true") return true;
  if (value === false || value === "0" || value === "false") return false;
  return null;
}

export const browserCompositingEnv: CompositingEnv = {
  webgl() {
    if (typeof document === "undefined") return null;
    const canvas = document.createElement("canvas");
    return (canvas.getContext("webgl") ?? canvas.getContext("experimental-webgl")) as GL | null;
  },
  override() {
    const flag = parseOverride((globalThis as Record<string, unknown>)[SOFTWARE_OVERRIDE_GLOBAL]);
    if (flag !== null) return flag;
    try {
      return parseOverride(globalThis.localStorage?.getItem(SOFTWARE_OVERRIDE_KEY));
    } catch {
      return null;
    }
  },
};

let cached: boolean | undefined;

/** Software compositing on this page; probed on the first call only. */
export function softwareCompositing(env: CompositingEnv = browserCompositingEnv): boolean {
  if (cached === undefined) cached = detectSoftwareCompositing(env);
  return cached;
}

/** Forget the cached answer (tests). */
export function resetSoftwareCompositing(): void {
  cached = undefined;
}
