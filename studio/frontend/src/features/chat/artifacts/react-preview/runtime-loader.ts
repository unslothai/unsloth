// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Picks and fetches the bundled library files a preview imports. Each file is fetched once per
// session; the frame is an opaque origin, so the parent inlines them into the page.

export type PreviewRuntimeManifest = {
  files: Record<string, { url: string; bytes: number; version: string }>;
  modules: Record<string, string>;
  tailwind: string;
};

export type FetchBytes = (url: string) => Promise<Uint8Array>;

const REACT_FILE = "react";

export function createRuntimeLoader(manifest: PreviewRuntimeManifest, fetchBytes: FetchBytes) {
  const pending = new Map<string, Promise<string>>();
  const order = Object.keys(manifest.files).filter((name) => name !== manifest.tailwind);

  /** File names for `deps` in load order (React always first), and the imports no file provides. */
  const resolve = (deps: readonly string[]): { names: string[]; unknown: string[] } => {
    const wanted = new Set<string>([REACT_FILE]);
    const unknown: string[] = [];
    for (const dep of deps) {
      const name = Object.hasOwn(manifest.modules, dep) ? manifest.modules[dep] : undefined;
      if (name) wanted.add(name);
      else if (!unknown.includes(dep)) unknown.push(dep);
    }
    return { names: order.filter((name) => wanted.has(name)), unknown };
  };

  const fetchFile = async (name: string): Promise<string> => {
    const file = manifest.files[name];
    if (!file) throw new Error(`Unknown preview library "${name}"`);
    const bytes = await fetchBytes(file.url);
    // A stale or wrong path can come back as the app's index.html; never inline that as a script.
    if (bytes.byteLength !== file.bytes || bytes[0] === 0x3c) {
      throw new Error(`Preview library "${name}" didn't load correctly`);
    }
    return new TextDecoder().decode(bytes);
  };

  const load = (name: string): Promise<string> => {
    let entry = pending.get(name);
    if (!entry) {
      entry = fetchFile(name);
      // A failed fetch is forgotten, so Run again retries it.
      entry.catch(() => pending.delete(name));
      pending.set(name, entry);
    }
    return entry;
  };

  return {
    resolve,
    load,
    /** Every specifier a preview can import, for the unknown-import message. */
    available: Object.keys(manifest.modules),
    tailwind: manifest.tailwind,
  };
}

export type RuntimeLoader = ReturnType<typeof createRuntimeLoader>;
