import { describe, expect, test } from "bun:test";
import * as fs from "node:fs/promises";
import * as path from "node:path";

import { createMemoryStore } from "./memory";

async function mkTmpDir(): Promise<string> {
  const root = await fs.mkdtemp(path.join("/tmp/", "opencode-memory-"));
  return root;
}

describe("store", () => {
  test("seeds and writes blocks", async () => {
    const dir = await mkTmpDir();
    const store = createMemoryStore(dir);
    await store.ensureSeed();

    const blocks = await store.listBlocks("all");
    const labels = blocks.map((b) => `${b.scope}:${b.label}`);

    expect(labels).toContain("global:persona");
    expect(labels).toContain("global:human");
    expect(labels).toContain("project:project");

    await store.setBlock("project", "project", "hello");
    const b = await store.getBlock("project", "project");
    expect(b.value).toBe("hello");
  });
});
