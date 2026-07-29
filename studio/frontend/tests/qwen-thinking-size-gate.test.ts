import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

// Mirrors the gate in use-chat-model-runtime.ts, apply-inference-status-to-store.ts and
// llama_cpp.py. Path segments are scanned right to left so the size nearest the leaf wins
// over a size-like parent dir, and a directory identifier (auto-switch snapshot sha, quant
// subdir) still resolves.
const SIZE_RE = /(?:^|[-_.])(\d+\.?\d*)b(?:$|[-_.])/;

function thinkingDefaultOff(modelId: string): boolean {
  const mid = modelId.toLowerCase();
  if (!mid.includes("qwen3.5") && !mid.includes("qwen3.6")) return false;
  const sizeMatch = mid
    .replace(/\\/g, "/")
    .split("/")
    .reduceRight<RegExpMatchArray | null>(
      (found, seg) => found ?? seg.match(SIZE_RE),
      null,
    );
  return !!sizeMatch && Number.parseFloat(sizeMatch[1]) <= 9;
}

test("35B-A3B keeps thinking on: total params win over MoE active params", () => {
  assert.equal(thinkingDefaultOff("unsloth/Qwen3.5-35B-A3B-GGUF"), false);
  assert.equal(thinkingDefaultOff("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"), false);
});

test("sub-9B turns thinking off, including directory identifiers", () => {
  assert.equal(thinkingDefaultOff("unsloth/Qwen3.5-4B-GGUF"), true);
  assert.equal(thinkingDefaultOff("unsloth/Qwen3.5-0.8B-GGUF"), true);
  assert.equal(thinkingDefaultOff("unsloth/Qwen3.5-9B-GGUF"), true);
  assert.equal(thinkingDefaultOff("/m/Qwen3.5-4B-GGUF/UD-Q4_K_XL"), true);
  assert.equal(
    thinkingDefaultOff("/c/models--unsloth--Qwen3.5-4B-GGUF/snapshots/bfc15c3"),
    true,
  );
  assert.equal(thinkingDefaultOff("C:\\models\\Qwen3.5-4B.gguf"), true);
});

test("a trailing separator does not lose the size", () => {
  assert.equal(thinkingDefaultOff("unsloth/Qwen3.5-4B/"), true);
  assert.equal(thinkingDefaultOff("/models/Qwen3.5-4B//"), true);
  assert.equal(thinkingDefaultOff("C:\\models\\Qwen3.5-4B\\"), true);
});

test("a size-like directory does not shadow the real size", () => {
  assert.equal(thinkingDefaultOff("/models/8bit/qwen3.6-27b.gguf"), false);
  assert.equal(thinkingDefaultOff("/models/8b/qwen3.6-27b.gguf"), false);
  // Directory identifier, so there is no file name to prefer.
  assert.equal(thinkingDefaultOff("/models/8b/Qwen3.5-35B-A3B/UD-Q4_K_XL"), false);
  assert.equal(
    thinkingDefaultOff("/models/4b/Qwen3.6-27B-GGUF/snapshots/bfc15c3"),
    false,
  );
});

test("non-qwen3.5/3.6 models are never gated", () => {
  assert.equal(thinkingDefaultOff("unsloth/Qwen3-4B-GGUF"), false);
  assert.equal(thinkingDefaultOff("unsloth/Qwen3.5-9.5B-GGUF"), false);
  assert.equal(thinkingDefaultOff(""), false);
});

test("all four copies of the gate pattern stay in sync", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const read = (rel: string) => readFileSync(path.join(here, rel), "utf8");
  const patterns = [
    read("../src/features/chat/hooks/use-chat-model-runtime.ts").match(
      /const sizeRe = \/(.+?)\/;/,
    ),
    read("../src/features/chat/lib/apply-inference-status-to-store.ts").match(
      /const sizeRe = \/(.+?)\/;/,
    ),
    read("../../backend/core/inference/llama_cpp.py").match(
      /size_re = r"(.+?)"\n/,
    ),
  ];
  for (const found of patterns) assert.ok(found, "size gate pattern not found");
  const expected = SIZE_RE.source;
  for (const found of patterns) assert.equal(found![1], expected);
});
