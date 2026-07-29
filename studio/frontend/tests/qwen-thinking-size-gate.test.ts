import assert from "node:assert/strict";
import { test } from "node:test";

// Mirrors the gate in use-chat-model-runtime.ts, apply-inference-status-to-store.ts and
// llama_cpp.py. Basename first so /8bit/ loses to the real size, full path as fallback so a
// directory identifier (auto-switch snapshot sha, quant subdir) still resolves.
function thinkingDefaultOff(modelId: string): boolean {
  const mid = modelId.toLowerCase();
  if (!mid.includes("qwen3.5") && !mid.includes("qwen3.6")) return false;
  const midSlash = mid.replace(/\\/g, "/");
  const sizeRe = /(?:^|[-_/.])(\d+\.?\d*)b(?:$|[-_/.])/;
  const sizeMatch =
    (midSlash.split("/").pop() || "").match(sizeRe) ?? midSlash.match(sizeRe);
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
});

test("non-qwen3.5/3.6 models are never gated", () => {
  assert.equal(thinkingDefaultOff("unsloth/Qwen3-4B-GGUF"), false);
  assert.equal(thinkingDefaultOff("unsloth/Qwen3.5-9.5B-GGUF"), false);
  assert.equal(thinkingDefaultOff(""), false);
});
