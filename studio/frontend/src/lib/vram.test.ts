import { describe, expect, it } from "vitest";
import {
  BNB_4BIT_LOADING_BYTES,
  FP16_LOADING_BYTES,
  LOADING_OVERHEAD_GB,
  checkVramFit,
  estimateLoadingVram,
} from "./vram";

// ---------------------------------------------------------------------------
// estimateLoadingVram  (QLoRA / 4-bit -- default)
// ---------------------------------------------------------------------------

describe("estimateLoadingVram (qlora)", () => {
  it("returns overhead only for 0 params", () => {
    expect(estimateLoadingVram(0)).toBe(LOADING_OVERHEAD_GB);
  });

  it("Qwen2.5-0.5B  (0.49B params)", () => {
    const est = estimateLoadingVram(0.49e9);
    // 0.49 * 0.9 + 1.4 = 1.841 -> rounds to 1.8
    expect(est).toBeCloseTo(1.8, 1);
  });

  it("Llama-3.2-1B  (1.24B params)", () => {
    const est = estimateLoadingVram(1.24e9);
    // 1.24 * 0.9 + 1.4 = 2.516 -> rounds to 2.5
    expect(est).toBeCloseTo(2.5, 1);
  });

  it("Llama-3.2-3B  (3.21B params)", () => {
    const est = estimateLoadingVram(3.21e9);
    // 3.21 * 0.9 + 1.4 = 4.289 -> rounds to 4.3
    expect(est).toBeCloseTo(4.3, 1);
  });

  it("Llama-3.1-8B  (8.03B params)", () => {
    const est = estimateLoadingVram(8.03e9);
    // 8.03 * 0.9 + 1.4 = 8.627 -> rounds to 8.6
    expect(est).toBeCloseTo(8.6, 1);
  });
});

// ---------------------------------------------------------------------------
// estimateLoadingVram  (LoRA / fp16)
// ---------------------------------------------------------------------------

describe("estimateLoadingVram (lora / fp16)", () => {
  it("returns overhead only for 0 params", () => {
    expect(estimateLoadingVram(0, "lora")).toBe(LOADING_OVERHEAD_GB);
  });

  it("1B model at fp16", () => {
    const est = estimateLoadingVram(1e9, "lora");
    // 1.0 * 2.0 + 1.4 = 3.4
    expect(est).toBeCloseTo(3.4, 1);
  });

  it("8B model at fp16", () => {
    const est = estimateLoadingVram(8e9, "lora");
    // 8.0 * 2.0 + 1.4 = 17.4
    expect(est).toBeCloseTo(17.4, 1);
  });

  it("full fine-tune uses same fp16 rate", () => {
    const full = estimateLoadingVram(3e9, "full");
    const lora = estimateLoadingVram(3e9, "lora");
    expect(full).toBe(lora);
  });
});

// ---------------------------------------------------------------------------
// Sanity: fp16 always requires more VRAM than 4-bit
// ---------------------------------------------------------------------------

describe("fp16 vs 4-bit ordering", () => {
  it("fp16 estimate is always larger for non-zero params", () => {
    for (const params of [0.5e9, 1e9, 3e9, 7e9, 13e9]) {
      const q4 = estimateLoadingVram(params, "qlora");
      const fp16 = estimateLoadingVram(params, "lora");
      expect(fp16).toBeGreaterThan(q4);
    }
  });
});

// ---------------------------------------------------------------------------
// checkVramFit
// ---------------------------------------------------------------------------

describe("checkVramFit", () => {
  it("fits when ratio <= 0.75", () => {
    expect(checkVramFit(6, 16)).toBe("fits");
    expect(checkVramFit(12, 16)).toBe("fits");
  });

  it("tight when ratio 0.75..1.0", () => {
    expect(checkVramFit(13, 16)).toBe("tight");
    expect(checkVramFit(16, 16)).toBe("tight");
  });

  it("exceeds when ratio > 1.0", () => {
    expect(checkVramFit(17, 16)).toBe("exceeds");
  });

  it("handles 0 available gracefully", () => {
    expect(checkVramFit(5, 0)).toBe("exceeds");
    expect(checkVramFit(0, 0)).toBe("fits");
  });
});

// ---------------------------------------------------------------------------
// Constants sanity
// ---------------------------------------------------------------------------

describe("constants", () => {
  it("4-bit rate is less than fp16 rate", () => {
    expect(BNB_4BIT_LOADING_BYTES).toBeLessThan(FP16_LOADING_BYTES);
  });

  it("overhead is positive", () => {
    expect(LOADING_OVERHEAD_GB).toBeGreaterThan(0);
  });
});
