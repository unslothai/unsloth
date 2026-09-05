import { describe, expect, it } from "vitest";

import { generationChunkCountsTowardTiming } from "./chat-generation-recovery";

describe("generationChunkCountsTowardTiming", () => {
  it("counts an ordinary content chunk", () => {
    expect(
      generationChunkCountsTowardTiming({
        choices: [{ delta: { content: "x" } }],
      }),
    ).toBe(true);
  });

  it("ignores the usage-only tail", () => {
    expect(generationChunkCountsTowardTiming({ usage: { total_tokens: 3 }, choices: [] })).toBe(
      false,
    );
  });

  it("ignores a pause or resume notice relayed by the durable run", () => {
    // The worker writes these when the upstream stream says `: preempt-paused` or
    // `: preempt-resumed`; they are a status line, not output, and must not start the
    // first-chunk clock or count as progress.
    expect(generationChunkCountsTowardTiming({ _admissionStatus: "paused" })).toBe(false);
    expect(generationChunkCountsTowardTiming({ _admissionStatus: "resumed" })).toBe(false);
  });
});
