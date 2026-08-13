import { describe, expect, it } from "vitest";

import { formatProfileCount } from "./stats-format";

describe("formatProfileCount", () => {
  it("formats profile statistics in Turkish", () => {
    expect(formatProfileCount(2, "message", "tr")).toBe("2 mesaj");
    expect(formatProfileCount(4, "step", "tr")).toBe("4 adım");
    expect(formatProfileCount(8, "token", "tr")).toBe("8 token");
  });
});
