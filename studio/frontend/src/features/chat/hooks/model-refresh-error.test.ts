import { describe, expect, it } from "vitest";
import { modelRefreshErrorMessage } from "./model-refresh-error";

describe("modelRefreshErrorMessage", () => {
  it("suppresses an optional background refresh failure", () => {
    expect(modelRefreshErrorMessage(new Error("Request failed (502)"), false)).toBeNull();
  });

  it("keeps explicit local-model failures visible", () => {
    expect(modelRefreshErrorMessage(new Error("Request failed (502)"))).toBe(
      "Request failed (502)",
    );
    expect(modelRefreshErrorMessage("unknown failure")).toBe(
      "Failed to load models",
    );
  });
});
