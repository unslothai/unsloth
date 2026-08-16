import { describe, expect, it } from "vitest";

import { stripPlatformCitationMarkers } from "./platform-citation-markers";

describe("Rag Platform citation markers", () => {
  it("removes raw backend chunk pointers without damaging punctuation", () => {
    expect(
      stripPlatformCitationMarkers(
        "Baran Acar [ID:0]. İzmir [ ID : 12 ]. LinkedIn[ID：2] [ID3]",
      ),
    ).toBe("Baran Acar. İzmir. LinkedIn");
  });

  it("leaves ordinary bracketed numbers untouched", () => {
    expect(stripPlatformCitationMarkers("Use values [0] and [12].")).toBe(
      "Use values [0] and [12].",
    );
  });
});
