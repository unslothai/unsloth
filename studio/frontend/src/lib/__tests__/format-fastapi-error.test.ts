// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { describe, expect, it } from "vitest";

import {
  formatFastApiDetail,
  readFastApiError,
} from "@/lib/format-fastapi-error";

function fakeResponse(
  status: number,
  body: unknown,
): Response {
  return {
    status,
    async json() {
      if (typeof body === "string") throw new SyntaxError("bad json");
      return body;
    },
  } as unknown as Response;
}

describe("formatFastApiDetail", () => {
  it("returns string detail verbatim", () => {
    expect(formatFastApiDetail("password too short")).toBe(
      "password too short",
    );
  });

  it("treats empty / nullish / non-array shapes as no message", () => {
    expect(formatFastApiDetail("")).toBeNull();
    expect(formatFastApiDetail(null)).toBeNull();
    expect(formatFastApiDetail(undefined)).toBeNull();
    expect(formatFastApiDetail(42)).toBeNull();
    expect(formatFastApiDetail(true)).toBeNull();
    expect(formatFastApiDetail({ foo: 1 })).toBeNull();
  });

  it("joins a single FastAPI 422 array entry as `field: msg`", () => {
    expect(
      formatFastApiDetail([
        { loc: ["body", "password"], msg: "field required" },
      ]),
    ).toBe("password: field required");
  });

  it("joins multi-entry detail with `; ` and strips the 'body' segment", () => {
    expect(
      formatFastApiDetail([
        { loc: ["body", "password"], msg: "field required" },
        { loc: ["body", "username"], msg: "must be lowercase" },
      ]),
    ).toBe("password: field required; username: must be lowercase");
  });

  it("joins nested loc segments with `.`", () => {
    expect(
      formatFastApiDetail([
        { loc: ["body", "settings", "lr"], msg: "must be > 0" },
      ]),
    ).toBe("settings.lr: must be > 0");
  });

  it("falls back to path when msg is missing", () => {
    expect(formatFastApiDetail([{ loc: ["body", "lr"] }])).toBe("lr");
  });

  it("falls back to msg when loc is missing", () => {
    expect(formatFastApiDetail([{ msg: "bad request" }])).toBe("bad request");
  });

  it("returns null for empty / no-content arrays", () => {
    expect(formatFastApiDetail([])).toBeNull();
    expect(formatFastApiDetail([{}, {}])).toBeNull();
  });

  it("skips mid-array junk and yields the valid entry", () => {
    expect(
      formatFastApiDetail([
        null,
        { loc: ["body", "x"], msg: "bad" },
        "not-an-object",
      ]),
    ).toBe("x: bad");
  });
});

describe("readFastApiError", () => {
  it("renders 422 array detail as `field: msg`", async () => {
    const r = fakeResponse(422, {
      detail: [{ loc: ["body", "p"], msg: "bad" }],
    });
    expect(await readFastApiError(r)).toBe("p: bad");
  });

  it("renders 400 string detail verbatim", async () => {
    expect(await readFastApiError(fakeResponse(400, { detail: "Invalid" }))).toBe(
      "Invalid",
    );
  });

  it("falls through to `message` when detail is missing", async () => {
    expect(await readFastApiError(fakeResponse(500, { message: "boom" }))).toBe(
      "boom",
    );
  });

  it("uses the fallback prefix + status code when body has neither", async () => {
    expect(await readFastApiError(fakeResponse(503, {}))).toBe(
      "Request failed (503)",
    );
  });

  it("handles non-JSON bodies via the fallback", async () => {
    expect(await readFastApiError(fakeResponse(502, "not json"))).toBe(
      "Request failed (502)",
    );
  });

  it("honours a custom fallback prefix", async () => {
    expect(
      await readFastApiError(fakeResponse(413, "binary"), "Upload too large"),
    ).toBe("Upload too large (413)");
  });

  it("falls through when detail is an empty array", async () => {
    expect(await readFastApiError(fakeResponse(422, { detail: [] }))).toBe(
      "Request failed (422)",
    );
  });

  it("prefers `message` when detail is an object (not a renderable shape)", async () => {
    expect(
      await readFastApiError(
        fakeResponse(400, { detail: { foo: 1 }, message: "msg-fallback" }),
      ),
    ).toBe("msg-fallback");
  });
});
