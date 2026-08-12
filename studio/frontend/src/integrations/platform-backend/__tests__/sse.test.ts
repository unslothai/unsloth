import { describe, expect, it } from "vitest";

import { PlatformSseParser } from "../sse";

describe("PlatformSseParser", () => {
  it("parses CRLF, fragmented frames and multi-line data", () => {
    const parser = new PlatformSseParser();

    expect(parser.feed("event: answer\r")).toEqual([]);
    expect(parser.feed("\ndata: first\r\ndata: sec")).toEqual([]);
    expect(parser.feed("ond\r\nid: 42\r\n\r\n")).toEqual([
      {
        data: "first\nsecond",
        event: "answer",
        id: "42",
        retry: undefined,
        terminal: false,
      },
    ]);
  });

  it("recognizes the Rag Platform data:true terminal envelope", () => {
    const parser = new PlatformSseParser();

    expect(
      parser.feed('data: {"code":0,"data":true,"message":"success"}\n\n'),
    ).toEqual([expect.objectContaining({ terminal: true })]);
  });

  it("flushes a final frame without a trailing blank line", () => {
    const parser = new PlatformSseParser();
    parser.feed("data: final");

    expect(parser.end()).toEqual([
      expect.objectContaining({ data: "final", terminal: false }),
    ]);
  });
});
