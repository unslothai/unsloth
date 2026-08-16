import { afterEach, describe, expect, it } from "vitest";

import {
  hasPreStreamRunReservation,
  releasePreStreamRunReservation,
  reservePreStreamRun,
  yieldWithReleasedPreStreamRunReservation,
} from "./pre-stream-run-reservation";

const tokens: symbol[] = [];

function reserve(threadId: string): symbol {
  const token = reservePreStreamRun([threadId], { usesLocalModel: false });
  expect(token).not.toBeNull();
  tokens.push(token!);
  return token!;
}

afterEach(() => {
  for (const token of tokens.splice(0)) releasePreStreamRunReservation(token);
});

describe("yieldWithReleasedPreStreamRunReservation", () => {
  it("releases the reservation after a successful platform stream", async () => {
    const token = reserve("session-success");
    const source = (async function* () {
      yield "first";
      yield "second";
    })();

    const values: string[] = [];
    for await (const value of yieldWithReleasedPreStreamRunReservation(
      source,
      token,
    )) {
      values.push(value);
    }

    expect(values).toEqual(["first", "second"]);
    expect(hasPreStreamRunReservation(["session-success"])).toBe(false);
  });

  it("releases the reservation when the stream fails", async () => {
    const token = reserve("session-failure");
    const source = (async function* () {
      throw new Error("stream failed");
      yield "unreachable";
    })();
    const run = yieldWithReleasedPreStreamRunReservation(source, token);

    await expect(run.next()).rejects.toThrow("stream failed");
    expect(hasPreStreamRunReservation(["session-failure"])).toBe(false);
  });

  it("releases the reservation when the consumer stops early", async () => {
    const token = reserve("session-abort");
    const source = (async function* () {
      yield "first";
      yield "second";
    })();
    const run = yieldWithReleasedPreStreamRunReservation(source, token);

    await expect(run.next()).resolves.toMatchObject({ value: "first" });
    await run.return();

    expect(hasPreStreamRunReservation(["session-abort"])).toBe(false);
  });
});
