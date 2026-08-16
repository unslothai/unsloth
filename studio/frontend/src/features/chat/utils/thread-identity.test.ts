import { describe, expect, it } from "vitest";
import { isThreadAttached } from "./thread-identity";

describe("isThreadAttached", () => {
  it("treats a missing target as the current thread", () => {
    expect(
      isThreadAttached({
        targetThreadId: undefined,
        mainThreadId: "__LOCALID_1",
        remoteThreadId: undefined,
      }),
    ).toBe(true);
  });

  it("matches the assistant-ui local thread id", () => {
    expect(
      isThreadAttached({
        targetThreadId: "__LOCALID_1",
        mainThreadId: "__LOCALID_1",
        remoteThreadId: "session-1",
      }),
    ).toBe(true);
  });

  it("matches the persisted backend session id", () => {
    expect(
      isThreadAttached({
        targetThreadId: "session-1",
        mainThreadId: "__LOCALID_1",
        remoteThreadId: "session-1",
      }),
    ).toBe(true);
  });

  it("rejects a different local and remote thread", () => {
    expect(
      isThreadAttached({
        targetThreadId: "session-2",
        mainThreadId: "__LOCALID_1",
        remoteThreadId: "session-1",
      }),
    ).toBe(false);
  });
});
