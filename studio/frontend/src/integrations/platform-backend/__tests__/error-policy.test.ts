import { describe, expect, it } from "vitest";

import { getPlatformUiError } from "../error-policy";
import { PlatformApiError } from "../errors";

function apiError(status: number | null, code: number | string) {
  return new PlatformApiError("backend detail must not be shown", {
    httpStatus: status,
    code,
    endpoint: "/system/status",
  });
}

describe("Rag Platform shared UI error policy", () => {
  it.each([
    [401, "HTTP_401", "authentication", false],
    [403, "HTTP_403", "permission", false],
    [429, "HTTP_429", "rate-limit", true],
    [503, "HTTP_503", "server", true],
    [null, "NETWORK_ERROR", "network", true],
    [null, "CLIENT_TIMEOUT", "timeout", true],
    [null, "CLIENT_ABORTED", "aborted", false],
  ])("maps %s/%s to %s", (status, code, kind, retryable) => {
    const result = getPlatformUiError(apiError(status, code));
    expect(result).toMatchObject({ kind, retryable });
    expect(result.message).not.toContain("backend detail");
  });
});
