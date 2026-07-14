// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { describe, expect, it } from "vitest";
import type { JobStatusResponse } from "../../api";
import { isJobStatusPublishable } from "./executions-view-helpers";

function status(overrides: Partial<JobStatusResponse> = {}): JobStatusResponse {
  return {
    job_id: "job-1",
    status: "completed",
    execution_type: "full",
    artifact_path: "/tmp/artifact",
    ...overrides,
  };
}

describe("isJobStatusPublishable", () => {
  it("requires a currently owned completed full job with an artifact", () => {
    expect(isJobStatusPublishable(status())).toBe(true);
    expect(isJobStatusPublishable(status({ status: "running" }))).toBe(false);
    expect(isJobStatusPublishable(status({ execution_type: "preview" }))).toBe(
      false,
    );
    expect(isJobStatusPublishable(status({ artifact_path: null }))).toBe(false);
  });
});
