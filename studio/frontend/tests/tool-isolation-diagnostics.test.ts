// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { runInNewContext } from "node:vm";
import test from "node:test";
import {
  parseBackendExecutionRecord,
  toolExecutionRecordLabel,
  toolExecutionRecordFromCard,
  stripUntrustedExecutionMetadata,
} from "../src/features/chat/types/api.ts";

const source = readFileSync(
  new URL("../src/features/chat/tool-isolation.ts", import.meta.url),
  "utf8",
);

test("backend nested records retain the distinct mode and shared-proc disclosure", () => {
  const value = {
    requested_mode: "container_isolation",
    effective_mode: "container_isolation",
    environment: "colab",
    backend: "srt",
    profile_id: "srt-0.0.75-nested-cap-drop-v1",
    probe_generation: "g1",
    os_isolation: true,
    retained_safeguards: [],
    authority_disclosure:
      "Uses the outer container's /proc and exposes additional process information.",
  };
  const record = parseBackendExecutionRecord(value);
  assert.equal(
    toolExecutionRecordLabel(record),
    "Container-compatible isolation · SRT",
  );
  assert.equal(record?.authority_disclosure, value.authority_disclosure);
  assert.equal(
    toolExecutionRecordLabel(
      parseBackendExecutionRecord({ ...value, os_isolation: false }),
    ),
    null,
  );
  assert.deepEqual(
    stripUntrustedExecutionMetadata({ __unsloth_execution_record: value }),
    {},
  );
});
const parser = source.slice(
  source.indexOf("function parseCapability("),
  source.indexOf("let capabilityRequest:"),
);
const parse = runInNewContext(
  stripTypeScriptTypes(parser) + "\nparseCapability;",
);
const permission = readFileSync(
  new URL("../src/features/chat/permission-mode-select.tsx", import.meta.url),
  "utf8",
);
const warning = runInNewContext(
  stripTypeScriptTypes(
    permission.slice(
      permission.indexOf("export const TOOL_ISOLATION_UNAVAILABLE_WARNING"),
      permission.indexOf("export function permissionModeOption"),
    ),
  ).replace(/^export /gm, "") + "\nlimitedModeWarning;",
);

for (const environment of ["colab", "linux"]) {
  test(`${environment}: capability disclosure reaches consent and backend record`, () => {
    const disclosure = `Python and Terminal run with Studio's permissions ${environment === "colab" ? "inside this Colab runtime" : "in this environment"}. They can access files and credentials available to Studio and use its network connection. Studio does not add an OS sandbox in Limited mode.`;
    const capability = parse({
      environment,
      protection_state: "unavailable",
      probe_generation: "test",
      environment_fingerprint: "test",
      reason_code: "dependency_missing",
      diagnostic: {
        code: "dependency_missing",
        stage: "dependency",
        dependency: "bubblewrap",
      },
      limited_disclosure: disclosure,
    });
    assert.equal(capability.diagnostic.dependency, "bubblewrap");
    assert.equal(warning(capability), disclosure);
    const value = {
      requested_mode: "limited",
      effective_mode: "limited",
      environment,
      backend: "process-guard",
      profile_id: "limited-software-safeguards-v1",
      probe_generation: "test",
      os_isolation: false,
      retained_safeguards: [],
      authority_disclosure: disclosure,
    };
    const record = parseBackendExecutionRecord(value);
    assert.equal(record?.authority_disclosure, disclosure);
    assert.equal(toolExecutionRecordLabel(record), "Limited · no OS isolation");
    assert.deepEqual(
      stripUntrustedExecutionMetadata({ __unsloth_execution_record: value }),
      {},
    );
    assert.equal(toolExecutionRecordFromCard("untrusted-record-fixture"), null);
  });
}
test("legacy capability keeps a generic warning and malformed diagnostics do not render", () => {
  const cap = parse({
    environment: "linux",
    protection_state: "unavailable",
    probe_generation: "test",
    environment_fingerprint: "test",
    diagnostic: { code: [], stage: "policy" },
  });
  assert.equal(cap.diagnostic, null);
  assert.doesNotMatch(warning(cap), /Colab/);
  assert.match(warning(cap), /files, credentials, and network/);
});
