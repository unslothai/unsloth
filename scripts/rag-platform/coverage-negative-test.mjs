#!/usr/bin/env node

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { validateCoverageRecords } from "./coverage-release-validator.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const matrix = JSON.parse(
  readFileSync(resolve(ROOT, "docs/rag-platform/endpoint-coverage-matrix.json"), "utf8"),
);
const fakeRoute = {
  ...matrix.records[0],
  method: "GET",
  path: "/api/v1/__phase15_negative_gate__",
  service: "go-api",
  runtime: "enabled",
  class: "unsupported",
  status: "unclassified",
  justification: "",
  test_evidence: [],
};
const failures = validateCoverageRecords([...matrix.records, fakeRoute]);

if (
  !failures.some((failure) => failure.includes("forbidden status unclassified")) ||
  !failures.some((failure) => failure.includes("reachable endpoint is unsupported"))
) {
  console.error("coverage negative test FAIL: fake route was not rejected");
  process.exit(1);
}

console.log(`coverage negative test PASS (${failures.length} expected violations)`);
