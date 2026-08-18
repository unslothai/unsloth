const FRONTEND_CLASSES = new Set(["frontend-screen", "frontend-action"]);
const FORBIDDEN_STATUSES = new Set(["planned", "in-progress", "unclassified"]);

export function validateCoverageRecords(records) {
  const failures = [];
  const seen = new Set();

  for (const record of records) {
    const recordKey = [
      record.method,
      record.path,
      record.service,
      record.is_alternate ? "alternate" : "primary",
      record.replaced_by ?? "",
    ].join("|");
    if (seen.has(recordKey)) failures.push(`duplicate coverage record: ${recordKey}`);
    seen.add(recordKey);

    if (FORBIDDEN_STATUSES.has(record.status)) {
      failures.push(`${record.method} ${record.path}: forbidden status ${record.status}`);
    }
    if (record.runtime === "enabled" && record.class === "unsupported") {
      failures.push(`${record.method} ${record.path}: reachable endpoint is unsupported`);
    }
    if (FRONTEND_CLASSES.has(record.class)) {
      if (!record.ui_path || !record.typed_service || !record.test_evidence?.length) {
        failures.push(`${record.method} ${record.path}: incomplete frontend evidence`);
      }
    } else if (!record.justification || !record.test_evidence?.length) {
      failures.push(`${record.method} ${record.path}: missing contract/security evidence`);
    }
    if (record.runtime === "disabled" && !record.runtime_disabled_reason) {
      failures.push(`${record.method} ${record.path}: runtime-disabled reason missing`);
    }
  }

  return failures;
}
