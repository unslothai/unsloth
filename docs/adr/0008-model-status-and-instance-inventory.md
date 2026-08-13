# ADR 0008: Model status and instance inventory

- Status: Accepted
- Date: 2026-08-13

## Context

The generated hybrid runtime originally routed instance-model PATCH and DELETE
requests to the Go API. The pinned Go image updates only the first matching
row for a status PATCH. The tenant-wide `GET /models` response intentionally omits inactive
models, so it cannot be the inventory rendered by the connection editor.
Historical model additions can also leave multiple database rows with the same
provider, instance, and model name (for example, one row per capability).
Updating only the first matching row leaves the logical model partly active.

## Decision

The connection editor uses `GET /providers/{provider}/instances/{instance}/models`
as its configured-model inventory and uses `GET /models` only to enrich active
records with tenant model identifiers. Inactive records remain visible and can
be enabled again. Default-model selectors exclude inactive records.

When a status PATCH identifies a model by name, the Go service updates every
row in that provider/instance/name group in one database statement. A request
that explicitly identifies a model by ID remains scoped to that row. DELETE
continues to remove every row matching the requested model name.

Until an owned backend image includes that Go source fix, the hybrid map has an
explicit method/path override that sends only the instance-model PATCH to the
pinned Python API on port 9380. That implementation already batch-updates the
same logical name group. GET, POST, and DELETE remain on Go; the override must
be removed after the owned image is rebuilt and its Go regression test passes.

## Consequences

- Disabling a model no longer removes it from the connection UI.
- Re-enabling works without adding a duplicate model.
- Historical duplicate capability rows receive a consistent status.
- Deleting a logical model removes all of its matching persisted rows.
- Route inventory reports the temporary Python PATCH as the active
  frontend-action and the shadowed Go PATCH as runtime-disabled with an active
  equivalent.
