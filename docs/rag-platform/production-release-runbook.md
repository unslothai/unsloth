# Rag Platform production release and rollback runbook

## Preconditions

- Frontend and backend release commits are clean, pushed to `acrbaran/*`, and
  protected by pull-request checks. The backend ref is on protected `main` or
  has a protected release tag.
- `Phase 15 production release` passes quality, route/coverage, 24-scenario
  evidence, dependency/secret/branding/performance, image vulnerability, SBOM,
  provenance and self-hosted runtime E2E jobs.
- `RAG_PLATFORM_PUBLIC_URL` is an HTTPS canary URL. TLS terminates only at a
  trusted edge/ingress and HTTP redirects permanently to HTTPS.
- Database, object storage, search index, key-material volume and runtime config
  backups have completed and their restore checksums are recorded outside Git.

## Release

1. Run the workflow manually with the reviewed backend commit/tag and HTTPS
   canary URL. Never use `latest`.
2. Confirm the produced image is scanned before publication, then published to
   GHCR under the backend commit and consumed by the canary through its immutable
   `ghcr.io/acrbaran/rag-platform-backend@sha256:...` digest. Its SBOM and
   provenance must name the same backend/frontend commits, and HIGH/CRITICAL
   scan findings must be zero.
3. Validate Compose without mutation:

   ```sh
   infra/rag-platform/platform-compose.sh --cpu config
   infra/rag-platform/platform-compose.sh --check-profiles
   ```

4. Deploy to the canary, wait for the container healthcheck, then run:

   ```sh
   RAG_PLATFORM_PUBLIC_URL=https://canary.example.invalid \
     node scripts/rag-platform/release-security-gate.mjs --runtime
   node scripts/rag-platform/phase-15-runtime-smoke.mjs \
     --release --full https://canary.example.invalid
   ```

5. Verify CSP/HSTS/nosniff headers, no wildcard CORS, immediate SSE deltas,
   bounded upload behavior, four direct services and hybrid route targets.
6. Shift traffic in small canary increments while watching 401/403/429/5xx,
   stream interruption, queue depth, parse latency and container health. Feature
   flags control rollout only; they do not waive implemented coverage.

## Backup and restore

Back up the MySQL schema, MinIO bucket/prefix, selected search backend snapshot,
`rag-platform-key-material` volume and owned deployment configuration before
traffic shift. Store encrypted backups outside both repositories. Restore into
an isolated project/network first, run health and read-only data integrity
checks, and only then point a canary at the restored stack. Never restore a
single component across an incompatible version pair.

## Rollback

1. Stop traffic shift and route all traffic back to the preceding healthy
   deployment; do not delete the failed deployment yet.
2. Redeploy the preceding pinned image/config pair from its provenance record.
3. Run four-service readiness, hybrid target, auth, dataset read, Session read
   and SSE smoke before reopening traffic.
4. Restore data only if the release performed an incompatible write and the
   rollback compatibility table requires it. Prefer forward repair when the
   previous version can read the new data.
5. Preserve Phase 15 migration exports and local ledgers. The migration UI
   never automatically deletes old Project/Thread/Message data.
6. Open a follow-up pull request with the incident evidence. Do not reset or
   force-push shared history.
