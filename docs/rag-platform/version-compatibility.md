# Rag Platform version compatibility

| Frontend release | Backend source/image | API | Proxy | Status |
| --- | --- | --- | --- | --- |
| Phase 15 local smoke | Python/MCP base `infiniflow/ragflow:v0.26.4` + owned Go at `a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2` + selected Python overlay / `rag-platform-backend:0.26.4-phase15-local-smoke` | v1 | hybrid (9380, 9381, 9383, 9384) | Verified locally: healthy, four direct services, same-origin security headers and full Phase 7–15 runtime smoke PASS; non-release because backend source is dirty |
| Phase 15 production candidate | Python/MCP base `v0.26.4` + clean protected `acrbaran/rag-backend` release commit / `rag-platform-backend:0.26.4` | v1 | hybrid (9380, 9381, 9383, 9384) | Release blocked until the backend changes are reviewed/committed and an HTTPS canary passes image scan, SBOM, provenance and runtime jobs |

Every release updates this table, route inventory baseline/diff, SBOM and
provenance together. The local-smoke row is evidence for implementation and
runtime parity only; it cannot satisfy protected-source or HTTPS release gates.
