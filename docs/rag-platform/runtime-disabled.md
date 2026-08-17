# Rag Platform — runtime-disabled backend routes

<!-- GENERATED FILE. Do not edit by hand.
     Regenerate: node scripts/rag-platform/route-inventory.mjs -->

Every backend route that the deployed stack cannot serve, with the reason it
cannot. Nothing here is skipped silently: a route is listed if either nginx
does not forward it to the service that implements it, or that service is not
running in the active scheme.

- Active proxy scheme: `hybrid` (from infra/rag-platform/.env.rag-platform)
- Proxy config: `infra/rag-platform/rag-platform.hybrid.conf`
- Source image: `infiniflow/ragflow:v0.26.4`
- Decision record: `docs/adr/0005-backend-proxy-scheme.md`

## Totals

| Metric | Count |
| --- | --- |
| routes discovered | 746 |
| reachable | 509 |
| runtime-disabled | 232 |
| — no reachable equivalent (capability lost) | 50 |
| — same concrete request served elsewhere (no capability lost) | 182 |
| not proxied by nginx | 5 |

## Why these routes are closed

The owned method-aware hybrid map selects one implementation for each
method+path. When both Python and Go register the same contract, the Go
implementation is selected and the duplicate Python registration appears
below as runtime-disabled with a reachable equivalent. This is intentional
deduplication, not a lost capability. The Go executable provenance and the
four direct service smoke probes are recorded in ADR 0005 and the Faz 0
result report.

46 route(s) are separate forward-source cases: they are declared only at backend worktree `a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2`, and are absent from deployed `v0.26.4`. Nine auth handlers return `CodeNotImplemented`; the two pipeline catalog handlers and the Phase 10 dataset compilation/artifact/navigation/skill handlers are implemented but absent from the pinned runtime. Live hybrid smoke returns HTTP 404 for the pipeline list/detail and seven auth paths; GitHub and Lark callback URLs return 302 through the active parameterised callback. The auth UI uses live channels without a false captcha/OTP step, the pipeline selector shows an explicit runtime-disabled reason, and 34 Phase 10 source-only routes remain hidden from product actions until the runtime image is upgraded.

## Phase 5 functional runtime gaps (reachable route, unusable browser contract)

These two routes are reachable at the proxy and therefore are not included in the
runtime-disabled total above, but their active v0.26.4 handler contract cannot
complete the user-facing browser operation. They are explicitly classified
`runtime-disabled` in the endpoint coverage matrix rather than presented as empty UI.

| Route | Source evidence | Proxy evidence | Smoke / product result |
| --- | --- | --- | --- |
| `GET /api/v1/documents` | `internal/router/router.go:291` binds the flat route to `ListDocuments`; `internal/handler/document.go:520` reads the absent `dataset_id` path param and runs dataset ownership against it. | Generated hybrid map sends GET `/api/v1/documents` to Go `9384`. | Authless live probe returns HTTP 401 from the Go session middleware, confirming target selection. The authenticated handler is source-provably unable to supply a flat collection; the UI shows `runtime-disabled` and uses dataset-scoped listing. |
| `GET /api/v1/documents/{id}` | `internal/handler/document.go:116-143` authenticates but discards the user and returns `GetDocumentByID` without `datasetService.Accessible`; neighboring PUT/DELETE handlers do perform that ownership check. | Generated hybrid map sends GET `/api/v1/documents/{id}` to Go `9384`. | The unsafe metadata read is not exposed in the frontend; the General documents tab shows a security-specific `runtime-disabled` notice while ownership-checked PUT/DELETE remain available. |
| `POST /api/v1/datasets/{id}/documents` (Go alternate) | Active Go v0.26.4 upload inserts the historical SQL document shape including `meta_fields`; the deployed DB/Python model uses the dedicated metadata service and has no such column. | Explicit generated runtime override sends the canonical upload to Python `9380`; Go remains a disabled alternate. | Live PDF/TXT/DOCX probe first failed on Go with MySQL 1054, then passed as a three-file upload on Python after nginx reload. |
| `POST /api/v1/datasets/{id}/documents/parse` (Go alternate) | Go v0.26.4 accepts legacy `{dataset_id, documents}` and publishes to its ingestor path, while the active parsing worker consumes Python task-executor jobs. | Explicit generated runtime override sends canonical `{document_ids}` parsing to Python `9380`; Go remains a disabled alternate. | Initial Go submission remained queued; the Python route produced observable progress and terminal 100% for PDF/TXT/DOCX. |
| `GET /api/v1/datasets/ingestion/tasks` | `internal/handler/document.go:1460` calls `ShouldBindJSON` for `dataset_id` on GET and never reads the query string. Browser Fetch forbids GET request bodies. | Generated hybrid map sends the route to Go `9384`. | Authless live probe returns HTTP 401 from the Go session middleware, confirming target selection. Browser contract test uses no GET body; document polling plus Python `POST /datasets/{id}/documents/stop` is the safe product path. |

## Phase 10 functional runtime gaps (registered route, unavailable prerequisites)

The global skill routes are registered and selected by hybrid nginx, but the
deployed v0.26.4 data/search prerequisites are absent. They are classified
`runtime-disabled` in the coverage matrix and render one explicit disabled
notice; no create/update/delete/index action is offered.

| Route family | Source / proxy evidence | Authenticated smoke result | Product result |
| --- | --- | --- | --- |
| `/api/v1/skills/spaces*`, `/skills/space/by-folder` | `internal/router/router.go` registers Go skill-space handlers; hybrid sends `/api/v1/skills/*` to Go `9384`. | `GET /skills/spaces` reaches the handler but returns HTTP 200/code 103, MySQL 1146: `rag_platform.skill_spaces` does not exist. | Global space CRUD and folder lookup controls are hidden behind a retryable runtime-disabled notice. |
| `/api/v1/skills/config`, `/skills/search`, `/skills/index`, `/skills/reindex` | `internal/handler/skill_search.go` registers config/search/index lifecycle; hybrid selects Go `9384`. | Authenticated search reaches the handler but returns code 103 because Elasticsearch `172.19.0.3:9200` refuses connections; write/index lifecycle also depends on the missing skill schema/search service. | Config/search/index/reindex/delete-index controls are hidden; dataset-owned compiled skill reads remain available separately. |

## Capability lost — no reachable route serves this method and path

Compared after canonicalising parameter syntax (`<id>`, `:id`, `*path` all
normalise to `{p}`), so a Go route is only listed here when no Python route
provides the same method and path shape.

### `datasets` (39)

| Method | Path | Service | Source | Proxy result |
| --- | --- | --- | --- | --- |
| GET | `/api/v1/datasets/:dataset_id/artifacts` | go-api@9384 | `internal/router/router.go:360` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| HEAD | `/api/v1/datasets/:dataset_id/artifacts` | go-api@9384 | `internal/router/router.go:359` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/artifacts/:page_type/:slug` | go-api@9384 | `internal/router/router.go:365` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| PUT | `/api/v1/datasets/:dataset_id/artifacts/:page_type/:slug` | go-api@9384 | `internal/router/router.go:366` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/artifacts/alteration` | go-api@9384 | `internal/router/router.go:363` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/artifacts/graph` | go-api@9384 | `internal/router/router.go:364` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| DELETE | `/api/v1/datasets/:dataset_id/artifacts/structure` | go-api@9384 | `internal/router/router.go:368` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/artifacts/structure` | go-api@9384 | `internal/router/router.go:367` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/artifacts/topics` | go-api@9384 | `internal/router/router.go:362` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/compilation/status` | go-api@9384 | `internal/router/router.go:356` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| DELETE | `/api/v1/datasets/:dataset_id/navigation` | go-api@9384 | `internal/router/router.go:372` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/navigation` | go-api@9384 | `internal/router/router.go:371` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| DELETE | `/api/v1/datasets/:dataset_id/navigation/:name` | go-api@9384 | `internal/router/router.go:373` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/navigation/:name/children` | go-api@9384 | `internal/router/router.go:374` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| DELETE | `/api/v1/datasets/:dataset_id/skills` | go-api@9384 | `internal/router/router.go:379` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/skills` | go-api@9384 | `internal/router/router.go:378` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| HEAD | `/api/v1/datasets/:dataset_id/skills` | go-api@9384 | `internal/router/router.go:377` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| DELETE | `/api/v1/datasets/:dataset_id/skills/:skill_kwd` | go-api@9384 | `internal/router/router.go:381` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/:dataset_id/skills/:skill_kwd` | go-api@9384 | `internal/router/router.go:380` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/artifacts` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:580` | proxy-shadowed onto go-api:9384 under scheme "hybrid" |
| HEAD | `/api/v1/datasets/<dataset_id>/artifacts` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:584` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/<page_type>/<path:slug>` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:667` | proxy-shadowed onto go-api:9384 under scheme "hybrid" |
| PUT | `/api/v1/datasets/<dataset_id>/artifacts/<page_type>/<path:slug>` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:770` | proxy-shadowed onto go-api:9384 under scheme "hybrid" |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/alteration` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:803` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/graph` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:612` | proxy-shadowed onto go-api:9384 under scheme "hybrid" |
| DELETE | `/api/v1/datasets/<dataset_id>/artifacts/structure` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:762` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/structure` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:715` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/topics` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:638` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| DELETE | `/api/v1/datasets/<dataset_id>/navigation` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:1052` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/navigation` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:962` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| POST | `/api/v1/datasets/<dataset_id>/navigation` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:1097` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| DELETE | `/api/v1/datasets/<dataset_id>/navigation/<path:name>` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:1074` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/navigation/<path:name>/children` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:1029` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/navigation/search` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:984` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| DELETE | `/api/v1/datasets/<dataset_id>/skills` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:917` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/skills` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:713` | proxy-shadowed onto go-api:9384 under scheme "hybrid" |
| HEAD | `/api/v1/datasets/<dataset_id>/skills` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:881` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| DELETE | `/api/v1/datasets/<dataset_id>/skills/<path:skill_kwd>` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:1129` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/skills/<path:skill_kwd>` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:735` | proxy-shadowed onto go-api:9384 under scheme "hybrid" |

### `auth` (7)

| Method | Path | Service | Source | Proxy result |
| --- | --- | --- | --- | --- |
| GET | `/api/v1/auth/azure/callback` | go-api@9384 | `internal/router/router_ee.go:35` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); handler AzureAuthCallback returns CodeNotImplemented |
| GET | `/api/v1/auth/azure/login` | go-api@9384 | `internal/router/router_ee.go:36` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); handler AzureAuthLogin returns CodeNotImplemented |
| GET | `/api/v1/auth/icbc/callback` | go-api@9384 | `internal/router/router_ee.go:34` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); handler ICBCAuthCallback returns CodeNotImplemented |
| GET | `/api/v1/auth/oauth/callback` | go-api@9384 | `internal/router/router_ee.go:31` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); handler OAuthCallback returns CodeNotImplemented |
| POST | `/api/v1/auth/register/captcha` | go-api@9384 | `internal/router/router_ee.go:37` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); handler Captcha returns CodeNotImplemented |
| POST | `/api/v1/auth/register/otp` | go-api@9384 | `internal/router/router_ee.go:38` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); handler SendOTP returns CodeNotImplemented |
| POST | `/api/v1/auth/register/otp/verify` | go-api@9384 | `internal/router/router_ee.go:39` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); handler VerifyOTP returns CodeNotImplemented |

### `pipelines` (2)

| Method | Path | Service | Source | Proxy result |
| --- | --- | --- | --- | --- |
| GET | `/api/v1/pipelines` | go-api@9384 | `internal/router/router.go:170` | implemented only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); live hybrid proxy probe returns HTTP 404 |
| GET | `/api/v1/pipelines/:id` | go-api@9384 | `internal/router/router.go:171` | implemented only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); live hybrid proxy probe returns HTTP 404 |

### `tasks` (2)

| Method | Path | Service | Source | Proxy result |
| --- | --- | --- | --- | --- |
| POST | `/api/v1/tasks/:session_id/cancel` | go-api@9384 | `internal/router/agent_routes.go:110` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke verifies the deployed route boundary |
| POST | `/api/v1/tasks/<task_id>/cancel` | python-api@9380 | `api/apps/restful_apis/task_api.py:30` | proxy-shadowed onto go-api:9384 under scheme "hybrid" |

## No capability lost — same method and path is served by a reachable route

Duplicate implementations of one contract, or a static source-only route
whose concrete request is handled by a reachable parameterised route. The
serving implementation shown below keeps the surface available.

| Method | Path | Unreachable | Serving instead | Source |
| --- | --- | --- | --- | --- |
| DELETE | `/api/v1/admin/roles/<role_name>` | python-admin@9381 | go-admin (`/api/v1/admin/roles/:role_name`) | `admin/server/routes.py:327` |
| PUT | `/api/v1/admin/roles/<role_name>` | python-admin@9381 | go-admin (`/api/v1/admin/roles/:role_name`) | `admin/server/routes.py:312` |
| DELETE | `/api/v1/admin/roles/<role_name>/permission` | python-admin@9381 | go-admin (`/api/v1/admin/roles/:role_name/permission`) | `admin/server/routes.py:376` |
| GET | `/api/v1/admin/roles/<role_name>/permission` | python-admin@9381 | go-admin (`/api/v1/admin/roles/:role_name/permission`) | `admin/server/routes.py:349` |
| POST | `/api/v1/admin/roles/<role_name>/permission` | python-admin@9381 | go-admin (`/api/v1/admin/roles/:role_name/permission`) | `admin/server/routes.py:360` |
| GET | `/api/v1/admin/sandbox/providers/<provider_id>/schema` | python-admin@9381 | go-admin (`/api/v1/admin/sandbox/providers/:provider_id/schema`) | `admin/server/routes.py:575` |
| GET | `/api/v1/admin/service_types/<service_type>` | python-admin@9381 | go-admin (`/api/v1/admin/service_types/:service_type`) | `admin/server/routes.py:252` |
| DELETE | `/api/v1/admin/services/<service_id>` | python-admin@9381 | go-admin (`/api/v1/admin/services/:service_id`) | `admin/server/routes.py:274` |
| GET | `/api/v1/admin/services/<service_id>` | python-admin@9381 | go-admin (`/api/v1/admin/services/:service_id`) | `admin/server/routes.py:263` |
| PUT | `/api/v1/admin/services/<service_id>` | python-admin@9381 | go-admin (`/api/v1/admin/services/:service_id`) | `admin/server/routes.py:285` |
| GET | `/api/v1/admin/users/<user_name>/permission` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/permission`) | `admin/server/routes.py:407` |
| PUT | `/api/v1/admin/users/<user_name>/role` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/role`) | `admin/server/routes.py:392` |
| DELETE | `/api/v1/admin/users/<username>` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username`) | `admin/server/routes.py:114` |
| GET | `/api/v1/admin/users/<username>` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username`) | `admin/server/routes.py:199` |
| PUT | `/api/v1/admin/users/<username>/activate` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/activate`) | `admin/server/routes.py:150` |
| DELETE | `/api/v1/admin/users/<username>/admin` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/admin`) | `admin/server/routes.py:183` |
| PUT | `/api/v1/admin/users/<username>/admin` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/admin`) | `admin/server/routes.py:167` |
| GET | `/api/v1/admin/users/<username>/agents` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/agents`) | `admin/server/routes.py:227` |
| GET | `/api/v1/admin/users/<username>/datasets` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/datasets`) | `admin/server/routes.py:213` |
| GET | `/api/v1/admin/users/<username>/keys` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/keys`) | `admin/server/routes.py:521` |
| POST | `/api/v1/admin/users/<username>/keys` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/keys`) | `admin/server/routes.py:489` |
| DELETE | `/api/v1/admin/users/<username>/keys/<key>` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/keys/:key`) | `admin/server/routes.py:534` |
| PUT | `/api/v1/admin/users/<username>/password` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/password`) | `admin/server/routes.py:131` |
| DELETE | `/api/v1/agents/<agent_id>` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id`) | `api/apps/restful_apis/agent_api.py:1021` |
| GET | `/api/v1/agents/<agent_id>` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id`) | `api/apps/restful_apis/agent_api.py:930` |
| PUT | `/api/v1/agents/<agent_id>` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id`) | `api/apps/restful_apis/agent_api.py:1030` |
| POST | `/api/v1/agents/<agent_id>/components/<component_id>/debug` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/components/:component_id/debug`) | `api/apps/restful_apis/agent_api.py:892` |
| GET | `/api/v1/agents/<agent_id>/components/<component_id>/input-form` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/components/:component_id/input-form`) | `api/apps/restful_apis/agent_api.py:875` |
| GET | `/api/v1/agents/<agent_id>/logs/<message_id>` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/logs/:message_id`) | `api/apps/restful_apis/agent_api.py:1002` |
| POST | `/api/v1/agents/<agent_id>/reset` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/reset`) | `api/apps/restful_apis/agent_api.py:1084` |
| DELETE | `/api/v1/agents/<agent_id>/sessions` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/sessions`) | `api/apps/restful_apis/agent_api.py:532` |
| GET | `/api/v1/agents/<agent_id>/sessions` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/sessions`) | `api/apps/restful_apis/agent_api.py:430` |
| POST | `/api/v1/agents/<agent_id>/sessions` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/sessions`) | `api/apps/restful_apis/agent_api.py:470` |
| DELETE | `/api/v1/agents/<agent_id>/sessions/<session_id>` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/sessions/:session_id`) | `api/apps/restful_apis/agent_api.py:521` |
| GET | `/api/v1/agents/<agent_id>/sessions/<session_id>` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/sessions/:session_id`) | `api/apps/restful_apis/agent_api.py:510` |
| POST | `/api/v1/agents/<agent_id>/upload` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/upload`) | `api/apps/restful_apis/agent_api.py:847` |
| GET | `/api/v1/agents/<agent_id>/versions` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/versions`) | `api/apps/restful_apis/agent_api.py:973` |
| GET | `/api/v1/agents/<agent_id>/versions/<version_id>` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/versions/:version_id`) | `api/apps/restful_apis/agent_api.py:988` |
| DELETE | `/api/v1/agents/<agent_id>/webhook` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook`) | `api/apps/restful_apis/agent_api.py:1679` |
| GET | `/api/v1/agents/<agent_id>/webhook` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook`) | `api/apps/restful_apis/agent_api.py:1679` |
| HEAD | `/api/v1/agents/<agent_id>/webhook` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook`) | `api/apps/restful_apis/agent_api.py:1679` |
| PATCH | `/api/v1/agents/<agent_id>/webhook` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook`) | `api/apps/restful_apis/agent_api.py:1679` |
| POST | `/api/v1/agents/<agent_id>/webhook` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook`) | `api/apps/restful_apis/agent_api.py:1679` |
| PUT | `/api/v1/agents/<agent_id>/webhook` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook`) | `api/apps/restful_apis/agent_api.py:1679` |
| GET | `/api/v1/agents/<agent_id>/webhook/logs` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook/logs`) | `api/apps/restful_apis/agent_api.py:2365` |
| DELETE | `/api/v1/agents/<agent_id>/webhook/test` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook/test`) | `api/apps/restful_apis/agent_api.py:1684` |
| GET | `/api/v1/agents/<agent_id>/webhook/test` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook/test`) | `api/apps/restful_apis/agent_api.py:1684` |
| HEAD | `/api/v1/agents/<agent_id>/webhook/test` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook/test`) | `api/apps/restful_apis/agent_api.py:1684` |
| PATCH | `/api/v1/agents/<agent_id>/webhook/test` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook/test`) | `api/apps/restful_apis/agent_api.py:1684` |
| POST | `/api/v1/agents/<agent_id>/webhook/test` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook/test`) | `api/apps/restful_apis/agent_api.py:1684` |
| PUT | `/api/v1/agents/<agent_id>/webhook/test` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/webhook/test`) | `api/apps/restful_apis/agent_api.py:1684` |
| PUT | `/api/v1/agents/<canvas_id>/tags` | python-api@9380 | go-api (`/api/v1/agents/:canvas_id/tags`) | `api/apps/restful_apis/agent_api.py:739` |
| GET | `/api/v1/agents/attachments/<attachment_id>/download` | python-api@9380 | go-api (`/api/v1/agents/attachments/:attachment_id/download`) | `api/apps/restful_apis/agent_api.py:2516` |
| GET | `/api/v1/agents/attachments/<attachment_id>/preview` | python-api@9380 | go-api (`/api/v1/agents/attachments/:attachment_id/preview`) | `api/apps/restful_apis/agent_api.py:2505` |
| GET | `/api/v1/auth/login/<channel>` | python-api@9380 | go-api (`/api/v1/auth/login/:channel`) | `api/apps/restful_apis/user_api.py:165` |
| GET | `/api/v1/auth/oauth/<channel>/callback` | python-api@9380 | go-api (`/api/v1/auth/oauth/:channel/callback`) | `api/apps/restful_apis/user_api.py:179` |
| GET | `/api/v1/auth/oauth/github/callback` | go-api@9384 | go-api (`/api/v1/auth/oauth/:channel/callback`) | `internal/router/router_ee.go:32` |
| GET | `/api/v1/auth/oauth/lark/callback` | go-api@9384 | go-api (`/api/v1/auth/oauth/:channel/callback`) | `internal/router/router_ee.go:33` |
| DELETE | `/api/v1/chat-channels/<channel_id>` | python-api@9380 | go-api (`/api/v1/chat-channels/:channel_id`) | `api/apps/restful_apis/chat_channel_api.py:102` |
| GET | `/api/v1/chat-channels/<channel_id>` | python-api@9380 | go-api (`/api/v1/chat-channels/:channel_id`) | `api/apps/restful_apis/chat_channel_api.py:56` |
| PATCH | `/api/v1/chat-channels/<channel_id>` | python-api@9380 | go-api (`/api/v1/chat-channels/:channel_id`) | `api/apps/restful_apis/chat_channel_api.py:69` |
| DELETE | `/api/v1/chats/<chat_id>` | python-api@9380 | go-api (`/api/v1/chats/:chat_id`) | `api/apps/restful_apis/chat_api.py:692` |
| GET | `/api/v1/chats/<chat_id>` | python-api@9380 | go-api (`/api/v1/chats/:chat_id`) | `api/apps/restful_apis/chat_api.py:494` |
| PATCH | `/api/v1/chats/<chat_id>` | python-api@9380 | go-api (`/api/v1/chats/:chat_id`) | `api/apps/restful_apis/chat_api.py:604` |
| PUT | `/api/v1/chats/<chat_id>` | python-api@9380 | go-api (`/api/v1/chats/:chat_id`) | `api/apps/restful_apis/chat_api.py:522` |
| DELETE | `/api/v1/chats/<chat_id>/sessions` | python-api@9380 | go-api (`/api/v1/chats/:chat_id/sessions`) | `api/apps/restful_apis/chat_api.py:863` |
| GET | `/api/v1/chats/<chat_id>/sessions` | python-api@9380 | go-api (`/api/v1/chats/:chat_id/sessions`) | `api/apps/restful_apis/chat_api.py:785` |
| POST | `/api/v1/chats/<chat_id>/sessions` | python-api@9380 | go-api (`/api/v1/chats/:chat_id/sessions`) | `api/apps/restful_apis/chat_api.py:753` |
| GET | `/api/v1/chats/<chat_id>/sessions/<session_id>` | python-api@9380 | go-api (`/api/v1/chats/:chat_id/sessions/:session_id`) | `api/apps/restful_apis/chat_api.py:810` |
| PATCH | `/api/v1/chats/<chat_id>/sessions/<session_id>` | python-api@9380 | go-api (`/api/v1/chats/:chat_id/sessions/:session_id`) | `api/apps/restful_apis/chat_api.py:834` |
| DELETE | `/api/v1/chats/<chat_id>/sessions/<session_id>/messages/<msg_id>` | python-api@9380 | go-api (`/api/v1/chats/:chat_id/sessions/:session_id/messages/:msg_id`) | `api/apps/restful_apis/chat_api.py:914` |
| PUT | `/api/v1/chats/<chat_id>/sessions/<session_id>/messages/<msg_id>/feedback` | python-api@9380 | go-api (`/api/v1/chats/:chat_id/sessions/:session_id/messages/:msg_id/feedback`) | `api/apps/restful_apis/chat_api.py:939` |
| POST | `/api/v1/connectors/:connector_id/rebuild` | go-api@9384 | python-api (`/api/v1/connectors/<connector_id>/rebuild`) | `internal/router/router.go:591` |
| POST | `/api/v1/connectors/:connector_id/test` | go-api@9384 | python-api (`/api/v1/connectors/<connector_id>/test`) | `internal/router/router.go:592` |
| DELETE | `/api/v1/connectors/<connector_id>` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id`) | `api/apps/restful_apis/connector_api.py:169` |
| GET | `/api/v1/connectors/<connector_id>` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id`) | `api/apps/restful_apis/connector_api.py:123` |
| PATCH | `/api/v1/connectors/<connector_id>` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id`) | `api/apps/restful_apis/connector_api.py:49` |
| GET | `/api/v1/connectors/<connector_id>/logs` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id/logs`) | `api/apps/restful_apis/connector_api.py:136` |
| DELETE | `/api/v1/datasets/:dataset_id/artifacts` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/artifacts`) | `internal/router/router.go:361` |
| GET | `/api/v1/datasets/:dataset_id/changes` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/changes`) | `internal/router/router.go:458` |
| DELETE | `/api/v1/datasets/:dataset_id/chunks` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/chunks`) | `internal/router/router.go:381` |
| GET | `/api/v1/datasets/:dataset_id/commits` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits`) | `internal/router/router.go:452` |
| POST | `/api/v1/datasets/:dataset_id/commits` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits`) | `internal/router/router.go:451` |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/<commit_id>`) | `internal/router/router.go:454` |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id/files` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/<commit_id>/files`) | `internal/router/router.go:455` |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id/files/:file_id/content` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/<commit_id>/files/<file_id>/content`) | `internal/router/router.go:457` |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id/tree` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/<commit_id>/tree`) | `internal/router/router.go:456` |
| GET | `/api/v1/datasets/:dataset_id/commits/diff` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/diff`) | `internal/router/router.go:453` |
| DELETE | `/api/v1/datasets/:dataset_id/documents` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/documents`) | `internal/router/router.go:366` |
| POST | `/api/v1/datasets/:dataset_id/documents` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/documents`) | `internal/router/router.go:363` |
| PATCH | `/api/v1/datasets/:dataset_id/documents/metadatas` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/documents/metadatas`) | `internal/router/router.go:385` |
| POST | `/api/v1/datasets/:dataset_id/documents/parse` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/documents/parse`) | `internal/router/router.go:375` |
| DELETE | `/api/v1/datasets/:dataset_id/index` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/index`) | `internal/router/router.go:342` |
| GET | `/api/v1/datasets/:dataset_id/ingestions/summary` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/ingestions/summary`) | `internal/router/router.go:353` |
| DELETE | `/api/v1/datasets/:dataset_id/tags` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/tags`) | `internal/router/router.go:336` |
| GET | `/api/v1/datasets/<dataset_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id`) | `api/apps/restful_apis/dataset_api.py:384` |
| PUT | `/api/v1/datasets/<dataset_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id`) | `api/apps/restful_apis/dataset_api.py:220` |
| DELETE | `/api/v1/datasets/<dataset_id>/<index_type>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/:index_type`) | `api/apps/restful_apis/dataset_api.py:854` |
| POST | `/api/v1/datasets/<dataset_id>/chunks` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/chunks`) | `api/apps/restful_apis/chunk_api.py:183` |
| GET | `/api/v1/datasets/<dataset_id>/documents` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents`) | `api/apps/restful_apis/document_api.py:703` |
| GET | `/api/v1/datasets/<dataset_id>/documents/<document_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id`) | `api/apps/restful_apis/document_api.py:2089` |
| PATCH | `/api/v1/datasets/<dataset_id>/documents/<document_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id`) | `api/apps/restful_apis/document_api.py:170` |
| DELETE | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id/chunks`) | `api/apps/restful_apis/chunk_api.py:931` |
| GET | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id/chunks`) | `api/apps/restful_apis/chunk_api.py:441` |
| PATCH | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id/chunks`) | `api/apps/restful_apis/chunk_api.py:1072` |
| POST | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id/chunks`) | `api/apps/restful_apis/chunk_api.py:842` |
| GET | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks/<chunk_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id/chunks/:chunk_id`) | `api/apps/restful_apis/chunk_api.py:529` |
| PATCH | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks/<chunk_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id/chunks/:chunk_id`) | `api/apps/restful_apis/chunk_api.py:984` |
| PUT | `/api/v1/datasets/<dataset_id>/documents/<document_id>/metadata/config` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id/metadata/config`) | `api/apps/restful_apis/document_api.py:1193` |
| POST | `/api/v1/datasets/<dataset_id>/documents/batch-update-status` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/batch-update-status`) | `api/apps/restful_apis/document_api.py:1924` |
| POST | `/api/v1/datasets/<dataset_id>/embedding` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/embedding`) | `api/apps/restful_apis/dataset_api.py:881` |
| POST | `/api/v1/datasets/<dataset_id>/embedding/check` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/embedding/check`) | `api/apps/restful_apis/dataset_api.py:896` |
| GET | `/api/v1/datasets/<dataset_id>/graph` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/graph`) | `api/apps/restful_apis/dataset_api.py:537` |
| GET | `/api/v1/datasets/<dataset_id>/index` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/index`) | `api/apps/restful_apis/dataset_api.py:835` |
| POST | `/api/v1/datasets/<dataset_id>/index` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/index`) | `api/apps/restful_apis/dataset_api.py:816` |
| GET | `/api/v1/datasets/<dataset_id>/ingestions` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/ingestions`) | `api/apps/restful_apis/dataset_api.py:916` |
| GET | `/api/v1/datasets/<dataset_id>/ingestions/<log_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/ingestions/:log_id`) | `api/apps/restful_apis/dataset_api.py:942` |
| GET | `/api/v1/datasets/<dataset_id>/metadata/config` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/metadata/config`) | `api/apps/restful_apis/dataset_api.py:959` |
| PUT | `/api/v1/datasets/<dataset_id>/metadata/config` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/metadata/config`) | `api/apps/restful_apis/dataset_api.py:1000` |
| GET | `/api/v1/datasets/<dataset_id>/metadata/summary` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/metadata/summary`) | `api/apps/restful_apis/document_api.py:307` |
| POST | `/api/v1/datasets/<dataset_id>/metadata/update` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/metadata/update`) | `api/apps/restful_apis/document_api.py:345` |
| POST | `/api/v1/datasets/<dataset_id>/search` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/search`) | `api/apps/restful_apis/dataset_api.py:507` |
| GET | `/api/v1/datasets/<dataset_id>/tags` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/tags`) | `api/apps/restful_apis/dataset_api.py:418` |
| PUT | `/api/v1/datasets/<dataset_id>/tags` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/tags`) | `api/apps/restful_apis/dataset_api.py:458` |
| GET | `/api/v1/documents/:id/preview` | go-api@9384 | python-api (`/api/v1/documents/<doc_id>/preview`) | `internal/router/router.go:214` |
| GET | `/api/v1/documents/<document_id>` | python-api@9380 | go-api (`/api/v1/documents/:id`) | `api/apps/restful_apis/document_api.py:2151` |
| GET | `/api/v1/documents/artifact/<filename>` | python-api@9380 | go-api (`/api/v1/documents/artifact/:filename`) | `api/apps/restful_apis/document_api.py:1871` |
| GET | `/api/v1/documents/images/<image_id>` | python-api@9380 | go-api (`/api/v1/documents/images/:image_id`) | `api/apps/restful_apis/document_api.py:1782` |
| GET | `/api/v1/files/:id/parent` | go-api@9384 | python-api (`/api/v1/files/<file_id>/parent`) | `internal/router/router.go:408` |
| GET | `/api/v1/files/<file_id>` | python-api@9380 | go-api (`/api/v1/files/:id`) | `api/apps/restful_apis/file_api.py:262` |
| GET | `/api/v1/files/<file_id>/ancestors` | python-api@9380 | go-api (`/api/v1/files/:id/ancestors`) | `api/apps/restful_apis/file_api.py:348` |
| GET | `/api/v1/files/<file_id>/versions` | python-api@9380 | go-api (`/api/v1/files/:id/versions`) | `api/apps/restful_apis/file_commit_api.py:370` |
| GET | `/api/v1/folders/:folder_id/changes` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/changes`) | `internal/router/router.go:431` |
| GET | `/api/v1/folders/:folder_id/commits` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits`) | `internal/router/router.go:425` |
| POST | `/api/v1/folders/:folder_id/commits` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits`) | `internal/router/router.go:424` |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/<commit_id>`) | `internal/router/router.go:427` |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id/files` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/<commit_id>/files`) | `internal/router/router.go:428` |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id/files/:file_id/content` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/<commit_id>/files/<file_id>/content`) | `internal/router/router.go:430` |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id/tree` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/<commit_id>/tree`) | `internal/router/router.go:429` |
| GET | `/api/v1/folders/:folder_id/commits/diff` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/diff`) | `internal/router/router.go:426` |
| DELETE | `/api/v1/mcp/servers/<mcp_id>` | python-api@9380 | go-api (`/api/v1/mcp/servers/:mcp_id`) | `api/apps/restful_apis/mcp_api.py:231` |
| GET | `/api/v1/mcp/servers/<mcp_id>` | python-api@9380 | go-api (`/api/v1/mcp/servers/:mcp_id`) | `api/apps/restful_apis/mcp_api.py:95` |
| PUT | `/api/v1/mcp/servers/<mcp_id>` | python-api@9380 | go-api (`/api/v1/mcp/servers/:mcp_id`) | `api/apps/restful_apis/mcp_api.py:172` |
| POST | `/api/v1/mcp/servers/<mcp_id>/test` | python-api@9380 | go-api (`/api/v1/mcp/servers/:mcp_id/test`) | `api/apps/restful_apis/mcp_api.py:321` |
| DELETE | `/api/v1/memories/<memory_id>` | python-api@9380 | go-api (`/api/v1/memories/:memory_id`) | `api/apps/restful_apis/memory_api.py:119` |
| GET | `/api/v1/memories/<memory_id>` | python-api@9380 | go-api (`/api/v1/memories/:memory_id`) | `api/apps/restful_apis/memory_api.py:162` |
| PUT | `/api/v1/memories/<memory_id>` | python-api@9380 | go-api (`/api/v1/memories/:memory_id`) | `api/apps/restful_apis/memory_api.py:78` |
| GET | `/api/v1/memories/<memory_id>/config` | python-api@9380 | go-api (`/api/v1/memories/:memory_id/config`) | `api/apps/restful_apis/memory_api.py:148` |
| DELETE | `/api/v1/messages/<memory_id>:<message_id>` | python-api@9380 | go-api (`/api/v1/messages/:memory_message`) | `api/apps/restful_apis/memory_api.py:216` |
| PUT | `/api/v1/messages/<memory_id>:<message_id>` | python-api@9380 | go-api (`/api/v1/messages/:memory_message`) | `api/apps/restful_apis/memory_api.py:230` |
| GET | `/api/v1/messages/<memory_id>:<message_id>/content` | python-api@9380 | go-api (`/api/v1/messages/:memory_message/content`) | `api/apps/restful_apis/memory_api.py:300` |
| POST | `/api/v1/openai/<chat_id>/chat/completions` | python-api@9380 | go-api (`/api/v1/openai/:chat_id/chat/completions`) | `api/apps/restful_apis/openai_api.py:237` |
| PATCH | `/api/v1/providers/:provider_name/instances/:instance_name/models/*model_name` | go-api@9384 | python-api (`/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models/<path:model_name>`) | `internal/router/router.go:525` |
| DELETE | `/api/v1/providers/<provider_id_or_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_name`) | `api/apps/restful_apis/provider_api.py:161` |
| GET | `/api/v1/providers/<provider_id_or_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_name`) | `api/apps/restful_apis/provider_api.py:123` |
| POST | `/api/v1/providers/<provider_id_or_name>/connection` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/connection`) | `api/apps/restful_apis/provider_api.py:359` |
| DELETE | `/api/v1/providers/<provider_id_or_name>/instances` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances`) | `api/apps/restful_apis/provider_api.py:515` |
| GET | `/api/v1/providers/<provider_id_or_name>/instances` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances`) | `api/apps/restful_apis/provider_api.py:427` |
| POST | `/api/v1/providers/<provider_id_or_name>/instances` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances`) | `api/apps/restful_apis/provider_api.py:288` |
| GET | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances/:instance_name`) | `api/apps/restful_apis/provider_api.py:471` |
| GET | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances/:instance_name/models`) | `api/apps/restful_apis/provider_api.py:576` |
| POST | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances/:instance_name/models`) | `api/apps/restful_apis/provider_api.py:691` |
| GET | `/api/v1/providers/<provider_id_or_name>/models` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/models`) | `api/apps/restful_apis/provider_api.py:200` |
| GET | `/api/v1/providers/<provider_id_or_name>/models/<path:model_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/models/:model_name`) | `api/apps/restful_apis/provider_api.py:245` |
| DELETE | `/api/v1/searches/<search_id>` | python-api@9380 | go-api (`/api/v1/searches/:search_id`) | `api/apps/restful_apis/search_api.py:179` |
| GET | `/api/v1/searches/<search_id>` | python-api@9380 | go-api (`/api/v1/searches/:search_id`) | `api/apps/restful_apis/search_api.py:101` |
| PUT | `/api/v1/searches/<search_id>` | python-api@9380 | go-api (`/api/v1/searches/:search_id`) | `api/apps/restful_apis/search_api.py:120` |
| POST | `/api/v1/searches/<search_id>/completion` | python-api@9380 | go-api (`/api/v1/searches/:search_id/completion`) | `api/apps/restful_apis/search_api.py:193` |
| POST | `/api/v1/searches/<search_id>/completions` | python-api@9380 | go-api (`/api/v1/searches/:search_id/completions`) | `api/apps/restful_apis/search_api.py:194` |
| DELETE | `/api/v1/system/tokens/<token>` | python-api@9380 | go-api (`/api/v1/system/tokens/:key`) | `api/apps/restful_apis/system_api.py:333` |
| PATCH | `/api/v1/tenants/<tenant_id>` | python-api@9380 | go-api (`/api/v1/tenants/:tenant_id`) | `api/apps/restful_apis/tenant_api.py:167` |
| DELETE | `/api/v1/tenants/<tenant_id>/users` | python-api@9380 | go-api (`/api/v1/tenants/:tenant_id/users`) | `api/apps/restful_apis/tenant_api.py:135` |
| GET | `/api/v1/tenants/<tenant_id>/users` | python-api@9380 | go-api (`/api/v1/tenants/:tenant_id/users`) | `api/apps/restful_apis/tenant_api.py:41` |
| POST | `/api/v1/tenants/<tenant_id>/users` | python-api@9380 | go-api (`/api/v1/tenants/:tenant_id/users`) | `api/apps/restful_apis/tenant_api.py:60` |
| GET | `/api/v1/workspace/:folder_id/changes` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/changes`) | `internal/router/router.go:444` |
| GET | `/api/v1/workspace/:folder_id/commits` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits`) | `internal/router/router.go:438` |
| POST | `/api/v1/workspace/:folder_id/commits` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits`) | `internal/router/router.go:437` |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/<commit_id>`) | `internal/router/router.go:440` |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id/files` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/<commit_id>/files`) | `internal/router/router.go:441` |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id/files/:file_id/content` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/<commit_id>/files/<file_id>/content`) | `internal/router/router.go:443` |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id/tree` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/<commit_id>/tree`) | `internal/router/router.go:442` |
| GET | `/api/v1/workspace/:folder_id/commits/diff` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/diff`) | `internal/router/router.go:439` |

