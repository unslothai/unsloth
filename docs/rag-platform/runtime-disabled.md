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
- Owned Go source: `worktree` (`a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2`; dirty: `true`)
- Decision record: `docs/adr/0005-backend-proxy-scheme.md`

## Totals

| Metric | Count |
| --- | --- |
| routes discovered | 692 |
| reachable | 475 |
| runtime-disabled | 212 |
| — no reachable equivalent (capability lost) | 11 |
| — same concrete request served elsewhere (no capability lost) | 201 |
| not proxied by nginx | 5 |

## Why these routes are closed

The owned method-aware hybrid map selects one implementation for each
method+path. When both Python and Go register the same contract, the Go
implementation is selected and the duplicate Python registration appears
below as runtime-disabled with a reachable equivalent. This is intentional
deduplication, not a lost capability. The Go executable provenance and the
four direct service smoke probes are recorded in ADR 0005 and the Faz 0
result report.

22 route(s) are current Python declarations at backend worktree `a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2` that are absent from the pinned `v0.26.4` Python base. Most are disabled alternates whose concrete method+path is served by the owned Go executable; those preserve capability and are listed separately below. Only declarations without a reachable equivalent contribute to the capability-lost table. Runtime smoke confirms the two Python-only navigation gaps return HTTP 404, while current Go pipeline, compilation, artifact, navigation and skill handlers are registered and reachable through the hybrid proxy.

## Owned Go runtime and selected Python overlay

The owned image builds every Go declaration from `worktree` at `a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2`. It also overlays 2 current Python AIMLAPI authorization declarations onto the pinned Python base. Go-internal developer import routes remain classified internal even when reachable; they are not browser actions.

`DELETE /api/v1/admin/ingestors` validates a fresh exact heartbeat, publishes a target-scoped NATS shutdown command and returns HTTP 202. Runtime smoke observed the addressed ingestor process exit and restart; the UI exposes it only behind audit reason and destructive confirmation.

## Phase 5 functional runtime gaps (reachable route, unusable browser contract)

These two routes are reachable at the proxy and therefore are not included in the
runtime-disabled total above, but their active owned-Go handler contract cannot
complete the user-facing browser operation. They are explicitly classified
`runtime-disabled` in the endpoint coverage matrix rather than presented as empty UI.

| Route | Source evidence | Proxy evidence | Smoke / product result |
| --- | --- | --- | --- |
| `GET /api/v1/documents` | `internal/router/router.go:291` binds the flat route to `ListDocuments`; `internal/handler/document.go:520` reads the absent `dataset_id` path param and runs dataset ownership against it. | Generated hybrid map sends GET `/api/v1/documents` to Go `9384`. | Authless live probe returns HTTP 401 from the Go session middleware, confirming target selection. The authenticated handler is source-provably unable to supply a flat collection; the UI shows `runtime-disabled` and uses dataset-scoped listing. |
| `GET /api/v1/documents/{id}` | `internal/handler/document.go:116-143` authenticates but discards the user and returns `GetDocumentByID` without `datasetService.Accessible`; neighboring PUT/DELETE handlers do perform that ownership check. | Generated hybrid map sends GET `/api/v1/documents/{id}` to Go `9384`. | The unsafe metadata read is not exposed in the frontend; the General documents tab shows a security-specific `runtime-disabled` notice while ownership-checked PUT/DELETE remain available. |
| `POST /api/v1/datasets/{id}/documents` (Go alternate) | The owned Go upload still inserts the historical SQL document shape including `meta_fields`; the deployed DB/Python model uses the dedicated metadata service and has no such column. | Explicit generated runtime override sends the canonical upload to Python `9380`; Go remains a disabled alternate. | Live PDF/TXT/DOCX probe first failed on Go with MySQL 1054, then passed as a three-file upload on Python after nginx reload. |
| `POST /api/v1/datasets/{id}/documents/parse` (Go alternate) | The owned Go route accepts legacy `{dataset_id, documents}` and publishes to its ingestor path, while the active parsing worker consumes Python task-executor jobs. | Explicit generated runtime override sends canonical `{document_ids}` parsing to Python `9380`; Go remains a disabled alternate. | Initial Go submission remained queued; the Python route produced observable progress and terminal 100% for PDF/TXT/DOCX. |
| `GET /api/v1/datasets/ingestion/tasks` | `internal/handler/document.go:1460` calls `ShouldBindJSON` for `dataset_id` on GET and never reads the query string. Browser Fetch forbids GET request bodies. | Generated hybrid map sends the route to Go `9384`. | Authless live probe returns HTTP 401 from the Go session middleware, confirming target selection. Browser contract test uses no GET body; document polling plus Python `POST /datasets/{id}/documents/stop` is the safe product path. |

## Phase 10 functional runtime gaps (registered route, unavailable prerequisites)

The global skill routes are registered and selected by hybrid nginx, but the
deployed database/search prerequisites are absent. They are classified
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

### `auth` (7)

| Method | Path | Service | Source | Proxy result |
| --- | --- | --- | --- | --- |
| GET | `/api/v1/auth/azure/callback` | go-api@9384 | `internal/router/router_ee.go:35` | backend handler returns CodeNotImplemented; route registration is present but no supported user operation exists |
| GET | `/api/v1/auth/azure/login` | go-api@9384 | `internal/router/router_ee.go:36` | backend handler returns CodeNotImplemented; route registration is present but no supported user operation exists |
| GET | `/api/v1/auth/icbc/callback` | go-api@9384 | `internal/router/router_ee.go:34` | backend handler returns CodeNotImplemented; route registration is present but no supported user operation exists |
| GET | `/api/v1/auth/oauth/callback` | go-api@9384 | `internal/router/router_ee.go:31` | backend handler returns CodeNotImplemented; route registration is present but no supported user operation exists |
| POST | `/api/v1/auth/register/captcha` | go-api@9384 | `internal/router/router_ee.go:37` | backend handler returns CodeNotImplemented; route registration is present but no supported user operation exists |
| POST | `/api/v1/auth/register/otp` | go-api@9384 | `internal/router/router_ee.go:38` | backend handler returns CodeNotImplemented; route registration is present but no supported user operation exists |
| POST | `/api/v1/auth/register/otp/verify` | go-api@9384 | `internal/router/router_ee.go:39` | backend handler returns CodeNotImplemented; route registration is present but no supported user operation exists |

### `datasets` (2)

| Method | Path | Service | Source | Proxy result |
| --- | --- | --- | --- | --- |
| POST | `/api/v1/datasets/<dataset_id>/navigation` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:1097` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |
| GET | `/api/v1/datasets/<dataset_id>/navigation/search` | python-api@9380 | `api/apps/restful_apis/dataset_api.py:984` | declared only in backend worktree a0e091e75051; absent from deployed v0.26.4 (cb93883f3f8c); authenticated hybrid/direct smoke returns HTTP 404 |

### `users` (2)

| Method | Path | Service | Source | Proxy result |
| --- | --- | --- | --- | --- |
| GET | `/api/v1/users/me/admin` | go-api@9384 | `internal/router/router.go:270` | backend handler returns CodeNotImplemented; route registration is present but no supported user operation exists |
| GET | `/api/v1/users/me/meta` | go-api@9384 | `internal/router/router.go:271` | backend handler returns CodeNotImplemented; route registration is present but no supported user operation exists |

## No capability lost — same method and path is served by a reachable route

Duplicate implementations of one contract, or a static source-only route
whose concrete request is handled by a reachable parameterised route. The
serving implementation shown below keeps the surface available.

| Method | Path | Unreachable | Serving instead | Source |
| --- | --- | --- | --- | --- |
| POST | `/api/v1/admin/logout` | python-admin@9381 | go-admin (`/api/v1/admin/logout`) | `admin/server/routes.py:55` |
| GET | `/api/v1/admin/sandbox/providers/<provider_id>/schema` | python-admin@9381 | go-admin (`/api/v1/admin/sandbox/providers/:provider_id/schema`) | `admin/server/routes.py:575` |
| GET | `/api/v1/admin/service_types/<service_type>` | python-admin@9381 | go-admin (`/api/v1/admin/service_types/:service_type`) | `admin/server/routes.py:252` |
| DELETE | `/api/v1/admin/services/<service_id>` | python-admin@9381 | go-admin (`/api/v1/admin/services/:service_name`) | `admin/server/routes.py:274` |
| GET | `/api/v1/admin/services/<service_id>` | python-admin@9381 | go-admin (`/api/v1/admin/services/:service_name`) | `admin/server/routes.py:263` |
| PUT | `/api/v1/admin/services/<service_id>` | python-admin@9381 | go-admin (`/api/v1/admin/services/:service_name`) | `admin/server/routes.py:285` |
| DELETE | `/api/v1/admin/users/<username>` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username`) | `admin/server/routes.py:114` |
| GET | `/api/v1/admin/users/<username>` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username`) | `admin/server/routes.py:199` |
| PUT | `/api/v1/admin/users/<username>/activate` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/activate`) | `admin/server/routes.py:150` |
| DELETE | `/api/v1/admin/users/<username>/admin` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/admin`) | `admin/server/routes.py:183` |
| PUT | `/api/v1/admin/users/<username>/admin` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/admin`) | `admin/server/routes.py:167` |
| PUT | `/api/v1/admin/users/<username>/password` | python-admin@9381 | go-admin (`/api/v1/admin/users/:username/password`) | `admin/server/routes.py:131` |
| POST | `/api/v1/agentbots/<agent_id>/completions` | python-api@9380 | go-api (`/api/v1/agentbots/:agent_id/completions`) | `api/apps/restful_apis/bot_api.py:162` |
| GET | `/api/v1/agentbots/<agent_id>/inputs` | python-api@9380 | go-api (`/api/v1/agentbots/:agent_id/inputs`) | `api/apps/restful_apis/bot_api.py:257` |
| GET | `/api/v1/agentbots/<shared_id>/logs/<message_id>` | python-api@9380 | go-api (`/api/v1/agentbots/:agent_id/logs/:message_id`) | `api/apps/restful_apis/bot_api.py:277` |
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
| GET | `/api/v1/chat-channels/<channel_id>/runtime` | python-api@9380 | go-api (`/api/v1/chat-channels/:channel_id/runtime`) | `api/apps/restful_apis/chat_channel_api.py:113` |
| POST | `/api/v1/chatbots/<dialog_id>/completions` | python-api@9380 | go-api (`/api/v1/chatbots/:dialog_id/completions`) | `api/apps/restful_apis/bot_api.py:63` |
| GET | `/api/v1/chatbots/<dialog_id>/info` | python-api@9380 | go-api (`/api/v1/chatbots/:dialog_id/info`) | `api/apps/restful_apis/bot_api.py:133` |
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
| GET | `/api/v1/compilation-template-groups` | python-api@9380 | go-api (`/api/v1/compilation-template-groups`) | `api/apps/restful_apis/compilation_template_group_api.py:69` |
| POST | `/api/v1/compilation-template-groups` | python-api@9380 | go-api (`/api/v1/compilation-template-groups`) | `api/apps/restful_apis/compilation_template_group_api.py:101` |
| DELETE | `/api/v1/compilation-template-groups/<group_id>` | python-api@9380 | go-api (`/api/v1/compilation-template-groups/:group_id`) | `api/apps/restful_apis/compilation_template_group_api.py:163` |
| GET | `/api/v1/compilation-template-groups/<group_id>` | python-api@9380 | go-api (`/api/v1/compilation-template-groups/:group_id`) | `api/apps/restful_apis/compilation_template_group_api.py:89` |
| PUT | `/api/v1/compilation-template-groups/<group_id>` | python-api@9380 | go-api (`/api/v1/compilation-template-groups/:group_id`) | `api/apps/restful_apis/compilation_template_group_api.py:128` |
| GET | `/api/v1/compilation-templates/builtins` | python-api@9380 | go-api (`/api/v1/compilation-templates/builtins`) | `api/apps/restful_apis/compilation_template_api.py:28` |
| GET | `/api/v1/compilation-templates/wiki-presets` | python-api@9380 | go-api (`/api/v1/compilation-templates/wiki-presets`) | `api/apps/restful_apis/compilation_template_api.py:57` |
| POST | `/api/v1/connectors/:connector_id/rebuild` | go-api@9384 | python-api (`/api/v1/connectors/<connector_id>/rebuild`) | `internal/router/router.go:640` |
| POST | `/api/v1/connectors/:connector_id/test` | go-api@9384 | python-api (`/api/v1/connectors/<connector_id>/test`) | `internal/router/router.go:641` |
| DELETE | `/api/v1/connectors/<connector_id>` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id`) | `api/apps/restful_apis/connector_api.py:169` |
| GET | `/api/v1/connectors/<connector_id>` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id`) | `api/apps/restful_apis/connector_api.py:123` |
| PATCH | `/api/v1/connectors/<connector_id>` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id`) | `api/apps/restful_apis/connector_api.py:49` |
| GET | `/api/v1/connectors/<connector_id>/logs` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id/logs`) | `api/apps/restful_apis/connector_api.py:136` |
| GET | `/api/v1/datasets/:dataset_id/changes` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/changes`) | `internal/router/router.go:501` |
| GET | `/api/v1/datasets/:dataset_id/commits` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits`) | `internal/router/router.go:495` |
| POST | `/api/v1/datasets/:dataset_id/commits` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits`) | `internal/router/router.go:494` |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/<commit_id>`) | `internal/router/router.go:497` |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id/files` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/<commit_id>/files`) | `internal/router/router.go:498` |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id/files/:file_id/content` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/<commit_id>/files/<file_id>/content`) | `internal/router/router.go:500` |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id/tree` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/<commit_id>/tree`) | `internal/router/router.go:499` |
| GET | `/api/v1/datasets/:dataset_id/commits/diff` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/diff`) | `internal/router/router.go:496` |
| POST | `/api/v1/datasets/:dataset_id/documents` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/documents`) | `internal/router/router.go:404` |
| PATCH | `/api/v1/datasets/:dataset_id/documents/metadatas` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/documents/metadatas`) | `internal/router/router.go:428` |
| POST | `/api/v1/datasets/:dataset_id/documents/parse` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/documents/parse`) | `internal/router/router.go:416` |
| GET | `/api/v1/datasets/:dataset_id/ingestions/summary` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/ingestions/summary`) | `internal/router/router.go:394` |
| GET | `/api/v1/datasets/<dataset_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id`) | `api/apps/restful_apis/dataset_api.py:384` |
| PUT | `/api/v1/datasets/<dataset_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id`) | `api/apps/restful_apis/dataset_api.py:220` |
| DELETE | `/api/v1/datasets/<dataset_id>/artifacts` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts`) | `api/apps/restful_apis/dataset_api.py:643` |
| GET | `/api/v1/datasets/<dataset_id>/artifacts` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts`) | `api/apps/restful_apis/dataset_api.py:580` |
| HEAD | `/api/v1/datasets/<dataset_id>/artifacts` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts`) | `api/apps/restful_apis/dataset_api.py:584` |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/<page_type>/<path:slug>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts/:page_type/:slug`) | `api/apps/restful_apis/dataset_api.py:667` |
| PUT | `/api/v1/datasets/<dataset_id>/artifacts/<page_type>/<path:slug>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts/:page_type/:slug`) | `api/apps/restful_apis/dataset_api.py:770` |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/alteration` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts/alteration`) | `api/apps/restful_apis/dataset_api.py:803` |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/graph` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts/graph`) | `api/apps/restful_apis/dataset_api.py:612` |
| DELETE | `/api/v1/datasets/<dataset_id>/artifacts/structure` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts/structure`) | `api/apps/restful_apis/dataset_api.py:762` |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/structure` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts/structure`) | `api/apps/restful_apis/dataset_api.py:715` |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/topics` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/artifacts/topics`) | `api/apps/restful_apis/dataset_api.py:638` |
| DELETE | `/api/v1/datasets/<dataset_id>/chunks` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/chunks`) | `api/apps/restful_apis/chunk_api.py:258` |
| POST | `/api/v1/datasets/<dataset_id>/chunks` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/chunks`) | `api/apps/restful_apis/chunk_api.py:183` |
| DELETE | `/api/v1/datasets/<dataset_id>/documents` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents`) | `api/apps/restful_apis/document_api.py:1103` |
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
| DELETE | `/api/v1/datasets/<dataset_id>/documents/<document_id>/structure/graph` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id/structure/graph`) | `api/apps/restful_apis/chunk_api.py:786` |
| GET | `/api/v1/datasets/<dataset_id>/documents/<document_id>/structure/graph` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/:document_id/structure/graph`) | `api/apps/restful_apis/chunk_api.py:556` |
| POST | `/api/v1/datasets/<dataset_id>/documents/batch-update-status` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/batch-update-status`) | `api/apps/restful_apis/document_api.py:1924` |
| POST | `/api/v1/datasets/<dataset_id>/documents/stop` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/stop`) | `api/apps/restful_apis/document_api.py:1621` |
| POST | `/api/v1/datasets/<dataset_id>/embedding/check` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/embedding/check`) | `api/apps/restful_apis/dataset_api.py:896` |
| GET | `/api/v1/datasets/<dataset_id>/graph` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/graph`) | `api/apps/restful_apis/dataset_api.py:537` |
| GET | `/api/v1/datasets/<dataset_id>/ingestions` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/ingestions`) | `api/apps/restful_apis/dataset_api.py:916` |
| GET | `/api/v1/datasets/<dataset_id>/ingestions/<log_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/ingestions/:log_id`) | `api/apps/restful_apis/dataset_api.py:942` |
| GET | `/api/v1/datasets/<dataset_id>/metadata/config` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/metadata/config`) | `api/apps/restful_apis/dataset_api.py:959` |
| PUT | `/api/v1/datasets/<dataset_id>/metadata/config` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/metadata/config`) | `api/apps/restful_apis/dataset_api.py:1000` |
| GET | `/api/v1/datasets/<dataset_id>/metadata/summary` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/metadata/summary`) | `api/apps/restful_apis/document_api.py:307` |
| POST | `/api/v1/datasets/<dataset_id>/metadata/update` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/metadata/update`) | `api/apps/restful_apis/document_api.py:345` |
| DELETE | `/api/v1/datasets/<dataset_id>/navigation` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/navigation`) | `api/apps/restful_apis/dataset_api.py:1052` |
| GET | `/api/v1/datasets/<dataset_id>/navigation` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/navigation`) | `api/apps/restful_apis/dataset_api.py:962` |
| DELETE | `/api/v1/datasets/<dataset_id>/navigation/<path:name>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/navigation/:name`) | `api/apps/restful_apis/dataset_api.py:1074` |
| GET | `/api/v1/datasets/<dataset_id>/navigation/<path:name>/children` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/navigation/:name/children`) | `api/apps/restful_apis/dataset_api.py:1029` |
| POST | `/api/v1/datasets/<dataset_id>/search` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/search`) | `api/apps/restful_apis/dataset_api.py:507` |
| DELETE | `/api/v1/datasets/<dataset_id>/skills` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/skills`) | `api/apps/restful_apis/dataset_api.py:917` |
| GET | `/api/v1/datasets/<dataset_id>/skills` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/skills`) | `api/apps/restful_apis/dataset_api.py:713` |
| HEAD | `/api/v1/datasets/<dataset_id>/skills` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/skills`) | `api/apps/restful_apis/dataset_api.py:881` |
| DELETE | `/api/v1/datasets/<dataset_id>/skills/<path:skill_kwd>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/skills/:skill_kwd`) | `api/apps/restful_apis/dataset_api.py:1129` |
| GET | `/api/v1/datasets/<dataset_id>/skills/<path:skill_kwd>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/skills/:skill_kwd`) | `api/apps/restful_apis/dataset_api.py:735` |
| DELETE | `/api/v1/datasets/<dataset_id>/tags` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/tags`) | `api/apps/restful_apis/dataset_api.py:435` |
| GET | `/api/v1/datasets/<dataset_id>/tags` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/tags`) | `api/apps/restful_apis/dataset_api.py:418` |
| PUT | `/api/v1/datasets/<dataset_id>/tags` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/tags`) | `api/apps/restful_apis/dataset_api.py:458` |
| GET | `/api/v1/documents/:id/preview` | go-api@9384 | python-api (`/api/v1/documents/<doc_id>/preview`) | `internal/router/router.go:221` |
| GET | `/api/v1/documents/<document_id>` | python-api@9380 | go-api (`/api/v1/documents/:id`) | `api/apps/restful_apis/document_api.py:2151` |
| GET | `/api/v1/documents/artifact/<filename>` | python-api@9380 | go-api (`/api/v1/documents/artifact/:filename`) | `api/apps/restful_apis/document_api.py:1871` |
| GET | `/api/v1/documents/images/<image_id>` | python-api@9380 | go-api (`/api/v1/documents/images/:image_id`) | `api/apps/restful_apis/document_api.py:1782` |
| GET | `/api/v1/files/:id/parent` | go-api@9384 | python-api (`/api/v1/files/<file_id>/parent`) | `internal/router/router.go:451` |
| GET | `/api/v1/files/<file_id>` | python-api@9380 | go-api (`/api/v1/files/:id`) | `api/apps/restful_apis/file_api.py:262` |
| GET | `/api/v1/files/<file_id>/ancestors` | python-api@9380 | go-api (`/api/v1/files/:id/ancestors`) | `api/apps/restful_apis/file_api.py:348` |
| GET | `/api/v1/files/<file_id>/versions` | python-api@9380 | go-api (`/api/v1/files/:id/versions`) | `api/apps/restful_apis/file_commit_api.py:370` |
| GET | `/api/v1/folders/:folder_id/changes` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/changes`) | `internal/router/router.go:474` |
| GET | `/api/v1/folders/:folder_id/commits` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits`) | `internal/router/router.go:468` |
| POST | `/api/v1/folders/:folder_id/commits` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits`) | `internal/router/router.go:467` |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/<commit_id>`) | `internal/router/router.go:470` |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id/files` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/<commit_id>/files`) | `internal/router/router.go:471` |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id/files/:file_id/content` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/<commit_id>/files/<file_id>/content`) | `internal/router/router.go:473` |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id/tree` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/<commit_id>/tree`) | `internal/router/router.go:472` |
| GET | `/api/v1/folders/:folder_id/commits/diff` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/diff`) | `internal/router/router.go:469` |
| DELETE | `/api/v1/mcp/servers/<mcp_id>` | python-api@9380 | go-api (`/api/v1/mcp/servers/:mcp_id`) | `api/apps/restful_apis/mcp_api.py:231` |
| GET | `/api/v1/mcp/servers/<mcp_id>` | python-api@9380 | go-api (`/api/v1/mcp/servers/:mcp_id`) | `api/apps/restful_apis/mcp_api.py:95` |
| PUT | `/api/v1/mcp/servers/<mcp_id>` | python-api@9380 | go-api (`/api/v1/mcp/servers/:mcp_id`) | `api/apps/restful_apis/mcp_api.py:172` |
| POST | `/api/v1/mcp/servers/<mcp_id>/test` | python-api@9380 | go-api (`/api/v1/mcp/servers/:mcp_id/test`) | `api/apps/restful_apis/mcp_api.py:321` |
| DELETE | `/api/v1/memories/<memory_id>` | python-api@9380 | go-api (`/api/v1/memories/:memory_id`) | `api/apps/restful_apis/memory_api.py:119` |
| GET | `/api/v1/memories/<memory_id>` | python-api@9380 | go-api (`/api/v1/memories/:memory_id`) | `api/apps/restful_apis/memory_api.py:162` |
| PUT | `/api/v1/memories/<memory_id>` | python-api@9380 | go-api (`/api/v1/memories/:memory_id`) | `api/apps/restful_apis/memory_api.py:78` |
| GET | `/api/v1/memories/<memory_id>/config` | python-api@9380 | go-api (`/api/v1/memories/:memory_id/config`) | `api/apps/restful_apis/memory_api.py:148` |
| DELETE | `/api/v1/messages/:memory_message` | go-api@9384 | python-api (`/api/v1/messages/<memory_id>:<message_id>`) | `internal/router/router.go:520` |
| PUT | `/api/v1/messages/:memory_message` | go-api@9384 | python-api (`/api/v1/messages/<memory_id>:<message_id>`) | `internal/router/router.go:521` |
| GET | `/api/v1/messages/:memory_message/content` | go-api@9384 | python-api (`/api/v1/messages/<memory_id>:<message_id>/content`) | `internal/router/router.go:522` |
| POST | `/api/v1/openai/<chat_id>/chat/completions` | python-api@9380 | go-api (`/api/v1/openai/:chat_id/chat/completions`) | `api/apps/restful_apis/openai_api.py:237` |
| PATCH | `/api/v1/providers/:provider_id_or_name/instances/:instance_id_or_name/models/*model_name` | go-api@9384 | python-api (`/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models/<path:model_name>`) | `internal/router/router.go:568` |
| DELETE | `/api/v1/providers/<provider_id_or_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name`) | `api/apps/restful_apis/provider_api.py:161` |
| GET | `/api/v1/providers/<provider_id_or_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name`) | `api/apps/restful_apis/provider_api.py:123` |
| POST | `/api/v1/providers/<provider_id_or_name>/connection` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name/connection`) | `api/apps/restful_apis/provider_api.py:359` |
| DELETE | `/api/v1/providers/<provider_id_or_name>/instances` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name/instances`) | `api/apps/restful_apis/provider_api.py:515` |
| GET | `/api/v1/providers/<provider_id_or_name>/instances` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name/instances`) | `api/apps/restful_apis/provider_api.py:427` |
| POST | `/api/v1/providers/<provider_id_or_name>/instances` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name/instances`) | `api/apps/restful_apis/provider_api.py:288` |
| GET | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name/instances/:instance_id_or_name`) | `api/apps/restful_apis/provider_api.py:471` |
| GET | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name/instances/:instance_id_or_name/models`) | `api/apps/restful_apis/provider_api.py:576` |
| POST | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name/instances/:instance_id_or_name/models`) | `api/apps/restful_apis/provider_api.py:691` |
| GET | `/api/v1/providers/<provider_id_or_name>/models` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name/models`) | `api/apps/restful_apis/provider_api.py:200` |
| GET | `/api/v1/providers/<provider_id_or_name>/models/<path:model_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_id_or_name/models/:model_name`) | `api/apps/restful_apis/provider_api.py:245` |
| DELETE | `/api/v1/searches/<search_id>` | python-api@9380 | go-api (`/api/v1/searches/:search_id`) | `api/apps/restful_apis/search_api.py:179` |
| GET | `/api/v1/searches/<search_id>` | python-api@9380 | go-api (`/api/v1/searches/:search_id`) | `api/apps/restful_apis/search_api.py:101` |
| PUT | `/api/v1/searches/<search_id>` | python-api@9380 | go-api (`/api/v1/searches/:search_id`) | `api/apps/restful_apis/search_api.py:120` |
| POST | `/api/v1/searches/<search_id>/completion` | python-api@9380 | go-api (`/api/v1/searches/:search_id/completion`) | `api/apps/restful_apis/search_api.py:193` |
| POST | `/api/v1/searches/<search_id>/completions` | python-api@9380 | go-api (`/api/v1/searches/:search_id/completions`) | `api/apps/restful_apis/search_api.py:194` |
| DELETE | `/api/v1/system/tokens/<token>` | python-api@9380 | go-api (`/api/v1/system/tokens/:key`) | `api/apps/restful_apis/system_api.py:333` |
| POST | `/api/v1/tasks/<task_id>/cancel` | python-api@9380 | go-api (`/api/v1/tasks/:session_id/cancel`) | `api/apps/restful_apis/task_api.py:30` |
| PATCH | `/api/v1/tenants/<tenant_id>` | python-api@9380 | go-api (`/api/v1/tenants/:tenant_id`) | `api/apps/restful_apis/tenant_api.py:167` |
| DELETE | `/api/v1/tenants/<tenant_id>/users` | python-api@9380 | go-api (`/api/v1/tenants/:tenant_id/users`) | `api/apps/restful_apis/tenant_api.py:135` |
| GET | `/api/v1/tenants/<tenant_id>/users` | python-api@9380 | go-api (`/api/v1/tenants/:tenant_id/users`) | `api/apps/restful_apis/tenant_api.py:41` |
| POST | `/api/v1/tenants/<tenant_id>/users` | python-api@9380 | go-api (`/api/v1/tenants/:tenant_id/users`) | `api/apps/restful_apis/tenant_api.py:60` |
| GET | `/api/v1/workspace/:folder_id/changes` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/changes`) | `internal/router/router.go:487` |
| GET | `/api/v1/workspace/:folder_id/commits` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits`) | `internal/router/router.go:481` |
| POST | `/api/v1/workspace/:folder_id/commits` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits`) | `internal/router/router.go:480` |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/<commit_id>`) | `internal/router/router.go:483` |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id/files` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/<commit_id>/files`) | `internal/router/router.go:484` |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id/files/:file_id/content` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/<commit_id>/files/<file_id>/content`) | `internal/router/router.go:486` |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id/tree` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/<commit_id>/tree`) | `internal/router/router.go:485` |
| GET | `/api/v1/workspace/:folder_id/commits/diff` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/diff`) | `internal/router/router.go:482` |

