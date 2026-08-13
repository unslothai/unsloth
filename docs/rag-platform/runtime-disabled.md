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
| routes discovered | 709 |
| reachable | 516 |
| runtime-disabled | 188 |
| — no reachable equivalent (capability lost) | 7 |
| — same concrete request served elsewhere (no capability lost) | 181 |
| not proxied by nginx | 5 |

## Why these routes are closed

The owned method-aware hybrid map selects one implementation for each
method+path. When both Python and Go register the same contract, the Go
implementation is selected and the duplicate Python registration appears
below as runtime-disabled with a reachable equivalent. This is intentional
deduplication, not a lost capability. The Go executable provenance and the
four direct service smoke probes are recorded in ADR 0005 and the Faz 0
result report.

9 auth route(s) are a separate forward-source case: they are declared only at backend worktree `a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2`, are absent from deployed `v0.26.4`, and their worktree handlers return `CodeNotImplemented`. Live hybrid smoke returns HTTP 404 for seven concrete paths; GitHub and Lark callback URLs return 302 through the active parameterised callback. The UI therefore uses live channels and exposes direct registration without a false captcha/OTP step.

## Capability lost — no reachable route serves this method and path

Compared after canonicalising parameter syntax (`<id>`, `:id`, `*path` all
normalise to `{p}`), so a Go route is only listed here when no Python route
provides the same method and path shape.

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
| DELETE | `/api/v1/connectors/<connector_id>` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id`) | `api/apps/restful_apis/connector_api.py:169` |
| GET | `/api/v1/connectors/<connector_id>` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id`) | `api/apps/restful_apis/connector_api.py:123` |
| PATCH | `/api/v1/connectors/<connector_id>` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id`) | `api/apps/restful_apis/connector_api.py:49` |
| GET | `/api/v1/connectors/<connector_id>/logs` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id/logs`) | `api/apps/restful_apis/connector_api.py:136` |
| POST | `/api/v1/connectors/<connector_id>/rebuild` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id/rebuild`) | `api/apps/restful_apis/connector_api.py:152` |
| POST | `/api/v1/connectors/<connector_id>/test` | python-api@9380 | go-api (`/api/v1/connectors/:connector_id/test`) | `api/apps/restful_apis/connector_api.py:181` |
| DELETE | `/api/v1/datasets/:dataset_id/chunks` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/chunks`) | `internal/router/router.go:381` |
| GET | `/api/v1/datasets/:dataset_id/commits/diff` | go-api@9384 | python-api (`/api/v1/datasets/<entity_id>/commits/diff`) | `internal/router/router.go:453` |
| DELETE | `/api/v1/datasets/:dataset_id/documents` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/documents`) | `internal/router/router.go:366` |
| PATCH | `/api/v1/datasets/:dataset_id/documents/metadatas` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/documents/metadatas`) | `internal/router/router.go:385` |
| DELETE | `/api/v1/datasets/:dataset_id/index` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/index`) | `internal/router/router.go:342` |
| GET | `/api/v1/datasets/:dataset_id/ingestions/summary` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/ingestions/summary`) | `internal/router/router.go:353` |
| DELETE | `/api/v1/datasets/:dataset_id/tags` | go-api@9384 | python-api (`/api/v1/datasets/<dataset_id>/tags`) | `internal/router/router.go:336` |
| GET | `/api/v1/datasets/<dataset_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id`) | `api/apps/restful_apis/dataset_api.py:384` |
| PUT | `/api/v1/datasets/<dataset_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id`) | `api/apps/restful_apis/dataset_api.py:220` |
| DELETE | `/api/v1/datasets/<dataset_id>/<index_type>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/:index_type`) | `api/apps/restful_apis/dataset_api.py:854` |
| POST | `/api/v1/datasets/<dataset_id>/chunks` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/chunks`) | `api/apps/restful_apis/chunk_api.py:183` |
| GET | `/api/v1/datasets/<dataset_id>/documents` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents`) | `api/apps/restful_apis/document_api.py:703` |
| POST | `/api/v1/datasets/<dataset_id>/documents` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents`) | `api/apps/restful_apis/document_api.py:427` |
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
| POST | `/api/v1/datasets/<dataset_id>/documents/parse` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/documents/parse`) | `api/apps/restful_apis/document_api.py:1508` |
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
| GET | `/api/v1/datasets/<entity_id>/changes` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/changes`) | `api/apps/restful_apis/file_commit_api.py:301` |
| GET | `/api/v1/datasets/<entity_id>/commits` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/commits`) | `api/apps/restful_apis/file_commit_api.py:140` |
| POST | `/api/v1/datasets/<entity_id>/commits` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/commits`) | `api/apps/restful_apis/file_commit_api.py:107` |
| GET | `/api/v1/datasets/<entity_id>/commits/<commit_id>` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/commits/:commit_id`) | `api/apps/restful_apis/file_commit_api.py:199` |
| GET | `/api/v1/datasets/<entity_id>/commits/<commit_id>/files` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/commits/:commit_id/files`) | `api/apps/restful_apis/file_commit_api.py:249` |
| GET | `/api/v1/datasets/<entity_id>/commits/<commit_id>/files/<file_id>/content` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/commits/:commit_id/files/:file_id/content`) | `api/apps/restful_apis/file_commit_api.py:328` |
| GET | `/api/v1/datasets/<entity_id>/commits/<commit_id>/tree` | python-api@9380 | go-api (`/api/v1/datasets/:dataset_id/commits/:commit_id/tree`) | `api/apps/restful_apis/file_commit_api.py:312` |
| GET | `/api/v1/documents/<doc_id>/preview` | python-api@9380 | go-api (`/api/v1/documents/:id/preview`) | `api/apps/restful_apis/document_api.py:2045` |
| GET | `/api/v1/documents/<document_id>` | python-api@9380 | go-api (`/api/v1/documents/:id`) | `api/apps/restful_apis/document_api.py:2151` |
| GET | `/api/v1/documents/artifact/<filename>` | python-api@9380 | go-api (`/api/v1/documents/artifact/:filename`) | `api/apps/restful_apis/document_api.py:1871` |
| GET | `/api/v1/documents/images/<image_id>` | python-api@9380 | go-api (`/api/v1/documents/images/:image_id`) | `api/apps/restful_apis/document_api.py:1782` |
| GET | `/api/v1/files/<file_id>` | python-api@9380 | go-api (`/api/v1/files/:id`) | `api/apps/restful_apis/file_api.py:262` |
| GET | `/api/v1/files/<file_id>/ancestors` | python-api@9380 | go-api (`/api/v1/files/:id/ancestors`) | `api/apps/restful_apis/file_api.py:348` |
| GET | `/api/v1/files/<file_id>/parent` | python-api@9380 | go-api (`/api/v1/files/:id/parent`) | `api/apps/restful_apis/file_api.py:317` |
| GET | `/api/v1/files/<file_id>/versions` | python-api@9380 | go-api (`/api/v1/files/:id/versions`) | `api/apps/restful_apis/file_commit_api.py:370` |
| GET | `/api/v1/folders/:folder_id/commits/diff` | go-api@9384 | python-api (`/api/v1/folders/<entity_id>/commits/diff`) | `internal/router/router.go:426` |
| GET | `/api/v1/folders/<entity_id>/changes` | python-api@9380 | go-api (`/api/v1/folders/:folder_id/changes`) | `api/apps/restful_apis/file_commit_api.py:301` |
| GET | `/api/v1/folders/<entity_id>/commits` | python-api@9380 | go-api (`/api/v1/folders/:folder_id/commits`) | `api/apps/restful_apis/file_commit_api.py:140` |
| POST | `/api/v1/folders/<entity_id>/commits` | python-api@9380 | go-api (`/api/v1/folders/:folder_id/commits`) | `api/apps/restful_apis/file_commit_api.py:107` |
| GET | `/api/v1/folders/<entity_id>/commits/<commit_id>` | python-api@9380 | go-api (`/api/v1/folders/:folder_id/commits/:commit_id`) | `api/apps/restful_apis/file_commit_api.py:199` |
| GET | `/api/v1/folders/<entity_id>/commits/<commit_id>/files` | python-api@9380 | go-api (`/api/v1/folders/:folder_id/commits/:commit_id/files`) | `api/apps/restful_apis/file_commit_api.py:249` |
| GET | `/api/v1/folders/<entity_id>/commits/<commit_id>/files/<file_id>/content` | python-api@9380 | go-api (`/api/v1/folders/:folder_id/commits/:commit_id/files/:file_id/content`) | `api/apps/restful_apis/file_commit_api.py:328` |
| GET | `/api/v1/folders/<entity_id>/commits/<commit_id>/tree` | python-api@9380 | go-api (`/api/v1/folders/:folder_id/commits/:commit_id/tree`) | `api/apps/restful_apis/file_commit_api.py:312` |
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
| DELETE | `/api/v1/providers/<provider_id_or_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_name`) | `api/apps/restful_apis/provider_api.py:161` |
| GET | `/api/v1/providers/<provider_id_or_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_name`) | `api/apps/restful_apis/provider_api.py:123` |
| POST | `/api/v1/providers/<provider_id_or_name>/connection` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/connection`) | `api/apps/restful_apis/provider_api.py:359` |
| DELETE | `/api/v1/providers/<provider_id_or_name>/instances` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances`) | `api/apps/restful_apis/provider_api.py:515` |
| GET | `/api/v1/providers/<provider_id_or_name>/instances` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances`) | `api/apps/restful_apis/provider_api.py:427` |
| POST | `/api/v1/providers/<provider_id_or_name>/instances` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances`) | `api/apps/restful_apis/provider_api.py:288` |
| GET | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances/:instance_name`) | `api/apps/restful_apis/provider_api.py:471` |
| GET | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances/:instance_name/models`) | `api/apps/restful_apis/provider_api.py:576` |
| POST | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances/:instance_name/models`) | `api/apps/restful_apis/provider_api.py:691` |
| PATCH | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models/<path:model_name>` | python-api@9380 | go-api (`/api/v1/providers/:provider_name/instances/:instance_name/models/*model_name`) | `api/apps/restful_apis/provider_api.py:764` |
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
| GET | `/api/v1/workspace/:folder_id/commits/diff` | go-api@9384 | python-api (`/api/v1/workspace/<entity_id>/commits/diff`) | `internal/router/router.go:439` |
| GET | `/api/v1/workspace/<entity_id>/changes` | python-api@9380 | go-api (`/api/v1/workspace/:folder_id/changes`) | `api/apps/restful_apis/file_commit_api.py:301` |
| GET | `/api/v1/workspace/<entity_id>/commits` | python-api@9380 | go-api (`/api/v1/workspace/:folder_id/commits`) | `api/apps/restful_apis/file_commit_api.py:140` |
| POST | `/api/v1/workspace/<entity_id>/commits` | python-api@9380 | go-api (`/api/v1/workspace/:folder_id/commits`) | `api/apps/restful_apis/file_commit_api.py:107` |
| GET | `/api/v1/workspace/<entity_id>/commits/<commit_id>` | python-api@9380 | go-api (`/api/v1/workspace/:folder_id/commits/:commit_id`) | `api/apps/restful_apis/file_commit_api.py:199` |
| GET | `/api/v1/workspace/<entity_id>/commits/<commit_id>/files` | python-api@9380 | go-api (`/api/v1/workspace/:folder_id/commits/:commit_id/files`) | `api/apps/restful_apis/file_commit_api.py:249` |
| GET | `/api/v1/workspace/<entity_id>/commits/<commit_id>/files/<file_id>/content` | python-api@9380 | go-api (`/api/v1/workspace/:folder_id/commits/:commit_id/files/:file_id/content`) | `api/apps/restful_apis/file_commit_api.py:328` |
| GET | `/api/v1/workspace/<entity_id>/commits/<commit_id>/tree` | python-api@9380 | go-api (`/api/v1/workspace/:folder_id/commits/:commit_id/tree`) | `api/apps/restful_apis/file_commit_api.py:312` |

