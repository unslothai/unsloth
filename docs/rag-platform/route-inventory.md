# Rag Platform — backend route inventory

<!-- GENERATED FILE. Do not edit by hand.
     Regenerate: node scripts/rag-platform/route-inventory.mjs -->

- Backend source: `/Users/baran/Desktop/rag-backend` at `v0.26.4` (`cb93883f3f8c975eecb2fed81210effeb3bdb06f`)
- Source image: `infiniflow/ragflow:v0.26.4` (API version `v1`)
- Forward source audit: backend worktree `a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2`; 11 source-only runtime-disabled route(s)
- Active proxy scheme: `hybrid` (from infra/rag-platform/.env.rag-platform)
- Proxy config: `infra/rag-platform/rag-platform.hybrid.conf`

## Totals

| Metric | Count |
| --- | --- |
| routes | 711 |
| go-admin (port 9383) | 114 |
| python-admin (port 9381) | 34 |
| python-api (port 9380) | 304 |
| go-api (port 9384) | 254 |
| mcp (port 9382) | 5 |
| runtime-enabled | 516 |
| runtime-disabled | 190 |
| — source-only forward declarations | 11 |
| not proxied by nginx | 5 |
| method+path with alternate implementations | 110 |

## Proxy location map (nginx evaluation order)

| Order | Location | Upstream |
| --- | --- | --- |
| 1 | `PATCH ~ ^/api/v1/datasets/[^/]+/documents/metadatas/?$` | 127.0.0.1:9380 |
| 2 | `GET ~ ^/api/v1/datasets/[^/]+/ingestions/summary/?$` | 127.0.0.1:9380 |
| 3 | `DELETE ~ ^/api/v1/datasets/[^/]+/knowledge_graph/?$` | 127.0.0.1:9380 |
| 4 | `GET ~ ^/api/v1/workspace/[^/]+/commits/diff/?$` | 127.0.0.1:9380 |
| 5 | `GET ~ ^/api/v1/datasets/[^/]+/commits/diff/?$` | 127.0.0.1:9380 |
| 6 | `GET ~ ^/api/v1/folders/[^/]+/commits/diff/?$` | 127.0.0.1:9380 |
| 7 | `DELETE ~ ^/api/v1/datasets/[^/]+/artifacts/?$` | 127.0.0.1:9380 |
| 8 | `DELETE ~ ^/api/v1/datasets/[^/]+/documents/?$` | 127.0.0.1:9380 |
| 9 | `GET ~ ^/api/v1/messages/[^/]+:[^/]+/content/?$` | 127.0.0.1:9380 |
| 10 | `DELETE ~ ^/api/v1/datasets/[^/]+/chunks/?$` | 127.0.0.1:9380 |
| 11 | `DELETE ~ ^/api/v1/datasets/[^/]+/index/?$` | 127.0.0.1:9380 |
| 12 | `DELETE ~ ^/api/v1/datasets/[^/]+/tags/?$` | 127.0.0.1:9380 |
| 13 | `DELETE ~ ^/api/v1/messages/[^/]+:[^/]+/?$` | 127.0.0.1:9380 |
| 14 | `PUT ~ ^/api/v1/messages/[^/]+:[^/]+/?$` | 127.0.0.1:9380 |
| 15 | `PATCH ~ ^/api/v1/providers/[^/]+/instances/[^/]+/models/.*/?$` | 127.0.0.1:9380 |
| 16 | `GET ~ ^/api/v1/admin/users/[^/]+/providers/[^/]+/instances/[^/]+/models/?$` | 127.0.0.1:9383 |
| 17 | `POST ~ ^/api/v1/datasets/[^/]+/documents/batch-update-status/?$` | 127.0.0.1:9384 |
| 18 | `GET ~ ^/api/v1/admin/providers/[^/]+/instances/[^/]+/connection/?$` | 127.0.0.1:9383 |
| 19 | `PUT ~ ^/api/v1/datasets/[^/]+/documents/[^/]+/metadata/config/?$` | 127.0.0.1:9384 |
| 20 | `GET ~ ^/api/v1/admin/providers/[^/]+/instances/[^/]+/balance/?$` | 127.0.0.1:9383 |
| 21 | `GET ~ ^/connectors/google-drive/oauth/web/callback/?$` | 127.0.0.1:9384 |
| 22 | `PATCH ~ ^/api/v1/admin/providers/[^/]+/instances/[^/]+/models/.*/?$` | 127.0.0.1:9383 |
| 23 | `PUT ~ ^/api/v1/chats/[^/]+/sessions/[^/]+/messages/[^/]+/feedback/?$` | 127.0.0.1:9384 |
| 24 | `DELETE ~ ^/api/v1/admin/providers/[^/]+/instances/[^/]+/models/?$` | 127.0.0.1:9383 |
| 25 | `GET ~ ^/api/v1/admin/providers/[^/]+/instances/[^/]+/models/?$` | 127.0.0.1:9383 |
| 26 | `GET ~ ^/api/v1/workspace/[^/]+/commits/[^/]+/files/[^/]+/content/?$` | 127.0.0.1:9384 |
| 27 | `POST ~ ^/api/v1/admin/providers/[^/]+/instances/[^/]+/models/?$` | 127.0.0.1:9383 |
| 28 | `GET ~ ^/api/v1/admin/users/[^/]+/providers/[^/]+/instances/?$` | 127.0.0.1:9383 |
| 29 | `GET ~ ^/api/v1/datasets/[^/]+/commits/[^/]+/files/[^/]+/content/?$` | 127.0.0.1:9384 |
| 30 | `GET ~ ^/api/v1/folders/[^/]+/commits/[^/]+/files/[^/]+/content/?$` | 127.0.0.1:9384 |
| 31 | `GET ~ ^/api/v1/providers/[^/]+/instances/[^/]+/connection/?$` | 127.0.0.1:9384 |
| 32 | `POST ~ ^/api/v1/tenant/insert_metadata_from_file/?$` | 127.0.0.1:9384 |
| 33 | `GET ~ ^/api/v1/admin/sandbox/providers/[^/]+/schema/?$` | 127.0.0.1:9383 |
| 34 | `GET ~ ^/api/v1/agents/[^/]+/components/[^/]+/input-form/?$` | 127.0.0.1:9384 |
| 35 | `POST ~ ^/api/v1/tenant/insert_chunks_from_file/?$` | 127.0.0.1:9384 |
| 36 | `GET ~ ^/api/v1/admin/ingestion/tasks/summary/?$` | 127.0.0.1:9383 |
| 37 | `GET ~ ^/api/v1/providers/[^/]+/instances/[^/]+/balance/?$` | 127.0.0.1:9384 |
| 38 | `PATCH ~ ^/api/v1/datasets/[^/]+/documents/metadatas/?$` | 127.0.0.1:9384 |
| 39 | `PATCH ~ ^/api/v1/providers/[^/]+/instances/[^/]+/models/.*/?$` | 127.0.0.1:9384 |
| 40 | `DELETE ~ ^/api/v1/providers/[^/]+/instances/[^/]+/models/?$` | 127.0.0.1:9384 |
| 41 | `GET ~ ^/api/v1/agents/attachments/[^/]+/download/?$` | 127.0.0.1:9384 |
| 42 | `GET ~ ^/api/v1/datasets/[^/]+/documents/[^/]+/chunks/[^/]+/?$` | 127.0.0.1:9384 |
| 43 | `GET ~ ^/api/v1/datasets/[^/]+/ingestions/summary/?$` | 127.0.0.1:9384 |
| 44 | `GET ~ ^/api/v1/providers/[^/]+/instances/[^/]+/models/?$` | 127.0.0.1:9384 |
| 45 | `GET ~ ^/api/v1/providers/[^/]+/instances/[^/]+/tasks/[^/]+/?$` | 127.0.0.1:9384 |
| 46 | `GET ~ ^/connectors/gmail/oauth/web/callback/?$` | 127.0.0.1:9384 |
| 47 | `PATCH ~ ^/api/v1/datasets/[^/]+/documents/[^/]+/chunks/[^/]+/?$` | 127.0.0.1:9384 |
| 48 | `POST ~ ^/api/v1/providers/[^/]+/instances/[^/]+/models/?$` | 127.0.0.1:9384 |
| 49 | `DELETE ~ ^/api/v1/admin/roles/[^/]+/default-models/?$` | 127.0.0.1:9383 |
| 50 | `DELETE ~ ^/api/v1/datasets/[^/]+/documents/[^/]+/chunks/?$` | 127.0.0.1:9384 |
| 51 | `GET ~ ^/api/v1/admin/providers/[^/]+/instances/[^/]+/?$` | 127.0.0.1:9383 |
| 52 | `GET ~ ^/api/v1/admin/roles/[^/]+/default-models/?$` | 127.0.0.1:9383 |
| 53 | `GET ~ ^/api/v1/admin/users/[^/]+/default-models/?$` | 127.0.0.1:9383 |
| 54 | `GET ~ ^/api/v1/agents/attachments/[^/]+/preview/?$` | 127.0.0.1:9384 |
| 55 | `GET ~ ^/api/v1/datasets/[^/]+/documents/[^/]+/chunks/?$` | 127.0.0.1:9384 |
| 56 | `GET ~ ^/api/v1/providers/[^/]+/instances/[^/]+/tasks/?$` | 127.0.0.1:9384 |
| 57 | `PATCH ~ ^/api/v1/admin/roles/[^/]+/default-models/?$` | 127.0.0.1:9383 |
| 58 | `PATCH ~ ^/api/v1/datasets/[^/]+/documents/[^/]+/chunks/?$` | 127.0.0.1:9384 |
| 59 | `POST ~ ^/api/v1/admin/providers/[^/]+/connection/?$` | 127.0.0.1:9383 |
| 60 | `POST ~ ^/api/v1/datasets/[^/]+/documents/[^/]+/chunks/?$` | 127.0.0.1:9384 |
| 61 | `PUT ~ ^/api/v1/admin/providers/[^/]+/instances/[^/]+/?$` | 127.0.0.1:9383 |
| 62 | `PUT ~ ^/api/v1/admin/system/license/config/?$` | 127.0.0.1:9383 |
| 63 | `DELETE ~ ^/api/v1/admin/providers/[^/]+/instances/?$` | 127.0.0.1:9383 |
| 64 | `DELETE ~ ^/api/v1/chats/[^/]+/sessions/[^/]+/messages/[^/]+/?$` | 127.0.0.1:9384 |
| 65 | `GET ~ ^/api/v1/admin/providers/[^/]+/instances/?$` | 127.0.0.1:9383 |
| 66 | `GET ~ ^/api/v1/auth/oauth/github/callback/?$` | 127.0.0.1:9384 |
| 67 | `GET ~ ^/api/v1/datasets/[^/]+/metadata/summary/?$` | 127.0.0.1:9384 |
| 68 | `GET ~ ^/connectors/box/oauth/web/callback/?$` | 127.0.0.1:9384 |
| 69 | `POST ~ ^/api/v1/admin/providers/[^/]+/instances/?$` | 127.0.0.1:9383 |
| 70 | `GET ~ ^/api/v1/admin/users/quota/summary/?$` | 127.0.0.1:9383 |
| 71 | `GET ~ ^/api/v1/datasets/[^/]+/metadata/config/?$` | 127.0.0.1:9384 |
| 72 | `GET ~ ^/api/v1/workspace/[^/]+/commits/[^/]+/files/?$` | 127.0.0.1:9384 |
| 73 | `POST ~ ^/api/v1/agents/[^/]+/components/[^/]+/debug/?$` | 127.0.0.1:9384 |
| 74 | `POST ~ ^/api/v1/datasets/[^/]+/documents/parse/?$` | 127.0.0.1:9384 |
| 75 | `POST ~ ^/api/v1/datasets/[^/]+/embedding/check/?$` | 127.0.0.1:9384 |
| 76 | `POST ~ ^/api/v1/datasets/[^/]+/metadata/update/?$` | 127.0.0.1:9384 |
| 77 | `POST ~ ^/api/v1/document/metadata/summary/?$` | 127.0.0.1:9384 |
| 78 | `PUT ~ ^/api/v1/datasets/[^/]+/metadata/config/?$` | 127.0.0.1:9384 |
| 79 | `DELETE ~ ^/api/v1/datasets/ingestion/tasks/?$` | 127.0.0.1:9384 |
| 80 | `GET ~ ^/api/v1/admin/providers/[^/]+/models/[^/]+/?$` | 127.0.0.1:9383 |
| 81 | `GET ~ ^/api/v1/admin/system/fingerprint/?$` | 127.0.0.1:9383 |
| 82 | `GET ~ ^/api/v1/admin/users/plan/summary/?$` | 127.0.0.1:9383 |
| 83 | `GET ~ ^/api/v1/auth/oauth/lark/callback/?$` | 127.0.0.1:9384 |
| 84 | `GET ~ ^/api/v1/datasets/[^/]+/commits/[^/]+/files/?$` | 127.0.0.1:9384 |
| 85 | `GET ~ ^/api/v1/datasets/ingestion/tasks/?$` | 127.0.0.1:9384 |
| 86 | `GET ~ ^/api/v1/workspace/[^/]+/commits/[^/]+/tree/?$` | 127.0.0.1:9384 |
| 87 | `POST ~ ^/api/v1/auth/register/otp/verify/?$` | 127.0.0.1:9384 |
| 88 | `POST ~ ^/api/v1/openai/[^/]+/chat/completions/?$` | 127.0.0.1:9384 |
| 89 | `PUT ~ ^/api/v1/datasets/ingestion/tasks/?$` | 127.0.0.1:9384 |
| 90 | `DELETE ~ ^/api/v1/admin/roles/[^/]+/permission/?$` | 127.0.0.1:9383 |
| 91 | `GET ~ ^/api/v1/admin/providers/[^/]+/models/?$` | 127.0.0.1:9383 |
| 92 | `GET ~ ^/api/v1/admin/roles/[^/]+/permission/?$` | 127.0.0.1:9383 |
| 93 | `GET ~ ^/api/v1/admin/users/[^/]+/permission/?$` | 127.0.0.1:9383 |
| 94 | `GET ~ ^/api/v1/datasets/[^/]+/commits/[^/]+/tree/?$` | 127.0.0.1:9384 |
| 95 | `GET ~ ^/api/v1/folders/[^/]+/commits/[^/]+/files/?$` | 127.0.0.1:9384 |
| 96 | `GET ~ ^/api/v1/workspace/[^/]+/commits/diff/?$` | 127.0.0.1:9384 |
| 97 | `POST ~ ^/api/v1/admin/roles/[^/]+/permission/?$` | 127.0.0.1:9383 |
| 98 | `GET ~ ^/api/v1/admin/users/[^/]+/providers/?$` | 127.0.0.1:9383 |
| 99 | `GET ~ ^/api/v1/datasets/[^/]+/commits/diff/?$` | 127.0.0.1:9384 |
| 100 | `GET ~ ^/api/v1/folders/[^/]+/commits/[^/]+/tree/?$` | 127.0.0.1:9384 |
| 101 | `GET ~ ^/api/v1/skills/space/by-folder/?$` | 127.0.0.1:9384 |
| 102 | `DELETE ~ ^/api/v1/admin/ingestion/tasks/?$` | 127.0.0.1:9383 |
| 103 | `DELETE ~ ^/api/v1/tenant/metadata_store/?$` | 127.0.0.1:9384 |
| 104 | `GET ~ ^/api/v1/admin/ingestion/tasks/?$` | 127.0.0.1:9383 |
| 105 | `GET ~ ^/api/v1/admin/users/[^/]+/activity/?$` | 127.0.0.1:9383 |
| 106 | `GET ~ ^/api/v1/admin/users/[^/]+/datasets/?$` | 127.0.0.1:9383 |
| 107 | `GET ~ ^/api/v1/admin/users/[^/]+/searches/?$` | 127.0.0.1:9383 |
| 108 | `GET ~ ^/api/v1/admin/users/documents/?$` | 127.0.0.1:9383 |
| 109 | `GET ~ ^/api/v1/datasets/[^/]+/ingestions/[^/]+/?$` | 127.0.0.1:9384 |
| 110 | `GET ~ ^/api/v1/folders/[^/]+/commits/diff/?$` | 127.0.0.1:9384 |
| 111 | `GET ~ ^/api/v1/providers/[^/]+/instances/[^/]+/?$` | 127.0.0.1:9384 |
| 112 | `POST ~ ^/api/v1/auth/register/captcha/?$` | 127.0.0.1:9384 |
| 113 | `POST ~ ^/api/v1/providers/[^/]+/connection/?$` | 127.0.0.1:9384 |
| 114 | `POST ~ ^/api/v1/searches/[^/]+/completions/?$` | 127.0.0.1:9384 |
| 115 | `POST ~ ^/api/v1/tenant/metadata_store/?$` | 127.0.0.1:9384 |
| 116 | `PUT ~ ^/api/v1/admin/ingestion/tasks/?$` | 127.0.0.1:9383 |
| 117 | `PUT ~ ^/api/v1/admin/users/[^/]+/activate/?$` | 127.0.0.1:9383 |
| 118 | `PUT ~ ^/api/v1/admin/users/[^/]+/password/?$` | 127.0.0.1:9383 |
| 119 | `PUT ~ ^/api/v1/providers/[^/]+/instances/[^/]+/?$` | 127.0.0.1:9384 |
| 120 | `DELETE ~ ^/api/v1/admin/users/[^/]+/tokens/[^/]+/?$` | 127.0.0.1:9383 |
| 121 | `DELETE ~ ^/api/v1/agents/[^/]+/webhook/test/?$` | 127.0.0.1:9384 |
| 122 | `DELETE ~ ^/api/v1/providers/[^/]+/instances/?$` | 127.0.0.1:9384 |
| 123 | `GET ~ ^/api/v1/admin/queue/messages/?$` | 127.0.0.1:9383 |
| 124 | `GET ~ ^/api/v1/admin/roles/resource/?$` | 127.0.0.1:9383 |
| 125 | `GET ~ ^/api/v1/admin/service_types/[^/]+/?$` | 127.0.0.1:9383 |
| 126 | `GET ~ ^/api/v1/admin/system/license/?$` | 127.0.0.1:9383 |
| 127 | `GET ~ ^/api/v1/admin/users/[^/]+/dataset/?$` | 127.0.0.1:9383 |
| 128 | `GET ~ ^/api/v1/admin/users/[^/]+/storage/?$` | 127.0.0.1:9383 |
| 129 | `GET ~ ^/api/v1/admin/users/[^/]+/summary/?$` | 127.0.0.1:9383 |
| 130 | `GET ~ ^/api/v1/admin/users/activity/?$` | 127.0.0.1:9383 |
| 131 | `GET ~ ^/api/v1/agents/[^/]+/webhook/logs/?$` | 127.0.0.1:9384 |
| 132 | `GET ~ ^/api/v1/agents/[^/]+/webhook/test/?$` | 127.0.0.1:9384 |
| 133 | `GET ~ ^/api/v1/auth/oauth/[^/]+/callback/?$` | 127.0.0.1:9384 |
| 134 | `GET ~ ^/api/v1/datasets/[^/]+/documents/[^/]+/?$` | 127.0.0.1:9384 |
| 135 | `GET ~ ^/api/v1/datasets/[^/]+/ingestions/?$` | 127.0.0.1:9384 |
| 136 | `GET ~ ^/api/v1/providers/[^/]+/instances/?$` | 127.0.0.1:9384 |
| 137 | `HEAD ~ ^/api/v1/agents/[^/]+/webhook/test/?$` | 127.0.0.1:9384 |
| 138 | `PATCH ~ ^/api/v1/agents/[^/]+/webhook/test/?$` | 127.0.0.1:9384 |
| 139 | `PATCH ~ ^/api/v1/datasets/[^/]+/documents/[^/]+/?$` | 127.0.0.1:9384 |
| 140 | `POST ~ ^/api/v1/admin/license/config/?$` | 127.0.0.1:9383 |
| 141 | `POST ~ ^/api/v1/admin/queue/messages/?$` | 127.0.0.1:9383 |
| 142 | `POST ~ ^/api/v1/admin/system/license/?$` | 127.0.0.1:9383 |
| 143 | `POST ~ ^/api/v1/agents/[^/]+/webhook/test/?$` | 127.0.0.1:9384 |
| 144 | `POST ~ ^/api/v1/audio/transcriptions/?$` | 127.0.0.1:9384 |
| 145 | `POST ~ ^/api/v1/document/delete_meta/?$` | 127.0.0.1:9384 |
| 146 | `POST ~ ^/api/v1/providers/[^/]+/instances/?$` | 127.0.0.1:9384 |
| 147 | `POST ~ ^/api/v1/searches/[^/]+/completion/?$` | 127.0.0.1:9384 |
| 148 | `PUT ~ ^/api/v1/admin/queue/messages/?$` | 127.0.0.1:9383 |
| 149 | `PUT ~ ^/api/v1/agents/[^/]+/webhook/test/?$` | 127.0.0.1:9384 |
| 150 | `DELETE ~ ^/api/v1/datasets/[^/]+/documents/?$` | 127.0.0.1:9384 |
| 151 | `GET ~ ^/api/v1/admin/users/[^/]+/agents/?$` | 127.0.0.1:9383 |
| 152 | `GET ~ ^/api/v1/admin/users/[^/]+/models/?$` | 127.0.0.1:9383 |
| 153 | `GET ~ ^/api/v1/admin/users/[^/]+/tokens/?$` | 127.0.0.1:9383 |
| 154 | `GET ~ ^/api/v1/admin/users/reports/?$` | 127.0.0.1:9383 |
| 155 | `GET ~ ^/api/v1/admin/users/storage/?$` | 127.0.0.1:9383 |
| 156 | `GET ~ ^/api/v1/admin/users/summary/?$` | 127.0.0.1:9383 |
| 157 | `GET ~ ^/api/v1/auth/azure/callback/?$` | 127.0.0.1:9384 |
| 158 | `GET ~ ^/api/v1/auth/login/channels/?$` | 127.0.0.1:9384 |
| 159 | `GET ~ ^/api/v1/auth/oauth/callback/?$` | 127.0.0.1:9384 |
| 160 | `GET ~ ^/api/v1/datasets/[^/]+/documents/?$` | 127.0.0.1:9384 |
| 161 | `GET ~ ^/api/v1/documents/artifact/[^/]+/?$` | 127.0.0.1:9384 |
| 162 | `GET ~ ^/api/v1/system/environments/?$` | 127.0.0.1:9384 |
| 163 | `GET ~ ^/api/v1/workspace/[^/]+/commits/[^/]+/?$` | 127.0.0.1:9384 |
| 164 | `POST ~ ^/api/v1/admin/users/[^/]+/tokens/?$` | 127.0.0.1:9383 |
| 165 | `POST ~ ^/api/v1/connectors/[^/]+/rebuild/?$` | 127.0.0.1:9384 |
| 166 | `POST ~ ^/api/v1/datasets/[^/]+/documents/?$` | 127.0.0.1:9384 |
| 167 | `POST ~ ^/api/v1/datasets/[^/]+/embedding/?$` | 127.0.0.1:9384 |
| 168 | `DELETE ~ ^/api/v1/admin/users/[^/]+/admin/?$` | 127.0.0.1:9383 |
| 169 | `DELETE ~ ^/api/v1/admin/users/[^/]+/keys/[^/]+/?$` | 127.0.0.1:9383 |
| 170 | `DELETE ~ ^/api/v1/tenant/chunk_store/?$` | 127.0.0.1:9384 |
| 171 | `GET ~ ^/api/v1/admin/data/storage/?$` | 127.0.0.1:9383 |
| 172 | `GET ~ ^/api/v1/admin/data/summary/?$` | 127.0.0.1:9383 |
| 173 | `GET ~ ^/api/v1/admin/environments/?$` | 127.0.0.1:9383 |
| 174 | `GET ~ ^/api/v1/admin/users/[^/]+/chats/?$` | 127.0.0.1:9383 |
| 175 | `GET ~ ^/api/v1/admin/users/[^/]+/files/?$` | 127.0.0.1:9383 |
| 176 | `GET ~ ^/api/v1/admin/users/[^/]+/index/?$` | 127.0.0.1:9383 |
| 177 | `GET ~ ^/api/v1/admin/users/[^/]+/quota/?$` | 127.0.0.1:9383 |
| 178 | `GET ~ ^/api/v1/auth/icbc/callback/?$` | 127.0.0.1:9384 |
| 179 | `GET ~ ^/api/v1/datasets/[^/]+/commits/[^/]+/?$` | 127.0.0.1:9384 |
| 180 | `GET ~ ^/api/v1/documents/[^/]+/preview/?$` | 127.0.0.1:9384 |
| 181 | `GET ~ ^/api/v1/providers/[^/]+/models/[^/]+/?$` | 127.0.0.1:9384 |
| 182 | `GET ~ ^/api/v1/workspace/[^/]+/changes/?$` | 127.0.0.1:9384 |
| 183 | `GET ~ ^/api/v1/workspace/[^/]+/commits/?$` | 127.0.0.1:9384 |
| 184 | `GET ~ ^/v1/file/all_parent_folder/?$` | 127.0.0.1:9384 |
| 185 | `POST ~ ^/api/v1/tenant/chunk_store/?$` | 127.0.0.1:9384 |
| 186 | `POST ~ ^/api/v1/workspace/[^/]+/commits/?$` | 127.0.0.1:9384 |
| 187 | `PUT ~ ^/api/v1/admin/users/[^/]+/admin/?$` | 127.0.0.1:9383 |
| 188 | `DELETE ~ ^/api/v1/admin/data/orphan/?$` | 127.0.0.1:9383 |
| 189 | `DELETE ~ ^/api/v1/admin/users/[^/]+/data/?$` | 127.0.0.1:9383 |
| 190 | `DELETE ~ ^/api/v1/agents/[^/]+/sessions/[^/]+/?$` | 127.0.0.1:9384 |
| 191 | `DELETE ~ ^/api/v1/agents/[^/]+/versions/[^/]+/?$` | 127.0.0.1:9384 |
| 192 | `GET ~ ^/api/v1/admin/all-models/[^/]+/?$` | 127.0.0.1:9383 |
| 193 | `GET ~ ^/api/v1/admin/data/orphan/?$` | 127.0.0.1:9383 |
| 194 | `GET ~ ^/api/v1/admin/fingerprint/?$` | 127.0.0.1:9383 |
| 195 | `GET ~ ^/api/v1/admin/users/[^/]+/keys/?$` | 127.0.0.1:9383 |
| 196 | `GET ~ ^/api/v1/admin/users/index/?$` | 127.0.0.1:9383 |
| 197 | `GET ~ ^/api/v1/admin/users/quota/?$` | 127.0.0.1:9383 |
| 198 | `GET ~ ^/api/v1/agents/[^/]+/sessions/[^/]+/?$` | 127.0.0.1:9384 |
| 199 | `GET ~ ^/api/v1/agents/[^/]+/versions/[^/]+/?$` | 127.0.0.1:9384 |
| 200 | `GET ~ ^/api/v1/datasets/[^/]+/changes/?$` | 127.0.0.1:9384 |
| 201 | `GET ~ ^/api/v1/datasets/[^/]+/commits/?$` | 127.0.0.1:9384 |
| 202 | `GET ~ ^/api/v1/documents/images/[^/]+/?$` | 127.0.0.1:9384 |
| 203 | `GET ~ ^/api/v1/folders/[^/]+/commits/[^/]+/?$` | 127.0.0.1:9384 |
| 204 | `GET ~ ^/api/v1/messages/[^/]+/content/?$` | 127.0.0.1:9384 |
| 205 | `GET ~ ^/api/v1/providers/[^/]+/models/?$` | 127.0.0.1:9384 |
| 206 | `GET ~ ^/api/v1/system/variables/[^/]+/?$` | 127.0.0.1:9384 |
| 207 | `POST ~ ^/api/v1/admin/users/[^/]+/keys/?$` | 127.0.0.1:9383 |
| 208 | `POST ~ ^/api/v1/auth/register/otp/?$` | 127.0.0.1:9384 |
| 209 | `POST ~ ^/api/v1/datasets/[^/]+/commits/?$` | 127.0.0.1:9384 |
| 210 | `POST ~ ^/api/v1/document/set_meta/?$` | 127.0.0.1:9384 |
| 211 | `POST ~ ^/api/v1/mcp/servers/[^/]+/test/?$` | 127.0.0.1:9384 |
| 212 | `POST ~ ^/v1/user/setting/password/?$` | 127.0.0.1:9384 |
| 213 | `PUT ~ ^/api/v1/admin/users/[^/]+/role/?$` | 127.0.0.1:9383 |
| 214 | `DELETE ~ ^/api/v1/admin/users/data/?$` | 127.0.0.1:9383 |
| 215 | `DELETE ~ ^/api/v1/agents/[^/]+/sessions/?$` | 127.0.0.1:9384 |
| 216 | `DELETE ~ ^/api/v1/datasets/[^/]+/chunks/?$` | 127.0.0.1:9384 |
| 217 | `GET ~ ^/api/v1/admin/all-models/?$` | 127.0.0.1:9383 |
| 218 | `GET ~ ^/api/v1/admin/config/log/?$` | 127.0.0.1:9383 |
| 219 | `GET ~ ^/api/v1/admin/data/index/?$` | 127.0.0.1:9383 |
| 220 | `GET ~ ^/api/v1/admin/providers/[^/]+/?$` | 127.0.0.1:9383 |
| 221 | `GET ~ ^/api/v1/admin/variables/[^/]+/?$` | 127.0.0.1:9383 |
| 222 | `GET ~ ^/api/v1/agents/[^/]+/sessions/?$` | 127.0.0.1:9384 |
| 223 | `GET ~ ^/api/v1/agents/[^/]+/versions/?$` | 127.0.0.1:9384 |
| 224 | `GET ~ ^/api/v1/agents/templates/?$` | 127.0.0.1:9384 |
| 225 | `GET ~ ^/api/v1/auth/azure/login/?$` | 127.0.0.1:9384 |
| 226 | `GET ~ ^/api/v1/chats/[^/]+/sessions/[^/]+/?$` | 127.0.0.1:9384 |
| 227 | `GET ~ ^/api/v1/connectors/[^/]+/logs/?$` | 127.0.0.1:9384 |
| 228 | `GET ~ ^/api/v1/files/[^/]+/ancestors/?$` | 127.0.0.1:9384 |
| 229 | `GET ~ ^/api/v1/folders/[^/]+/changes/?$` | 127.0.0.1:9384 |
| 230 | `GET ~ ^/api/v1/folders/[^/]+/commits/?$` | 127.0.0.1:9384 |
| 231 | `GET ~ ^/api/v1/memories/[^/]+/config/?$` | 127.0.0.1:9384 |
| 232 | `GET ~ ^/api/v1/system/variables/?$` | 127.0.0.1:9384 |
| 233 | `PATCH ~ ^/api/v1/chats/[^/]+/sessions/[^/]+/?$` | 127.0.0.1:9384 |
| 234 | `POST ~ ^/api/v1/agents/[^/]+/sessions/?$` | 127.0.0.1:9384 |
| 235 | `POST ~ ^/api/v1/chat/completions/?$` | 127.0.0.1:9384 |
| 236 | `POST ~ ^/api/v1/connectors/[^/]+/test/?$` | 127.0.0.1:9384 |
| 237 | `POST ~ ^/api/v1/datasets/[^/]+/chunks/?$` | 127.0.0.1:9384 |
| 238 | `POST ~ ^/api/v1/datasets/[^/]+/search/?$` | 127.0.0.1:9384 |
| 239 | `POST ~ ^/api/v1/folders/[^/]+/commits/?$` | 127.0.0.1:9384 |
| 240 | `POST ~ ^/v1/user/set_tenant_info/?$` | 127.0.0.1:9384 |
| 241 | `PUT ~ ^/api/v1/admin/config/log/?$` | 127.0.0.1:9383 |
| 242 | `PUT ~ ^/api/v1/system/variables/?$` | 127.0.0.1:9384 |
| 243 | `DELETE ~ ^/api/v1/admin/ingestors/?$` | 127.0.0.1:9383 |
| 244 | `DELETE ~ ^/api/v1/admin/providers/?$` | 127.0.0.1:9383 |
| 245 | `DELETE ~ ^/api/v1/admin/services/[^/]+/?$` | 127.0.0.1:9383 |
| 246 | `DELETE ~ ^/api/v1/agents/[^/]+/webhook/?$` | 127.0.0.1:9384 |
| 247 | `DELETE ~ ^/api/v1/chats/[^/]+/sessions/?$` | 127.0.0.1:9384 |
| 248 | `DELETE ~ ^/api/v1/datasets/[^/]+/index/?$` | 127.0.0.1:9384 |
| 249 | `GET ~ ^/api/v1/admin/ingestors/?$` | 127.0.0.1:9383 |
| 250 | `GET ~ ^/api/v1/admin/providers/?$` | 127.0.0.1:9383 |
| 251 | `GET ~ ^/api/v1/admin/services/[^/]+/?$` | 127.0.0.1:9383 |
| 252 | `GET ~ ^/api/v1/agents/[^/]+/webhook/?$` | 127.0.0.1:9384 |
| 253 | `GET ~ ^/api/v1/agents/download/?$` | 127.0.0.1:9384 |
| 254 | `GET ~ ^/api/v1/chats/[^/]+/sessions/?$` | 127.0.0.1:9384 |
| 255 | `GET ~ ^/api/v1/datasets/[^/]+/graph/?$` | 127.0.0.1:9384 |
| 256 | `GET ~ ^/api/v1/datasets/[^/]+/index/?$` | 127.0.0.1:9384 |
| 257 | `GET ~ ^/api/v1/files/[^/]+/versions/?$` | 127.0.0.1:9384 |
| 258 | `HEAD ~ ^/api/v1/agents/[^/]+/webhook/?$` | 127.0.0.1:9384 |
| 259 | `PATCH ~ ^/api/v1/agents/[^/]+/webhook/?$` | 127.0.0.1:9384 |
| 260 | `POST ~ ^/api/v1/admin/providers/?$` | 127.0.0.1:9383 |
| 261 | `POST ~ ^/api/v1/admin/services/[^/]+/?$` | 127.0.0.1:9383 |
| 262 | `POST ~ ^/api/v1/agents/[^/]+/publish/?$` | 127.0.0.1:9384 |
| 263 | `POST ~ ^/api/v1/agents/[^/]+/webhook/?$` | 127.0.0.1:9384 |
| 264 | `POST ~ ^/api/v1/chats/[^/]+/sessions/?$` | 127.0.0.1:9384 |
| 265 | `POST ~ ^/api/v1/datasets/[^/]+/index/?$` | 127.0.0.1:9384 |
| 266 | `POST ~ ^/api/v1/datasets/search/?$` | 127.0.0.1:9384 |
| 267 | `PUT ~ ^/api/v1/admin/services/[^/]+/?$` | 127.0.0.1:9383 |
| 268 | `PUT ~ ^/api/v1/agents/[^/]+/webhook/?$` | 127.0.0.1:9384 |
| 269 | `DELETE ~ ^/api/v1/chat-channels/[^/]+/?$` | 127.0.0.1:9384 |
| 270 | `DELETE ~ ^/api/v1/datasets/[^/]+/tags/?$` | 127.0.0.1:9384 |
| 271 | `DELETE ~ ^/api/v1/skills/spaces/[^/]+/?$` | 127.0.0.1:9384 |
| 272 | `DELETE ~ ^/api/v1/system/tokens/[^/]+/?$` | 127.0.0.1:9384 |
| 273 | `DELETE ~ ^/api/v1/tenants/[^/]+/users/?$` | 127.0.0.1:9384 |
| 274 | `GET ~ ^/api/v1/admin/services/?$` | 127.0.0.1:9383 |
| 275 | `GET ~ ^/api/v1/agents/prompts/?$` | 127.0.0.1:9384 |
| 276 | `GET ~ ^/api/v1/chat-channels/[^/]+/?$` | 127.0.0.1:9384 |
| 277 | `GET ~ ^/api/v1/datasets/[^/]+/tags/?$` | 127.0.0.1:9384 |
| 278 | `GET ~ ^/api/v1/skills/spaces/[^/]+/?$` | 127.0.0.1:9384 |
| 279 | `GET ~ ^/api/v1/system/configs/?$` | 127.0.0.1:9384 |
| 280 | `GET ~ ^/api/v1/tenants/[^/]+/users/?$` | 127.0.0.1:9384 |
| 281 | `GET ~ ^/v1/file/parent_folder/?$` | 127.0.0.1:9384 |
| 282 | `PATCH ~ ^/api/v1/chat-channels/[^/]+/?$` | 127.0.0.1:9384 |
| 283 | `POST ~ ^/api/v1/agents/[^/]+/upload/?$` | 127.0.0.1:9384 |
| 284 | `POST ~ ^/api/v1/skills/reindex/?$` | 127.0.0.1:9384 |
| 285 | `POST ~ ^/api/v1/tenants/[^/]+/users/?$` | 127.0.0.1:9384 |
| 286 | `POST ~ ^/v1/connector/[^/]+/rebuild/?$` | 127.0.0.1:9384 |
| 287 | `PUT ~ ^/api/v1/datasets/[^/]+/tags/?$` | 127.0.0.1:9384 |
| 288 | `PUT ~ ^/api/v1/skills/spaces/[^/]+/?$` | 127.0.0.1:9384 |
| 289 | `GET ~ ^/api/v1/admin/configs/?$` | 127.0.0.1:9383 |
| 290 | `GET ~ ^/api/v1/admin/license/?$` | 127.0.0.1:9383 |
| 291 | `GET ~ ^/api/v1/admin/version/?$` | 127.0.0.1:9383 |
| 292 | `GET ~ ^/api/v1/agents/[^/]+/logs/[^/]+/?$` | 127.0.0.1:9384 |
| 293 | `GET ~ ^/api/v1/files/[^/]+/parent/?$` | 127.0.0.1:9384 |
| 294 | `GET ~ ^/api/v1/skills/config/?$` | 127.0.0.1:9384 |
| 295 | `GET ~ ^/api/v1/skills/spaces/?$` | 127.0.0.1:9384 |
| 296 | `POST ~ ^/api/v1/admin/license/?$` | 127.0.0.1:9383 |
| 297 | `POST ~ ^/api/v1/admin/reports/?$` | 127.0.0.1:9383 |
| 298 | `POST ~ ^/api/v1/agents/[^/]+/reset/?$` | 127.0.0.1:9384 |
| 299 | `POST ~ ^/api/v1/chat/to_model/?$` | 127.0.0.1:9384 |
| 300 | `POST ~ ^/api/v1/document/list/?$` | 127.0.0.1:9384 |
| 301 | `POST ~ ^/api/v1/skills/config/?$` | 127.0.0.1:9384 |
| 302 | `POST ~ ^/api/v1/skills/search/?$` | 127.0.0.1:9384 |
| 303 | `POST ~ ^/api/v1/skills/spaces/?$` | 127.0.0.1:9384 |
| 304 | `DELETE ~ ^/api/v1/admin/roles/[^/]+/?$` | 127.0.0.1:9383 |
| 305 | `DELETE ~ ^/api/v1/admin/users/[^/]+/?$` | 127.0.0.1:9383 |
| 306 | `DELETE ~ ^/api/v1/mcp/servers/[^/]+/?$` | 127.0.0.1:9384 |
| 307 | `DELETE ~ ^/api/v1/skills/index/?$` | 127.0.0.1:9384 |
| 308 | `DELETE ~ ^/api/v1/system/keys/[^/]+/?$` | 127.0.0.1:9384 |
| 309 | `GET ~ ^/api/v1/admin/roles/[^/]+/?$` | 127.0.0.1:9383 |
| 310 | `GET ~ ^/api/v1/admin/users/[^/]+/?$` | 127.0.0.1:9383 |
| 311 | `GET ~ ^/api/v1/mcp/servers/[^/]+/?$` | 127.0.0.1:9384 |
| 312 | `GET ~ ^/v1/file/root_folder/?$` | 127.0.0.1:9384 |
| 313 | `GET ~ ^/v1/user/tenant_info/?$` | 127.0.0.1:9384 |
| 314 | `POST ~ ^/api/v1/admin/logout/?$` | 127.0.0.1:9383 |
| 315 | `POST ~ ^/api/v1/audio/speech/?$` | 127.0.0.1:9384 |
| 316 | `POST ~ ^/api/v1/chunk/update/?$` | 127.0.0.1:9384 |
| 317 | `POST ~ ^/api/v1/skills/index/?$` | 127.0.0.1:9384 |
| 318 | `PUT ~ ^/api/v1/admin/roles/[^/]+/?$` | 127.0.0.1:9383 |
| 319 | `PUT ~ ^/api/v1/agents/[^/]+/tags/?$` | 127.0.0.1:9384 |
| 320 | `PUT ~ ^/api/v1/mcp/servers/[^/]+/?$` | 127.0.0.1:9384 |
| 321 | `DELETE ~ ^/api/v1/agents/[^/]+/run/?$` | 127.0.0.1:9384 |
| 322 | `DELETE ~ ^/api/v1/connectors/[^/]+/?$` | 127.0.0.1:9384 |
| 323 | `GET ~ ^/api/v1/admin/queue/?$` | 127.0.0.1:9383 |
| 324 | `GET ~ ^/api/v1/admin/users/?$` | 127.0.0.1:9383 |
| 325 | `GET ~ ^/api/v1/agents/tags/?$` | 127.0.0.1:9384 |
| 326 | `GET ~ ^/api/v1/all-models/[^/]+/?$` | 127.0.0.1:9384 |
| 327 | `GET ~ ^/api/v1/auth/login/[^/]+/?$` | 127.0.0.1:9384 |
| 328 | `GET ~ ^/api/v1/connectors/[^/]+/?$` | 127.0.0.1:9384 |
| 329 | `GET ~ ^/api/v1/system/keys/?$` | 127.0.0.1:9384 |
| 330 | `GET ~ ^/api/v1/tenant/list/?$` | 127.0.0.1:9384 |
| 331 | `PATCH ~ ^/api/v1/connectors/[^/]+/?$` | 127.0.0.1:9384 |
| 332 | `POST ~ ^/api/v1/admin/login/?$` | 127.0.0.1:9383 |
| 333 | `POST ~ ^/api/v1/admin/users/?$` | 127.0.0.1:9383 |
| 334 | `POST ~ ^/api/v1/agents/[^/]+/run/?$` | 127.0.0.1:9384 |
| 335 | `POST ~ ^/api/v1/system/keys/?$` | 127.0.0.1:9384 |
| 336 | `DELETE ~ ^/api/v1/datasets/[^/]+/[^/]+/?$` | 127.0.0.1:9384 |
| 337 | `DELETE ~ ^/api/v1/documents/[^/]+/?$` | 127.0.0.1:9384 |
| 338 | `DELETE ~ ^/api/v1/providers/[^/]+/?$` | 127.0.0.1:9384 |
| 339 | `GET ~ ^/api/v1/admin/auth/?$` | 127.0.0.1:9383 |
| 340 | `GET ~ ^/api/v1/admin/ping/?$` | 127.0.0.1:9383 |
| 341 | `GET ~ ^/api/v1/all-models/?$` | 127.0.0.1:9384 |
| 342 | `GET ~ ^/api/v1/components/?$` | 127.0.0.1:9384 |
| 343 | `GET ~ ^/api/v1/documents/[^/]+/?$` | 127.0.0.1:9384 |
| 344 | `GET ~ ^/api/v1/pipelines/[^/]+/?$` | 127.0.0.1:9384 |
| 345 | `GET ~ ^/api/v1/providers/[^/]+/?$` | 127.0.0.1:9384 |
| 346 | `GET ~ ^/v1/connector/list/?$` | 127.0.0.1:9384 |
| 347 | `GET ~ ^/v1/system/configs/?$` | 127.0.0.1:9384 |
| 348 | `POST ~ ^/api/v1/chunk/list/?$` | 127.0.0.1:9384 |
| 349 | `POST ~ ^/api/v1/embeddings/?$` | 127.0.0.1:9384 |
| 350 | `POST ~ ^/api/v1/file/parse/?$` | 127.0.0.1:9384 |
| 351 | `PUT ~ ^/api/v1/documents/[^/]+/?$` | 127.0.0.1:9384 |
| 352 | `DELETE ~ ^/api/v1/memories/[^/]+/?$` | 127.0.0.1:9384 |
| 353 | `DELETE ~ ^/api/v1/messages/[^/]+/?$` | 127.0.0.1:9384 |
| 354 | `DELETE ~ ^/api/v1/searches/[^/]+/?$` | 127.0.0.1:9384 |
| 355 | `GET ~ ^/api/v1/datasets/[^/]+/?$` | 127.0.0.1:9384 |
| 356 | `GET ~ ^/api/v1/documents/?$` | 127.0.0.1:9384 |
| 357 | `GET ~ ^/api/v1/memories/[^/]+/?$` | 127.0.0.1:9384 |
| 358 | `GET ~ ^/api/v1/pipelines/?$` | 127.0.0.1:9384 |
| 359 | `GET ~ ^/api/v1/searches/[^/]+/?$` | 127.0.0.1:9384 |
| 360 | `POST ~ ^/api/v1/documents/?$` | 127.0.0.1:9384 |
| 361 | `PUT ~ ^/api/v1/datasets/[^/]+/?$` | 127.0.0.1:9384 |
| 362 | `PUT ~ ^/api/v1/memories/[^/]+/?$` | 127.0.0.1:9384 |
| 363 | `PUT ~ ^/api/v1/messages/[^/]+/?$` | 127.0.0.1:9384 |
| 364 | `PUT ~ ^/api/v1/searches/[^/]+/?$` | 127.0.0.1:9384 |
| 365 | `PATCH ~ ^/api/v1/tenants/[^/]+/?$` | 127.0.0.1:9384 |
| 366 | `POST ~ ^/api/v1/file/ocr/?$` | 127.0.0.1:9384 |
| 367 | `POST ~ ^/v1/user/setting/?$` | 127.0.0.1:9384 |
| 368 | `DELETE ~ ^/api/v1/agents/[^/]+/?$` | 127.0.0.1:9384 |
| 369 | `GET ~ ^/api/v1/agents/[^/]+/?$` | 127.0.0.1:9384 |
| 370 | `GET ~ ^/v1/tenant/list/?$` | 127.0.0.1:9384 |
| 371 | `GET ~ ^/v1/user/logout/?$` | 127.0.0.1:9384 |
| 372 | `PUT ~ ^/api/v1/agents/[^/]+/?$` | 127.0.0.1:9384 |
| 373 | `DELETE ~ ^/api/v1/chats/[^/]+/?$` | 127.0.0.1:9384 |
| 374 | `GET ~ ^/api/v1/chats/[^/]+/?$` | 127.0.0.1:9384 |
| 375 | `GET ~ ^/api/v1/files/[^/]+/?$` | 127.0.0.1:9384 |
| 376 | `GET ~ ^/v1/connector/[^/]+/?$` | 127.0.0.1:9384 |
| 377 | `PATCH ~ ^/api/v1/chats/[^/]+/?$` | 127.0.0.1:9384 |
| 378 | `PATCH ~ ^/api/v1/models/?$` | 127.0.0.1:9384 |
| 379 | `POST ~ ^/api/v1/rerank/?$` | 127.0.0.1:9384 |
| 380 | `PUT ~ ^/api/v1/chats/[^/]+/?$` | 127.0.0.1:9384 |
| 381 | `GET ~ ^/v1/user/info/?$` | 127.0.0.1:9384 |
| 382 | `POST ~ ^/api/v1/mcp/?$` | 127.0.0.1:9384 |
| 383 | `GET ~ ^/health/?$` | 127.0.0.1:9384 |
| 384 | `~ ^/api/v1/admin(?:/|$)` | 127.0.0.1:9381 |
| 385 | `~ ^/(?:v1|api)(?:/|$)` | 127.0.0.1:9380 |

## Routes

| Method | Path | Service | Port | Proxy mode | Proxy destination | Auth / role | Runtime | Source | Alternates | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GET | `/api/v1/admin/all-models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:201` | — | — |
| GET | `/api/v1/admin/all-models/:model_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:202` | — | — |
| GET | `/api/v1/admin/auth` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:60` | python-admin@9381 (`admin/server/routes.py:67`) | — |
| GET | `/api/v1/admin/config/log` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:88` | — | — |
| PUT | `/api/v1/admin/config/log` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:89` | — | — |
| GET | `/api/v1/admin/configs` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:86` | python-admin@9381 (`admin/server/routes.py:463`) | — |
| GET | `/api/v1/admin/data/index` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:147` | — | — |
| DELETE | `/api/v1/admin/data/orphan` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:148` | — | — |
| GET | `/api/v1/admin/data/orphan` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:145` | — | — |
| GET | `/api/v1/admin/data/storage` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:146` | — | — |
| GET | `/api/v1/admin/data/summary` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:144` | — | — |
| GET | `/api/v1/admin/environments` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:92` | python-admin@9381 (`admin/server/routes.py:476`) | — |
| GET | `/api/v1/admin/fingerprint` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:211` | — | — |
| DELETE | `/api/v1/admin/ingestion/tasks` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:218` | — | — |
| GET | `/api/v1/admin/ingestion/tasks` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:220` | — | — |
| PUT | `/api/v1/admin/ingestion/tasks` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:219` | — | — |
| GET | `/api/v1/admin/ingestion/tasks/summary` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:143` | — | — |
| DELETE | `/api/v1/admin/ingestors` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:106` | — | — |
| GET | `/api/v1/admin/ingestors` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:105` | — | — |
| GET | `/api/v1/admin/license` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:215` | — | — |
| POST | `/api/v1/admin/license` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:213` | — | — |
| POST | `/api/v1/admin/license/config` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:214` | — | — |
| GET | `/api/v1/admin/log_levels` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:658` | — | — |
| PUT | `/api/v1/admin/log_levels` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:670` | — | — |
| POST | `/api/v1/admin/login` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | public | enabled | `internal/admin/router.go:45` | python-admin@9381 (`admin/server/routes.py:43`) | — |
| GET | `/api/v1/admin/logout` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:55` | — | — |
| POST | `/api/v1/admin/logout` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:58` | — | — |
| GET | `/api/v1/admin/ping` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | public | enabled | `internal/admin/router.go:44` | python-admin@9381 (`admin/server/routes.py:38`) | — |
| DELETE | `/api/v1/admin/providers` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:182` | — | — |
| GET | `/api/v1/admin/providers` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:179` | — | — |
| POST | `/api/v1/admin/providers` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:180` | — | — |
| GET | `/api/v1/admin/providers/:provider_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:181` | — | — |
| POST | `/api/v1/admin/providers/:provider_name/connection` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:192` | — | — |
| DELETE | `/api/v1/admin/providers/:provider_name/instances` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:188` | — | — |
| GET | `/api/v1/admin/providers/:provider_name/instances` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:187` | — | — |
| POST | `/api/v1/admin/providers/:provider_name/instances` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:186` | — | — |
| GET | `/api/v1/admin/providers/:provider_name/instances/:instance_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:189` | — | — |
| PUT | `/api/v1/admin/providers/:provider_name/instances/:instance_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:193` | — | — |
| GET | `/api/v1/admin/providers/:provider_name/instances/:instance_name/balance` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:190` | — | — |
| GET | `/api/v1/admin/providers/:provider_name/instances/:instance_name/connection` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:191` | — | — |
| DELETE | `/api/v1/admin/providers/:provider_name/instances/:instance_name/models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:198` | — | — |
| GET | `/api/v1/admin/providers/:provider_name/instances/:instance_name/models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:195` | — | — |
| POST | `/api/v1/admin/providers/:provider_name/instances/:instance_name/models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:197` | — | — |
| PATCH | `/api/v1/admin/providers/:provider_name/instances/:instance_name/models/*model_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:196` | — | — |
| GET | `/api/v1/admin/providers/:provider_name/models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:183` | — | — |
| GET | `/api/v1/admin/providers/:provider_name/models/:model_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:184` | — | — |
| GET | `/api/v1/admin/queue` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:99` | — | — |
| GET | `/api/v1/admin/queue/messages` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:101` | — | — |
| POST | `/api/v1/admin/queue/messages` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:100` | — | — |
| PUT | `/api/v1/admin/queue/messages` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:102` | — | — |
| POST | `/api/v1/admin/reports` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | public | enabled | `internal/admin/router.go:47` | — | — |
| GET | `/api/v1/admin/roles` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:338` | go-admin@9383 (`internal/admin/router.go:163`) | — |
| POST | `/api/v1/admin/roles` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:296` | go-admin@9383 (`internal/admin/router.go:164`) | — |
| DELETE | `/api/v1/admin/roles/:role_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:167` | — | — |
| GET | `/api/v1/admin/roles/:role_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:165` | — | — |
| PUT | `/api/v1/admin/roles/:role_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:166` | — | — |
| DELETE | `/api/v1/admin/roles/:role_name/default-models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:174` | — | — |
| GET | `/api/v1/admin/roles/:role_name/default-models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:172` | — | — |
| PATCH | `/api/v1/admin/roles/:role_name/default-models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:173` | — | — |
| DELETE | `/api/v1/admin/roles/:role_name/permission` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:170` | — | — |
| GET | `/api/v1/admin/roles/:role_name/permission` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:168` | — | — |
| POST | `/api/v1/admin/roles/:role_name/permission` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:169` | — | — |
| DELETE | `/api/v1/admin/roles/<role_name>` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:327` | — | — |
| PUT | `/api/v1/admin/roles/<role_name>` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:312` | — | — |
| DELETE | `/api/v1/admin/roles/<role_name>/permission` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:376` | — | — |
| GET | `/api/v1/admin/roles/<role_name>/permission` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:349` | — | — |
| POST | `/api/v1/admin/roles/<role_name>/permission` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:360` | — | — |
| GET | `/api/v1/admin/roles/resource` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:171` | — | — |
| GET | `/api/v1/admin/sandbox/config` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:589` | go-admin@9383 (`internal/admin/router.go:111`) | — |
| POST | `/api/v1/admin/sandbox/config` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:603` | go-admin@9383 (`internal/admin/router.go:112`) | — |
| GET | `/api/v1/admin/sandbox/providers` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:561` | go-admin@9383 (`internal/admin/router.go:109`) | — |
| GET | `/api/v1/admin/sandbox/providers/:provider_id/schema` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:110` | — | — |
| GET | `/api/v1/admin/sandbox/providers/<provider_id>/schema` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:575` | — | — |
| POST | `/api/v1/admin/sandbox/test` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:635` | go-admin@9383 (`internal/admin/router.go:113`) | — |
| GET | `/api/v1/admin/service_types/:service_type` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:74` | — | — |
| GET | `/api/v1/admin/service_types/<service_type>` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:252` | — | — |
| GET | `/api/v1/admin/services` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:73` | python-admin@9381 (`admin/server/routes.py:241`) | — |
| DELETE | `/api/v1/admin/services/:service_id` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:76` | — | — |
| GET | `/api/v1/admin/services/:service_id` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:75` | — | — |
| POST | `/api/v1/admin/services/:service_id` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:78` | — | — |
| PUT | `/api/v1/admin/services/:service_id` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:77` | — | — |
| DELETE | `/api/v1/admin/services/<service_id>` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:274` | — | — |
| GET | `/api/v1/admin/services/<service_id>` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:263` | — | — |
| PUT | `/api/v1/admin/services/<service_id>` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:285` | — | — |
| GET | `/api/v1/admin/system/fingerprint` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:205` | — | — |
| GET | `/api/v1/admin/system/license` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:207` | — | — |
| POST | `/api/v1/admin/system/license` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:206` | — | — |
| PUT | `/api/v1/admin/system/license/config` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:208` | — | — |
| GET | `/api/v1/admin/users` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:63` | python-admin@9381 (`admin/server/routes.py:76`) | — |
| POST | `/api/v1/admin/users` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:64` | python-admin@9381 (`admin/server/routes.py:87`) | — |
| DELETE | `/api/v1/admin/users/:username` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:66` | — | — |
| GET | `/api/v1/admin/users/:username` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:65` | — | — |
| PUT | `/api/v1/admin/users/:username/activate` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:68` | — | — |
| GET | `/api/v1/admin/users/:username/activity` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:116` | — | — |
| DELETE | `/api/v1/admin/users/:username/admin` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:70` | — | — |
| PUT | `/api/v1/admin/users/:username/admin` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:69` | — | — |
| GET | `/api/v1/admin/users/:username/agents` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:125` | — | — |
| GET | `/api/v1/admin/users/:username/chats` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:126` | — | — |
| DELETE | `/api/v1/admin/users/:username/data` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:149` | — | — |
| GET | `/api/v1/admin/users/:username/dataset` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:117` | — | — |
| GET | `/api/v1/admin/users/:username/datasets` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:124` | — | — |
| GET | `/api/v1/admin/users/:username/default-models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:133` | — | — |
| GET | `/api/v1/admin/users/:username/files` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:129` | — | — |
| GET | `/api/v1/admin/users/:username/index` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:121` | — | — |
| GET | `/api/v1/admin/users/:username/keys` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:155` | — | — |
| POST | `/api/v1/admin/users/:username/keys` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:153` | — | — |
| DELETE | `/api/v1/admin/users/:username/keys/:key` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:154` | — | — |
| GET | `/api/v1/admin/users/:username/models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:128` | — | — |
| PUT | `/api/v1/admin/users/:username/password` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:67` | — | — |
| GET | `/api/v1/admin/users/:username/permission` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:123` | — | — |
| GET | `/api/v1/admin/users/:username/providers` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:130` | — | — |
| GET | `/api/v1/admin/users/:username/providers/:provider_name/instances` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:131` | — | — |
| GET | `/api/v1/admin/users/:username/providers/:provider_name/instances/:instance_name/models` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:132` | — | — |
| GET | `/api/v1/admin/users/:username/quota` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:120` | — | — |
| PUT | `/api/v1/admin/users/:username/role` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:122` | — | — |
| GET | `/api/v1/admin/users/:username/searches` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:127` | — | — |
| GET | `/api/v1/admin/users/:username/storage` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:119` | — | — |
| GET | `/api/v1/admin/users/:username/summary` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:118` | — | — |
| GET | `/api/v1/admin/users/:username/tokens` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:157` | — | — |
| POST | `/api/v1/admin/users/:username/tokens` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:159` | — | — |
| DELETE | `/api/v1/admin/users/:username/tokens/:token` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:160` | — | — |
| GET | `/api/v1/admin/users/<user_name>/permission` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:407` | — | — |
| PUT | `/api/v1/admin/users/<user_name>/role` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:392` | — | — |
| DELETE | `/api/v1/admin/users/<username>` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:114` | — | — |
| GET | `/api/v1/admin/users/<username>` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:199` | — | — |
| PUT | `/api/v1/admin/users/<username>/activate` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:150` | — | — |
| DELETE | `/api/v1/admin/users/<username>/admin` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:183` | — | — |
| PUT | `/api/v1/admin/users/<username>/admin` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:167` | — | — |
| GET | `/api/v1/admin/users/<username>/agents` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:227` | — | — |
| GET | `/api/v1/admin/users/<username>/datasets` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:213` | — | — |
| GET | `/api/v1/admin/users/<username>/keys` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:521` | — | — |
| POST | `/api/v1/admin/users/<username>/keys` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:489` | — | — |
| DELETE | `/api/v1/admin/users/<username>/keys/<key>` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:534` | — | — |
| PUT | `/api/v1/admin/users/<username>/password` | python-admin | 9381 | hybrid | 127.0.0.1:9383 | admin-session | **runtime-disabled** | `admin/server/routes.py:131` | — | — |
| GET | `/api/v1/admin/users/activity` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:135` | — | — |
| DELETE | `/api/v1/admin/users/data` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:150` | — | — |
| GET | `/api/v1/admin/users/documents` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:138` | — | — |
| GET | `/api/v1/admin/users/index` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:139` | — | — |
| GET | `/api/v1/admin/users/plan/summary` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:141` | — | — |
| GET | `/api/v1/admin/users/quota` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:140` | — | — |
| GET | `/api/v1/admin/users/quota/summary` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:142` | — | — |
| GET | `/api/v1/admin/users/reports` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:136` | — | — |
| GET | `/api/v1/admin/users/storage` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:137` | — | — |
| GET | `/api/v1/admin/users/summary` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:134` | — | — |
| GET | `/api/v1/admin/variables` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:440` | go-admin@9383 (`internal/admin/router.go:81`) | — |
| PUT | `/api/v1/admin/variables` | python-admin | 9381 | hybrid | 127.0.0.1:9381 | admin-session | enabled | `admin/server/routes.py:418` | go-admin@9383 (`internal/admin/router.go:82`) | — |
| GET | `/api/v1/admin/variables/:var_name` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:83` | — | — |
| GET | `/api/v1/admin/version` | go-admin | 9383 | hybrid | 127.0.0.1:9383 | admin-session | enabled | `internal/admin/router.go:95` | python-admin@9381 (`admin/server/routes.py:550`) | — |
| POST | `/api/v1/agentbots/<agent_id>/completions` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/bot_api.py:162` | — | — |
| GET | `/api/v1/agentbots/<agent_id>/inputs` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/bot_api.py:257` | — | — |
| GET | `/api/v1/agentbots/<shared_id>/logs/<message_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/bot_api.py:277` | — | — |
| GET | `/api/v1/agents` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/agent_api.py:674` | go-api@9384 (`internal/router/agent_routes.go:48`) | — |
| POST | `/api/v1/agents` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/agent_api.py:780` | go-api@9384 (`internal/router/agent_routes.go:49`) | — |
| POST | `/api/v1/agents_openai/<agent_id>/chat/completions` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:126` | — | backward-compat shim retained by upstream for older clients |
| DELETE | `/api/v1/agents/:canvas_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:52` | — | — |
| GET | `/api/v1/agents/:canvas_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:50` | — | — |
| PUT | `/api/v1/agents/:canvas_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:51` | — | — |
| POST | `/api/v1/agents/:canvas_id/components/:component_id/debug` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:67` | — | — |
| GET | `/api/v1/agents/:canvas_id/components/:component_id/input-form` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:66` | — | — |
| GET | `/api/v1/agents/:canvas_id/logs/:message_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:82` | — | — |
| POST | `/api/v1/agents/:canvas_id/publish` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:55` | — | — |
| POST | `/api/v1/agents/:canvas_id/reset` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:57` | — | — |
| DELETE | `/api/v1/agents/:canvas_id/run` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:54` | — | — |
| POST | `/api/v1/agents/:canvas_id/run` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:53` | — | — |
| DELETE | `/api/v1/agents/:canvas_id/sessions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:78` | — | — |
| GET | `/api/v1/agents/:canvas_id/sessions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:75` | — | — |
| POST | `/api/v1/agents/:canvas_id/sessions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:76` | — | — |
| DELETE | `/api/v1/agents/:canvas_id/sessions/:session_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:79` | — | — |
| GET | `/api/v1/agents/:canvas_id/sessions/:session_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:77` | — | — |
| PUT | `/api/v1/agents/:canvas_id/tags` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:56` | — | — |
| POST | `/api/v1/agents/:canvas_id/upload` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:63` | — | — |
| GET | `/api/v1/agents/:canvas_id/versions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:70` | — | — |
| DELETE | `/api/v1/agents/:canvas_id/versions/:version_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:72` | — | — |
| GET | `/api/v1/agents/:canvas_id/versions/:version_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:71` | — | — |
| DELETE | `/api/v1/agents/:canvas_id/webhook` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:90` | — | registerAnyMethod: six verbs share one handler |
| GET | `/api/v1/agents/:canvas_id/webhook` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:90` | — | registerAnyMethod: six verbs share one handler |
| HEAD | `/api/v1/agents/:canvas_id/webhook` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:90` | — | registerAnyMethod: six verbs share one handler |
| PATCH | `/api/v1/agents/:canvas_id/webhook` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:90` | — | registerAnyMethod: six verbs share one handler |
| POST | `/api/v1/agents/:canvas_id/webhook` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:90` | — | registerAnyMethod: six verbs share one handler |
| PUT | `/api/v1/agents/:canvas_id/webhook` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:90` | — | registerAnyMethod: six verbs share one handler |
| GET | `/api/v1/agents/:canvas_id/webhook/logs` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:83` | — | — |
| DELETE | `/api/v1/agents/:canvas_id/webhook/test` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:91` | — | registerAnyMethod: six verbs share one handler |
| GET | `/api/v1/agents/:canvas_id/webhook/test` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:91` | — | registerAnyMethod: six verbs share one handler |
| HEAD | `/api/v1/agents/:canvas_id/webhook/test` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:91` | — | registerAnyMethod: six verbs share one handler |
| PATCH | `/api/v1/agents/:canvas_id/webhook/test` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:91` | — | registerAnyMethod: six verbs share one handler |
| POST | `/api/v1/agents/:canvas_id/webhook/test` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:91` | — | registerAnyMethod: six verbs share one handler |
| PUT | `/api/v1/agents/:canvas_id/webhook/test` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:91` | — | registerAnyMethod: six verbs share one handler |
| DELETE | `/api/v1/agents/<agent_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1021` | — | — |
| GET | `/api/v1/agents/<agent_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:930` | — | — |
| PUT | `/api/v1/agents/<agent_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1030` | — | — |
| POST | `/api/v1/agents/<agent_id>/completions` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:636` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/agents/<agent_id>/components/<component_id>/debug` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:892` | — | — |
| GET | `/api/v1/agents/<agent_id>/components/<component_id>/input-form` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:875` | — | — |
| GET | `/api/v1/agents/<agent_id>/logs/<message_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1002` | — | — |
| POST | `/api/v1/agents/<agent_id>/reset` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1084` | — | — |
| DELETE | `/api/v1/agents/<agent_id>/sessions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:532` | — | — |
| GET | `/api/v1/agents/<agent_id>/sessions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:430` | — | — |
| POST | `/api/v1/agents/<agent_id>/sessions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:470` | — | — |
| DELETE | `/api/v1/agents/<agent_id>/sessions/<session_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:521` | — | — |
| GET | `/api/v1/agents/<agent_id>/sessions/<session_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:510` | — | — |
| POST | `/api/v1/agents/<agent_id>/upload` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:847` | — | — |
| GET | `/api/v1/agents/<agent_id>/versions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:973` | — | — |
| GET | `/api/v1/agents/<agent_id>/versions/<version_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:988` | — | — |
| DELETE | `/api/v1/agents/<agent_id>/webhook` | python-api | 9380 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1679` | — | — |
| GET | `/api/v1/agents/<agent_id>/webhook` | python-api | 9380 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1679` | — | — |
| HEAD | `/api/v1/agents/<agent_id>/webhook` | python-api | 9380 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1679` | — | — |
| PATCH | `/api/v1/agents/<agent_id>/webhook` | python-api | 9380 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1679` | — | — |
| POST | `/api/v1/agents/<agent_id>/webhook` | python-api | 9380 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1679` | — | — |
| PUT | `/api/v1/agents/<agent_id>/webhook` | python-api | 9380 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1679` | — | — |
| GET | `/api/v1/agents/<agent_id>/webhook/logs` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:2365` | — | — |
| DELETE | `/api/v1/agents/<agent_id>/webhook/test` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1684` | — | — |
| GET | `/api/v1/agents/<agent_id>/webhook/test` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1684` | — | — |
| HEAD | `/api/v1/agents/<agent_id>/webhook/test` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1684` | — | — |
| PATCH | `/api/v1/agents/<agent_id>/webhook/test` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1684` | — | — |
| POST | `/api/v1/agents/<agent_id>/webhook/test` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1684` | — | — |
| PUT | `/api/v1/agents/<agent_id>/webhook/test` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:1684` | — | — |
| PUT | `/api/v1/agents/<canvas_id>/tags` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:739` | — | — |
| GET | `/api/v1/agents/attachments/:attachment_id/download` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:61` | — | — |
| GET | `/api/v1/agents/attachments/:attachment_id/preview` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:62` | — | — |
| GET | `/api/v1/agents/attachments/<attachment_id>/download` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:2516` | — | — |
| GET | `/api/v1/agents/attachments/<attachment_id>/preview` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/agent_api.py:2505` | — | — |
| POST | `/api/v1/agents/chat/completions` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/agent_api.py:1294` | go-api@9384 (`internal/router/agent_routes.go:99`) | — |
| GET | `/api/v1/agents/download` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:60` | python-api@9380 (`api/apps/restful_apis/agent_api.py:584`) | — |
| GET | `/api/v1/agents/prompts` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:44` | python-api@9380 (`api/apps/restful_apis/agent_api.py:653`) | — |
| POST | `/api/v1/agents/rerun` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/agent_api.py:1115` | go-api@9384 (`internal/router/agent_routes.go:100`) | — |
| GET | `/api/v1/agents/tags` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:45` | python-api@9380 (`api/apps/restful_apis/agent_api.py:721`) | — |
| GET | `/api/v1/agents/templates` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/agent_routes.go:43` | python-api@9380 (`api/apps/restful_apis/agent_api.py:647`) | — |
| POST | `/api/v1/agents/test_db_connection` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/agent_api.py:1161` | go-api@9384 (`internal/router/agent_routes.go:101`) | — |
| GET | `/api/v1/all-models` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:550` | — | — |
| GET | `/api/v1/all-models/:model_name` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:551` | — | — |
| POST | `/api/v1/audio/speech` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:532` | — | — |
| POST | `/api/v1/audio/transcriptions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:531` | — | — |
| GET | `/api/v1/auth/azure/callback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router_ee.go:35` | — | backend worktree-only at a0e091e75051; AzureAuthCallback is CodeNotImplemented at internal/handler/user_auth_ee.go:41 |
| GET | `/api/v1/auth/azure/login` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router_ee.go:36` | — | backend worktree-only at a0e091e75051; AzureAuthLogin is CodeNotImplemented at internal/handler/user_auth_ee.go:45 |
| GET | `/api/v1/auth/icbc/callback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router_ee.go:34` | — | backend worktree-only at a0e091e75051; ICBCAuthCallback is CodeNotImplemented at internal/handler/user_auth_ee.go:37 |
| POST | `/api/v1/auth/login` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/user_api.py:61` | go-api@9384 (`internal/router/router.go:167`) | — |
| GET | `/api/v1/auth/login/:channel` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | enabled | `internal/router/router.go:173` | — | — |
| GET | `/api/v1/auth/login/<channel>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `api/apps/restful_apis/user_api.py:165` | — | — |
| GET | `/api/v1/auth/login/channels` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | enabled | `internal/router/router.go:164` | python-api@9380 (`api/apps/restful_apis/user_api.py:144`) | — |
| POST | `/api/v1/auth/logout` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/user_api.py:274` | go-api@9384 (`internal/router/router.go:250`) | — |
| GET | `/api/v1/auth/oauth/:channel/callback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | enabled | `internal/router/router.go:174` | — | — |
| GET | `/api/v1/auth/oauth/<channel>/callback` | python-api | 9380 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `api/apps/restful_apis/user_api.py:179` | — | — |
| GET | `/api/v1/auth/oauth/callback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router_ee.go:31` | — | backend worktree-only at a0e091e75051; OAuthCallback is CodeNotImplemented at internal/handler/user_auth_ee.go:25 |
| GET | `/api/v1/auth/oauth/github/callback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router_ee.go:32` | — | backend worktree-only at a0e091e75051; GitHubAuthCallback is CodeNotImplemented at internal/handler/user_auth_ee.go:29 |
| GET | `/api/v1/auth/oauth/lark/callback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router_ee.go:33` | — | backend worktree-only at a0e091e75051; LarkAuthCallback is CodeNotImplemented at internal/handler/user_auth_ee.go:33 |
| POST | `/api/v1/auth/password/forgot/captcha` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/user_api.py:654` | go-api@9384 (`internal/router/router.go:187`) | — |
| POST | `/api/v1/auth/password/forgot/otp` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/user_api.py:683` | go-api@9384 (`internal/router/router.go:188`) | — |
| POST | `/api/v1/auth/password/forgot/otp/verify` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/user_api.py:753` | go-api@9384 (`internal/router/router.go:189`) | — |
| POST | `/api/v1/auth/password/reset` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/user_api.py:814` | go-api@9384 (`internal/router/router.go:190`) | — |
| POST | `/api/v1/auth/register/captcha` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router_ee.go:37` | — | backend worktree-only at a0e091e75051; Captcha is CodeNotImplemented at internal/handler/user_auth_ee.go:49 |
| POST | `/api/v1/auth/register/otp` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router_ee.go:38` | — | backend worktree-only at a0e091e75051; SendOTP is CodeNotImplemented at internal/handler/user_auth_ee.go:53 |
| POST | `/api/v1/auth/register/otp/verify` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router_ee.go:39` | — | backend worktree-only at a0e091e75051; VerifyOTP is CodeNotImplemented at internal/handler/user_auth_ee.go:57 |
| GET | `/api/v1/chat-channels` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_channel_api.py:49` | go-api@9384 (`internal/router/router.go:676`) | — |
| POST | `/api/v1/chat-channels` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_channel_api.py:34` | go-api@9384 (`internal/router/router.go:675`) | — |
| DELETE | `/api/v1/chat-channels/:channel_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:679` | — | — |
| GET | `/api/v1/chat-channels/:channel_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:677` | — | — |
| PATCH | `/api/v1/chat-channels/:channel_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:678` | — | — |
| DELETE | `/api/v1/chat-channels/<channel_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_channel_api.py:102` | — | — |
| GET | `/api/v1/chat-channels/<channel_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_channel_api.py:56` | — | — |
| PATCH | `/api/v1/chat-channels/<channel_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_channel_api.py:69` | — | — |
| GET | `/api/v1/chat-channels/<channel_id>/runtime` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_channel_api.py:113` | — | — |
| POST | `/api/v1/chat/audio/speech` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_api.py:1006` | — | — |
| POST | `/api/v1/chat/audio/transcription` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_api.py:1034` | — | — |
| POST | `/api/v1/chat/completions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:320` | python-api@9380 (`api/apps/restful_apis/chat_api.py:1152`) | — |
| POST | `/api/v1/chat/mindmap` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_api.py:1096` | go-api@9384 (`internal/router/router.go:321`) | — |
| POST | `/api/v1/chat/recommendation` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_api.py:1114` | go-api@9384 (`internal/router/router.go:322`) | — |
| POST | `/api/v1/chat/to_model` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:528` | — | — |
| POST | `/api/v1/chatbots/<dialog_id>/completions` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/bot_api.py:63` | — | — |
| GET | `/api/v1/chatbots/<dialog_id>/info` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/bot_api.py:133` | — | — |
| DELETE | `/api/v1/chats` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_api.py:706` | go-api@9384 (`internal/router/router.go:304`) | — |
| GET | `/api/v1/chats` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_api.py:442` | go-api@9384 (`internal/router/router.go:302`) | — |
| POST | `/api/v1/chats` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chat_api.py:358` | go-api@9384 (`internal/router/router.go:303`) | — |
| POST | `/api/v1/chats_openai/<chat_id>/chat/completions` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:108` | — | backward-compat shim retained by upstream for older clients |
| DELETE | `/api/v1/chats/:chat_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:305` | — | — |
| GET | `/api/v1/chats/:chat_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:306` | — | — |
| PATCH | `/api/v1/chats/:chat_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:308` | — | — |
| PUT | `/api/v1/chats/:chat_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:307` | — | — |
| DELETE | `/api/v1/chats/:chat_id/sessions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:311` | — | — |
| GET | `/api/v1/chats/:chat_id/sessions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:309` | — | — |
| POST | `/api/v1/chats/:chat_id/sessions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:310` | — | — |
| GET | `/api/v1/chats/:chat_id/sessions/:session_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:312` | — | — |
| PATCH | `/api/v1/chats/:chat_id/sessions/:session_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:313` | — | — |
| DELETE | `/api/v1/chats/:chat_id/sessions/:session_id/messages/:msg_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:314` | — | — |
| PUT | `/api/v1/chats/:chat_id/sessions/:session_id/messages/:msg_id/feedback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:315` | — | — |
| DELETE | `/api/v1/chats/<chat_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:692` | — | — |
| GET | `/api/v1/chats/<chat_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:494` | — | — |
| PATCH | `/api/v1/chats/<chat_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:604` | — | — |
| PUT | `/api/v1/chats/<chat_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:522` | — | — |
| POST | `/api/v1/chats/<chat_id>/completions` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:91` | — | backward-compat shim retained by upstream for older clients |
| DELETE | `/api/v1/chats/<chat_id>/sessions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:863` | — | — |
| GET | `/api/v1/chats/<chat_id>/sessions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:785` | — | — |
| POST | `/api/v1/chats/<chat_id>/sessions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:753` | — | — |
| GET | `/api/v1/chats/<chat_id>/sessions/<session_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:810` | — | — |
| PATCH | `/api/v1/chats/<chat_id>/sessions/<session_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:834` | — | — |
| PUT | `/api/v1/chats/<chat_id>/sessions/<session_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:262` | — | backward-compat shim retained by upstream for older clients |
| DELETE | `/api/v1/chats/<chat_id>/sessions/<session_id>/messages/<msg_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:914` | — | — |
| PUT | `/api/v1/chats/<chat_id>/sessions/<session_id>/messages/<msg_id>/feedback` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chat_api.py:939` | — | — |
| POST | `/api/v1/chunk/list` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:668` | — | — |
| POST | `/api/v1/chunk/update` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:669` | — | Internal API only for GO |
| GET | `/api/v1/compilation_template_groups` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/compilation_template_group_api.py:69` | — | — |
| POST | `/api/v1/compilation_template_groups` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/compilation_template_group_api.py:101` | — | — |
| DELETE | `/api/v1/compilation_template_groups/<group_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/compilation_template_group_api.py:163` | — | — |
| GET | `/api/v1/compilation_template_groups/<group_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/compilation_template_group_api.py:89` | — | — |
| PUT | `/api/v1/compilation_template_groups/<group_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/compilation_template_group_api.py:128` | — | — |
| GET | `/api/v1/compilation_templates/builtins` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/compilation_template_api.py:28` | — | — |
| GET | `/api/v1/compilation_templates/wiki_presets` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/compilation_template_api.py:57` | — | — |
| GET | `/api/v1/components` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:570` | — | — |
| GET | `/api/v1/connectors` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/connector_api.py:116` | go-api@9384 (`internal/router/router.go:581`) | — |
| POST | `/api/v1/connectors` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/connector_api.py:89` | go-api@9384 (`internal/router/router.go:582`) | — |
| DELETE | `/api/v1/connectors/:connector_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:590` | — | — |
| GET | `/api/v1/connectors/:connector_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:587` | — | — |
| PATCH | `/api/v1/connectors/:connector_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:588` | — | — |
| GET | `/api/v1/connectors/:connector_id/logs` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:589` | — | — |
| POST | `/api/v1/connectors/:connector_id/rebuild` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:591` | — | — |
| POST | `/api/v1/connectors/:connector_id/test` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:592` | — | — |
| DELETE | `/api/v1/connectors/<connector_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/connector_api.py:169` | — | — |
| GET | `/api/v1/connectors/<connector_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/connector_api.py:123` | — | — |
| PATCH | `/api/v1/connectors/<connector_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/connector_api.py:49` | — | — |
| GET | `/api/v1/connectors/<connector_id>/logs` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/connector_api.py:136` | — | — |
| POST | `/api/v1/connectors/<connector_id>/rebuild` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/connector_api.py:152` | — | — |
| POST | `/api/v1/connectors/<connector_id>/test` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/connector_api.py:181` | — | — |
| GET | `/api/v1/connectors/box/oauth/web/callback` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/connector_api.py:621` | go-api@9384 (`internal/router/router.go:182`) | — |
| POST | `/api/v1/connectors/box/oauth/web/result` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/connector_api.py:664` | go-api@9384 (`internal/router/router.go:586`) | — |
| POST | `/api/v1/connectors/box/oauth/web/start` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/connector_api.py:576` | go-api@9384 (`internal/router/router.go:585`) | — |
| GET | `/api/v1/connectors/gmail/oauth/web/callback` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/connector_api.py:445` | go-api@9384 (`internal/router/router.go:180`) | — |
| GET | `/api/v1/connectors/google-drive/oauth/web/callback` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/connector_api.py:500` | go-api@9384 (`internal/router/router.go:181`) | — |
| POST | `/api/v1/connectors/google/oauth/web/result` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/connector_api.py:555` | go-api@9384 (`internal/router/router.go:584`) | — |
| POST | `/api/v1/connectors/google/oauth/web/start` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/connector_api.py:364` | go-api@9384 (`internal/router/router.go:583`) | — |
| DELETE | `/api/v1/datasets` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:161` | go-api@9384 (`internal/router/router.go:346`) | — |
| GET | `/api/v1/datasets` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:307` | go-api@9384 (`internal/router/router.go:329`) | — |
| POST | `/api/v1/datasets` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:81` | go-api@9384 (`internal/router/router.go:345`) | — |
| GET | `/api/v1/datasets/:dataset_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:331` | — | — |
| PUT | `/api/v1/datasets/:dataset_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:332` | — | — |
| DELETE | `/api/v1/datasets/:dataset_id/:index_type` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:343` | — | — |
| GET | `/api/v1/datasets/:dataset_id/changes` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:458` | — | — |
| DELETE | `/api/v1/datasets/:dataset_id/chunks` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:381` | — | — |
| POST | `/api/v1/datasets/:dataset_id/chunks` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:373` | — | — |
| GET | `/api/v1/datasets/:dataset_id/commits` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:452` | — | — |
| POST | `/api/v1/datasets/:dataset_id/commits` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:451` | — | — |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:454` | — | — |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id/files` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:455` | — | — |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id/files/:file_id/content` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:457` | — | — |
| GET | `/api/v1/datasets/:dataset_id/commits/:commit_id/tree` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:456` | — | — |
| GET | `/api/v1/datasets/:dataset_id/commits/diff` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:453` | — | — |
| DELETE | `/api/v1/datasets/:dataset_id/documents` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:366` | — | — |
| GET | `/api/v1/datasets/:dataset_id/documents` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:362` | — | — |
| POST | `/api/v1/datasets/:dataset_id/documents` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:363` | — | — |
| GET | `/api/v1/datasets/:dataset_id/documents/:document_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:364` | — | — |
| PATCH | `/api/v1/datasets/:dataset_id/documents/:document_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:365` | — | — |
| DELETE | `/api/v1/datasets/:dataset_id/documents/:document_id/chunks` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:382` | — | — |
| GET | `/api/v1/datasets/:dataset_id/documents/:document_id/chunks` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:370` | — | — |
| PATCH | `/api/v1/datasets/:dataset_id/documents/:document_id/chunks` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:371` | — | — |
| POST | `/api/v1/datasets/:dataset_id/documents/:document_id/chunks` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:367` | — | — |
| GET | `/api/v1/datasets/:dataset_id/documents/:document_id/chunks/:chunk_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:372` | — | — |
| PATCH | `/api/v1/datasets/:dataset_id/documents/:document_id/chunks/:chunk_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:374` | — | — |
| PUT | `/api/v1/datasets/:dataset_id/documents/:document_id/metadata/config` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:383` | — | — |
| POST | `/api/v1/datasets/:dataset_id/documents/batch-update-status` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:339` | — | — |
| PATCH | `/api/v1/datasets/:dataset_id/documents/metadatas` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:385` | — | — |
| POST | `/api/v1/datasets/:dataset_id/documents/parse` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:375` | — | — |
| POST | `/api/v1/datasets/:dataset_id/embedding` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:337` | — | — |
| POST | `/api/v1/datasets/:dataset_id/embedding/check` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:338` | — | — |
| GET | `/api/v1/datasets/:dataset_id/graph` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:333` | — | — |
| DELETE | `/api/v1/datasets/:dataset_id/index` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:342` | — | — |
| GET | `/api/v1/datasets/:dataset_id/index` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:340` | — | — |
| POST | `/api/v1/datasets/:dataset_id/index` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:341` | — | — |
| GET | `/api/v1/datasets/:dataset_id/ingestions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:354` | — | — |
| GET | `/api/v1/datasets/:dataset_id/ingestions/:log_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:355` | — | — |
| GET | `/api/v1/datasets/:dataset_id/ingestions/summary` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:353` | — | — |
| GET | `/api/v1/datasets/:dataset_id/metadata/config` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:358` | — | — |
| PUT | `/api/v1/datasets/:dataset_id/metadata/config` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:359` | — | — |
| GET | `/api/v1/datasets/:dataset_id/metadata/summary` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:350` | — | — |
| POST | `/api/v1/datasets/:dataset_id/metadata/update` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:384` | — | — |
| POST | `/api/v1/datasets/:dataset_id/search` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:348` | — | — |
| DELETE | `/api/v1/datasets/:dataset_id/tags` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:336` | — | — |
| GET | `/api/v1/datasets/:dataset_id/tags` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:334` | — | — |
| PUT | `/api/v1/datasets/:dataset_id/tags` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:335` | — | — |
| GET | `/api/v1/datasets/<dataset_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:384` | — | — |
| PUT | `/api/v1/datasets/<dataset_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:220` | — | — |
| DELETE | `/api/v1/datasets/<dataset_id>/<index_type>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:854` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/any_artifact` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:559` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/any_skill` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:694` | — | — |
| DELETE | `/api/v1/datasets/<dataset_id>/artifacts` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:643` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/artifacts` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:580` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/<page_type>/<path:slug>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:667` | — | — |
| PUT | `/api/v1/datasets/<dataset_id>/artifacts/<page_type>/<path:slug>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:770` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/artifacts/graph` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:612` | — | — |
| DELETE | `/api/v1/datasets/<dataset_id>/chunks` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chunk_api.py:258` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/chunks` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chunk_api.py:183` | — | — |
| DELETE | `/api/v1/datasets/<dataset_id>/documents` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/document_api.py:1103` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/documents` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:703` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/documents` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:427` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/documents/<document_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:2089` | — | — |
| PATCH | `/api/v1/datasets/<dataset_id>/documents/<document_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:170` | — | — |
| PUT | `/api/v1/datasets/<dataset_id>/documents/<document_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:562` | — | backward-compat shim retained by upstream for older clients |
| DELETE | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chunk_api.py:931` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chunk_api.py:441` | — | — |
| PATCH | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chunk_api.py:1072` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chunk_api.py:842` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks/<chunk_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chunk_api.py:529` | — | — |
| PATCH | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks/<chunk_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/chunk_api.py:984` | — | — |
| PUT | `/api/v1/datasets/<dataset_id>/documents/<document_id>/chunks/<chunk_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:499` | — | backward-compat shim retained by upstream for older clients |
| PUT | `/api/v1/datasets/<dataset_id>/documents/<document_id>/metadata/config` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:1193` | — | — |
| DELETE | `/api/v1/datasets/<dataset_id>/documents/<document_id>/structure/graph` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chunk_api.py:786` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/documents/<document_id>/structure/graph` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chunk_api.py:556` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/documents/batch-update-status` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:1924` | — | — |
| PATCH | `/api/v1/datasets/<dataset_id>/documents/metadatas` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/document_api.py:1307` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/documents/parse` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:1508` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/documents/stop` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/document_api.py:1621` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/embedding` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:881` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/embedding/check` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:896` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/graph` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:537` | — | — |
| DELETE | `/api/v1/datasets/<dataset_id>/index` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:855` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/index` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:835` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/index` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:816` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/ingestions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:916` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/ingestions/<log_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:942` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/ingestions/summary` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:401` | — | — |
| DELETE | `/api/v1/datasets/<dataset_id>/knowledge_graph` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:168` | — | backward-compat shim retained by upstream for older clients |
| GET | `/api/v1/datasets/<dataset_id>/knowledge_graph` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:151` | — | backward-compat shim retained by upstream for older clients |
| GET | `/api/v1/datasets/<dataset_id>/metadata/config` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:959` | — | — |
| PUT | `/api/v1/datasets/<dataset_id>/metadata/config` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:1000` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/metadata/summary` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:307` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/metadata/update` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:345` | — | — |
| POST | `/api/v1/datasets/<dataset_id>/run_graphrag` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:185` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/datasets/<dataset_id>/run_raptor` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:221` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/datasets/<dataset_id>/search` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:507` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/skills` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:713` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/skills/<path:skill_kwd>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:735` | — | — |
| DELETE | `/api/v1/datasets/<dataset_id>/tags` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:435` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/tags` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:418` | — | — |
| PUT | `/api/v1/datasets/<dataset_id>/tags` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/dataset_api.py:458` | — | — |
| GET | `/api/v1/datasets/<dataset_id>/trace_graphrag` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:203` | — | backward-compat shim retained by upstream for older clients |
| GET | `/api/v1/datasets/<dataset_id>/trace_raptor` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:239` | — | backward-compat shim retained by upstream for older clients |
| GET | `/api/v1/datasets/<entity_id>/changes` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:301` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/datasets/<entity_id>/commits` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:140` | — | Registered for 3 prefixes by a shared helper |
| POST | `/api/v1/datasets/<entity_id>/commits` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:107` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/datasets/<entity_id>/commits/<commit_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:199` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/datasets/<entity_id>/commits/<commit_id>/files` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:249` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/datasets/<entity_id>/commits/<commit_id>/files/<file_id>/content` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:328` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/datasets/<entity_id>/commits/<commit_id>/tree` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:312` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/datasets/<entity_id>/commits/diff` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/file_commit_api.py:280` | — | Registered for 3 prefixes by a shared helper |
| DELETE | `/api/v1/datasets/ingestion/tasks` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:378` | — | — |
| GET | `/api/v1/datasets/ingestion/tasks` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:376` | — | — |
| PUT | `/api/v1/datasets/ingestion/tasks` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:377` | — | — |
| GET | `/api/v1/datasets/metadata/flattened` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:59` | go-api@9384 (`internal/router/router.go:349`) | — |
| POST | `/api/v1/datasets/search` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:347` | python-api@9380 (`api/apps/restful_apis/dataset_api.py:484`) | — |
| GET | `/api/v1/datasets/tags/aggregation` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dataset_api.py:37` | go-api@9384 (`internal/router/router.go:330`) | — |
| GET | `/api/v1/dify/retrieval` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dify_retrieval_api.py:111` | go-api@9384 (`internal/router/router.go:695`) | — |
| POST | `/api/v1/dify/retrieval` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/dify_retrieval_api.py:111` | go-api@9384 (`internal/router/router.go:694`) | — |
| GET | `/api/v1/dify/retrieval/health` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/dify_retrieval_api.py:315` | go-api@9384 (`internal/router/router.go:192`) | — |
| POST | `/api/v1/document/delete_meta` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:662` | — | Internal API only for GO |
| GET | `/api/v1/document/download/<doc_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:597` | — | backward-compat shim retained by upstream for older clients |
| GET | `/api/v1/document/get/<doc_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:580` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/document/list` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:659` | — | — |
| POST | `/api/v1/document/metadata/summary` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:660` | — | — |
| POST | `/api/v1/document/set_meta` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:661` | — | — |
| GET | `/api/v1/documents` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:291` | — | — |
| POST | `/api/v1/documents` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:289` | — | — |
| DELETE | `/api/v1/documents/:id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:295` | — | — |
| GET | `/api/v1/documents/:id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:293` | — | — |
| PUT | `/api/v1/documents/:id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:294` | — | — |
| GET | `/api/v1/documents/:id/preview` | go-api | 9384 | hybrid | 127.0.0.1:9384 | beta-token | enabled | `internal/router/router.go:214` | — | — |
| GET | `/api/v1/documents/<doc_id>/preview` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required(AUTH_JWT,AUTH_API,AUTH_BETA) | **runtime-disabled** | `api/apps/restful_apis/document_api.py:2045` | — | — |
| GET | `/api/v1/documents/<document_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:2151` | — | — |
| GET | `/api/v1/documents/artifact/:filename` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:292` | — | — |
| GET | `/api/v1/documents/artifact/<filename>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/document_api.py:1871` | — | — |
| GET | `/api/v1/documents/images/:image_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | beta-token | enabled | `internal/router/router.go:215` | — | — |
| GET | `/api/v1/documents/images/<image_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required(AUTH_JWT,AUTH_API,AUTH_BETA) | **runtime-disabled** | `api/apps/restful_apis/document_api.py:1782` | — | — |
| POST | `/api/v1/documents/ingest` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/document_api.py:1431` | go-api@9384 (`internal/router/router.go:296`) | — |
| POST | `/api/v1/documents/upload` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/document_api.py:107` | go-api@9384 (`internal/router/router.go:290`) | — |
| POST | `/api/v1/embeddings` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:529` | — | — |
| GET | `/api/v1/file/all_parent_folder` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:319` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/file/convert` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:403` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/file/create` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:373` | — | backward-compat shim retained by upstream for older clients |
| GET | `/api/v1/file/get/<file_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:287` | — | backward-compat shim retained by upstream for older clients |
| GET | `/api/v1/file/list` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:305` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/file/mv` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:416` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/file/ocr` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:533` | — | — |
| GET | `/api/v1/file/parent_folder` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:339` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/file/parse` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:534` | — | — |
| POST | `/api/v1/file/rename` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:431` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/file/rm` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:459` | — | backward-compat shim retained by upstream for older clients |
| GET | `/api/v1/file/root_folder` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:359` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/file/upload` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:388` | — | backward-compat shim retained by upstream for older clients |
| POST | `/api/v1/file/upload_info` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:523` | — | backward-compat shim retained by upstream for older clients |
| DELETE | `/api/v1/files` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/file_api.py:154` | go-api@9384 (`internal/router/router.go:404`) | — |
| GET | `/api/v1/files` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/file_api.py:99` | go-api@9384 (`internal/router/router.go:403`) | — |
| POST | `/api/v1/files` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/file_api.py:45` | go-api@9384 (`internal/router/router.go:402`) | — |
| GET | `/api/v1/files/:id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:409` | — | — |
| GET | `/api/v1/files/:id/ancestors` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:407` | — | — |
| GET | `/api/v1/files/:id/parent` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:408` | — | — |
| GET | `/api/v1/files/:id/versions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:410` | — | — |
| GET | `/api/v1/files/<file_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_api.py:262` | — | — |
| GET | `/api/v1/files/<file_id>/ancestors` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_api.py:348` | — | — |
| GET | `/api/v1/files/<file_id>/parent` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_api.py:317` | — | — |
| GET | `/api/v1/files/<file_id>/versions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:370` | — | — |
| POST | `/api/v1/files/link-to-datasets` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/file2document_api.py:85` | go-api@9384 (`internal/router/router.go:406`) | — |
| POST | `/api/v1/files/move` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/file_api.py:208` | go-api@9384 (`internal/router/router.go:405`) | — |
| GET | `/api/v1/folders/:folder_id/changes` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:431` | — | — |
| GET | `/api/v1/folders/:folder_id/commits` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:425` | — | — |
| POST | `/api/v1/folders/:folder_id/commits` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:424` | — | — |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:427` | — | — |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id/files` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:428` | — | — |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id/files/:file_id/content` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:430` | — | — |
| GET | `/api/v1/folders/:folder_id/commits/:commit_id/tree` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:429` | — | — |
| GET | `/api/v1/folders/:folder_id/commits/diff` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:426` | — | — |
| GET | `/api/v1/folders/<entity_id>/changes` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:301` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/folders/<entity_id>/commits` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:140` | — | Registered for 3 prefixes by a shared helper |
| POST | `/api/v1/folders/<entity_id>/commits` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:107` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/folders/<entity_id>/commits/<commit_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:199` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/folders/<entity_id>/commits/<commit_id>/files` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:249` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/folders/<entity_id>/commits/<commit_id>/files/<file_id>/content` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:328` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/folders/<entity_id>/commits/<commit_id>/tree` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:312` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/folders/<entity_id>/commits/diff` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/file_commit_api.py:280` | — | Registered for 3 prefixes by a shared helper |
| DELETE | `/api/v1/langfuse/api-key` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/langfuse_api.py:85` | go-api@9384 (`internal/router/router.go:688`) | — |
| GET | `/api/v1/langfuse/api-key` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/langfuse_api.py:61` | go-api@9384 (`internal/router/router.go:687`) | — |
| POST | `/api/v1/langfuse/api-key` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/langfuse_api.py:26` | go-api@9384 (`internal/router/router.go:685`) | — |
| PUT | `/api/v1/langfuse/api-key` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/langfuse_api.py:26` | go-api@9384 (`internal/router/router.go:686`) | — |
| POST | `/api/v1/mcp` | go-api | 9384 | hybrid | 127.0.0.1:9384 | beta-token | enabled | `internal/router/router.go:222` | — | — |
| GET | `/api/v1/mcp/servers` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/mcp_api.py:70` | go-api@9384 (`internal/router/router.go:607`) | — |
| POST | `/api/v1/mcp/servers` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/mcp_api.py:115` | go-api@9384 (`internal/router/router.go:606`) | — |
| DELETE | `/api/v1/mcp/servers/:mcp_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:610` | — | — |
| GET | `/api/v1/mcp/servers/:mcp_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:608` | — | — |
| PUT | `/api/v1/mcp/servers/:mcp_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:609` | — | — |
| POST | `/api/v1/mcp/servers/:mcp_id/test` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:612` | — | — |
| DELETE | `/api/v1/mcp/servers/<mcp_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/mcp_api.py:231` | — | — |
| GET | `/api/v1/mcp/servers/<mcp_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/mcp_api.py:95` | — | — |
| PUT | `/api/v1/mcp/servers/<mcp_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/mcp_api.py:172` | — | — |
| POST | `/api/v1/mcp/servers/<mcp_id>/test` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/mcp_api.py:321` | — | — |
| POST | `/api/v1/mcp/servers/import` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/mcp_api.py:246` | go-api@9384 (`internal/router/router.go:611`) | — |
| GET | `/api/v1/memories` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/memory_api.py:133` | go-api@9384 (`internal/router/router.go:467`) | — |
| POST | `/api/v1/memories` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/memory_api.py:29` | go-api@9384 (`internal/router/router.go:464`) | — |
| DELETE | `/api/v1/memories/:memory_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:466` | — | — |
| GET | `/api/v1/memories/:memory_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:469` | — | — |
| PUT | `/api/v1/memories/:memory_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:465` | — | — |
| GET | `/api/v1/memories/:memory_id/config` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:468` | — | — |
| DELETE | `/api/v1/memories/<memory_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/memory_api.py:119` | — | — |
| GET | `/api/v1/memories/<memory_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/memory_api.py:162` | — | — |
| PUT | `/api/v1/memories/<memory_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/memory_api.py:78` | — | — |
| GET | `/api/v1/memories/<memory_id>/config` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/memory_api.py:148` | — | — |
| GET | `/api/v1/messages` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/memory_api.py:277` | go-api@9384 (`internal/router/router.go:475`) | — |
| POST | `/api/v1/messages` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/memory_api.py:184` | go-api@9384 (`internal/router/router.go:476`) | — |
| DELETE | `/api/v1/messages/:memory_message` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:477` | — | — |
| PUT | `/api/v1/messages/:memory_message` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:478` | — | — |
| GET | `/api/v1/messages/:memory_message/content` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:479` | — | — |
| DELETE | `/api/v1/messages/<memory_id>:<message_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/memory_api.py:216` | — | — |
| PUT | `/api/v1/messages/<memory_id>:<message_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/memory_api.py:230` | — | — |
| GET | `/api/v1/messages/<memory_id>:<message_id>/content` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/memory_api.py:300` | — | — |
| GET | `/api/v1/messages/search` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/memory_api.py:253` | go-api@9384 (`internal/router/router.go:480`) | — |
| GET | `/api/v1/models` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/models_api.py:30` | go-api@9384 (`internal/router/router.go:541`) | — |
| PATCH | `/api/v1/models` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:542` | — | — |
| GET | `/api/v1/models/default` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/models_api.py:84` | go-api@9384 (`internal/router/router.go:544`) | — |
| PATCH | `/api/v1/models/default` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/models_api.py:137` | go-api@9384 (`internal/router/router.go:545`) | — |
| POST | `/api/v1/openai/:chat_id/chat/completions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:324` | — | — |
| POST | `/api/v1/openai/<chat_id>/chat/completions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/openai_api.py:237` | — | — |
| GET | `/api/v1/pipelines` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router.go:170` | — | backend worktree-only implemented pipeline catalog at a0e091e75051; ListPipelines is implemented in internal/handler/pipeline.go |
| GET | `/api/v1/pipelines/:id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | **runtime-disabled** | `internal/router/router.go:171` | — | backend worktree-only implemented pipeline catalog at a0e091e75051; GetPipeline is implemented in internal/handler/pipeline.go |
| GET | `/api/v1/plugin/tools` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/plugin_api.py:24` | go-api@9384 (`internal/router/router.go:561`) | — |
| GET | `/api/v1/providers` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/provider_api.py:30` | go-api@9384 (`internal/router/router.go:508`) | — |
| PUT | `/api/v1/providers` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/provider_api.py:71` | go-api@9384 (`internal/router/router.go:509`) | — |
| DELETE | `/api/v1/providers/:provider_name` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:511` | — | — |
| GET | `/api/v1/providers/:provider_name` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:510` | — | — |
| POST | `/api/v1/providers/:provider_name/connection` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:519` | — | — |
| DELETE | `/api/v1/providers/:provider_name/instances` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:523` | — | — |
| GET | `/api/v1/providers/:provider_name/instances` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:515` | — | — |
| POST | `/api/v1/providers/:provider_name/instances` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:514` | — | — |
| GET | `/api/v1/providers/:provider_name/instances/:instance_name` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:516` | — | — |
| PUT | `/api/v1/providers/:provider_name/instances/:instance_name` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:522` | — | — |
| GET | `/api/v1/providers/:provider_name/instances/:instance_name/balance` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:517` | — | — |
| GET | `/api/v1/providers/:provider_name/instances/:instance_name/connection` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:518` | — | — |
| DELETE | `/api/v1/providers/:provider_name/instances/:instance_name/models` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:527` | — | — |
| GET | `/api/v1/providers/:provider_name/instances/:instance_name/models` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:524` | — | — |
| POST | `/api/v1/providers/:provider_name/instances/:instance_name/models` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:526` | — | — |
| PATCH | `/api/v1/providers/:provider_name/instances/:instance_name/models/*model_name` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:525` | — | — |
| GET | `/api/v1/providers/:provider_name/instances/:instance_name/tasks` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:520` | — | — |
| GET | `/api/v1/providers/:provider_name/instances/:instance_name/tasks/:task_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:521` | — | — |
| GET | `/api/v1/providers/:provider_name/models` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:512` | — | — |
| GET | `/api/v1/providers/:provider_name/models/:model_name` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:513` | — | — |
| DELETE | `/api/v1/providers/<provider_id_or_name>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:161` | — | — |
| GET | `/api/v1/providers/<provider_id_or_name>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:123` | — | — |
| POST | `/api/v1/providers/<provider_id_or_name>/connection` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:359` | — | — |
| DELETE | `/api/v1/providers/<provider_id_or_name>/instances` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:515` | — | — |
| GET | `/api/v1/providers/<provider_id_or_name>/instances` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:427` | — | — |
| POST | `/api/v1/providers/<provider_id_or_name>/instances` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:288` | — | — |
| GET | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:471` | — | — |
| GET | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:576` | — | — |
| POST | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:691` | — | — |
| PUT | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/provider_api.py:631` | — | — |
| PATCH | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models/<path:model_name>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/provider_api.py:764` | — | — |
| POST | `/api/v1/providers/<provider_id_or_name>/instances/<instance_id_or_name>/models/<path:model_name>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/provider_api.py:834` | — | — |
| GET | `/api/v1/providers/<provider_id_or_name>/models` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:200` | — | — |
| GET | `/api/v1/providers/<provider_id_or_name>/models/<path:model_name>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/provider_api.py:245` | — | — |
| POST | `/api/v1/rerank` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:530` | — | — |
| POST | `/api/v1/retrieval` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/chunk_api.py:311` | — | — |
| POST | `/api/v1/searchbots/ask` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/bot_api.py:328` | go-api@9384 (`internal/router/router.go:203`) | — |
| GET | `/api/v1/searchbots/detail` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/bot_api.py:549` | go-api@9384 (`internal/router/router.go:161`) | — |
| POST | `/api/v1/searchbots/mindmap` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/bot_api.py:572` | go-api@9384 (`internal/router/router.go:204`) | — |
| POST | `/api/v1/searchbots/related_questions` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/bot_api.py:507` | go-api@9384 (`internal/router/router.go:201`) | — |
| POST | `/api/v1/searchbots/retrieval_test` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/bot_api.py:364` | go-api@9384 (`internal/router/router.go:202`) | — |
| GET | `/api/v1/searches` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/search_api.py:76` | go-api@9384 (`internal/router/router.go:391`) | — |
| POST | `/api/v1/searches` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/search_api.py:42` | go-api@9384 (`internal/router/router.go:392`) | — |
| DELETE | `/api/v1/searches/:search_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:395` | — | — |
| GET | `/api/v1/searches/:search_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:393` | — | — |
| PUT | `/api/v1/searches/:search_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:394` | — | — |
| POST | `/api/v1/searches/:search_id/completion` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:396` | — | — |
| POST | `/api/v1/searches/:search_id/completions` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:397` | — | — |
| DELETE | `/api/v1/searches/<search_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/search_api.py:179` | — | — |
| GET | `/api/v1/searches/<search_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/search_api.py:101` | — | — |
| PUT | `/api/v1/searches/<search_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/search_api.py:120` | — | — |
| POST | `/api/v1/searches/<search_id>/completion` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/search_api.py:193` | — | — |
| POST | `/api/v1/searches/<search_id>/completions` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/search_api.py:194` | — | — |
| POST | `/api/v1/sessions/related_questions` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:480` | — | backward-compat shim retained by upstream for older clients |
| GET | `/api/v1/skills/config` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:495` | — | — |
| POST | `/api/v1/skills/config` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:496` | — | — |
| DELETE | `/api/v1/skills/index` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:501` | — | — |
| POST | `/api/v1/skills/index` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:500` | — | — |
| POST | `/api/v1/skills/reindex` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:502` | — | — |
| POST | `/api/v1/skills/search` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:499` | — | — |
| GET | `/api/v1/skills/space/by-folder` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:492` | — | — |
| GET | `/api/v1/skills/spaces` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:487` | — | — |
| POST | `/api/v1/skills/spaces` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:488` | — | — |
| DELETE | `/api/v1/skills/spaces/:space_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:491` | — | — |
| GET | `/api/v1/skills/spaces/:space_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:489` | — | — |
| PUT | `/api/v1/skills/spaces/:space_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:490` | — | — |
| GET | `/api/v1/system/config` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/system_api.py:204` | go-api@9384 (`internal/router/router.go:156`) | — |
| GET | `/api/v1/system/config/log` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/system_api.py:371` | go-api@9384 (`internal/router/router.go:623`) | — |
| PUT | `/api/v1/system/config/log` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/system_api.py:386` | go-api@9384 (`internal/router/router.go:624`) | — |
| GET | `/api/v1/system/configs` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:617` | — | — |
| GET | `/api/v1/system/environments` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:633` | — | — |
| GET | `/api/v1/system/healthz` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/system_api.py:229` | go-api@9384 (`internal/router/router.go:158`) | — |
| GET | `/api/v1/system/keys` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:648` | — | — |
| POST | `/api/v1/system/keys` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:650` | — | — |
| DELETE | `/api/v1/system/keys/:key` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:652` | — | — |
| GET | `/api/v1/system/oceanbase/status` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/system_api.py:174` | — | — |
| GET | `/api/v1/system/ping` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/system_api.py:38` | go-api@9384 (`internal/router/router.go:155`) | — |
| GET | `/api/v1/system/stats` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/stats_api.py:24` | go-api@9384 (`internal/router/router.go:619`) | — |
| GET | `/api/v1/system/status` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/system_api.py:65` | go-api@9384 (`internal/router/router.go:618`) | — |
| GET | `/api/v1/system/tokens` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/system_api.py:235` | go-api@9384 (`internal/router/router.go:638`) | — |
| POST | `/api/v1/system/tokens` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/system_api.py:283` | go-api@9384 (`internal/router/router.go:640`) | — |
| DELETE | `/api/v1/system/tokens/:key` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:642` | — | — |
| DELETE | `/api/v1/system/tokens/<token>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/system_api.py:333` | — | — |
| GET | `/api/v1/system/variables` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:628` | — | — |
| PUT | `/api/v1/system/variables` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:629` | — | — |
| GET | `/api/v1/system/variables/:var_name` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:630` | — | — |
| GET | `/api/v1/system/version` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/system_api.py:43` | go-api@9384 (`internal/router/router.go:157`) | — |
| PATCH | `/api/v1/tasks/<task_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/task_api.py:37` | — | — |
| POST | `/api/v1/tasks/<task_id>/cancel` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/task_api.py:30` | — | — |
| DELETE | `/api/v1/tenant/chunk_store` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:279` | — | Internal API only for GO |
| POST | `/api/v1/tenant/chunk_store` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:278` | — | Internal API only for GO |
| POST | `/api/v1/tenant/insert_chunks_from_file` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:282` | — | Internal API only for GO |
| POST | `/api/v1/tenant/insert_metadata_from_file` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:283` | — | Internal API only for GO |
| GET | `/api/v1/tenant/list` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:277` | — | — |
| DELETE | `/api/v1/tenant/metadata_store` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:281` | — | Internal API only for GO |
| POST | `/api/v1/tenant/metadata_store` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:280` | — | Internal API only for GO |
| GET | `/api/v1/tenants` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/tenant_api.py:155` | go-api@9384 (`internal/router/router.go:267`) | — |
| PATCH | `/api/v1/tenants/:tenant_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:268` | — | — |
| DELETE | `/api/v1/tenants/:tenant_id/users` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:271` | — | — |
| GET | `/api/v1/tenants/:tenant_id/users` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:269` | — | — |
| POST | `/api/v1/tenants/:tenant_id/users` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:270` | — | — |
| PATCH | `/api/v1/tenants/<tenant_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/tenant_api.py:167` | — | — |
| DELETE | `/api/v1/tenants/<tenant_id>/users` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/tenant_api.py:135` | — | — |
| GET | `/api/v1/tenants/<tenant_id>/users` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/tenant_api.py:41` | — | — |
| POST | `/api/v1/tenants/<tenant_id>/users` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/tenant_api.py:60` | — | — |
| GET | `/api/v1/thumbnails` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required(AUTH_JWT,AUTH_API,AUTH_BETA) | enabled | `api/apps/restful_apis/document_api.py:1268` | go-api@9384 (`internal/router/router.go:216`) | — |
| POST | `/api/v1/users` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/restful_apis/user_api.py:468` | go-api@9384 (`internal/router/router.go:177`) | — |
| GET | `/api/v1/users/me` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/user_api.py:381` | go-api@9384 (`internal/router/router.go:256`) | — |
| PATCH | `/api/v1/users/me` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/user_api.py:302` | go-api@9384 (`internal/router/router.go:258`) | — |
| GET | `/api/v1/users/me/models` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/user_api.py:567` | go-api@9384 (`internal/router/router.go:260`) | — |
| PATCH | `/api/v1/users/me/models` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/user_api.py:605` | go-api@9384 (`internal/router/router.go:262`) | — |
| GET | `/api/v1/workspace/:folder_id/changes` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:444` | — | — |
| GET | `/api/v1/workspace/:folder_id/commits` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:438` | — | — |
| POST | `/api/v1/workspace/:folder_id/commits` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:437` | — | — |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:440` | — | — |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id/files` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:441` | — | — |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id/files/:file_id/content` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:443` | — | — |
| GET | `/api/v1/workspace/:folder_id/commits/:commit_id/tree` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:442` | — | — |
| GET | `/api/v1/workspace/:folder_id/commits/diff` | go-api | 9384 | hybrid | 127.0.0.1:9380 | session | **runtime-disabled** | `internal/router/router.go:439` | — | — |
| GET | `/api/v1/workspace/<entity_id>/changes` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:301` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/workspace/<entity_id>/commits` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:140` | — | Registered for 3 prefixes by a shared helper |
| POST | `/api/v1/workspace/<entity_id>/commits` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:107` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/workspace/<entity_id>/commits/<commit_id>` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:199` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/workspace/<entity_id>/commits/<commit_id>/files` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:249` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/workspace/<entity_id>/commits/<commit_id>/files/<file_id>/content` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:328` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/workspace/<entity_id>/commits/<commit_id>/tree` | python-api | 9380 | hybrid | 127.0.0.1:9384 | login_required | **runtime-disabled** | `api/apps/restful_apis/file_commit_api.py:312` | — | Registered for 3 prefixes by a shared helper |
| GET | `/api/v1/workspace/<entity_id>/commits/diff` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/restful_apis/file_commit_api.py:280` | — | Registered for 3 prefixes by a shared helper |
| GET | `/connectors/box/oauth/web/callback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | enabled | `internal/router/router.go:151` | — | — |
| GET | `/connectors/gmail/oauth/web/callback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | enabled | `internal/router/router.go:149` | — | — |
| GET | `/connectors/google-drive/oauth/web/callback` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | enabled | `internal/router/router.go:150` | — | — |
| GET | `/health` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | enabled | `internal/router/router.go:138` | go-admin@9383 (`internal/admin/router.go:38`) | — |
| DELETE | `/mcp` | mcp | 9382 | hybrid | direct:9382 (not proxied by nginx) | mcp-api-key | not-proxied | `mcp/server/server.py:783` | — | MCP transport endpoint; own port, opt-in via --enable-mcpserver |
| GET | `/mcp` | mcp | 9382 | hybrid | direct:9382 (not proxied by nginx) | mcp-api-key | not-proxied | `mcp/server/server.py:783` | — | MCP transport endpoint; own port, opt-in via --enable-mcpserver |
| POST | `/mcp` | mcp | 9382 | hybrid | direct:9382 (not proxied by nginx) | mcp-api-key | not-proxied | `mcp/server/server.py:783` | mcp@9382 (`mcp/server/server.py:784`) | MCP transport endpoint; own port, opt-in via --enable-mcpserver |
| POST | `/messages/` | mcp | 9382 | hybrid | direct:9382 (not proxied by nginx) | mcp-api-key | not-proxied | `mcp/server/server.py:749` | — | starlette Mount (sub-application), opt-in via --enable-mcpserver |
| GET | `/sse` | mcp | 9382 | hybrid | direct:9382 (not proxied by nginx) | mcp-api-key | not-proxied | `mcp/server/server.py:748` | — | MCP transport endpoint; own port, opt-in via --enable-mcpserver |
| GET | `/v1/connector/:connector_id` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:599` | — | — |
| POST | `/v1/connector/:connector_id/rebuild` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:600` | — | — |
| GET | `/v1/connector/list` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:598` | — | — |
| GET | `/v1/document/download/<attachment_id>` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:614` | — | backward-compat shim retained by upstream for older clients |
| POST | `/v1/document/upload_info` | python-api | 9380 | hybrid | 127.0.0.1:9380 | login_required | enabled | `api/apps/backward_compat.py:541` | — | backward-compat shim retained by upstream for older clients |
| GET | `/v1/file/all_parent_folder` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:418` | — | — |
| GET | `/v1/file/parent_folder` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:417` | — | — |
| GET | `/v1/file/root_folder` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:416` | — | — |
| GET | `/v1/system/configs` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | enabled | `internal/router/router.go:141` | — | — |
| GET | `/v1/system/healthz` | python-api | 9380 | hybrid | 127.0.0.1:9380 | public | enabled | `api/apps/backward_compat.py:73` | — | backward-compat shim retained by upstream for older clients |
| GET | `/v1/tenant/list` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:235` | — | — |
| GET | `/v1/user/info` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:231` | — | — |
| GET | `/v1/user/logout` | go-api | 9384 | hybrid | 127.0.0.1:9384 | public | enabled | `internal/router/router.go:145` | — | — |
| POST | `/v1/user/set_tenant_info` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:241` | — | — |
| POST | `/v1/user/setting` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:237` | — | — |
| POST | `/v1/user/setting/password` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:239` | — | — |
| GET | `/v1/user/tenant_info` | go-api | 9384 | hybrid | 127.0.0.1:9384 | session | enabled | `internal/router/router.go:233` | — | — |

