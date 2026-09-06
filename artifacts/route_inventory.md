# Studio route isolation inventory

Generated from imported APIRouters under studio/backend/routes. Paths are router-relative; mount aliases outside routes (main.py, hub, picker, MCP mounts) are outside this inventory.

719 route/method pairs; 109 have object-like parameters; 13 have factories; 96 are uncovered.

Each object route produces five cases: owner reading Alice's resource, Alice, Bob, unauthenticated, and a deactivated Alice with a previously issued JWT. Uncovered cases fail at factory lookup under a strict worker xfail; they do not send a request. A factory means exercised, not necessarily passing. See pytest outcomes for pending behavior.

Worker numbers are provisional domain assignments; confirm them against integration ownership. Worker 10 owns adding the remaining factories; domain workers own the underlying behavior.

| Module | Method | Router path | Object parameters | Factory / gap | Domain worker |
| --- | --- | --- | --- | --- | --- |
| routes.auth | DELETE | `/api-keys/{key_id}` | key_id | api-key | 01 |
| routes.auth | GET | `/api-keys` | - | no object-like path parameter | 01 |
| routes.auth | GET | `/identity` | - | no object-like path parameter | 01 |
| routes.auth | GET | `/status` | - | no object-like path parameter | 01 |
| routes.auth | POST | `/api-keys` | - | no object-like path parameter | 01 |
| routes.auth | POST | `/change-password` | - | no object-like path parameter | 01 |
| routes.auth | POST | `/desktop-initial-password` | - | no object-like path parameter | 01 |
| routes.auth | POST | `/desktop-login` | - | no object-like path parameter | 01 |
| routes.auth | POST | `/login` | - | no object-like path parameter | 01 |
| routes.auth | POST | `/logout` | - | no object-like path parameter | 01 |
| routes.auth | POST | `/refresh` | - | no object-like path parameter | 01 |
| routes.chat_generation_runs | GET | `/active` | - | no object-like path parameter | 02 |
| routes.chat_generation_runs | GET | `/{run_id}` | run_id | **uncovered** | 02 |
| routes.chat_generation_runs | POST | `` | - | no object-like path parameter | 02 |
| routes.chat_generation_runs | POST | `/{run_id}/cancel` | run_id | **uncovered** | 02 |
| routes.chat_generation_runs | POST | `/{run_id}/events` | run_id | **uncovered** | 02 |
| routes.chat_history | DELETE | `` | - | no object-like path parameter | 02 |
| routes.chat_history | DELETE | `/attachments/{message_id}/{attachment_id}` | message_id, attachment_id | **uncovered** | 02 |
| routes.chat_history | DELETE | `/projects/{project_id}` | project_id | **uncovered** | 02 |
| routes.chat_history | DELETE | `/threads` | - | no object-like path parameter | 02 |
| routes.chat_history | GET | `/attachments` | - | no object-like path parameter | 02 |
| routes.chat_history | GET | `/attachments/{message_id}/{attachment_id}/file` | message_id, attachment_id | **uncovered** | 02 |
| routes.chat_history | GET | `/count` | - | no object-like path parameter | 02 |
| routes.chat_history | GET | `/export` | - | no object-like path parameter | 02 |
| routes.chat_history | GET | `/import-ledger` | - | no object-like path parameter | 02 |
| routes.chat_history | GET | `/projects` | - | no object-like path parameter | 02 |
| routes.chat_history | GET | `/projects/{project_id}` | project_id | project | 02 |
| routes.chat_history | GET | `/settings` | - | no object-like path parameter | 02 |
| routes.chat_history | GET | `/threads` | - | no object-like path parameter | 02 |
| routes.chat_history | GET | `/threads/{thread_id}` | thread_id | chat | 02 |
| routes.chat_history | GET | `/threads/{thread_id}/forks` | thread_id | **uncovered** | 02 |
| routes.chat_history | GET | `/threads/{thread_id}/messages` | thread_id | chat | 02 |
| routes.chat_history | GET | `/threads/{thread_id}/messages/{message_id}` | thread_id, message_id | chat | 02 |
| routes.chat_history | GET | `/threads/{thread_id}/messages/{message_id}/forks` | thread_id, message_id | **uncovered** | 02 |
| routes.chat_history | PATCH | `/projects/{project_id}` | project_id | project | 02 |
| routes.chat_history | PATCH | `/threads/{thread_id}` | thread_id | chat | 02 |
| routes.chat_history | POST | `/import-ledger` | - | no object-like path parameter | 02 |
| routes.chat_history | POST | `/messages:batch` | - | no object-like path parameter | 02 |
| routes.chat_history | POST | `/projects` | - | no object-like path parameter | 02 |
| routes.chat_history | POST | `/settings/compare-and-set` | - | no object-like path parameter | 02 |
| routes.chat_history | POST | `/threads` | - | no object-like path parameter | 02 |
| routes.chat_history | POST | `/threads/{thread_id}/fork` | thread_id | **uncovered** | 02 |
| routes.chat_history | PUT | `/settings` | - | no object-like path parameter | 02 |
| routes.chat_history | PUT | `/threads/{thread_id}/messages` | thread_id | chat | 02 |
| routes.chat_history | PUT | `/threads/{thread_id}/messages/{message_id}` | thread_id, message_id | chat | 02 |
| routes.data_recipe.jobs | GET | `/jobs/current` | - | no object-like path parameter | 05 |
| routes.data_recipe.jobs | GET | `/jobs/{job_id}/analysis` | job_id | **uncovered** | 05 |
| routes.data_recipe.jobs | GET | `/jobs/{job_id}/dataset` | job_id | **uncovered** | 05 |
| routes.data_recipe.jobs | GET | `/jobs/{job_id}/events` | job_id | **uncovered** | 05 |
| routes.data_recipe.jobs | GET | `/jobs/{job_id}/status` | job_id | **uncovered** | 05 |
| routes.data_recipe.jobs | POST | `/jobs` | - | no object-like path parameter | 05 |
| routes.data_recipe.jobs | POST | `/jobs/{job_id}/cancel` | job_id | **uncovered** | 05 |
| routes.data_recipe.jobs | POST | `/jobs/{job_id}/events` | job_id | **uncovered** | 05 |
| routes.data_recipe.jobs | POST | `/jobs/{job_id}/publish` | job_id | **uncovered** | 05 |
| routes.data_recipe.mcp | POST | `/mcp/tools` | - | no object-like path parameter | 05 |
| routes.data_recipe.seed | DELETE | `/seed/unstructured-block/{block_id}` | block_id | **uncovered** | 05 |
| routes.data_recipe.seed | DELETE | `/seed/unstructured-file/{block_id}/{file_id}` | block_id, file_id | **uncovered** | 05 |
| routes.data_recipe.seed | GET | `/seed/github/env-token` | - | no object-like path parameter | 05 |
| routes.data_recipe.seed | POST | `/seed/inspect` | - | no object-like path parameter | 05 |
| routes.data_recipe.seed | POST | `/seed/inspect-upload` | - | no object-like path parameter | 05 |
| routes.data_recipe.seed | POST | `/seed/upload-unstructured-file` | - | no object-like path parameter | 05 |
| routes.data_recipe.validate | POST | `/validate` | - | no object-like path parameter | 05 |
| routes.datasets | GET | `/download-progress` | - | no object-like path parameter | 04 |
| routes.datasets | GET | `/local` | - | no object-like path parameter | 04 |
| routes.datasets | POST | `/ai-assist-mapping` | - | no object-like path parameter | 04 |
| routes.datasets | POST | `/check-format` | - | no object-like path parameter | 04 |
| routes.datasets | POST | `/upload` | - | no object-like path parameter | 04 |
| routes.export | GET | `/logs` | - | no object-like path parameter | 05 |
| routes.export | GET | `/logs/stream` | - | no object-like path parameter | 05 |
| routes.export | GET | `/status` | - | no object-like path parameter | 05 |
| routes.export | POST | `/cancel` | - | no object-like path parameter | 05 |
| routes.export | POST | `/cleanup` | - | no object-like path parameter | 05 |
| routes.export | POST | `/export/base` | - | no object-like path parameter | 05 |
| routes.export | POST | `/export/gguf` | - | no object-like path parameter | 05 |
| routes.export | POST | `/export/lora` | - | no object-like path parameter | 05 |
| routes.export | POST | `/export/merged` | - | no object-like path parameter | 05 |
| routes.export | POST | `/load-checkpoint` | - | no object-like path parameter | 05 |
| routes.export | POST | `/logs/stream` | - | no object-like path parameter | 05 |
| routes.inference | DELETE | `/audio/gallery` | - | no object-like path parameter | 06 |
| routes.inference | DELETE | `/audio/gallery/{audio_id}` | audio_id | **uncovered** | 06 |
| routes.inference | DELETE | `/images/gallery` | - | no object-like path parameter | 06 |
| routes.inference | DELETE | `/images/gallery/{image_id}` | image_id | **uncovered** | 06 |
| routes.inference | DELETE | `/monitor` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/active-generations` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/artifact-preview-frame` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/audio/gallery` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/audio/gallery/{audio_id}/file` | audio_id | **uncovered** | 06 |
| routes.inference | GET | `/audio/stt/status` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/images/gallery` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/images/gallery/{image_id}/file` | image_id | **uncovered** | 06 |
| routes.inference | GET | `/images/gallery/{image_id}/file-signed` | image_id | **uncovered** | 06 |
| routes.inference | GET | `/images/generate-progress` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/images/info` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/images/load-progress` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/images/status` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/llama-flags` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/load-progress` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/models` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/models/` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/models/{model_id:path}` | model_id | **uncovered** | 06 |
| routes.inference | GET | `/monitor` | - | no object-like path parameter | 06 |
| routes.inference | GET | `/monitor/{entry_id}` | entry_id | **uncovered** | 06 |
| routes.inference | GET | `/sandbox/{session_id}` | session_id | **uncovered** | 06 |
| routes.inference | GET | `/sandbox/{session_id}/{filename:path}` | session_id, filename | **uncovered** | 06 |
| routes.inference | GET | `/search-images/{image_id}` | image_id | **uncovered** | 06 |
| routes.inference | GET | `/status` | - | no object-like path parameter | 06 |
| routes.inference | HEAD | `/sandbox/{session_id}/{filename:path}` | session_id, filename | **uncovered** | 06 |
| routes.inference | PATCH | `/audio/gallery/{audio_id}` | audio_id | **uncovered** | 06 |
| routes.inference | PATCH | `/images/gallery/{image_id}` | image_id | **uncovered** | 06 |
| routes.inference | POST | `/audio/download-plan` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/generate` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/speech` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/stt/download` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/stt/download/cancel` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/stt/load` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/stt/unload` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/stt/validate` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/transcribe` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/transcribe/raw` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/audio/transcriptions` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/cancel` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/chat/completions` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/chat/count_tokens` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/completions` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/embeddings` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/estimate-memory` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/external/openai/containers/create` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/external/openai/containers/delete` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/external/openai/containers/list` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/generate/stream` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/images/download-plan` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/images/generate` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/images/generate/cancel` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/images/generations` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/images/load` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/images/unload` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/install-latest-transformers` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/load` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/messages` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/messages/count_tokens` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/responses` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/sandbox/{session_id}/reveal` | session_id | **uncovered** | 06 |
| routes.inference | POST | `/search-images/lookup` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/tool-confirm` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/transformers-upgrade-check` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/unload` | - | no object-like path parameter | 06 |
| routes.inference | POST | `/validate` | - | no object-like path parameter | 06 |
| routes.llama | GET | `/backend` | - | no object-like path parameter | 06 |
| routes.llama | GET | `/update-changelog` | - | no object-like path parameter | 06 |
| routes.llama | GET | `/update-status` | - | no object-like path parameter | 06 |
| routes.llama | POST | `/backend` | - | no object-like path parameter | 06 |
| routes.llama | POST | `/update` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/apply-template` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/apply-template/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/audio/transcriptions` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/audio/transcriptions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/chat/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/chat/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/completion` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/completion/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/cors-proxy` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/cors-proxy/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/detokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/detokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/embedding` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/embedding/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/embeddings` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/embeddings/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/infill` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/infill/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/lora-adapters` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/lora-adapters/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/metrics` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/metrics/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/models` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/models/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/models/load` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/models/load/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/models/sse` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/models/sse/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/models/unload` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/models/unload/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/responses` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/responses/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/slots` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/slots/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/slots/{id_slot}` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/slots/{id_slot}/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/tokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/tokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/tools` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/tools/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/chat/completions/control` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/chat/completions/control/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/stream` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/stream/` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/streams/lookup` | - | no object-like path parameter | 06 |
| routes.llama_compat | DELETE | `/v1/streams/lookup/` | - | no object-like path parameter | 06 |
| routes.llama_compat | GET | `/props` | - | no object-like path parameter | 06 |
| routes.llama_compat | GET | `/props/` | - | no object-like path parameter | 06 |
| routes.llama_compat | GET | `/v1/props` | - | no object-like path parameter | 06 |
| routes.llama_compat | GET | `/v1/props/` | - | no object-like path parameter | 06 |
| routes.llama_compat | GET | `/version` | - | no object-like path parameter | 06 |
| routes.llama_compat | GET | `/version/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/apply-template` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/apply-template/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/audio/transcriptions` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/audio/transcriptions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/chat/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/chat/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/completion` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/completion/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/cors-proxy` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/cors-proxy/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/detokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/detokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/embedding` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/embedding/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/embeddings` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/embeddings/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/infill` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/infill/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/lora-adapters` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/lora-adapters/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/metrics` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/metrics/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/models` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/models/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/models/load` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/models/load/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/models/sse` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/models/sse/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/models/unload` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/models/unload/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/responses` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/responses/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/slots` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/slots/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/slots/{id_slot}` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/slots/{id_slot}/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/tokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/tokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/tools` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/tools/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/chat/completions/control` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/chat/completions/control/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/stream` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/stream/` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/streams/lookup` | - | no object-like path parameter | 06 |
| routes.llama_compat | HEAD | `/v1/streams/lookup/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/apply-template` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/apply-template/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/audio/transcriptions` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/audio/transcriptions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/chat/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/chat/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/completion` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/completion/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/cors-proxy` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/cors-proxy/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/detokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/detokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/embedding` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/embedding/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/embeddings` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/embeddings/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/infill` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/infill/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/lora-adapters` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/lora-adapters/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/metrics` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/metrics/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/models` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/models/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/models/load` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/models/load/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/models/sse` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/models/sse/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/models/unload` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/models/unload/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/responses` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/responses/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/slots` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/slots/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/slots/{id_slot}` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/slots/{id_slot}/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/tokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/tokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/tools` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/tools/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/chat/completions/control` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/chat/completions/control/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/stream` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/stream/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/streams/lookup` | - | no object-like path parameter | 06 |
| routes.llama_compat | PATCH | `/v1/streams/lookup/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/apply-template` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/apply-template/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/audio/transcriptions` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/audio/transcriptions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/chat/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/chat/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/completion` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/completion/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/cors-proxy` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/cors-proxy/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/detokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/detokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/embedding` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/embedding/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/embeddings` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/embeddings/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/infill` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/infill/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/lora-adapters` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/lora-adapters/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/metrics` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/metrics/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/models` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/models/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/models/load` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/models/load/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/models/sse` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/models/sse/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/models/unload` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/models/unload/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/responses` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/responses/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/slots` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/slots/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/slots/{id_slot}` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/slots/{id_slot}/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/tokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/tokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/tools` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/tools/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/chat/completions/control` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/chat/completions/control/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/stream` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/stream/` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/streams/lookup` | - | no object-like path parameter | 06 |
| routes.llama_compat | POST | `/v1/streams/lookup/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/apply-template` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/apply-template/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/audio/transcriptions` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/audio/transcriptions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/chat/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/chat/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/completion` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/completion/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/completions` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/completions/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/cors-proxy` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/cors-proxy/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/detokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/detokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/embedding` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/embedding/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/embeddings` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/embeddings/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/infill` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/infill/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/lora-adapters` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/lora-adapters/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/metrics` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/metrics/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/models` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/models/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/models/load` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/models/load/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/models/sse` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/models/sse/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/models/unload` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/models/unload/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/responses` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/responses/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/slots` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/slots/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/slots/{id_slot}` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/slots/{id_slot}/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/tokenize` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/tokenize/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/tools` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/tools/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/chat/completions/control` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/chat/completions/control/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/chat/completions/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/chat/completions/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/health` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/health/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/rerank` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/rerank/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/reranking` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/reranking/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/responses/input_tokens` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/responses/input_tokens/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/stream` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/stream/` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/streams/lookup` | - | no object-like path parameter | 06 |
| routes.llama_compat | PUT | `/v1/streams/lookup/` | - | no object-like path parameter | 06 |
| routes.mcp_servers | DELETE | `/{server_id}` | server_id | mcp | 07 |
| routes.mcp_servers | GET | `/` | - | no object-like path parameter | 07 |
| routes.mcp_servers | POST | `/` | - | no object-like path parameter | 07 |
| routes.mcp_servers | POST | `/import` | - | no object-like path parameter | 07 |
| routes.mcp_servers | POST | `/stdio/decode` | - | no object-like path parameter | 07 |
| routes.mcp_servers | POST | `/stdio/encode` | - | no object-like path parameter | 07 |
| routes.mcp_servers | POST | `/test` | - | no object-like path parameter | 07 |
| routes.mcp_servers | POST | `/{server_id}/refresh` | server_id | **uncovered** | 07 |
| routes.mcp_servers | PUT | `/{server_id}` | server_id | mcp | 07 |
| routes.models | DELETE | `/delete-cached` | - | no object-like path parameter | 08 |
| routes.models | DELETE | `/delete-finetuned` | - | no object-like path parameter | 08 |
| routes.models | DELETE | `/scan-folders/{folder_id}` | folder_id | **uncovered** | 08 |
| routes.models | GET | `/browse-folders` | - | no object-like path parameter | 08 |
| routes.models | GET | `/cached-gguf` | - | no object-like path parameter | 08 |
| routes.models | GET | `/cached-model-path` | - | no object-like path parameter | 08 |
| routes.models | GET | `/cached-models` | - | no object-like path parameter | 08 |
| routes.models | GET | `/check-embedding/{model_name:path}` | - | no object-like path parameter | 08 |
| routes.models | GET | `/check-vision/{model_name:path}` | - | no object-like path parameter | 08 |
| routes.models | GET | `/checkpoints` | - | no object-like path parameter | 08 |
| routes.models | GET | `/config/{model_name:path}` | - | no object-like path parameter | 08 |
| routes.models | GET | `/diffusion-controlnets` | - | no object-like path parameter | 08 |
| routes.models | GET | `/diffusion-loras` | - | no object-like path parameter | 08 |
| routes.models | GET | `/download-progress` | - | no object-like path parameter | 08 |
| routes.models | GET | `/export-size` | - | no object-like path parameter | 08 |
| routes.models | GET | `/gguf-download-progress` | - | no object-like path parameter | 08 |
| routes.models | GET | `/gguf-variants` | - | no object-like path parameter | 08 |
| routes.models | GET | `/kv-cache-estimate` | - | no object-like path parameter | 08 |
| routes.models | GET | `/list` | - | no object-like path parameter | 08 |
| routes.models | GET | `/local` | - | no object-like path parameter | 08 |
| routes.models | GET | `/loras` | - | no object-like path parameter | 08 |
| routes.models | GET | `/loras/{lora_path:path}/base-model` | - | no object-like path parameter | 08 |
| routes.models | GET | `/recommended-folders` | - | no object-like path parameter | 08 |
| routes.models | GET | `/scan-folders` | - | no object-like path parameter | 08 |
| routes.models | POST | `/discard-remote-code` | - | no object-like path parameter | 08 |
| routes.models | POST | `/remote-code-scan` | - | no object-like path parameter | 08 |
| routes.models | POST | `/reveal-cached-model` | - | no object-like path parameter | 08 |
| routes.models | POST | `/scan-folders` | - | no object-like path parameter | 08 |
| routes.openai_codex_auth | DELETE | `/{provider_id}/oauth` | provider_id | **uncovered** | 03 |
| routes.openai_codex_auth | DELETE | `/{provider_id}/oauth/flows/{flow_id}` | provider_id, flow_id | **uncovered** | 03 |
| routes.openai_codex_auth | GET | `/{provider_id}/codex/models` | provider_id | **uncovered** | 03 |
| routes.openai_codex_auth | GET | `/{provider_id}/oauth/flows/{flow_id}` | provider_id, flow_id | **uncovered** | 03 |
| routes.openai_codex_auth | POST | `/{provider_id}/oauth/flows/{flow_id}/complete` | provider_id, flow_id | **uncovered** | 03 |
| routes.openai_codex_auth | POST | `/{provider_id}/oauth/start` | provider_id | **uncovered** | 03 |
| routes.preview | GET | `` | - | no object-like path parameter | 08 |
| routes.preview | GET | `/_assets/{asset_path:path}` | - | no object-like path parameter | 08 |
| routes.preview | GET | `/{run}` | - | no object-like path parameter | 08 |
| routes.preview | GET | `/{run}/v1/models` | - | no object-like path parameter | 08 |
| routes.preview | GET | `/{run}/{checkpoint}` | - | no object-like path parameter | 08 |
| routes.preview | GET | `/{run}/{checkpoint}/v1/models` | - | no object-like path parameter | 08 |
| routes.preview | POST | `/{run}/v1/chat/completions` | - | no object-like path parameter | 08 |
| routes.preview | POST | `/{run}/{checkpoint}/v1/chat/completions` | - | no object-like path parameter | 08 |
| routes.profile_stats | GET | `/stats` | - | no object-like path parameter | 02 |
| routes.prompts | DELETE | `/entries/{entry_id}` | entry_id | **uncovered** | 02 |
| routes.prompts | DELETE | `/lists/{list_id}` | list_id | **uncovered** | 02 |
| routes.prompts | GET | `/entries` | - | no object-like path parameter | 02 |
| routes.prompts | GET | `/lists` | - | no object-like path parameter | 02 |
| routes.prompts | POST | `/entries/bulk` | - | no object-like path parameter | 02 |
| routes.prompts | POST | `/lists/bulk` | - | no object-like path parameter | 02 |
| routes.prompts | PUT | `/entries/{entry_id}` | entry_id | **uncovered** | 02 |
| routes.prompts | PUT | `/lists/{list_id}` | list_id | **uncovered** | 02 |
| routes.providers | DELETE | `/{provider_id}` | provider_id | **uncovered** | 03 |
| routes.providers | GET | `/` | - | no object-like path parameter | 03 |
| routes.providers | GET | `/pricing` | - | no object-like path parameter | 03 |
| routes.providers | GET | `/public-key` | - | no object-like path parameter | 03 |
| routes.providers | GET | `/registry` | - | no object-like path parameter | 03 |
| routes.providers | POST | `/` | - | no object-like path parameter | 03 |
| routes.providers | POST | `/models` | - | no object-like path parameter | 03 |
| routes.providers | POST | `/test` | - | no object-like path parameter | 03 |
| routes.providers | PUT | `/{provider_id}` | provider_id | **uncovered** | 03 |
| routes.providers | PUT | `/{provider_id}/api-key/migrate` | provider_id | **uncovered** | 03 |
| routes.rag | DELETE | `/documents/{document_id}` | document_id | **uncovered** | 04 |
| routes.rag | DELETE | `/knowledge-bases/{kb_id}` | kb_id | **uncovered** | 04 |
| routes.rag | DELETE | `/linked-folders/{folder_id}` | folder_id | **uncovered** | 04 |
| routes.rag | GET | `/documents` | - | no object-like path parameter | 04 |
| routes.rag | GET | `/documents/{document_id}/file-signed` | document_id | **uncovered** | 04 |
| routes.rag | GET | `/documents/{document_id}/file-url` | document_id | **uncovered** | 04 |
| routes.rag | GET | `/documents/{document_id}/preview-target` | document_id | **uncovered** | 04 |
| routes.rag | GET | `/jobs/{job_id}` | job_id | **uncovered** | 04 |
| routes.rag | GET | `/jobs/{job_id}/events` | job_id | **uncovered** | 04 |
| routes.rag | GET | `/knowledge-bases` | - | no object-like path parameter | 04 |
| routes.rag | GET | `/knowledge-bases/{kb_id}/documents` | kb_id | **uncovered** | 04 |
| routes.rag | GET | `/linked-folder-jobs/{job_id}` | job_id | **uncovered** | 04 |
| routes.rag | GET | `/linked-folder-jobs/{job_id}/events` | job_id | **uncovered** | 04 |
| routes.rag | GET | `/linked-folders` | - | no object-like path parameter | 04 |
| routes.rag | GET | `/projects/{project_id}/documents` | project_id | **uncovered** | 04 |
| routes.rag | GET | `/threads/{thread_id}/documents` | thread_id | **uncovered** | 04 |
| routes.rag | PATCH | `/knowledge-bases/{kb_id}` | kb_id | **uncovered** | 04 |
| routes.rag | PATCH | `/linked-folders/{folder_id}` | folder_id | **uncovered** | 04 |
| routes.rag | POST | `/jobs/{job_id}/events` | job_id | **uncovered** | 04 |
| routes.rag | POST | `/knowledge-bases` | - | no object-like path parameter | 04 |
| routes.rag | POST | `/knowledge-bases/{kb_id}/documents` | kb_id | **uncovered** | 04 |
| routes.rag | POST | `/knowledge-bases/{kb_id}/linked-folders` | kb_id | **uncovered** | 04 |
| routes.rag | POST | `/linked-folder-jobs/{job_id}/events` | job_id | **uncovered** | 04 |
| routes.rag | POST | `/linked-folders/{folder_id}/rebuild` | folder_id | **uncovered** | 04 |
| routes.rag | POST | `/linked-folders/{folder_id}/sync` | folder_id | **uncovered** | 04 |
| routes.rag | POST | `/projects/{project_id}/documents` | project_id | **uncovered** | 04 |
| routes.rag | POST | `/projects/{project_id}/linked-folders` | project_id | **uncovered** | 04 |
| routes.rag | POST | `/search` | - | no object-like path parameter | 04 |
| routes.rag | POST | `/threads/{thread_id}/documents` | thread_id | **uncovered** | 04 |
| routes.research_runs | GET | `/active` | - | no object-like path parameter | 07 |
| routes.research_runs | GET | `/{run_id}` | run_id | **uncovered** | 07 |
| routes.research_runs | GET | `/{run_id}/events` | run_id | **uncovered** | 07 |
| routes.research_runs | POST | `` | - | no object-like path parameter | 07 |
| routes.research_runs | POST | `/{run_id}/approve` | run_id | **uncovered** | 07 |
| routes.research_runs | POST | `/{run_id}/cancel` | run_id | **uncovered** | 07 |
| routes.research_runs | POST | `/{run_id}/events` | run_id | **uncovered** | 07 |
| routes.research_runs | POST | `/{run_id}/retry` | run_id | **uncovered** | 07 |
| routes.research_runs | PUT | `/{run_id}/plan` | run_id | **uncovered** | 07 |
| routes.settings | DELETE | `/embedding-model` | - | no object-like path parameter | 08 |
| routes.settings | DELETE | `/generation-presets/{kind}/custom` | - | no object-like path parameter | 08 |
| routes.settings | DELETE | `/hugging-face-token` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/chat-preferences` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/coding-agents` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/current-date-prompt` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/debug/logs` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/debug/logs/sources` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/download-transport` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/embedding-model` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/embedding-model/resolve` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/generation-presets/image` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/generation-presets/video` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/helper-precache` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/hugging-face-cache` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/hugging-face-token` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/keyless-api-access` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/lan-access` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/last-local-model` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/llama-cpp-path` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/model-memory` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/openai-auto-switch` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/openai-auto-switch/overrides` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/personalization` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/preview-sharing` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/remote-access` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/upload-limit` | - | no object-like path parameter | 08 |
| routes.settings | GET | `/vram-budget` | - | no object-like path parameter | 08 |
| routes.settings | POST | `/chat-preferences/migrate` | - | no object-like path parameter | 08 |
| routes.settings | POST | `/embedding-model/unload` | - | no object-like path parameter | 08 |
| routes.settings | POST | `/lan-access/start` | - | no object-like path parameter | 08 |
| routes.settings | POST | `/lan-access/stop` | - | no object-like path parameter | 08 |
| routes.settings | POST | `/preview-links/rotate` | - | no object-like path parameter | 08 |
| routes.settings | POST | `/remote-access/start` | - | no object-like path parameter | 08 |
| routes.settings | POST | `/remote-access/stop` | - | no object-like path parameter | 08 |
| routes.settings | POST | `/xet-notice/reserve` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/chat-preferences` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/current-date-prompt` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/download-transport` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/embedding-model` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/generation-presets/image` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/generation-presets/image/custom` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/generation-presets/video` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/generation-presets/video/custom` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/helper-precache` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/hugging-face-cache` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/hugging-face-token` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/hugging-face-token/migrate` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/keyless-api-access` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/lan-access/auto-start` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/lan-access/port` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/last-local-model` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/llama-cpp-path` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/model-memory` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/openai-auto-switch` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/openai-auto-switch/overrides` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/personalization` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/preview-sharing` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/remote-access/auto-start` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/upload-limit` | - | no object-like path parameter | 08 |
| routes.settings | PUT | `/vram-budget` | - | no object-like path parameter | 08 |
| routes.training | DELETE | `/diffusion/dataset/{name}/image/{filename}` | name, filename | **uncovered** | 05 |
| routes.training | GET | `/diffusion/dataset-examples` | - | no object-like path parameter | 05 |
| routes.training | GET | `/diffusion/dataset/{name}/image/{filename}` | name, filename | **uncovered** | 05 |
| routes.training | GET | `/diffusion/dataset/{name}/images` | name | **uncovered** | 05 |
| routes.training | GET | `/diffusion/info` | - | no object-like path parameter | 05 |
| routes.training | GET | `/diffusion/runs` | - | no object-like path parameter | 05 |
| routes.training | GET | `/diffusion/runs/{job_id}` | job_id | **uncovered** | 05 |
| routes.training | GET | `/diffusion/status` | - | no object-like path parameter | 05 |
| routes.training | GET | `/hardware` | - | no object-like path parameter | 05 |
| routes.training | GET | `/hardware/visible` | - | no object-like path parameter | 05 |
| routes.training | GET | `/metrics` | - | no object-like path parameter | 05 |
| routes.training | GET | `/progress` | - | no object-like path parameter | 05 |
| routes.training | GET | `/start-requests/{start_request_id}` | start_request_id | **uncovered** | 05 |
| routes.training | GET | `/status` | - | no object-like path parameter | 05 |
| routes.training | POST | `/diffusion/dataset` | - | no object-like path parameter | 05 |
| routes.training | POST | `/diffusion/dataset/import-example` | - | no object-like path parameter | 05 |
| routes.training | POST | `/diffusion/start` | - | no object-like path parameter | 05 |
| routes.training | POST | `/diffusion/stop` | - | no object-like path parameter | 05 |
| routes.training | POST | `/progress` | - | no object-like path parameter | 05 |
| routes.training | POST | `/reset` | - | no object-like path parameter | 05 |
| routes.training | POST | `/start` | - | no object-like path parameter | 05 |
| routes.training | POST | `/start-requests/{start_request_id}/acknowledge` | start_request_id | **uncovered** | 05 |
| routes.training | POST | `/start-requests/{start_request_id}/cancel` | start_request_id | **uncovered** | 05 |
| routes.training | POST | `/stop` | - | no object-like path parameter | 05 |
| routes.training | PUT | `/diffusion/dataset/{name}/caption/{filename}` | name, filename | **uncovered** | 05 |
| routes.training_history | DELETE | `/runs/{run_id}` | run_id | **uncovered** | 05 |
| routes.training_history | GET | `/runs` | - | no object-like path parameter | 05 |
| routes.training_history | GET | `/runs/{run_id}` | run_id | training | 05 |
| routes.training_history | PATCH | `/runs/{run_id}` | run_id | training | 05 |
| routes.video | DELETE | `/video/gallery` | - | no object-like path parameter | 06 |
| routes.video | DELETE | `/video/gallery/{video_id}` | video_id | **uncovered** | 06 |
| routes.video | DELETE | `/videos/{video_id}` | video_id | **uncovered** | 06 |
| routes.video | GET | `/video/gallery` | - | no object-like path parameter | 06 |
| routes.video | GET | `/video/gallery/{video_id}/export` | video_id | **uncovered** | 06 |
| routes.video | GET | `/video/gallery/{video_id}/file` | video_id | **uncovered** | 06 |
| routes.video | GET | `/video/gallery/{video_id}/file-signed` | video_id | **uncovered** | 06 |
| routes.video | GET | `/video/gallery/{video_id}/signed-url` | video_id | **uncovered** | 06 |
| routes.video | GET | `/video/generate-progress` | - | no object-like path parameter | 06 |
| routes.video | GET | `/video/load-progress` | - | no object-like path parameter | 06 |
| routes.video | GET | `/video/status` | - | no object-like path parameter | 06 |
| routes.video | GET | `/videos` | - | no object-like path parameter | 06 |
| routes.video | GET | `/videos/{video_id}` | video_id | **uncovered** | 06 |
| routes.video | GET | `/videos/{video_id}/content` | video_id | **uncovered** | 06 |
| routes.video | PATCH | `/video/gallery/{video_id}` | video_id | **uncovered** | 06 |
| routes.video | POST | `/video/download-plan` | - | no object-like path parameter | 06 |
| routes.video | POST | `/video/generate` | - | no object-like path parameter | 06 |
| routes.video | POST | `/video/generate/cancel` | - | no object-like path parameter | 06 |
| routes.video | POST | `/video/load` | - | no object-like path parameter | 06 |
| routes.video | POST | `/video/unload` | - | no object-like path parameter | 06 |
| routes.video | POST | `/videos` | - | no object-like path parameter | 06 |
| routes.whisper | GET | `/update-status` | - | no object-like path parameter | 06 |
| routes.youtube | POST | `/transcript` | - | no object-like path parameter | 04 |
