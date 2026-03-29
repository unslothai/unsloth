# External Providers — Frontend Integration Spec

## Overview

The backend proxies chat requests to external LLM providers (OpenAI, Mistral, Gemini, Cohere, Together AI, Fireworks AI, Perplexity). **API keys are never stored on the backend** — the frontend holds them in localStorage and encrypts them before every request using the server's RSA public key.

---

## 1. API Key Encryption

The server generates an RSA-2048 key pair on startup (rotates on restart). The frontend must:

1. **Fetch the public key** on app load (and after any 400 "public key may have changed" error)
2. **Encrypt API keys** with RSA-OAEP + SHA-256 before including in any request
3. **Base64-encode** the ciphertext

```ts
// Fetch once on load, cache in memory
const res = await authFetch("GET", "/api/providers/public-key");
const pem: string = res.public_key;

// Helper: PEM string -> ArrayBuffer
function pemToBuffer(pem: string): ArrayBuffer {
  const b64 = pem.replace(/-----[^-]+-----/g, "").replace(/\s/g, "");
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}

// Import the key (do once)
const cryptoKey = await crypto.subtle.importKey(
  "spki",
  pemToBuffer(pem),
  { name: "RSA-OAEP", hash: "SHA-256" },
  false,
  ["encrypt"],
);

// Encrypt before each request
async function encryptApiKey(plaintext: string): Promise<string> {
  const encoded = new TextEncoder().encode(plaintext);
  const encrypted = await crypto.subtle.encrypt(
    { name: "RSA-OAEP" },
    cryptoKey,
    encoded,
  );
  return btoa(String.fromCharCode(...new Uint8Array(encrypted)));
}
```

No npm packages needed — `crypto.subtle` is a native browser API.

---

## 2. Endpoints

All endpoints require auth (`Authorization: Bearer <jwt>`).

### GET `/api/providers/public-key`

Returns the RSA public key for encrypting API keys.

```json
// Response
{ "public_key": "-----BEGIN PUBLIC KEY-----\nMIIBI..." }
```

### GET `/api/providers/registry`

Returns all supported provider types with defaults. Use this to populate the "Add Provider" dropdown.

```json
// Response
[
  {
    "provider_type": "openai",
    "display_name": "OpenAI",
    "base_url": "https://api.openai.com/v1",
    "default_models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3-mini"],
    "supports_streaming": true,
    "supports_vision": true,
    "supports_tool_calling": true
  }
  // ... 6 more providers
]
```

**Supported `provider_type` values:** `openai`, `mistral`, `google`, `cohere`, `together`, `fireworks`, `perplexity`

### GET `/api/providers`

List saved provider configs. These store display name + base URL — **not** the API key.

```json
// Response
[
  {
    "id": "a1b2c3d4e5f67890",
    "provider_type": "openai",
    "display_name": "My OpenAI Key",
    "base_url": "https://api.openai.com/v1",
    "is_enabled": true,
    "created_at": "2026-03-29T...",
    "updated_at": "2026-03-29T..."
  }
]
```

### POST `/api/providers`

Create a saved provider config. No API key sent here.

```json
// Request
{
  "provider_type": "openai",
  "display_name": "My OpenAI Key",
  "base_url": null  // omit to use registry default
}

// Response (201)
{
  "id": "a1b2c3d4e5f67890",
  "provider_type": "openai",
  "display_name": "My OpenAI Key",
  "base_url": "https://api.openai.com/v1",
  "is_enabled": true,
  "created_at": "2026-03-29T...",
  "updated_at": "2026-03-29T..."
}
```

### PUT `/api/providers/{id}`

Update a provider config. All fields optional.

```json
// Request
{
  "display_name": "Work OpenAI",
  "base_url": null,
  "is_enabled": false
}

// Response -> same shape as GET items
```

### DELETE `/api/providers/{id}`

Delete a provider config. Returns `204 No Content`.

### POST `/api/providers/test`

Test if an API key works. Encrypted key required.

```json
// Request
{
  "provider_type": "openai",
  "encrypted_api_key": "<base64 RSA-encrypted key>",
  "base_url": null  // optional override
}

// Response
{
  "success": true,
  "message": "Connected successfully. Found 42 model(s).",
  "models_count": 42
}
```

### POST `/api/providers/models`

List available models from a provider. Encrypted key required.

```json
// Request
{
  "provider_type": "openai",
  "encrypted_api_key": "<base64 RSA-encrypted key>",
  "base_url": null
}

// Response
[
  {
    "id": "gpt-4o",
    "display_name": "gpt-4o",
    "context_length": 128000,
    "owned_by": "openai"
  }
  // ...
]
```

---

## 3. Chatting with an External Provider

Use the **existing** `POST /v1/chat/completions` endpoint — just add provider fields. The backend detects them and proxies to the external API instead of local inference.

```json
// Request
{
  "messages": [
    { "role": "user", "content": "Hello!" }
  ],
  "stream": true,
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 1024,
  "presence_penalty": 0.0,

  // --- These trigger external routing ---
  "provider_type": "openai",           // required (or provider_id)
  "external_model": "gpt-4o-mini",     // required - model ID at the provider
  "encrypted_api_key": "<base64...>",  // required - RSA-encrypted
  "provider_id": "a1b2c3d4e5f67890",  // optional - saved config ID (for base_url lookup)
  "provider_base_url": null            // optional - override base URL
}
```

### Routing logic

- If `encrypted_api_key` + (`provider_type` or `provider_id`) are present -> routes to external provider
- Otherwise -> routes to local inference (existing behavior, unchanged)

### Response format

Standard OpenAI SSE streaming — same format the frontend already handles. No changes needed to the stream parser.

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"Hi"},"index":0}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":" there!"},"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

### Error format

If the provider returns an error, it comes as an SSE event:

```
data: {"error":{"message":"Invalid API key","type":"provider_error","code":"401","provider":"openai"}}
```

### Fields to skip for external providers

When chatting with an external provider, these local-only fields should **not** be sent (they will be ignored but are unnecessary):

- `top_k`, `min_p`, `repetition_penalty`
- `image_base64`, `audio_base64`
- `use_adapter`
- `enable_thinking`
- `enable_tools`, `enabled_tools`, `auto_heal_tool_calls`, `max_tool_calls_per_message`, `tool_call_timeout`
- `session_id`

Only send standard OpenAI fields: `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `presence_penalty`.

---

## 4. Frontend Storage Model

The frontend should store in **localStorage**:

| Key | Value | Notes |
|-----|-------|-------|
| Provider API keys | `{ [provider_id]: "sk-abc123..." }` | Plaintext in localStorage, encrypted before sending |
| Active provider | `provider_id` or `null` | `null` = local inference |
| Active external model | `"gpt-4o-mini"` or `null` | Model ID at the selected provider |

---

## 5. Typical User Flow

1. User opens **Provider Settings page**
2. Frontend calls `GET /api/providers/registry` -> shows available provider types
3. User picks "OpenAI", enters API key, names it "My OpenAI"
4. Frontend calls `POST /api/providers` to save config (without key)
5. Frontend stores the API key in localStorage keyed by the returned `id`
6. Frontend encrypts key -> calls `POST /api/providers/test` -> shows success/fail
7. User goes to **Chat**, selects the provider + model from the model selector
8. On each message, frontend encrypts the key and adds `provider_type`, `external_model`, `encrypted_api_key` to the existing chat completions request
9. Response streams back in the same SSE format as local inference — no parser changes needed

---

## 6. UI Considerations

### Provider Settings Page (separate page in app nav)

- List configured providers with enable/disable toggles
- "Add Provider" form: dropdown (from registry), API key input, display name
- "Test Connection" button per provider
- Edit / Delete actions per provider
- Show provider status (connected / error)

### Chat Model Selector

- Add a "Cloud" or "API" section alongside local models
- Group external models by provider (e.g. "OpenAI / gpt-4o")
- When external model selected: set `activeProviderId` + `activeExternalModel` in store
- When local model selected: clear provider state (back to `null`)

### Chat Page Adaptations

When an external provider is active:

- **Hide** local-only UI: context length bar, GGUF settings, LoRA controls, KV cache dtype
- **Hide** local-only features: reasoning toggle, tool calling controls (unless provider supports them)
- **Show** simplified params: temperature, top_p, max_tokens, presence_penalty only
- **Skip** auto-load logic (no local model needed)
- **Show** provider badge/icon next to model name

---

## 7. Error Handling

| Scenario | How to detect | What to do |
|----------|---------------|------------|
| Public key rotated (server restarted) | 400 with "public key may have changed" | Re-fetch `GET /api/providers/public-key`, re-encrypt, retry |
| Invalid API key | SSE error with `code: "401"` | Show "API key invalid or expired" message |
| Provider down | SSE error with `code: "502"` or `code: "504"` | Show "Provider unavailable, try again later" |
| Rate limited | SSE error with `code: "429"` | Show "Rate limited, please wait" |
| Unknown provider type | 400 from POST endpoints | Should not happen if using registry dropdown |
