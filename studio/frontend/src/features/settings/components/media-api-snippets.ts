// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { psSingle, shSingle } from "./agent-command.ts";
import type { AudioApiLang, AudioApiOs } from "./audio-api-snippets";

export type MediaApiKind = "images" | "video";

export const MEDIA_API_KINDS: readonly MediaApiKind[] = ["images", "video"];

export const MEDIA_API_PLACEHOLDER_MODELS: Record<MediaApiKind, string> = {
  images: "unsloth/FLUX.2-klein-4B-GGUF",
  video: "unsloth/Wan2.2-TI2V-5B-GGUF",
};

export interface MediaApiSnippetInput {
  base: string;
  apiKey: string;
  model: string;
  lang: AudioApiLang;
  os: AudioApiOs;
}

const IMAGE_PROMPT = "A cozy wooden cabin on a snowy mountain at dusk";
const VIDEO_PROMPT = "A paper boat drifting down a rain-soaked street";
const j = (value: string) => JSON.stringify(value);

function jsonBody(body: Record<string, string>, indent: string): string {
  const lines = Object.entries(body).map(
    ([key, value]) => `${indent}  ${j(key)}: ${j(value)}`,
  );
  return `{\n${lines.join(",\n")}\n${indent}}`;
}

function imagesSnippet({
  base,
  apiKey,
  model,
  lang,
  os,
}: MediaApiSnippetInput) {
  const body = {
    model,
    prompt: IMAGE_PROMPT,
    size: "1024x1024",
    response_format: "b64_json",
  };
  if (lang === "python") {
    return `import base64

from openai import OpenAI

client = OpenAI(base_url=${j(`${base}/v1`)}, api_key=${j(apiKey)})

result = client.images.generate(
    model=${j(model)},
    prompt=${j(IMAGE_PROMPT)},
    size="1024x1024",
    response_format="b64_json",
)
with open("image.png", "wb") as image:
    image.write(base64.b64decode(result.data[0].b64_json))`;
  }
  if (lang === "javascript") {
    return `import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ baseURL: ${j(`${base}/v1`)}, apiKey: ${j(apiKey)} });

const result = await client.images.generate({
  model: ${j(model)},
  prompt: ${j(IMAGE_PROMPT)},
  size: "1024x1024",
  response_format: "b64_json",
});
fs.writeFileSync("image.png", Buffer.from(result.data[0].b64_json, "base64"));`;
  }
  if (os === "windows") {
    return `$base = '${psSingle(base)}'
$key = '${psSingle(apiKey)}'
$body = @'
${jsonBody(body, "")}
'@
Set-Content -Path body.json -Value $body -Encoding ascii
$result = curl.exe -sSf "$base/v1/images/generations" \`
  -H "Authorization: Bearer $key" \`
  -H "Content-Type: application/json" \`
  -d "@body.json" | ConvertFrom-Json
[IO.File]::WriteAllBytes("$PWD\\image.png", [Convert]::FromBase64String($result.data[0].b64_json))`;
  }
  return `BASE='${shSingle(base)}'
KEY='${shSingle(apiKey)}'
curl -sSf "$BASE/v1/images/generations" \\
  -H "Authorization: Bearer $KEY" \\
  -H "Content-Type: application/json" \\
  -d '${shSingle(jsonBody(body, "  "))}' \\
  | jq -r '.data[0].b64_json' | base64 --decode > image.png`;
}

function videoSnippet({ base, apiKey, model, lang, os }: MediaApiSnippetInput) {
  const body = { model, prompt: VIDEO_PROMPT };
  if (lang === "python") {
    return `import time

import httpx

studio = httpx.Client(
    base_url=${j(base)}, headers={"Authorization": "Bearer " + ${j(apiKey)}}, timeout=None
)


def call(method, path, **kwargs):
    response = studio.request(method, path, **kwargs)
    response.raise_for_status()
    return response


job = call("POST", "/v1/videos", json={"model": ${j(model)}, "prompt": ${j(VIDEO_PROMPT)}}).json()
while job["status"] in ("queued", "in_progress"):
    time.sleep(5)
    job = call("GET", f"/v1/videos/{job['id']}").json()
if job["status"] != "completed":
    raise RuntimeError(job.get("error"))
with open("video.mp4", "wb") as video:
    video.write(call("GET", f"/v1/videos/{job['id']}/content").content)`;
  }
  if (lang === "javascript") {
    return `import fs from "node:fs";

const BASE = ${j(base)};
const KEY = ${j(apiKey)};

async function call(path, init = {}) {
  const response = await fetch(BASE + path, {
    ...init,
    headers: { Authorization: "Bearer " + KEY, ...init.headers },
  });
  if (!response.ok) throw new Error(await response.text());
  return response;
}

let job = await (
  await call("/v1/videos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: ${j(model)}, prompt: ${j(VIDEO_PROMPT)} }),
  })
).json();
while (job.status === "queued" || job.status === "in_progress") {
  await new Promise((resolve) => setTimeout(resolve, 5000));
  job = await (await call(\`/v1/videos/\${job.id}\`)).json();
}
if (job.status !== "completed") throw new Error(JSON.stringify(job.error));
const video = await call(\`/v1/videos/\${job.id}/content\`);
fs.writeFileSync("video.mp4", Buffer.from(await video.arrayBuffer()));`;
  }
  if (os === "windows") {
    return `$base = '${psSingle(base)}'
$key = '${psSingle(apiKey)}'
$body = @'
${jsonBody(body, "")}
'@
Set-Content -Path body.json -Value $body -Encoding ascii
$job = curl.exe -sSf "$base/v1/videos" \`
  -H "Authorization: Bearer $key" \`
  -H "Content-Type: application/json" \`
  -d "@body.json" | ConvertFrom-Json
while ($job.status -in "queued", "in_progress") {
  Start-Sleep -Seconds 5
  $job = curl.exe -sSf "$base/v1/videos/$($job.id)" -H "Authorization: Bearer $key" | ConvertFrom-Json
}
if ($job.status -eq "completed") {
  curl.exe -sSf "$base/v1/videos/$($job.id)/content" -H "Authorization: Bearer $key" -o video.mp4
} else {
  $job.error
}`;
  }
  return `BASE='${shSingle(base)}'
KEY='${shSingle(apiKey)}'
ID=$(curl -sSf "$BASE/v1/videos" \\
  -H "Authorization: Bearer $KEY" \\
  -H "Content-Type: application/json" \\
  -d '${shSingle(jsonBody(body, "  "))}' | jq -r .id)
STATUS=queued
while [ "$STATUS" = queued ] || [ "$STATUS" = in_progress ]; do
  sleep 5
  STATUS=$(curl -sSf "$BASE/v1/videos/$ID" -H "Authorization: Bearer $KEY" | jq -r .status)
done
[ "$STATUS" = completed ] && curl -sSf "$BASE/v1/videos/$ID/content" \\
  -H "Authorization: Bearer $KEY" -o video.mp4`;
}

export function buildMediaApiSnippet(
  kind: MediaApiKind,
  input: MediaApiSnippetInput,
): string {
  return kind === "images" ? imagesSnippet(input) : videoSnippet(input);
}
