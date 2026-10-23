// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ApiSamplingDefaults = Partial<{
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repetition_penalty: number;
  presence_penalty: number;
}>;

// The OpenAI client takes these as kwargs; the Unsloth extensions only go through extra_body.
const CLIENT_FIELDS = ["temperature", "top_p", "presence_penalty"] as const;
const EXTRA_FIELDS = ["top_k", "min_p", "repetition_penalty"] as const;

export function apiExampleModelName(
  id: string,
  ggufVariant: string | null,
): string {
  return ggufVariant && !id.includes(":") ? `${id}:${ggufVariant}` : id;
}

export function buildApiSamplingExample({
  baseUrl,
  model,
  sampling,
}: {
  baseUrl: string;
  model: string;
  sampling: ApiSamplingDefaults;
}): string {
  const args = CLIENT_FIELDS.filter((field) => sampling[field] != null).map(
    (field) => `    ${field}=${sampling[field]},`,
  );
  const extras = EXTRA_FIELDS.filter((field) => sampling[field] != null).map(
    (field) => `        "${field}": ${sampling[field]},`,
  );
  return [
    "from openai import OpenAI",
    "",
    "client = OpenAI(",
    `    base_url=${JSON.stringify(baseUrl)},`,
    '    api_key="sk-unsloth-YOUR_KEY",',
    ")",
    "",
    "response = client.chat.completions.create(",
    `    model=${JSON.stringify(model)},`,
    '    messages=[{"role": "user", "content": "Hello!"}],',
    ...args,
    ...(extras.length > 0 ? ["    extra_body={", ...extras, "    },"] : []),
    ")",
    "print(response.choices[0].message.content)",
  ].join("\n");
}
