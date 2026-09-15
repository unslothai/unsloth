// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// chat-api reaches .tsx dialogs, which bare node cannot parse. Register after bundler-resolver.
export const chatApiStub = { status: {}, validated: [], resident: false };

export async function getInferenceStatus() {
  return chatApiStub.status;
}

export async function validateModel(request) {
  chatApiStub.validated.push(request.model_path);
  return { valid: true, resident: chatApiStub.resident };
}

export function resolve(specifier, context, next) {
  return /(^|\/)features\/chat\/api\/chat-api$/.test(specifier)
    ? next(import.meta.url, context)
    : next(specifier, context);
}
