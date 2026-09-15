// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export declare const chatApiStub: {
  status: Record<string, unknown>;
  validated: string[];
  resident: boolean;
};
export declare function getInferenceStatus(): Promise<Record<string, unknown>>;
export declare function validateModel(request: {
  // biome-ignore lint/style/useNamingConvention: api schema
  model_path: string;
}): Promise<{ valid: boolean; resident: boolean }>;
export declare function resolve(
  specifier: string,
  context: unknown,
  next: (specifier: string, context: unknown) => unknown,
): unknown;
