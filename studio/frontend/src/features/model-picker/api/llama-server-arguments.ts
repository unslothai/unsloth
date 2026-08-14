// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { ActiveLlamaServerArgumentsResponse } from "../model-config/active-arguments-hydration";
import type { LlamaServerArgumentsResponse } from "../model-config/llama-extra-args";

type StableManagedPolicy = Pick<
  LlamaServerArgumentsResponse,
  "managed_flags" | "managed_flag_groups"
>;

let cachedManagedPolicy: StableManagedPolicy | null = null;
let cachedAuthoritativeCatalog: LlamaServerArgumentsResponse | null = null;

function cloneCatalog(
  catalog: LlamaServerArgumentsResponse,
): LlamaServerArgumentsResponse {
  return {
    ...catalog,
    arguments: catalog.arguments.map((argument) => ({
      ...argument,
      aliases: [...argument.aliases],
      choices: [...argument.choices],
    })),
    managed_flags: [...catalog.managed_flags],
    managed_flag_groups: catalog.managed_flag_groups.map((group) => [...group]),
  };
}

export function cachedLlamaServerManagedPolicy(): StableManagedPolicy | null {
  return cachedManagedPolicy
    ? {
        managed_flags: [...cachedManagedPolicy.managed_flags],
        managed_flag_groups: cachedManagedPolicy.managed_flag_groups.map(
          (group) => [...group],
        ),
      }
    : null;
}

export function cachedAuthoritativeLlamaServerArguments(): LlamaServerArgumentsResponse | null {
  return cachedAuthoritativeCatalog
    ? cloneCatalog(cachedAuthoritativeCatalog)
    : null;
}

export async function fetchLlamaServerArguments(): Promise<LlamaServerArgumentsResponse> {
  const response = await authFetch("/api/inference/llama-server/arguments");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(
        response,
        "Failed to inspect the installed llama.cpp build",
      ),
    );
  }
  const result = (await response.json()) as LlamaServerArgumentsResponse;
  cachedManagedPolicy = {
    managed_flags: [...result.managed_flags],
    managed_flag_groups: result.managed_flag_groups.map((group) => [...group]),
  };
  cachedAuthoritativeCatalog = result.authoritative
    ? cloneCatalog(result)
    : null;
  return result;
}

export async function fetchActiveLlamaServerArguments(): Promise<ActiveLlamaServerArgumentsResponse> {
  const response = await authFetch(
    "/api/inference/llama-server/active-arguments",
  );
  if (!response.ok) {
    throw new Error(
      await readFastApiError(
        response,
        "Failed to hydrate the active llama.cpp arguments",
      ),
    );
  }
  return response.json() as Promise<ActiveLlamaServerArgumentsResponse>;
}
