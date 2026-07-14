// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  create: vi.fn(),
}));

vi.mock("@/features/user-assets", () => ({
  createServerRecipe: mocks.create,
  deleteServerRecipe: vi.fn(),
  getServerRecipe: vi.fn(),
  listServerRecipes: vi.fn(),
  updateServerRecipe: vi.fn(),
}));

import { saveRecipe } from "./recipes-db";

describe("recipe persistence sanitization", () => {
  beforeEach(() => {
    mocks.create.mockReset();
    mocks.create.mockImplementation(async (input) => ({
      ...input,
      revision: 1,
      createdAt: 1,
      updatedAt: 1,
    }));
  });

  it("preserves allowed stdio environment values while removing secrets", async () => {
    const result = await saveRecipe({
      id: "recipe-1",
      name: "Recipe",
      payload: {
        mcp_servers: [
          {
            provider_type: "stdio",
            env: {
              NODE_ENV: "production",
              FEATURE_FLAG: "enabled",
              API_TOKEN: "do-not-store",
            },
          },
        ],
      } as never,
    });

    expect(mocks.create).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: {
          mcp_servers: [
            {
              provider_type: "stdio",
              env: {
                NODE_ENV: "production",
                FEATURE_FLAG: "enabled",
              },
            },
          ],
        },
      }),
    );
    expect(result.removedCredentialPaths).toEqual([
      "mcp_servers.0.env.api_token",
    ]);
  });
});
