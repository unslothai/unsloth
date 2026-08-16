import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getChatSettings: vi.fn(),
  saveChatSettingsPatch: vi.fn(),
}));

vi.mock("../api/chat-settings-api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api/chat-settings-api")>()),
  getChatSettings: apiMocks.getChatSettings,
  saveChatSettingsPatch: apiMocks.saveChatSettingsPatch,
}));

import {
  loadChatSettingsWithLegacyImport,
  savePersistedChatSettingsPatch,
} from "./chat-settings-storage";

describe("platform chat settings storage", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    vi.stubEnv("VITE_RAG_PLATFORM_ENABLED", "true");
    vi.stubEnv("VITE_RAG_PLATFORM_AUTH_ENABLED", "true");
  });

  afterEach(() => vi.unstubAllEnvs());

  it("loads and saves settings locally without calling unsupported Studio endpoints", async () => {
    localStorage.setItem(
      "unsloth_chat_inference_params",
      JSON.stringify({ temperature: 0.2, topP: 0.8 }),
    );

    await expect(
      savePersistedChatSettingsPatch({
        inferenceParams: { temperature: 0.7 },
        autoTitle: false,
      }),
    ).resolves.toMatchObject({
      inferenceParams: { temperature: 0.7, topP: 0.8 },
      autoTitle: false,
    });

    await expect(loadChatSettingsWithLegacyImport()).resolves.toMatchObject({
      inferenceParams: { temperature: 0.7, topP: 0.8 },
      autoTitle: false,
    });
    expect(apiMocks.getChatSettings).not.toHaveBeenCalled();
    expect(apiMocks.saveChatSettingsPatch).not.toHaveBeenCalled();
  });

  it("sanitizes unsupported fields before writing local settings", async () => {
    await savePersistedChatSettingsPatch({
      inferenceParams: {
        temperature: 0.4,
        trustRemoteCode: true,
      } as never,
    });

    expect(localStorage.getItem("rag_platform_chat_settings")).not.toContain(
      "trustRemoteCode",
    );
  });
});
