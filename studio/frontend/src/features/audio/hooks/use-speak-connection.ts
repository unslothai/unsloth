// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo, useState } from "react";
import {
  buildExternalModelId,
  isDecisionConnection,
  useExternalProvidersStore,
} from "@/features/chat";
import type { ExternalModelOption } from "@/features/model-picker/components/model-selector/types";
import { connectionSpeechModels } from "../speak-connection-models";

export interface SpeakConnection {
  providerId: string;
  model: string;
  providerName: string;
  voice: string;
  speed: number;
}

export function useSpeakConnection() {
  const connections = useExternalProvidersStore((s) => s.providers);
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const models = useMemo<ExternalModelOption[]>(
    () =>
      connectionsEnabled
        ? connections
            .filter((c) => !isDecisionConnection(c))
            .flatMap((c) =>
              connectionSpeechModels(c).map((model) => ({
                id: buildExternalModelId(c.id, model),
                name: model,
                providerId: c.id,
                providerName: c.name,
                providerType: c.providerType,
              })),
            )
        : [],
    [connections, connectionsEnabled],
  );
  const [modelId, setModelId] = useState<string | null>(null);
  const [voice, setVoice] = useState("");
  const [speed, setSpeed] = useState(1);
  const connection = useMemo<SpeakConnection | null>(() => {
    const option = models.find((model) => model.id === modelId);
    return option
      ? {
          providerId: option.providerId,
          model: option.name,
          providerName: option.providerName,
          voice,
          speed,
        }
      : null;
  }, [models, modelId, voice, speed]);
  return { models, modelId, setModelId, connection, setVoice, setSpeed };
}
