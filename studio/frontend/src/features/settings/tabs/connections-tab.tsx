import { ChatProvidersSettings } from "@/features/chat/chat-providers-dialog";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import {
  isPlatformAuthEnabled,
  isPlatformModelToolsEnabled,
} from "@/integrations/platform-backend";
import { useCallback, useState } from "react";
import { PlatformModelsSettings } from "../components/platform-models-settings";

export function ConnectionsTab() {
  const providers = useExternalProvidersStore((s) => s.providers);
  const setProviders = useExternalProvidersStore((s) => s.setProviders);
  const platformModelsEnabled =
    isPlatformAuthEnabled() && isPlatformModelToolsEnabled();
  const [platformRevision, setPlatformRevision] = useState(0);
  const [platformSummary, setPlatformSummary] = useState({
    connections: 0,
    models: 0,
  });
  const handlePlatformSummary = useCallback(
    (summary: { connections: number; models: number }) =>
      setPlatformSummary(summary),
    [],
  );

  return (
    <div data-testid="connections-tab-content">
      <ChatProvidersSettings
        providers={providers}
        onProvidersChange={setProviders}
        platformConnectionCount={platformSummary.connections}
        platformModelCount={platformSummary.models}
        platformConnection={
          platformModelsEnabled
            ? ({ close }) => (
                <PlatformModelsSettings
                  mode="create"
                  refreshKey={platformRevision}
                  onCreated={() => {
                    setPlatformRevision((revision) => revision + 1);
                    close();
                  }}
                />
              )
            : undefined
        }
        platformConnections={
          platformModelsEnabled ? (
            <PlatformModelsSettings
              mode="manage"
              refreshKey={platformRevision}
              onSummaryChange={handlePlatformSummary}
            />
          ) : undefined
        }
      />
    </div>
  );
}
