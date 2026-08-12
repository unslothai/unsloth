import { ChatProvidersSettings } from "@/features/chat/chat-providers-dialog";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { PlatformBackendConnectionStatus } from "@/integrations/platform-backend";

export function ConnectionsTab() {
  const providers = useExternalProvidersStore((s) => s.providers);
  const setProviders = useExternalProvidersStore((s) => s.setProviders);

  return (
    <div className="flex flex-col gap-4">
      <PlatformBackendConnectionStatus />
      <ChatProvidersSettings
        providers={providers}
        onProvidersChange={setProviders}
      />
    </div>
  );
}
