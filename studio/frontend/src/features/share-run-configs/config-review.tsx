import type { PerModelConfig } from "@/features/model-picker";
import { SHARED_CONFIG_FIELDS, SHARED_CONFIG_KEYS } from "./fields";

export function SharedRunConfigReview({
  config,
}: { config: Partial<PerModelConfig> | null }) {
  if (!config) {
    return null;
  }
  const keys = SHARED_CONFIG_KEYS.filter((key) => Object.hasOwn(config, key));
  return (
    <details open className="mb-5 rounded-lg border p-3 text-sm">
      <summary className="cursor-pointer font-medium">
        {keys.length
          ? `Settings changed by link (${keys.length})`
          : "Link settings already match this editor"}
      </summary>
      {keys.length > 0 && (
        <>
          <p className="my-2 text-xs text-muted-foreground">
            These values came from the link. Review text and extra arguments,
            and edit the settings below before loading.
          </p>
          <dl className="max-h-48 space-y-2 overflow-y-auto">
            {keys.map((key) => (
              <div key={key}>
                <dt className="font-medium">
                  {SHARED_CONFIG_FIELDS[key].label}
                </dt>
                <dd className="whitespace-pre-wrap break-words text-xs text-muted-foreground">
                  {config[key] === null
                    ? "Default"
                    : typeof config[key] === "string" && config[key] !== ""
                      ? config[key]
                      : JSON.stringify(config[key])}
                </dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </details>
  );
}
