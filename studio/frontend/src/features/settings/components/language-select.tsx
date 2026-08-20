


import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import {
  AUTO_LOCALE,
  LOCALES,
  isLocalePreference,
  setLocale,
  useLocale,
  useLocaleCatalogFailed,
  useLocalePreference,
  usePendingLocalePreference,
  useT,
} from "@/i18n";

export function LanguageSelect() {
  const t = useT();
  const locale = useLocale();
  const preference = useLocalePreference();
  const pendingPreference = usePendingLocalePreference();
  const catalogFailed = useLocaleCatalogFailed();
  // Name no entry at all whenever the preference's own catalog failed and a
  // fallback was adopted. A controlled Select never fires onValueChange for the
  // value it already holds, so whichever entry it names is the one entry the
  // user cannot pick, and after a failure both of them are needed: the failed
  // preference is the retry, and the fallback in effect is how English becomes
  // a real preference instead of something the app is doing temporarily. The
  // empty string is Radix's "no selection", so every entry stays pickable and
  // the trigger shows the language actually in effect as the placeholder.
  // Whether the catalog failed has to come from the store rather than a
  // `preference !== locale` comparison, because a failed `auto` still reads as
  // `auto`, and its fallback locale is indistinguishable from a successful
  // detection that landed on the same language.
  const fallbackLabel = catalogFailed ? LOCALES[locale].nativeLabel : "";

  return (
    <Select
      value={pendingPreference ?? (catalogFailed ? "" : preference)}
      onValueChange={(value) => {
        if (isLocalePreference(value)) setLocale(value);
      }}
    >
      <SelectTrigger
        aria-label={t("settings.appearance.language.label")}
        aria-busy={pendingPreference !== null}
        // The placeholder is a real language here, not an empty menu, so it
        // keeps the normal trigger colour rather than the muted one.
        className="w-40 data-[placeholder]:text-foreground"
        size="sm"
      >
        <SelectValue placeholder={fallbackLabel} />
        {pendingPreference !== null ? <Spinner className="size-3.5" /> : null}
      </SelectTrigger>
      <SelectContent
        style={{
          maxHeight: "min(18rem, var(--radix-select-content-available-height))",
        }}
      >
        <SelectItem value={AUTO_LOCALE}>
          {t("settings.appearance.language.autoDetect")}
        </SelectItem>
        {Object.entries(LOCALES).map(([value, metadata]) => (
          <SelectItem key={value} value={value}>
            {metadata.nativeLabel}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
