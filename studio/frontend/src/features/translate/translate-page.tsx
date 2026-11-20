// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ComposerModelPicker } from "@/components/assistant-ui/composer-model-picker/composer-model-picker";
import {
  ComposerModelPickerContext,
  type ComposerModelPickerSource,
} from "@/components/assistant-ui/composer-model-picker/context";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
// Imported directly: the feature index re-exports the Audio page, kept out of this chunk.
import { FieldLabel } from "@/features/audio/components/field";
import {
  chatLocalModelOptions,
  providerModelDisplayName,
  useChatRuntimeStore,
  useExternalProvidersStore,
  useModelDisplayNamesVersion,
} from "@/features/chat";
import {
  getInferenceStatus,
  listGgufVariants,
  unloadModel,
} from "@/features/chat/api/chat-api";
import {
  buildExternalModelId,
  isDecisionConnection,
  isExternalModelId,
  parseExternalModelId,
} from "@/features/chat/external-providers";
import { useDeviceInventorySources } from "@/features/hub";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import type {
  LoraModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker";
import {
  MAX_SERVER_DOCUMENT_BYTES,
  SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS,
} from "@/features/chat/open-document-accept";
import { useNativeFileDrop } from "@/features/native-intents";
import { extractDocumentText } from "@/features/rag/api/rag-api";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { ReadAloudIcon } from "@/lib/action-bar-icons";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Settings01Icon } from "@/lib/hugeicons-derived";
import { StopIcon } from "@/lib/media-control-icons";
import { MicIcon } from "@/lib/mic-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { cn } from "@/lib/utils";
import {
  AiTranslateIcon,
  ArrowLeft02Icon,
  ArrowLeftRightIcon,
  Cancel01Icon,
  Message01Icon,
  RemoveCircleIcon,
  Search01Icon,
  Upload01Icon,
  VolumeMute02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { Command as CommandPrimitive } from "cmdk";
import { motion, useReducedMotion } from "motion/react";
import {
  type ReactNode,
  type RefObject,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import { chatAboutTranslations } from "./chat-handoff";
import { TranslateHistoryList } from "./history-list";
import { detectTranslateLanguage } from "./detect-language";
import { ICON_ACTION, IconAction } from "./icon-action";
import {
  TRANSLATE_LANGUAGES,
  findTranslateLanguage,
  translateLanguageLabel,
  translateSpeechTag,
} from "./languages";
import {
  CHAT_TRANSLATE_MODEL,
  CUSTOM_TRANSLATE_MODEL,
  type CustomTranslateModel,
  DEDICATED_TRANSLATE_MODELS,
  type DownloadedTranslateModel,
  downloadedQuant,
  downloadedTranslateModels,
  nonTextModelIds,
  type TranslateModelId,
  dedicatedTranslateModel,
  formatModelBytes,
  isTranslateModelId,
  translateInstruction,
  translateModelKey,
  translatePromptKind,
  translatePromptModel,
} from "./models";
import {
  type TranslatePhase,
  translateModelDownloaded,
} from "./translate-runner";
import {
  AUTO_SOURCE,
  TRANSLATE_SOURCE_MAX_CHARS,
  customPromptOf,
  languageTabs,
  translatePromptOptions,
  translationIsStale,
  useTranslateStore,
} from "./translate-store";
import {
  SOURCE_PLACEHOLDER,
  TARGET_PLACEHOLDER,
  TRANSLATE_MODES,
  TRANSLATE_TONES,
  type TranslateMode,
  type TranslateTone,
  templateOptions,
} from "./translate-text";
import { useDictation, useReadAloud } from "./use-translate-voice";

const AUTO_TRANSLATE_DELAY_MS = 800;
const TEXT_FILE_EXTENSIONS =
  ".txt,.md,.markdown,.srt,.vtt,.csv,.tsv,.json,.html,.xml,.yaml,.yml,.log,.tex,.rst";
const SERVER_FILE_EXTENSIONS = [
  ".pdf,.docx,.xlsx,.pptx,.odt,.ods,.rtf,.eml",
  SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS,
  ".png,.jpg,.jpeg,.webp,.gif,.bmp,.tif,.tiff",
].join(",");
const OPEN_FILE_EXTENSIONS = `${TEXT_FILE_EXTENSIONS},${SERVER_FILE_EXTENSIONS}`;

function readsInBrowser(file: File): boolean {
  const name = file.name.toLowerCase();
  if (SERVER_FILE_EXTENSIONS.split(",").some((ext) => name.endsWith(ext))) {
    return false;
  }
  return (
    file.type.startsWith("text/") ||
    TEXT_FILE_EXTENSIONS.split(",").some((ext) => name.endsWith(ext))
  );
}

const IS_MAC =
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad/.test(navigator.platform);

const ACTION_BUTTON = "relative z-10 transition-opacity disabled:opacity-40";
const GO_BUTTON =
  "ms-1 h-9 rounded-full px-4 text-ui-14 bg-control-accent text-control-accent-foreground hover:bg-control-accent/85";
const STOP_BUTTON =
  "ms-1 h-9 rounded-full px-3.5 text-ui-14 hover:bg-muted dark:hover:bg-muted";
const HINT = "text-ui-12 leading-normal text-muted-foreground";
const PANEL = "glass-inset rounded-xl bg-wash-6";
const PANE_HEIGHT =
  "min-h-52 max-h-[70dvh] sm:min-h-[calc(18rem*var(--ui-space-scale,1))] md:max-h-[max(18rem,calc(100dvh-10rem))]";
const PANE_HEIGHT_WITH_PROMPT =
  "min-h-52 max-h-[60dvh] sm:min-h-[calc(13rem*var(--ui-space-scale,1))] md:max-h-[max(13rem,calc(100dvh-30rem))]";
// The box clips and an inner element scrolls: a scrollbar on the box itself squares off its corners.
// Its outline is an inset shadow, not a border, so the scrollbar reaches the box's very edge.
const SOURCE_BOX =
  "glass-inset overflow-hidden rounded-xl bg-background shadow-[inset_0_0_0_1px_var(--border)] dark:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:shadow-none";
// A thin scrollbar keeps a margin beside its thumb; the scroller overhangs the box by that much.
const EDGE_SCROLL = "hover-scrollbar w-[calc(100%+2px)] max-w-none";
const SOURCE_TEXT =
  "min-h-full rounded-none border-0 bg-transparent dark:bg-transparent dark:focus-visible:bg-transparent";
const STOP_GLYPH = "size-3.5 fill-current";
// The second mask layer keeps the scrollbar opaque.
const BOTTOM_FADE =
  "data-[fade-bottom]:[mask-image:linear-gradient(to_bottom,#000_calc(100%-2.5rem),transparent),linear-gradient(#000,#000)] data-[fade-bottom]:[mask-size:100%_100%,12px_100%] data-[fade-bottom]:[mask-position:0_0,right] data-[fade-bottom]:[mask-repeat:no-repeat]";

function useBottomFade(ref: RefObject<HTMLElement | null>, content: string) {
  // biome-ignore lint/correctness/useExhaustiveDependencies: new content can change the scroll height.
  useLayoutEffect(() => {
    const scroller = ref.current;
    if (!scroller) return;
    const update = () =>
      scroller.toggleAttribute(
        "data-fade-bottom",
        scroller.scrollTop + scroller.clientHeight < scroller.scrollHeight - 1,
      );
    update();
    scroller.addEventListener("scroll", update, { passive: true });
    const observer = new ResizeObserver(update);
    observer.observe(scroller);
    if (scroller.firstElementChild)
      observer.observe(scroller.firstElementChild);
    return () => {
      scroller.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, [ref, content]);
}
const TAB =
  "relative flex h-9 shrink-0 cursor-pointer items-center whitespace-nowrap rounded-full px-3.5 text-ui-14 font-medium text-muted-foreground transition-colors hover:bg-[rgb(0_0_0_/_calc(0.04*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring";
const LIST_ROW =
  "h-9 cursor-pointer break-inside-avoid rounded-full py-0 text-ui-14 data-[checked=true]:font-medium data-[checked=true]:text-control-accent";
const CONTROL_HOVER =
  "hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.08*var(--contrast-wash-gain,1)))]";
const FOOTER_ICON = cn("size-9 shrink-0", CONTROL_HOVER);

const TONE_LABEL: Record<TranslateTone, TranslationKey> = {
  default: "translate.options.toneDefault",
  formal: "translate.options.toneFormal",
  casual: "translate.options.toneCasual",
};

const MODE_LABEL: Record<TranslateMode, TranslationKey> = {
  automatic: "translate.options.modeAutomatic",
  literal: "translate.options.modeLiteral",
  semantic: "translate.options.modeSemantic",
};

const MODEL_NOTE: Record<
  (typeof DEDICATED_TRANSLATE_MODELS)[number]["id"],
  TranslationKey
> = {
  "hy-mt2-1.8b": "translate.model.fast",
  "hy-mt2-7b": "translate.model.quality",
};

function LanguagePicker({
  value,
  onChange,
  label,
  allowAuto,
  tabs,
  trailing,
  className,
  detected,
}: {
  value: string;
  onChange: (code: string) => void;
  label: string;
  allowAuto?: boolean;
  tabs: ReactNode;
  trailing?: ReactNode;
  className?: string;
  detected?: string | null;
}) {
  const t = useT();
  const locale = useLocale();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const listFade = useScrollFades();
  const [active, setActive] = useState(value);
  const languages = useMemo(
    () =>
      TRANSLATE_LANGUAGES.map((language) => ({
        ...language,
        label: translateLanguageLabel(language, locale),
      })).sort((a, b) => a.label.localeCompare(b.label, locale)),
    [locale],
  );
  const needle = query.trim().toLowerCase();
  const shown = needle
    ? languages.filter((language) =>
        [language.label, language.name, language.code].some((name) =>
          name.toLowerCase().includes(needle),
        ),
      )
    : languages;
  const autoLabel = t("translate.detectLanguage");
  const showAuto =
    allowAuto === true && (!needle || autoLabel.toLowerCase().includes(needle));
  const labelOf = (code: string) =>
    languages.find((language) => language.code === code)?.label ?? code;
  const currentLabel =
    allowAuto && value === AUTO_SOURCE
      ? detected
        ? t("translate.detectedLanguage", { language: labelOf(detected) })
        : autoLabel
      : labelOf(value);
  const pick = (code: string) => {
    onChange(code);
    setOpen(false);
    setQuery("");
  };

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) setActive(value);
        else setQuery("");
      }}
    >
      <PopoverAnchor asChild={true}>
        <div className={cn("flex min-w-0 items-center gap-1", className)}>
          {tabs}
          <PopoverTrigger asChild={true}>
            <button
              type="button"
              aria-label={`${label}: ${currentLabel}`}
              title={label}
              className={cn(
                TAB,
                "w-9 justify-center px-0 data-[state=open]:bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:data-[state=open]:bg-[rgb(255_255_255_/_calc(0.08*var(--contrast-wash-gain,1)))] data-[state=open]:text-foreground",
              )}
            >
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                strokeWidth={2}
                className="size-4.5 transition-transform in-data-[state=open]:rotate-180"
              />
            </button>
          </PopoverTrigger>
          {trailing}
        </div>
      </PopoverAnchor>
      <PopoverContent
        align="start"
        sideOffset={12}
        collisionPadding={16}
        className="w-[var(--radix-popper-anchor-width)] max-w-none gap-0 overflow-hidden rounded-xl p-0"
      >
        {/* Filtered here: the labels are in the interface language, the search also takes English. */}
        <Command
          shouldFilter={false}
          value={active}
          onValueChange={setActive}
          className="rounded-none bg-transparent p-0"
        >
          {/* ! beats the global combobox pointer. */}
          <div
            onMouseDown={(event) => {
              const input = event.currentTarget.querySelector("input");
              if (input && event.target !== input) {
                event.preventDefault();
                input.focus();
              }
            }}
            className="flex h-12 cursor-text! items-center gap-2.5 border-b border-border/60 px-4"
          >
            <HugeiconsIcon
              icon={Search01Icon}
              strokeWidth={2}
              className="size-4 shrink-0 text-muted-foreground"
            />
            <CommandPrimitive.Input
              value={query}
              onValueChange={setQuery}
              aria-label={t("translate.searchLanguages")}
              placeholder={t("translate.searchLanguages")}
              className="h-full min-w-0 flex-1 cursor-text! bg-transparent text-ui-14 outline-hidden placeholder:text-muted-foreground"
            />
          </div>
          <CommandList
            ref={listFade.attach}
            onScroll={listFade.onScroll}
            className={cn(
              "hover-scrollbar panel-scroll-fade-action max-h-[min(28rem,calc(var(--radix-popover-content-available-height)-3rem))] p-2 **:[[cmdk-list-sizer]]:columns-[10.5rem] **:[[cmdk-list-sizer]]:gap-x-1",
              listFade.className,
            )}
          >
            {!showAuto && shown.length === 0 && (
              <CommandEmpty>{t("translate.noLanguages")}</CommandEmpty>
            )}
            {showAuto && (
              <CommandItem
                value={AUTO_SOURCE}
                onSelect={() => pick(AUTO_SOURCE)}
                data-checked={value === AUTO_SOURCE}
                className={LIST_ROW}
              >
                <span className="min-w-0 flex-1 truncate">{autoLabel}</span>
              </CommandItem>
            )}
            {shown.map((language) => (
              <CommandItem
                key={language.code}
                value={language.code}
                onSelect={() => pick(language.code)}
                data-checked={language.code === value}
                className={LIST_ROW}
              >
                <span className="min-w-0 flex-1 truncate">
                  {language.label}
                </span>
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

function LanguageTabs({
  value,
  tabs,
  onChange,
  label,
  allowAuto,
  detected,
  className,
  children,
}: {
  value: string;
  tabs: readonly string[];
  onChange: (code: string) => void;
  label: string;
  allowAuto?: boolean;
  detected?: string | null;
  className?: string;
  children?: ReactNode;
}) {
  const t = useT();
  const locale = useLocale();
  const pillId = useId();
  const detectedLanguage =
    value === AUTO_SOURCE && detected
      ? findTranslateLanguage(detected)
      : undefined;
  const reduced = useReducedMotion();
  const tab = (code: string, name: string) => {
    const active = code === value;
    return (
      <button
        key={code}
        type="button"
        role="tab"
        aria-selected={active}
        onClick={() => onChange(code)}
        className={cn(TAB, active && "text-foreground hover:bg-transparent")}
      >
        {active && (
          <motion.span
            layoutId={pillId}
            className="hub-tab-toggle-pill absolute inset-0 rounded-full"
            transition={
              reduced
                ? { duration: 0 }
                : { type: "spring", stiffness: 500, damping: 35, mass: 0.5 }
            }
          />
        )}
        <span className="relative z-10">{name}</span>
      </button>
    );
  };
  return (
    <LanguagePicker
      value={value}
      onChange={onChange}
      label={label}
      allowAuto={allowAuto}
      detected={detected}
      className={className}
      trailing={children}
      tabs={
        // Clipped, not scrolled (the chevron lists dropped tabs); the padding keeps the pill's shadow.
        <div
          role="tablist"
          aria-label={label}
          className="-mx-2 -my-2 flex min-w-0 items-center gap-0.5 overflow-hidden px-2 py-2"
        >
          {allowAuto &&
            tab(
              AUTO_SOURCE,
              detectedLanguage
                ? t("translate.detectedTab", {
                    language: translateLanguageLabel(detectedLanguage, locale),
                  })
                : t("translate.detectLanguage"),
            )}
          {tabs.map((code) => {
            const language = findTranslateLanguage(code);
            return language
              ? tab(code, translateLanguageLabel(language, locale))
              : null;
          })}
        </div>
      }
    />
  );
}

function StyleMenu() {
  const t = useT();
  const { mode, tone } = useTranslateStore(
    useShallow((state) => ({ mode: state.prefs.mode, tone: state.prefs.tone })),
  );
  const setPrefs = useTranslateStore((state) => state.setPrefs);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          title={`${t("translate.options.mode")} · ${t("translate.options.tone")}`}
          className={cn(
            "h-9 min-w-0 shrink cursor-pointer gap-1.5 rounded-full px-3 text-ui-14 font-medium text-muted-foreground hover:text-foreground data-[state=open]:text-foreground @max-[26rem]:ms-auto",
            CONTROL_HOVER,
            "data-[state=open]:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] dark:data-[state=open]:bg-[rgb(255_255_255_/_calc(0.08*var(--contrast-wash-gain,1)))]",
          )}
        >
          <span className="truncate">
            {t(MODE_LABEL[mode])} · {t(TONE_LABEL[tone])}
          </span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={2}
            className="size-4 transition-transform in-data-[state=open]:rotate-180"
          />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        side="top"
        sideOffset={6}
        className="w-48"
      >
        <DropdownMenuLabel>{t("translate.options.mode")}</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          value={mode}
          onValueChange={(next) => setPrefs({ mode: next as TranslateMode })}
        >
          {TRANSLATE_MODES.map((option) => (
            <DropdownMenuRadioItem key={option} value={option}>
              {t(MODE_LABEL[option])}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
        <DropdownMenuSeparator />
        <DropdownMenuLabel>{t("translate.options.tone")}</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          value={tone}
          onValueChange={(next) => setPrefs({ tone: next as TranslateTone })}
        >
          {TRANSLATE_TONES.map((option) => (
            <DropdownMenuRadioItem key={option} value={option}>
              {t(TONE_LABEL[option])}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

type Downloaded = Partial<Record<TranslateModelId, boolean>>;

const ignore = () => undefined;

/** The hidden chip only anchors the panel under the Translate menu, which is the trigger. */
function OtherModelPicker({
  open,
  onOpenChange,
  selection,
  onPick,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selection: { id: string; variant: string | null };
  onPick: (id: string, meta: ModelSelectorChangeMeta) => void;
}) {
  const models = useChatRuntimeStore((s) => s.models);
  const loras = useChatRuntimeStore((s) => s.loras);
  const inventory = useDeviceInventorySources(["localModels"]);
  const refreshLocalModels = inventory.refresh;
  const localRows = inventory.localModels.rows;
  const providers = useExternalProvidersStore((s) => s.providers);
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const namesVersion = useModelDisplayNamesVersion();
  // biome-ignore lint/correctness/useExhaustiveDependencies: learned display names arrive by version.
  const externalModels = useMemo<ComposerModelPickerSource["externalModels"]>(
    () =>
      connectionsEnabled
        ? providers
            .filter((provider) => !isDecisionConnection(provider))
            .flatMap((provider) =>
              provider.models.map((model) => ({
                id: buildExternalModelId(provider.id, model),
                name: providerModelDisplayName(provider.providerType, model),
                providerId: provider.id,
                providerName: provider.name,
                providerType: provider.providerType,
              })),
            )
        : [],
    [providers, connectionsEnabled, namesVersion],
  );
  const externalConnections = useMemo<
    ComposerModelPickerSource["externalConnections"]
  >(
    () =>
      connectionsEnabled
        ? providers.map((provider) => ({
            id: provider.id,
            name: provider.name,
            providerType: provider.providerType,
            availableModels: provider.availableModels,
          }))
        : [],
    [providers, connectionsEnabled],
  );
  const hiddenIds = useMemo(() => nonTextModelIds(localRows), [localRows]);
  const loraModels = useMemo<LoraModelOption[]>(
    () => [
      ...loras.map((lora) => ({
        id: lora.id,
        name: lora.name,
        baseModel: lora.baseModel,
        updatedAt: lora.updatedAt,
        source: lora.source,
        exportType: lora.exportType,
      })),
      ...chatLocalModelOptions(localRows),
    ],
    [loras, localRows],
  );
  const source = useMemo<ComposerModelPickerSource>(
    () => ({
      models: models.map(({ id, name, description, isGguf }) => ({
        id,
        name,
        description,
        isGguf,
      })),
      loraModels,
      externalModels,
      externalConnections,
      connections: connectionsEnabled,
      browseOnly: true,
      hiddenIds,
      onPick,
      eject: ignore,
      ejectAll: ignore,
      open,
      onOpenChange,
      configRequest: null,
      onConfigRequestAdopted: ignore,
      onFoldersChange: refreshLocalModels,
      onModelsChange: refreshLocalModels,
      deleteDisabled: false,
      activeModelConfig: null,
      loadedContextLength: null,
      loaded: false,
      selection: {
        id: selection.id,
        ggufVariant: selection.variant,
        loaded: false,
      },
    }),
    [
      models,
      loraModels,
      externalModels,
      externalConnections,
      connectionsEnabled,
      hiddenIds,
      onPick,
      open,
      onOpenChange,
      refreshLocalModels,
      selection.id,
      selection.variant,
    ],
  );
  return (
    <div className="invisible absolute inset-0 flex justify-end overflow-hidden">
      <ComposerModelPickerContext.Provider value={source}>
        <ComposerModelPicker side="bottom" align="end" />
      </ComposerModelPickerContext.Provider>
    </div>
  );
}

function pickedModelName(id: string): string {
  const external = parseExternalModelId(id)?.modelId;
  if (external) return external;
  const name = id.split(/[\\/]/).filter(Boolean).pop() ?? id;
  return name.replace(/-GGUF$/i, "");
}

function ModelMenu({
  value,
  custom,
  onChange,
  onPickCustom,
  downloaded,
}: {
  value: TranslateModelId;
  custom: CustomTranslateModel | null;
  onChange: (id: TranslateModelId) => void;
  onPickCustom: (model: CustomTranslateModel) => void;
  downloaded: Downloaded;
}) {
  const t = useT();
  const busy = useTranslateStore((state) => state.phase !== null);
  const preparing = useTranslateStore(
    (state) => state.phase !== null && state.phase.kind !== "translating",
  );
  const [pickerOpen, setPickerOpen] = useState(false);
  const [loaded, setLoaded] = useState<
    { id: string; variant: string | null }[]
  >([]);
  const refreshLoaded = useCallback(() => {
    getInferenceStatus()
      .then((status) =>
        Promise.all(
          (status.loaded ?? []).map(async (id) => ({
            id,
            // The quant in memory: another quant of the same repo is not this model.
            variant:
              (await getInferenceStatus(undefined, id).catch(() => null))
                ?.gguf_variant ?? null,
          })),
        ),
      )
      .then(setLoaded)
      .catch(() => undefined);
  }, []);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `busy` is the trigger, not an input.
  useEffect(refreshLoaded, [refreshLoaded, busy]);
  const inventory = useDeviceInventorySources(["localModels"]);
  const localRows = inventory.localModels.rows;
  const mine = useMemo(() => downloadedTranslateModels(localRows), [localRows]);
  const customMine =
    value === CUSTOM_TRANSLATE_MODEL && custom !== null
      ? (mine.find((model) => modelIdsMatch(model.id, custom.id)) ?? null)
      : null;
  const customIsMine = customMine !== null;
  const localModel =
    value === CUSTOM_TRANSLATE_MODEL
      ? custom && !isExternalModelId(custom.id)
        ? custom.id
        : null
      : (dedicatedTranslateModel(value)?.repo ?? null);
  const isLoaded = (repo: string, variant?: string | null) =>
    loaded.some(
      (model) =>
        modelIdsMatch(model.id, repo) &&
        (!variant ||
          !model.variant ||
          model.variant.toLowerCase() === variant.toLowerCase()),
    );
  const selectedVariant =
    value === CUSTOM_TRANSLATE_MODEL
      ? custom?.variant
      : dedicatedTranslateModel(value)?.variant;
  // A model loaded by its path is reported under its Hub-style name.
  const ejectable =
    localModel !== null &&
    (isLoaded(localModel, selectedVariant) ||
      (customMine?.repo != null && isLoaded(customMine.repo, selectedVariant)));
  const selectedKey = translateModelKey(value, custom);
  const setModelReady = useTranslateStore((state) => state.setModelReady);
  useEffect(() => {
    if (ejectable) setModelReady(selectedKey, true);
  }, [ejectable, selectedKey, setModelReady]);
  const eject = async () => {
    if (!localModel) return;
    try {
      await unloadModel({ model_path: localModel });
      setLoaded([]);
      // Typing must not load it straight back.
      setModelReady(selectedKey, false);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };
  // A later choice wins over a quant lookup still in flight.
  const pickSeq = useRef(0);
  const pickMine = async (model: DownloadedTranslateModel) => {
    const seq = ++pickSeq.current;
    let variant: string | null = null;
    let sizeBytes = model.sizeBytes;
    if (model.needsQuant) {
      // The quant already on disk, so picking a downloaded model never starts a download.
      const listing = await listGgufVariants(model.id, undefined, {
        localOnly: true,
        localPath: model.localPath,
      }).catch(() => null);
      if (seq !== pickSeq.current) return;
      const onDisk = listing ? downloadedQuant(listing.variants) : null;
      if (!onDisk) {
        setPickerOpen(true);
        return;
      }
      variant = onDisk.quant;
      sizeBytes = onDisk.sizeBytes;
    }
    onPickCustom({ id: model.id, name: model.name, variant, sizeBytes });
  };
  const mineValue = (id: string) => `mine:${id}`;
  const selectedValue = customMine ? mineValue(customMine.id) : value;
  const openPickerOnClose = useRef(false);
  const label =
    value === CUSTOM_TRANSLATE_MODEL
      ? (custom?.name ?? "")
      : (dedicatedTranslateModel(value)?.label ?? t("translate.model.chat"));
  const pick = (id: string, meta: ModelSelectorChangeMeta) => {
    pickSeq.current++;
    setPickerOpen(false);
    const variant = meta.ggufVariant ?? null;
    const builtIn = DEDICATED_TRANSLATE_MODELS.find(
      (model) =>
        model.repo.toLowerCase() === id.toLowerCase() &&
        (variant ?? model.variant).toLowerCase() ===
          model.variant.toLowerCase(),
    );
    if (builtIn) {
      onChange(builtIn.id);
      return;
    }
    onPickCustom({
      id,
      name: pickedModelName(id),
      variant,
      sizeBytes: meta.expectedBytes ?? null,
    });
  };
  const customNote = custom
    ? [
        custom.variant,
        !isExternalModelId(custom.id) && isLoaded(custom.id, custom.variant)
          ? t("translate.model.loaded")
          : formatModelBytes(custom.sizeBytes ?? 0),
      ]
        .filter(Boolean)
        .join(" · ")
    : "";
  return (
    <div className="relative">
      <OtherModelPicker
        open={pickerOpen}
        onOpenChange={setPickerOpen}
        selection={{
          id:
            (value === CUSTOM_TRANSLATE_MODEL
              ? custom?.id
              : dedicatedTranslateModel(value)?.repo) ?? "",
          variant:
            (value === CUSTOM_TRANSLATE_MODEL
              ? custom?.variant
              : dedicatedTranslateModel(value)?.variant) ?? null,
        }}
        onPick={pick}
      />
      <DropdownMenu
        onOpenChange={(open) => {
          if (open) refreshLoaded();
        }}
      >
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            aria-label={`${t("translate.model.label")}: ${label}`}
            title={t("translate.model.label")}
            className="unsloth-thinking-pill composer-model-chip unsloth-model-selector-trigger max-w-56 pl-3.5 pr-3"
          >
            {preparing && (
              <Spinner className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0" />
            )}
            <span className="unsloth-thinking-label composer-model-chip-name truncate">
              {label}
            </span>
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.75}
              className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="w-80"
          // Opened once the menu has closed: its returning focus would dismiss the panel.
          onCloseAutoFocus={(event) => {
            if (!openPickerOnClose.current) return;
            openPickerOnClose.current = false;
            event.preventDefault();
            setPickerOpen(true);
          }}
        >
          <DropdownMenuRadioGroup
            value={selectedValue}
            onValueChange={(next) => {
              const picked = mine.find((model) => mineValue(model.id) === next);
              if (picked) {
                void pickMine(picked);
              } else if (isTranslateModelId(next)) {
                pickSeq.current++;
                onChange(next);
              }
            }}
          >
            <DropdownMenuLabel>
              {t("translate.model.recommended")}
            </DropdownMenuLabel>
            {DEDICATED_TRANSLATE_MODELS.map((model) => (
              <DropdownMenuRadioItem
                key={model.id}
                value={model.id}
                className="items-start"
              >
                <span className="grid min-w-0 gap-0.5">
                  <span className="text-ui-13 font-medium text-foreground">
                    {model.label}
                  </span>
                  <span className="text-ui-12 leading-normal text-muted-foreground">
                    {[
                      t(MODEL_NOTE[model.id]),
                      isLoaded(model.repo, model.variant)
                        ? t("translate.model.loaded")
                        : downloaded[model.id]
                          ? t("translate.model.downloaded")
                          : downloaded[model.id] === false
                            ? t("translate.model.downloadSize", {
                                size: formatModelBytes(model.sizeBytes),
                              })
                            : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </span>
              </DropdownMenuRadioItem>
            ))}
            {(mine.length > 0 || (custom && !customIsMine)) && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuLabel>
                  {t("translate.model.yourModels")}
                </DropdownMenuLabel>
              </>
            )}
            {mine.map((model) => (
              <DropdownMenuRadioItem
                key={model.id}
                value={mineValue(model.id)}
                className="items-start"
              >
                <span className="grid min-w-0 gap-0.5">
                  <span className="truncate text-ui-13 font-medium text-foreground">
                    {model.name}
                  </span>
                  <span className="text-ui-12 leading-normal text-muted-foreground">
                    {[
                      customMine === model ? custom?.variant : null,
                      isLoaded(
                        model.id,
                        customMine === model ? custom?.variant : null,
                      ) ||
                      (model.repo !== null &&
                        isLoaded(
                          model.repo,
                          customMine === model ? custom?.variant : null,
                        ))
                        ? t("translate.model.loaded")
                        : t("translate.model.downloaded"),
                      customMine === model && custom?.sizeBytes
                        ? formatModelBytes(custom.sizeBytes)
                        : formatModelBytes(model.sizeBytes ?? 0),
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </span>
              </DropdownMenuRadioItem>
            ))}
            {custom && !customIsMine && (
              <DropdownMenuRadioItem
                value={CUSTOM_TRANSLATE_MODEL}
                className="items-start"
              >
                <span className="grid min-w-0 gap-0.5">
                  <span className="truncate text-ui-13 font-medium text-foreground">
                    {custom.name}
                  </span>
                  {customNote && (
                    <span className="text-ui-12 leading-normal text-muted-foreground">
                      {customNote}
                    </span>
                  )}
                </span>
              </DropdownMenuRadioItem>
            )}
          </DropdownMenuRadioGroup>
          <DropdownMenuSeparator />
          {ejectable && (
            <DropdownMenuItem
              variant="destructive"
              disabled={busy}
              onSelect={() => void eject()}
            >
              <HugeiconsIcon icon={RemoveCircleIcon} />
              {t("translate.model.eject")}
            </DropdownMenuItem>
          )}
          <DropdownMenuItem
            onSelect={() => {
              openPickerOnClose.current = true;
            }}
          >
            {t("translate.model.browse")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

function OptionRow({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: (id: string) => ReactNode;
}) {
  const id = useId();
  return (
    <div className="grid gap-2">
      <FieldLabel
        htmlFor={id}
        className="flex items-center justify-between gap-3 text-ui-13 font-medium text-foreground"
      >
        {label}
        {children(id)}
      </FieldLabel>
      {description && <p className={HINT}>{description}</p>}
    </div>
  );
}

function OptionsPopover() {
  const t = useT();
  const { autoTranslate, saveHistory, showPrompt } = useTranslateStore(
    useShallow((state) => ({
      autoTranslate: state.prefs.autoTranslate,
      saveHistory: state.prefs.saveHistory,
      showPrompt: state.prefs.showPrompt,
    })),
  );
  const setPrefs = useTranslateStore((state) => state.setPrefs);
  return (
    <Popover>
      <PopoverTrigger asChild={true}>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          title={t("translate.options.label")}
          aria-label={t("translate.options.label")}
          className={ICON_ACTION}
        >
          <HugeiconsIcon icon={Settings01Icon} className="size-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" sideOffset={4} className="w-80 gap-5">
        <OptionRow
          label={t("translate.options.auto")}
          description={t("translate.options.autoDescription")}
        >
          {(id) => (
            <Switch
              id={id}
              checked={autoTranslate}
              onCheckedChange={(checked) =>
                setPrefs({ autoTranslate: checked })
              }
            />
          )}
        </OptionRow>
        <OptionRow
          label={t("translate.options.saveHistory")}
          description={t("translate.options.saveHistoryDescription")}
        >
          {(id) => (
            <Switch
              id={id}
              checked={saveHistory}
              onCheckedChange={(checked) => setPrefs({ saveHistory: checked })}
            />
          )}
        </OptionRow>
        <OptionRow
          label={t("translate.options.showPrompt")}
          description={t("translate.options.showPromptDescription")}
        >
          {(id) => (
            <Switch
              id={id}
              checked={showPrompt}
              onCheckedChange={(checked) => setPrefs({ showPrompt: checked })}
            />
          )}
        </OptionRow>
      </PopoverContent>
    </Popover>
  );
}

function useRunStatusText(phase: TranslatePhase | null, model: string) {
  const t = useT();
  if (!phase) return null;
  if (phase.kind === "downloading")
    return t("translate.status.downloading", { model });
  if (phase.kind === "loading") return t("translate.status.loading", { model });
  if (phase.kind === "translating")
    return phase.parts > 1
      ? t("translate.status.translatingPart", {
          part: phase.part,
          parts: phase.parts,
        })
      : t("translate.status.translating");
  return t("translate.status.preparing");
}

export function TranslatePage() {
  const t = useT();
  const locale = useLocale();
  const { prefs, source, output, phase, error, result, readyModels } =
    useTranslateStore(
      useShallow((state) => ({
        prefs: state.prefs,
        source: state.source,
        output: state.output,
        phase: state.phase,
        error: state.error,
        result: state.result,
        readyModels: state.readyModels,
      })),
    );
  const attempt = useTranslateStore((state) => state.attempt);
  const {
    setSource,
    setPrefs,
    setFrom,
    setTarget,
    swap,
    outputToSource,
    translate,
    stop,
    clear,
  } = useTranslateStore(
    useShallow((state) => ({
      setSource: state.setSource,
      outputToSource: state.outputToSource,
      setPrefs: state.setPrefs,
      setFrom: state.setFrom,
      setTarget: state.setTarget,
      swap: state.swap,
      translate: state.translate,
      stop: state.stop,
      clear: state.clear,
    })),
  );
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);
  const [downloaded, setDownloaded] = useState<Downloaded>({});
  const fileInput = useRef<HTMLInputElement>(null);
  const outputRef = useRef<HTMLDivElement>(null);
  const sourceScrollRef = useRef<HTMLDivElement>(null);
  useBottomFade(sourceScrollRef, source);
  useBottomFade(outputRef, output);

  const busy = phase !== null;
  const paneHeight = prefs.showPrompt ? PANE_HEIGHT_WITH_PROMPT : PANE_HEIGHT;
  const canTranslate = source.trim() !== "";
  const detected = useMemo(
    () => (prefs.from === AUTO_SOURCE ? detectTranslateLanguage(source) : null),
    [prefs.from, source],
  );
  const canSwap =
    prefs.from !== AUTO_SOURCE ||
    (detected !== null && detected !== prefs.target);
  const stale = translationIsStale({ result, source, prefs });
  const modelKey = translateModelKey(prefs.modelId, prefs.customModel);
  const dedicated = dedicatedTranslateModel(prefs.modelId);
  const modelLabel =
    prefs.modelId === CUSTOM_TRANSLATE_MODEL
      ? (prefs.customModel?.name ?? "")
      : (dedicated?.label ?? t("translate.model.chat"));
  // The run's own model: the selection may change while it downloads.
  const runModelKey = attempt?.modelKey ?? modelKey;
  const runModelLabel =
    runModelKey === modelKey
      ? modelLabel
      : (DEDICATED_TRANSLATE_MODELS.find((model) => model.id === runModelKey)
          ?.label ??
        (runModelKey === CHAT_TRANSLATE_MODEL
          ? t("translate.model.chat")
          : (attempt?.customName ?? prefs.customModel?.name ?? "")));

  useEffect(() => {
    if (busy) return;
    const abort = new AbortController();
    for (const model of DEDICATED_TRANSLATE_MODELS) {
      void translateModelDownloaded(model, abort.signal).then((onDisk) => {
        if (onDisk === null || abort.signal.aborted) return;
        setDownloaded((prev) => ({ ...prev, [model.id]: onDisk }));
      });
    }
    return () => abort.abort();
  }, [busy]);

  useEffect(() => {
    if (!prefs.autoTranslate || busy || !canTranslate) return;
    if (!readyModels.includes(modelKey) || prefs.from === prefs.target) return;
    // Only a draft that differs from the last one sent: a stopped or failed run is not retried.
    if (
      attempt &&
      attempt.source === source &&
      attempt.from === prefs.from &&
      attempt.target === prefs.target &&
      attempt.mode === prefs.mode &&
      attempt.tone === prefs.tone &&
      attempt.custom === customPromptOf(prefs) &&
      attempt.modelKey === modelKey
    ) {
      return;
    }
    const timer = window.setTimeout(
      () => void translate({ auto: true }),
      AUTO_TRANSLATE_DELAY_MS,
    );
    return () => window.clearTimeout(timer);
  }, [
    prefs,
    modelKey,
    busy,
    canTranslate,
    readyModels,
    attempt,
    source,
    translate,
  ]);

  useEffect(() => {
    const pane = outputRef.current;
    if (pane && phase?.kind === "translating")
      pane.scrollTop = pane.scrollHeight;
  }, [output, phase]);

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const loadText = (text: string) => {
    if (text.includes("\u0000")) {
      toast.error(t("translate.fileNotText"));
      return;
    }
    if (text.length > TRANSLATE_SOURCE_MAX_CHARS) {
      toast.info(
        t("translate.truncated", {
          max: TRANSLATE_SOURCE_MAX_CHARS.toLocaleString(locale),
        }),
      );
    }
    setSource(text);
  };

  const [readingFile, setReadingFile] = useState<string | null>(null);
  const openFile = async (file: File | undefined) => {
    if (!file) return;
    if (readsInBrowser(file)) {
      try {
        loadText(
          await file.slice(0, TRANSLATE_SOURCE_MAX_CHARS * 4 + 4).text(),
        );
      } catch {
        toast.error(t("translate.fileNotText"));
      }
      return;
    }
    if (file.size > MAX_SERVER_DOCUMENT_BYTES) {
      toast.error(t("translate.fileNotText"));
      return;
    }
    setReadingFile(file.name);
    // Read-only meanwhile; a swap or history pick still wins.
    const before = useTranslateStore.getState().source;
    try {
      const { text, warning } = await extractDocumentText(file);
      if (useTranslateStore.getState().source !== before) {
        return;
      }
      if (text.trim()) {
        loadText(text);
        if (warning) toast.warning(warning);
      } else toast.error(t("translate.fileNotText"));
    } catch (error) {
      toast.error(
        error instanceof Error && error.message
          ? error.message
          : t("translate.fileNotText"),
      );
    } finally {
      setReadingFile(null);
    }
  };

  // Marking main on mount spares main:has(.translate-page), which Chromium re-matched per
  // keystroke.
  useLayoutEffect(() => {
    const host = document
      .querySelector(".translate-page")
      ?.parentElement?.closest("main");
    if (!host) {
      return;
    }
    host.setAttribute("data-translate-page", "");
    return () => host.removeAttribute("data-translate-page");
  }, []);

  const {
    ref: dropRef,
    dragging,
    dragHandlers,
  } = useNativeFileDrop({
    onFiles: (files) => void openFile(files[0]),
    accept: OPEN_FILE_EXTENSIONS,
    multiple: false,
    disabled: readingFile !== null,
    disabledReason: readingFile
      ? t("translate.readingFile", { name: readingFile })
      : undefined,
  });

  const errorMessage = error
    ? error.code === "other"
      ? /failed to load model|llama-server failed to start/i.test(error.message)
        ? t("translate.errors.loadFailed")
        : error.message
      : t(`translate.errors.${error.code}`)
    : "";
  const promptOptions = translatePromptOptions(prefs);
  const promptId = useId();
  const promptModel = translatePromptModel(prefs.modelId, prefs.customModel);
  const promptKind = translatePromptKind(promptModel);
  const customPrompt = customPromptOf(prefs);
  const defaultPrompt = promptOptions
    ? translateInstruction(promptModel, templateOptions(promptOptions))
    : "";
  // What the box shows while typing: emptying it must not refill it with the built prompt.
  const [promptDraft, setPromptDraft] = useState<string | null>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new kind of model starts from its own prompt.
  useEffect(() => setPromptDraft(null), [promptKind]);
  const setCustomPrompt = (value: string | null) => {
    const { [promptKind]: _dropped, ...others } = prefs.customPrompts;
    const edited =
      value !== null && value.trim() !== "" && value !== defaultPrompt;
    setPrefs({
      customPrompts: edited ? { ...others, [promptKind]: value } : others,
    });
  };

  const dictation = useDictation(
    (transcript) => {
      const draft = useTranslateStore.getState().source;
      setSource(
        draft && !/\s$/.test(draft)
          ? `${draft} ${transcript}`
          : draft + transcript,
      );
    },
    prefs.from === AUTO_SOURCE ? undefined : translateSpeechTag(prefs.from),
  );
  const readAloud = useReadAloud();
  const stopReading = readAloud.stop;
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs on every new output.
  useEffect(() => stopReading(), [output, stopReading]);

  const chatAboutResult = () => {
    if (!result || stale) return;
    chatAboutTranslations(
      [
        {
          at: Date.now(),
          source: result.source,
          output,
          target: result.target,
          from: result.from === AUTO_SOURCE ? null : result.from,
        },
      ],
      navigate,
    );
  };

  const shortcut = `${IS_MAC ? "⌘" : "Ctrl"}+Enter`;
  const runStatus = useRunStatusText(phase, runModelLabel);
  const [confirmDownload, setConfirmDownload] = useState<
    (typeof DEDICATED_TRANSLATE_MODELS)[number] | null
  >(null);
  const checkingDownload = useRef(false);
  const onDiskNow = async (
    model: (typeof DEDICATED_TRANSLATE_MODELS)[number],
  ): Promise<boolean | null> => {
    const known = downloaded[model.id];
    if (known !== undefined) {
      return known;
    }
    const onDisk = await translateModelDownloaded(model);
    if (onDisk !== null) {
      setDownloaded((prev) => ({ ...prev, [model.id]: onDisk }));
    }
    return onDisk;
  };
  const startTranslate = async () => {
    if (checkingDownload.current) {
      return;
    }
    if (dedicated && !readyModels.includes(modelKey)) {
      checkingDownload.current = true;
      const onDisk = await onDiskNow(dedicated).finally(() => {
        checkingDownload.current = false;
      });
      // Another model picked meanwhile: start nothing.
      if (useTranslateStore.getState().prefs.modelId !== dedicated.id) {
        return;
      }
      if (onDisk === false) {
        setConfirmDownload(dedicated);
        return;
      }
    }
    void translate();
  };

  return (
    <main
      ref={dropRef}
      {...dragHandlers}
      className="translate-page mx-auto flex w-full max-w-[calc(96rem*var(--ui-space-scale,1))] flex-col gap-6 px-6 pb-10 pt-8 font-heading max-sm:px-4 sm:px-8 lg:px-12"
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
            {t("translate.title")}
          </h1>
          <Badge
            variant="secondary"
            className="h-6 -translate-y-0.5 px-2.5 pt-1.5 pb-0.5 text-ui-12 text-muted-foreground"
          >
            {t("shell.beta")}
          </Badge>
        </div>
        <div className="flex items-center gap-1">
          <ModelMenu
            value={prefs.modelId}
            custom={prefs.customModel}
            onChange={(modelId) => setPrefs({ modelId })}
            onPickCustom={(customModel) =>
              setPrefs({ modelId: CUSTOM_TRANSLATE_MODEL, customModel })
            }
            downloaded={downloaded}
          />
          <OptionsPopover />
        </div>
      </div>

      <div className="grid gap-2">
        <div className="relative grid grid-cols-1 gap-x-3 gap-y-3 md:grid-cols-2">
          <IconAction
            icon={ArrowLeftRightIcon}
            label={
              canSwap ? t("translate.swap") : t("translate.swapNeedsSource")
            }
            disabled={!canSwap || busy}
            onClick={() => swap(detected)}
            className="absolute start-1/2 top-0 z-10 -translate-x-1/2 max-md:hidden rtl:translate-x-1/2"
          />
          <section
            aria-label={t("translate.sourceLabel")}
            className="relative flex min-w-0 flex-col gap-3"
          >
            <LanguageTabs
              value={prefs.from}
              tabs={languageTabs(prefs.from, prefs.recentSources)}
              onChange={setFrom}
              label={t("translate.sourceLanguage")}
              allowAuto={true}
              detected={detected}
              className="md:pe-6"
            />
            <div
              className={cn(
                SOURCE_BOX,
                paneHeight,
                "@container relative flex flex-1 flex-col",
                dragging && "shadow-[inset_0_0_0_1px_var(--ring)]",
              )}
            >
              <div
                ref={sourceScrollRef}
                className={cn(
                  EDGE_SCROLL,
                  BOTTOM_FADE,
                  "min-h-0 flex-1 cursor-text overflow-y-auto",
                )}
                onPointerUp={(event) => {
                  if (event.target === event.currentTarget)
                    event.currentTarget.querySelector("textarea")?.focus();
                }}
              >
                <Textarea
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      (event.metaKey || event.ctrlKey) &&
                      !busy
                    ) {
                      event.preventDefault();
                      void startTranslate();
                    }
                  }}
                  maxLength={TRANSLATE_SOURCE_MAX_CHARS}
                  readOnly={readingFile !== null}
                  placeholder={
                    readingFile
                      ? t("translate.readingFile", { name: readingFile })
                      : t("translate.sourcePlaceholder")
                  }
                  aria-label={t("translate.sourceLabel")}
                  spellCheck={false}
                  dir="auto"
                  className={cn(SOURCE_TEXT, "pe-11")}
                />
              </div>
              {source !== "" && readingFile === null && (
                <IconAction
                  icon={Cancel01Icon}
                  label={t("translate.clear")}
                  onClick={clear}
                  className="absolute end-2 top-2"
                />
              )}
              <div className="flex shrink-0 items-center gap-1 px-2.5 pt-1.5 pb-2.5">
                {/* Always shown: a click explains why it cannot dictate or offers local dictation. */}
                {dictation.dictating ? (
                  <IconAction
                    icon={dictation.finalizing ? undefined : StopIcon}
                    iconClassName={STOP_GLYPH}
                    label={t("translate.stopDictating")}
                    onClick={dictation.stop}
                    className={cn(
                      FOOTER_ICON,
                      "text-destructive hover:text-destructive",
                    )}
                  >
                    <Spinner className="size-5" />
                  </IconAction>
                ) : (
                  <IconAction
                    label={t("translate.dictate")}
                    onClick={dictation.start}
                    disabled={readingFile !== null}
                    className={FOOTER_ICON}
                  >
                    <MicIcon className="size-5" />
                  </IconAction>
                )}
                <IconAction
                  icon={readingFile ? undefined : Upload01Icon}
                  iconClassName="size-5"
                  className={FOOTER_ICON}
                  label={
                    readingFile
                      ? t("translate.readingFile", { name: readingFile })
                      : t("translate.openFile")
                  }
                  disabled={readingFile !== null}
                  onClick={() => fileInput.current?.click()}
                >
                  <Spinner className="size-5" />
                </IconAction>
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <span className="group/count ms-auto cursor-default whitespace-nowrap pe-2 font-mono text-ui-12 tabular-nums text-muted-foreground @max-[26rem]:hidden">
                      {source.length.toLocaleString(locale)}
                      <span
                        className={cn(
                          "group-hover/count:inline",
                          source.length < TRANSLATE_SOURCE_MAX_CHARS * 0.9 &&
                            "hidden",
                        )}
                      >
                        {" / "}
                        {TRANSLATE_SOURCE_MAX_CHARS.toLocaleString(locale)}
                      </span>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="top">
                    {t("translate.charLimit", {
                      max: TRANSLATE_SOURCE_MAX_CHARS.toLocaleString(locale),
                    })}
                  </TooltipContent>
                </Tooltip>

                <StyleMenu />
                <Button
                  type="button"
                  size="sm"
                  className={cn(
                    ACTION_BUTTON,
                    "shrink-0",
                    busy ? STOP_BUTTON : GO_BUTTON,
                  )}
                  variant={busy ? "outline" : "primary"}
                  disabled={!busy && !canTranslate}
                  onClick={busy ? stop : () => void startTranslate()}
                  aria-keyshortcuts="Control+Enter Meta+Enter"
                  title={
                    busy
                      ? undefined
                      : `${t("translate.translate")} (${shortcut})`
                  }
                >
                  {busy ? (
                    <>
                      <HugeiconsIcon
                        icon={StopIcon}
                        className={cn("mr-2", STOP_GLYPH)}
                      />
                      {t("translate.stop")}
                    </>
                  ) : (
                    t("translate.translate")
                  )}
                </Button>
              </div>
            </div>
            <input
              ref={fileInput}
              type="file"
              accept={`text/*,${OPEN_FILE_EXTENSIONS}`}
              className="hidden"
              onChange={(event) => {
                void openFile(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
            {dragging && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-xl bg-background/80 text-ui-13 text-muted-foreground">
                {t("translate.dropFile")}
              </div>
            )}
          </section>

          <section
            aria-label={t("translate.outputLabel")}
            aria-busy={busy}
            className="flex min-w-0 flex-col gap-3"
          >
            <LanguageTabs
              value={prefs.target}
              tabs={languageTabs(prefs.target, prefs.recentTargets)}
              onChange={setTarget}
              label={t("translate.targetLanguage")}
              className="md:ps-6"
            >
              <IconAction
                icon={ArrowLeftRightIcon}
                label={
                  canSwap ? t("translate.swap") : t("translate.swapNeedsSource")
                }
                disabled={!canSwap || busy}
                onClick={() => swap(detected)}
                className="ms-auto md:hidden"
              />
            </LanguageTabs>
            <div
              className={cn(
                PANEL,
                paneHeight,
                "flex flex-1 flex-col overflow-hidden",
              )}
            >
              <div
                ref={outputRef}
                className={cn(
                  EDGE_SCROLL,
                  BOTTOM_FADE,
                  "min-h-0 flex-1 overflow-y-auto px-3.5 py-3",
                )}
              >
                {output !== "" ? (
                  <p
                    dir="auto"
                    aria-live="polite"
                    className={cn(
                      "whitespace-pre-wrap break-words text-base [overflow-wrap:anywhere] md:text-sm",
                      stale && "text-muted-foreground",
                    )}
                  >
                    {output}
                  </p>
                ) : (
                  <p
                    aria-live="polite"
                    className="flex items-center gap-2 text-base text-muted-foreground md:text-sm"
                  >
                    {runStatus && <Spinner className="size-3.5" />}
                    {runStatus ?? t("translate.outputEmpty")}
                  </p>
                )}
              </div>
              {output !== "" && (
                <div className="flex shrink-0 items-center gap-1 px-2.5 pt-1.5 pb-2.5">
                  {readAloud.available &&
                    (readAloud.speaking ? (
                      <IconAction
                        iconClassName="size-5"
                        icon={VolumeMute02Icon}
                        label={t("translate.stopReading")}
                        onClick={readAloud.stop}
                        className={cn(
                          FOOTER_ICON,
                          "text-destructive hover:text-destructive",
                        )}
                      />
                    ) : (
                      <IconAction
                        className={FOOTER_ICON}
                        iconClassName="size-5"
                        icon={ReadAloudIcon}
                        label={t("translate.readAloud")}
                        onClick={() =>
                          readAloud.speak(
                            output,
                            translateSpeechTag(
                              (result ?? attempt)?.target ?? prefs.target,
                            ),
                          )
                        }
                      />
                    ))}
                  <span className="ms-auto" />
                  {result && !stale && !busy && (
                    <IconAction
                      className={FOOTER_ICON}
                      iconClassName="size-5"
                      icon={Message01Icon}
                      label={t("translate.chat")}
                      onClick={chatAboutResult}
                    />
                  )}
                  {!busy && (
                    <IconAction
                      icon={ArrowLeft02Icon}
                      iconClassName="size-5"
                      label={t("translate.useAsSource")}
                      onClick={outputToSource}
                      className={cn(FOOTER_ICON, "rtl:rotate-180")}
                    />
                  )}
                  <IconAction
                    icon={copied ? Tick02Icon : Copy01Icon}
                    iconClassName="size-5"
                    className={FOOTER_ICON}
                    label={copied ? t("translate.copied") : t("translate.copy")}
                    onClick={() => void copyToClipboard(output).then(setCopied)}
                  />
                </div>
              )}
            </div>
          </section>
        </div>

        <div className="mx-auto flex w-full max-w-2xl flex-col gap-2.5">
          <div className="min-h-[1lh] text-center text-ui-12 leading-normal">
            {busy ? null : error ? (
              <div role="alert" className="grid gap-0.5">
                <p className="text-ui-13 font-medium text-destructive">
                  {t("translate.errors.title")}
                </p>
                <p className="text-muted-foreground [overflow-wrap:anywhere]">
                  {errorMessage}
                </p>
              </div>
            ) : output !== "" && result === null ? (
              <p className="text-muted-foreground">{t("translate.stopped")}</p>
            ) : output !== "" && stale ? (
              <p className="text-muted-foreground">{t("translate.stale")}</p>
            ) : null}
          </div>
        </div>
      </div>

      {prefs.showPrompt && promptOptions && (
        <section className="grid gap-2">
          <div className="flex items-center justify-between gap-3">
            <FieldLabel
              htmlFor={promptId}
              className="text-ui-13 font-medium text-foreground"
            >
              {t("translate.promptLabel")}
            </FieldLabel>
            {customPrompt !== null && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="text-ui-12 text-muted-foreground"
                onClick={() => {
                  setPromptDraft(null);
                  setCustomPrompt(null);
                }}
              >
                {t("translate.promptReset")}
              </Button>
            )}
          </div>
          <Textarea
            id={promptId}
            value={promptDraft ?? customPrompt ?? defaultPrompt}
            onChange={(event) => {
              setPromptDraft(event.target.value);
              setCustomPrompt(event.target.value);
            }}
            onBlur={() => setPromptDraft(null)}
            spellCheck={false}
            className="min-h-28 font-mono text-ui-13 md:text-ui-13"
          />
          <p className={HINT}>
            {t("translate.promptHint", {
              target: TARGET_PLACEHOLDER,
              source: SOURCE_PLACEHOLDER,
            })}
            {customPrompt !== null && ` ${t("translate.promptEdited")}`}
          </p>
        </section>
      )}

      <AlertDialog
        open={confirmDownload !== null}
        onOpenChange={(open) => {
          if (!open) setConfirmDownload(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogMedia>
              <HugeiconsIcon
                icon={AiTranslateIcon}
                className="size-5 text-muted-foreground"
              />
            </AlertDialogMedia>
            <AlertDialogTitle>
              {t("translate.downloadConfirmTitle", {
                model: confirmDownload?.label ?? "",
              })}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {confirmDownload &&
                t("translate.firstRunHint", {
                  model: confirmDownload.label,
                  size: formatModelBytes(confirmDownload.sizeBytes),
                })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="dark"
              onClick={() => {
                setConfirmDownload(null);
                void translate();
              }}
            >
              {t("translate.downloadConfirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <TranslateHistoryList />
    </main>
  );
}
