// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ComposerModelPicker } from "@/components/assistant-ui/composer-model-picker/composer-model-picker";
import {
  ComposerModelPickerContext,
  type ComposerModelPickerSource,
} from "@/components/assistant-ui/composer-model-picker/context";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandInput,
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
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { useNativeFileDrop } from "@/features/native-intents";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { ReadAloudIcon } from "@/lib/action-bar-icons";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Settings01Icon } from "@/lib/hugeicons-derived";
import { StopIcon } from "@/lib/media-control-icons";
import { MicIcon } from "@/lib/mic-icon";
import { MenuTickIcon, Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowLeft02Icon,
  ArrowLeftRightIcon,
  Cancel01Icon,
  ClipboardPasteIcon,
  Message01Icon,
  RemoveCircleIcon,
  Upload01Icon,
  VolumeMute02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import { chatAboutTranslations } from "./chat-handoff";
import { TranslateHistoryList } from "./history-list";
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

const IS_MAC =
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad/.test(navigator.platform);

const ACTION_BUTTON =
  "relative z-10 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100";
const GO_BUTTON =
  "h-11 px-8 text-ui-14 bg-control-accent text-control-accent-foreground hover:bg-control-accent/85";
const STOP_BUTTON = "text-ui-13 hover:bg-muted dark:hover:bg-muted";
const HINT = "text-ui-12 leading-normal text-muted-foreground";
const PANEL = "rounded-xl bg-wash-6";
const PANE_HEIGHT = "h-52 sm:h-96 md:h-[max(30rem,calc(100dvh-18rem))]";
// The box clips and an inner element scrolls: a scrollbar on the box itself squares off its corners.
// Its outline is an inset shadow, not a border, so the scrollbar reaches the box's very edge.
const SOURCE_BOX =
  "overflow-hidden rounded-xl bg-background shadow-[inset_0_0_0_1px_var(--border)] transition-[box-shadow,background-color] focus-within:shadow-[inset_0_0_0_1px_var(--ring)] dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:shadow-none dark:focus-within:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))] dark:focus-within:shadow-none";
// A thin scrollbar keeps a margin beside its thumb; the scroller overhangs the box by that much.
const EDGE_SCROLL = "hover-scrollbar h-full w-[calc(100%+2px)] max-w-none";
const SOURCE_TEXT = cn(
  EDGE_SCROLL,
  "rounded-none border-0 bg-transparent dark:bg-transparent dark:focus-visible:bg-transparent",
);
const STOP_GLYPH = "size-3.5 fill-current";

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
}: {
  value: string;
  onChange: (code: string) => void;
  label: string;
  allowAuto?: boolean;
}) {
  const t = useT();
  const locale = useLocale();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const languages = useMemo(
    () =>
      TRANSLATE_LANGUAGES.map((language) => ({
        ...language,
        label: translateLanguageLabel(language, locale),
      })),
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
  const currentLabel =
    allowAuto && value === AUTO_SOURCE
      ? autoLabel
      : (languages.find((language) => language.code === value)?.label ?? value);
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
        if (!next) setQuery("");
      }}
    >
      <PopoverTrigger asChild={true}>
        <Button
          type="button"
          variant="picker"
          size="sm"
          aria-label={label}
          title={label}
          className={cn(
            "min-w-40 justify-between rounded-full border text-ui-13",
            allowAuto && value === AUTO_SOURCE && "text-muted-foreground",
          )}
        >
          <span className="min-w-0 truncate">{currentLabel}</span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground"
          />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={4}
        className="w-64 gap-0 rounded-lg p-1"
      >
        {/* Filtered here: the labels are in the interface language, the search also takes English. */}
        <Command
          shouldFilter={false}
          className="rounded-none bg-transparent p-0"
        >
          <CommandInput
            value={query}
            onValueChange={setQuery}
            aria-label={t("translate.searchLanguages")}
            placeholder={t("translate.searchLanguages")}
          />
          <CommandList>
            {!showAuto && shown.length === 0 && (
              <CommandEmpty>{t("translate.noLanguages")}</CommandEmpty>
            )}
            {showAuto && (
              <CommandItem
                value={AUTO_SOURCE}
                onSelect={() => pick(AUTO_SOURCE)}
              >
                <span className="min-w-0 flex-1 truncate">{autoLabel}</span>
                {value === AUTO_SOURCE && (
                  <HugeiconsIcon
                    icon={MenuTickIcon}
                    strokeWidth={2}
                    className="size-4"
                  />
                )}
              </CommandItem>
            )}
            {shown.map((language) => (
              <CommandItem
                key={language.code}
                value={language.code}
                onSelect={() => pick(language.code)}
              >
                <span className="min-w-0 flex-1 truncate">
                  {language.label}
                </span>
                {language.code === value && (
                  <HugeiconsIcon
                    icon={MenuTickIcon}
                    strokeWidth={2}
                    className="size-4"
                  />
                )}
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
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
            className="unsloth-thinking-pill composer-model-chip max-w-56"
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
  const { mode, tone, autoTranslate, saveHistory, showPrompt } =
    useTranslateStore(
      useShallow((state) => ({
        mode: state.prefs.mode,
        tone: state.prefs.tone,
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
        <OptionRow label={t("translate.options.mode")}>
          {(id) => (
            <Select
              value={mode}
              onValueChange={(next) =>
                setPrefs({ mode: next as TranslateMode })
              }
            >
              <SelectTrigger id={id} size="sm" className="w-36">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TRANSLATE_MODES.map((option) => (
                  <SelectItem key={option} value={option}>
                    {t(MODE_LABEL[option])}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </OptionRow>
        <OptionRow label={t("translate.options.tone")}>
          {(id) => (
            <Select
              value={tone}
              onValueChange={(next) =>
                setPrefs({ tone: next as TranslateTone })
              }
            >
              <SelectTrigger id={id} size="sm" className="w-36">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TRANSLATE_TONES.map((option) => (
                  <SelectItem key={option} value={option}>
                    {t(TONE_LABEL[option])}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </OptionRow>
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

function RunStatus({ phase, model }: { phase: TranslatePhase; model: string }) {
  const t = useT();
  const downloading = phase.kind === "downloading";
  const status = downloading
    ? t("translate.status.downloading", { model })
    : phase.kind === "loading"
      ? t("translate.status.loading", { model })
      : phase.kind === "translating"
        ? phase.parts > 1
          ? t("translate.status.translatingPart", {
              part: phase.part,
              parts: phase.parts,
            })
          : t("translate.status.translating")
        : t("translate.status.preparing");
  return (
    <>
      <output
        aria-live="polite"
        aria-atomic="true"
        className="text-center text-ui-12 text-muted-foreground"
      >
        {status}
        {downloading && (
          <span className="ml-1.5 font-mono tabular-nums">
            {t("translate.status.downloadProgress", {
              done: formatModelBytes(phase.downloadedBytes) || "0 MB",
              total: formatModelBytes(phase.totalBytes),
            })}
          </span>
        )}
      </output>
      <Progress
        value={
          downloading && phase.totalBytes > 0
            ? Math.min(100, (phase.downloadedBytes / phase.totalBytes) * 100)
            : 0
        }
        indeterminate={!downloading || phase.downloadedBytes <= 0}
        aria-label={status}
        className="h-1.5"
      />
    </>
  );
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

  const busy = phase !== null;
  const canTranslate = source.trim() !== "";
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

  const openFile = async (file: File | undefined) => {
    if (!file) return;
    try {
      // Sliced first: only the kept characters are ever read into memory.
      loadText(await file.slice(0, TRANSLATE_SOURCE_MAX_CHARS * 4 + 4).text());
    } catch {
      toast.error(t("translate.fileNotText"));
    }
  };

  const {
    ref: dropRef,
    dragging,
    dragHandlers,
  } = useNativeFileDrop({
    onFiles: (files) => void openFile(files[0]),
    accept: TEXT_FILE_EXTENSIONS,
    multiple: false,
  });

  const paste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) loadText(source + text);
    } catch {
      // Clipboard reads need a permission the user may refuse; Ctrl+V still works.
    }
  };

  const errorMessage = error
    ? error.code === "other"
      ? error.message
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
  const firstRun =
    !busy && dedicated !== null && downloaded[dedicated.id] === false;

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-6 pb-10 pt-8 font-heading max-sm:px-4 sm:px-10 4xl:max-w-6xl">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
          {t("translate.title")}
        </h1>
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

      <div className="grid gap-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <LanguagePicker
            value={prefs.from}
            onChange={(from) => setPrefs({ from })}
            label={t("translate.sourceLanguage")}
            allowAuto={true}
          />
          <IconAction
            icon={ArrowLeftRightIcon}
            label={
              prefs.from === AUTO_SOURCE
                ? t("translate.swapNeedsSource")
                : t("translate.swap")
            }
            disabled={prefs.from === AUTO_SOURCE || busy}
            onClick={swap}
          />
          <LanguagePicker
            value={prefs.target}
            onChange={setTarget}
            label={t("translate.targetLanguage")}
          />
          {prefs.recentTargets.length > 0 && (
            <fieldset className="flex items-center gap-1.5 ps-2 max-sm:hidden">
              <legend className="float-start pe-0.5 text-ui-12 leading-8 text-muted-foreground">
                {t("translate.recentTargets")}
              </legend>
              {prefs.recentTargets.map((code) => {
                const language = findTranslateLanguage(code);
                if (!language) return null;
                const name = translateLanguageLabel(language, locale);
                return (
                  <Button
                    key={code}
                    type="button"
                    variant="outline"
                    size="sm"
                    title={`${t("translate.targetLanguage")}: ${name}`}
                    className="rounded-full text-ui-12 font-normal text-muted-foreground hover:text-foreground"
                    onClick={() => setTarget(code)}
                  >
                    {name}
                  </Button>
                );
              })}
            </fieldset>
          )}
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <section
            ref={dropRef}
            {...dragHandlers}
            aria-label={t("translate.sourceLabel")}
            className="relative flex flex-col gap-1.5"
          >
            <div
              className={cn(
                SOURCE_BOX,
                PANE_HEIGHT,
                dragging && "shadow-[inset_0_0_0_1px_var(--ring)]",
              )}
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
                    void translate();
                  }
                }}
                maxLength={TRANSLATE_SOURCE_MAX_CHARS}
                placeholder={t("translate.sourcePlaceholder")}
                aria-label={t("translate.sourceLabel")}
                spellCheck={false}
                dir="auto"
                fieldSizing="fixed"
                className={SOURCE_TEXT}
              />
            </div>
            <div className="flex items-center">
              {dictation.supported &&
                (dictation.dictating ? (
                  <IconAction
                    icon={dictation.finalizing ? undefined : StopIcon}
                    iconClassName={STOP_GLYPH}
                    label={t("translate.stopDictating")}
                    onClick={dictation.stop}
                    className="text-destructive hover:text-destructive"
                  >
                    <Spinner className="size-4" />
                  </IconAction>
                ) : (
                  <IconAction
                    label={t("translate.dictate")}
                    onClick={dictation.start}
                  >
                    <MicIcon className="size-4" />
                  </IconAction>
                ))}
              <IconAction
                icon={Upload01Icon}
                label={t("translate.openFile")}
                onClick={() => fileInput.current?.click()}
              />
              <IconAction
                icon={ClipboardPasteIcon}
                label={t("translate.paste")}
                onClick={() => void paste()}
              />
              {source !== "" && (
                <IconAction
                  icon={Cancel01Icon}
                  label={t("translate.clear")}
                  onClick={clear}
                />
              )}
              <span className="ml-auto pr-2 font-mono text-ui-11p5 tabular-nums text-muted-foreground">
                {source.length.toLocaleString(locale)} /{" "}
                {TRANSLATE_SOURCE_MAX_CHARS.toLocaleString(locale)}
              </span>
            </div>
            <input
              ref={fileInput}
              type="file"
              accept={`text/*,${TEXT_FILE_EXTENSIONS}`}
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
            className="flex flex-col gap-1.5"
          >
            <div className={cn(PANEL, PANE_HEIGHT, "overflow-hidden")}>
              <div
                ref={outputRef}
                className={cn(EDGE_SCROLL, "overflow-y-auto px-3.5 py-3")}
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
                  <p className="text-base text-muted-foreground md:text-sm">
                    {t("translate.outputEmpty")}
                  </p>
                )}
              </div>
            </div>
            {output !== "" && (
              <div className="flex items-center justify-end">
                {result && !stale && !busy && (
                  <IconAction
                    icon={Message01Icon}
                    label={t("translate.chat")}
                    onClick={chatAboutResult}
                  />
                )}
                {!busy && (
                  <IconAction
                    icon={ArrowLeft02Icon}
                    label={t("translate.useAsSource")}
                    onClick={outputToSource}
                    className="rtl:rotate-180"
                  />
                )}
                {readAloud.available &&
                  (readAloud.speaking ? (
                    <IconAction
                      icon={VolumeMute02Icon}
                      label={t("translate.stopReading")}
                      onClick={readAloud.stop}
                      className="text-destructive hover:text-destructive"
                    />
                  ) : (
                    <IconAction
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
                <IconAction
                  icon={copied ? Tick02Icon : Copy01Icon}
                  label={copied ? t("translate.copied") : t("translate.copy")}
                  onClick={() => void copyToClipboard(output).then(setCopied)}
                />
              </div>
            )}
          </section>
        </div>

        <div className="mx-auto flex w-full max-w-sm flex-col gap-2.5 pt-1">
          {phase && <RunStatus phase={phase} model={runModelLabel} />}
          <div className="flex items-center justify-center">
            <Button
              type="button"
              className={cn(ACTION_BUTTON, busy ? STOP_BUTTON : GO_BUTTON)}
              variant={busy ? "outline" : "primary"}
              disabled={!busy && !canTranslate}
              onClick={busy ? stop : () => void translate()}
              aria-keyshortcuts="Control+Enter Meta+Enter"
              title={
                busy ? undefined : `${t("translate.translate")} (${shortcut})`
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
            ) : firstRun && dedicated ? (
              <p className="text-muted-foreground">
                {t("translate.firstRunHint", {
                  model: dedicated.label,
                  size: formatModelBytes(dedicated.sizeBytes),
                })}
              </p>
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

      <TranslateHistoryList />
    </main>
  );
}
