// SPDX-License-Identifier: AGPL-3.0-only
import { useRef, useState } from "react";
import { Upload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { useT, type TranslationKey } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { MenuTickIcon } from "@/lib/tick-icon";
// Shared settings primitives: avoid importing the settings barrel into this feature.
// eslint-disable-next-line no-restricted-imports
import { SettingsSection } from "@/features/settings/components/settings-section";
// eslint-disable-next-line no-restricted-imports
import { SettingsRow } from "@/features/settings/components/settings-row";
// eslint-disable-next-line no-restricted-imports
import { useTheme } from "@/features/settings/stores/theme-store";
import { WallpaperAdjustDialog } from "./adjust-dialog";
import { ChatBackground } from "./index";
import {
	DEFAULT_ADJUSTMENTS,
	EFFECTS,
	fadeKey,
	visibilityKey,
	LENS_MODES,
	type LensMode,
	MATERIALS,
	type WindowMaterial,
	IMAGE_SIZING,
	type ImageSizing,
	SCOPES,
	type BackgroundScope,
	type BackgroundEffect,
} from "./model";
import {
	patchBackground,
	resetChatBackground,
	useChatBackground,
} from "./storage";
import { uploadBackgroundImage } from "./upload";
import {
	nativeMaterialAvailable,
	useGlassStrength,
	useMaterialStore,
} from "./material";
import { MATERIAL_LABELS } from "./material-labels";

const labels: Record<BackgroundEffect, TranslationKey> = {
	None: "chatBackground.none",
	Dither: "chatBackground.dither",
	ASCII: "chatBackground.ascii",
	Halftone: "chatBackground.halftone",
	Scanlines: "chatBackground.scanlines",
};
const scopeLabels: Record<BackgroundScope, TranslationKey> = {
	newChat: "chatBackground.scopeNewChat",
	chats: "chatBackground.scopeChats",
	all: "chatBackground.scopeAll",
};
const sizingLabels: Record<ImageSizing, TranslationKey> = {
	cover: "chatBackground.fill",
	contain: "chatBackground.fit",
	width: "chatBackground.fitWidth",
	height: "chatBackground.fitHeight",
	stretch: "chatBackground.stretch",
	original: "chatBackground.original",
};
function GlassStyleRow() {
	const t = useT();
	const { style, nativeFailed, setMaterial } = useMaterialStore();
	const native = nativeMaterialAvailable();
	const options = MATERIALS.filter(
		(m) => m !== "solid" && (m !== "macos" || native),
	);
	const shown = options.includes(style) ? style : "liquid";
	const note = nativeFailed
		? "chatBackground.materialNativeFailed"
		: shown === "macos"
			? "chatBackground.materialMacosNote"
			: null;
	return (
		<SettingsRow
			label={t("chatBackground.materialStyle")}
			description={note ? t(note) : undefined}
		>
			<Select
				value={shown}
				onValueChange={(next) => setMaterial(next as WindowMaterial)}
			>
				<SelectTrigger
					aria-label={t("chatBackground.materialStyle")}
					className="w-52"
				>
					<SelectValue />
				</SelectTrigger>
				<SelectContent>
					{options.map((m) => (
						<SelectItem key={m} value={m}>
							{t(MATERIAL_LABELS[m])}
						</SelectItem>
					))}
				</SelectContent>
			</Select>
		</SettingsRow>
	);
}
export function GlassControls() {
	const t = useT();
	const { material, setEnabled } = useMaterialStore();
	const on = material !== "solid";
	return (
		<SettingsSection
			title={t("chatBackground.glassTheme")}
			badge={<Badge variant="accent">{t("shell.beta")}</Badge>}
			action={
				<Switch
					aria-label={t("chatBackground.glassTheme")}
					checked={on}
					onCheckedChange={setEnabled}
				/>
			}
		>
			{on && (
				<>
					<GlassStyleRow />
					<GlassScopeRow />
					<MaterialStrengthRow />
					<LensRow />
					<MaterialMenusRow />
				</>
			)}
		</SettingsSection>
	);
}
function GlassScopeRow() {
	const t = useT();
	const { scope, setScope } = useMaterialStore();
	return (
		<SettingsRow label={t("chatBackground.scope")}>
			<Select
				value={scope}
				onValueChange={(next) => setScope(next as BackgroundScope)}
			>
				<SelectTrigger aria-label={t("chatBackground.scope")} className="w-52">
					<SelectValue />
				</SelectTrigger>
				<SelectContent>
					{SCOPES.map((option) => (
						<SelectItem key={option} value={option}>
							{t(scopeLabels[option])}
						</SelectItem>
					))}
				</SelectContent>
			</Select>
		</SettingsRow>
	);
}
function MaterialMenusRow() {
	const t = useT();
	const { material, menus, setMenus } = useMaterialStore();
	const off =
		material === "solid" || (material === "macos" && !nativeMaterialAvailable());
	return (
		<SettingsRow
			label={t("chatBackground.materialMenus")}
			description={t("chatBackground.materialMenusDescription")}
		>
			<Switch
				aria-label={t("chatBackground.materialMenus")}
				checked={menus && !off}
				disabled={off}
				onCheckedChange={setMenus}
			/>
		</SettingsRow>
	);
}
function MaterialStrengthRow() {
	const t = useT();
	const { material, setStrength } = useMaterialStore();
	const strength = useGlassStrength();
	const { resolved } = useTheme();
	const off =
		material === "solid" || (material === "macos" && !nativeMaterialAvailable());
	return (
		<SettingsRow label={t("chatBackground.materialStrength")}>
			<div className="flex w-52 items-center gap-3">
				<Slider
					disabled={off}
					aria-label={t("chatBackground.materialStrength")}
					min={0}
					max={100}
					step={1}
					value={[strength]}
					onValueChange={([v]) => setStrength(v, resolved)}
				/>
				<span className="w-12 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
					{strength}%
				</span>
			</div>
		</SettingsRow>
	);
}
const lensLabels: Record<LensMode, TranslationKey> = {
	off: "chatBackground.lensOff",
	magnify: "chatBackground.lensMagnify",
	shrink: "chatBackground.lensShrink",
};
function LensRow() {
	const t = useT();
	const { material, lens, setLens } = useMaterialStore();
	const off =
		material === "solid" || (material === "macos" && !nativeMaterialAvailable());
	return (
		<SettingsRow
			label={t("chatBackground.lens")}
			description={t("chatBackground.lensDescription")}
		>
			<Select
				disabled={off}
				value={lens}
				onValueChange={(next) => setLens(next as LensMode)}
			>
				<SelectTrigger aria-label={t("chatBackground.lens")} className="w-52">
					<SelectValue />
				</SelectTrigger>
				<SelectContent>
					{LENS_MODES.map((mode) => (
						<SelectItem key={mode} value={mode}>
							{t(lensLabels[mode])}
						</SelectItem>
					))}
				</SelectContent>
			</Select>
		</SettingsRow>
	);
}
const SLIDERS: {
	key: "visibility" | "conversation" | "blur";
	label: TranslationKey;
	min: number;
	max: number;
	unit: string;
}[] = [
	{ key: "visibility", label: "chatBackground.visibility", min: 0, max: 100, unit: "%" },
	{ key: "conversation", label: "chatBackground.conversationVisibility", min: 0, max: 100, unit: "%" },
	{ key: "blur", label: "chatBackground.blur", min: 0, max: 20, unit: "px" },
];
const errors = {
	storage: "chatBackground.storageError",
	file: "chatBackground.fileError",
	dimensions: "chatBackground.dimensionsError",
	decode: "chatBackground.decodeError",
	effect: "chatBackground.effectError",
} as const;
export function ChatBackgroundControls() {
	const t = useT();
	const { value, error, ready, busy } = useChatBackground();
	const input = useRef<HTMLInputElement>(null);
	const adjustments = value ?? DEFAULT_ADJUSTMENTS;
	const { resolved } = useTheme();
	const glass = useMaterialStore((s) => s.material !== "solid");
	const [adjusting, setAdjusting] = useState(false);
	return (
		<SettingsSection
			title={t("chatBackground.title")}
			badge={<Badge variant="accent">{t("shell.navigation.newBadge")}</Badge>}
			action={
				// With an image: one-word, icon-free actions; empty keeps the icon Upload.
				<div className="flex max-w-full flex-wrap items-center justify-end gap-2">
					{value && (
						<Button
							className="text-foreground"
							type="button"
							variant="outline"
							size="sm"
							disabled={busy}
							onClick={() => setAdjusting(true)}
						>
							{t("chatBackground.adjust")}
						</Button>
					)}
					<Button
						className="text-foreground"
						type="button"
						variant="outline"
						size="sm"
						disabled={!ready || busy}
						onClick={() => input.current?.click()}
					>
						{!value && (
							<HugeiconsIcon
								icon={Upload01Icon}
								strokeWidth={1.75}
								className="size-4"
								aria-hidden="true"
							/>
						)}
						{busy
							? t("common.loading")
							: t(value ? "chatBackground.replace" : "chatBackground.upload")}
					</Button>
					{value && (
						<Button
							className="px-1 text-muted-foreground hover:text-foreground hover:no-underline"
							type="button"
							variant="link"
							size="sm"
							disabled={busy}
							onClick={() => void resetChatBackground()}
						>
							{t("chatBackground.remove")}
						</Button>
					)}
				</div>
			}
		>
			<input
				ref={input}
				className="sr-only"
				tabIndex={-1}
				aria-label={t("chatBackground.upload")}
				type="file"
				accept="image/jpeg,image/png,image/webp"
				disabled={!ready || busy}
				onChange={(event) => {
					const file = event.target.files?.[0];
					event.target.value = "";
					if (!file) return;
					void uploadBackgroundImage(file, DEFAULT_ADJUSTMENTS);
				}}
			/>
			{error && (
				<p role="alert" className="mt-3 mb-3 text-sm text-destructive">
					{t(errors[error])}
				</p>
			)}
			<div
				className={`${error ? "" : "mt-3 "}grid grid-cols-1 gap-3 sm:grid-cols-2`}
				data-wallpaper-previews=""
			>
				{[false, true].map((conversation) => (
					<div key={String(conversation)} className="min-w-0">
						<p className="mb-2 text-xs text-muted-foreground">
							{t(
								conversation
									? "chatBackground.conversation"
									: "chatBackground.newChat",
							)}
						</p>
						<div
							className="wallpaper-preview relative h-48 overflow-hidden rounded-xl border border-border bg-background"
							data-wallpaper-conversation={conversation || undefined}
						>
							<ChatBackground
								active={!(conversation && adjustments.scope === "newChat")}
							/>
							{conversation && <div className="wallpaper-preview-wash absolute inset-0" />}
							{!value && (
								<p className="absolute inset-x-4 top-4 text-center text-xs text-muted-foreground">
									{t("chatBackground.noBackground")}
								</p>
							)}
							<div className="wallpaper-preview-content absolute inset-x-4 bottom-4 space-y-3 text-xs">
								{conversation && (
									<p className="p-3">
										{t("chatBackground.response")}
									</p>
								)}
								<div className="wallpaper-preview-composer rounded-xl p-3">
									{t("chatBackground.prompt")}
								</div>
							</div>
						</div>
					</div>
				))}
			</div>
			{value && (
				<>
					<SettingsRow label={t("chatBackground.scope")} className="mt-3 items-start">
						<Select
							disabled={busy}
							value={adjustments.scope}
							onValueChange={(scope) =>
								patchBackground({ scope: scope as BackgroundScope })
							}
						>
							<SelectTrigger
								aria-label={t("chatBackground.scope")}
								className="w-52"
							>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								{SCOPES.map((scope) => (
									<SelectItem key={scope} value={scope}>
										{t(scopeLabels[scope])}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</SettingsRow>
					<SettingsRow label={t("chatBackground.effect")} className="items-start">
						<Select
							disabled={!value || busy}
							value={adjustments.effect}
							onValueChange={(effect) =>
								patchBackground({ effect: effect as BackgroundEffect })
							}
						>
							<SelectTrigger
								aria-label={t("chatBackground.effect")}
								className="w-52"
							>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								{EFFECTS.map((effect) => (
									<SelectItem key={effect} value={effect}>
										{t(labels[effect])}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</SettingsRow>
					<SettingsRow label={t("chatBackground.sizing")} className="items-start">
						<Popover>
							<PopoverTrigger asChild={true}>
								<Button
									variant="picker"
									disabled={!value || busy}
									aria-label={t("chatBackground.sizing")}
									className="w-52 justify-between font-normal"
								>
									{t(sizingLabels[adjustments.sizing])}
									<HugeiconsIcon
										icon={ChevronDownStandardIcon}
										strokeWidth={2}
										className="size-4 text-muted-foreground"
									/>
								</Button>
							</PopoverTrigger>
							<PopoverContent align="end" className="w-80 gap-0 rounded-lg p-1 font-heading">
								<div role="radiogroup" aria-label={t("chatBackground.sizing")}>
									{IMAGE_SIZING.map((sizing) => (
										<button
											key={sizing}
											type="button"
											role="radio"
											aria-checked={adjustments.sizing === sizing}
											className="relative flex w-full cursor-pointer items-center rounded-md py-2 pr-9 pl-3 text-left text-sm outline-hidden hover:bg-accent focus-visible:bg-accent"
											onClick={() => patchBackground({ sizing })}
										>
											{t(sizingLabels[sizing])}
											{adjustments.sizing === sizing && (
												<HugeiconsIcon
													icon={MenuTickIcon}
													strokeWidth={2}
													className="absolute right-3 size-4"
												/>
											)}
										</button>
									))}
								</div>
							</PopoverContent>
						</Popover>
					</SettingsRow>
					{SLIDERS.map(({ key: slider, label, min, max, unit }) => {
						const key =
							slider === "visibility"
								? visibilityKey(glass, resolved === "dark")
								: slider;
						return (
							<SettingsRow key={slider} className="items-start" label={t(label)}>
							<div className="flex w-52 items-center gap-3">
								<Slider
									disabled={!value || busy}
									aria-label={t(label)}
									min={min}
									max={max}
									step={1}
									value={[adjustments[key]]}
									onValueChange={([v]) => patchBackground({ [key]: v })}
								/>
								<span className="w-12 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
									{adjustments[key]}
									{unit}
								</span>
							</div>
							</SettingsRow>
						);
					})}
					<SettingsRow label={t("chatBackground.fade")} className="items-start">
						<Switch
							aria-label={t("chatBackground.fade")}
							checked={adjustments[fadeKey(glass)]}
							disabled={!value || busy}
							onCheckedChange={(fade) => patchBackground({ [fadeKey(glass)]: fade })}
						/>
					</SettingsRow>
				</>
			)}
			{value && (
				<WallpaperAdjustDialog
					open={adjusting}
					onOpenChange={setAdjusting}
					value={value}
					fade={adjustments[fadeKey(glass)]}
				/>
			)}
		</SettingsSection>
	);
}
