//! System-wide voice typing.
//!
//! A resident global shortcut starts a dictation session in the main webview and the
//! finished transcript is typed into whatever app currently has focus, via the
//! clipboard. The shortcut is detected with a low-level key poll on Windows: the only
//! way to see both press and release for an arbitrary chord without registering it
//! with the OS (registered hotkeys fire on press only, and would also swallow the key
//! from the focused app).
//!
//! This module owns the config, the shortcut poller, the overlay window and the
//! transcript insertion. The settings UI, recording and transcription live in the
//! frontend, which drives everything through the commands below.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

mod hud;
pub(crate) mod paste;

/// True while the frontend is capturing (or still transcribing) audio.
static SESSION_ACTIVE: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActivationMode {
    /// Hold the chord for the whole dictation (push-to-talk).
    Hold,
    /// Every press starts or ends the dictation.
    Toggle,
    /// Quick tap toggles the session on, holding talks and stops on release.
    Hybrid,
}

struct Config {
    enabled: bool,
    shortcut: String,
    mode: ActivationMode,
    /// Whether the overlay window may be shown; off keeps everything headless.
    hud: bool,
}

/// Starts disabled: the frontend pushes the persisted settings right after boot.
static CONFIG: Mutex<Config> = Mutex::new(Config {
    enabled: false,
    shortcut: String::new(),
    mode: ActivationMode::Hybrid,
    hud: true,
});

fn hud_enabled() -> bool {
    let config = CONFIG.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    config.hud
}

#[derive(Clone, Serialize)]
struct SignalPayload {
    action: &'static str,
}

fn emit_signal(app: &AppHandle, action: &'static str) {
    let _ = app.emit_to("main", "voice-typing:signal", SignalPayload { action });
}

/// Wire the shortcut poller. The overlay window is created on first use.
pub fn init(app: &AppHandle) {
    #[cfg(target_os = "windows")]
    shortcut_poller::spawn(app.clone());
    #[cfg(not(target_os = "windows"))]
    let _ = app;
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceTypingInfo {
    os: String,
    /// Whether a system-wide shortcut can be armed on this platform.
    shortcut_supported: bool,
    /// Whether press and release are both observable (push-to-talk needs this).
    hold_supported: bool,
    /// Whether a transcript can be typed into the focused app.
    paste_supported: bool,
}

#[tauri::command]
pub fn voice_typing_info() -> VoiceTypingInfo {
    let shortcut_supported = cfg!(target_os = "windows");
    VoiceTypingInfo {
        os: std::env::consts::OS.to_string(),
        shortcut_supported,
        hold_supported: shortcut_supported,
        paste_supported: true,
    }
}

#[tauri::command]
pub fn voice_typing_configure(
    app: AppHandle,
    enabled: bool,
    shortcut: String,
    mode: String,
    hud: bool,
) -> Result<(), String> {
    let mode = match mode.as_str() {
        "hold" => ActivationMode::Hold,
        "toggle" => ActivationMode::Toggle,
        "hybrid" => ActivationMode::Hybrid,
        other => return Err(format!("Unknown activation mode: {other}")),
    };
    // Parse eagerly so a bad chord is rejected while the user is still editing it.
    let chord = parse_chord(&shortcut)?;
    // Platforms without a shortcut layer keep the feature dormant instead of failing
    // the call: the settings section is hidden there anyway.
    let enabled = enabled && cfg!(target_os = "windows");
    {
        let mut config = CONFIG.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        config.enabled = enabled;
        config.shortcut = chord.canonical();
        config.mode = mode;
        config.hud = hud;
    }
    if !enabled || !hud {
        hud::hide(&app);
    }
    if !enabled {
        SESSION_ACTIVE.store(false, Ordering::SeqCst);
    }
    Ok(())
}

/// The frontend reports every phase change here so the overlay and the shortcut
/// poller (which cancels on Escape) know what is happening.
#[tauri::command]
pub fn voice_typing_hud_state(app: AppHandle, state: String) {
    match state.as_str() {
        "recording" => {
            SESSION_ACTIVE.store(true, Ordering::SeqCst);
            if hud_enabled() {
                hud::show(&app);
            }
        }
        "transcribing" | "done" => SESSION_ACTIVE.store(false, Ordering::SeqCst),
        "idle" => {
            SESSION_ACTIVE.store(false, Ordering::SeqCst);
            hud::hide(&app);
            return;
        }
        _ => return,
    }
    if hud_enabled() {
        hud::emit_state(&app, &state);
    }
}

#[tauri::command]
pub fn voice_typing_hud_level(app: AppHandle, level: f32) {
    hud::emit_level(&app, level.clamp(0.0, 1.0));
}

// ---------------------------------------------------------------------------
// Shortcut chords
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
struct Chord {
    ctrl: bool,
    shift: bool,
    alt: bool,
    super_key: bool,
    vk: u16,
    /// Display name of the non-modifier key, e.g. "Space", "A", "F4".
    key_label: String,
}

impl Chord {
    fn canonical(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.ctrl {
            parts.push("Ctrl".to_string());
        }
        if self.shift {
            parts.push("Shift".to_string());
        }
        if self.alt {
            parts.push("Alt".to_string());
        }
        if self.super_key {
            parts.push("Super".to_string());
        }
        parts.push(self.key_label.clone());
        parts.join("+")
    }
}

/// Parses the frontend's chord format (`Ctrl+Shift+Space`) into virtual-key codes.
/// Kept canonical in one place: the settings UI builds the same strings.
fn parse_chord(raw: &str) -> Result<Chord, String> {
    let mut chord = Chord {
        ctrl: false,
        shift: false,
        alt: false,
        super_key: false,
        vk: 0,
        key_label: String::new(),
    };
    let mut key: Option<(u16, String)> = None;
    for token in raw.split('+') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let lower = token.to_ascii_lowercase();
        match lower.as_str() {
            "ctrl" | "control" => chord.ctrl = true,
            "shift" => chord.shift = true,
            "alt" | "option" => chord.alt = true,
            "super" | "meta" | "win" | "cmd" | "command" => chord.super_key = true,
            "escape" | "esc" => {
                return Err("Escape is reserved for cancelling a dictation".to_string());
            }
            _ => {
                if key.is_some() {
                    return Err("Use exactly one key besides the modifiers".to_string());
                }
                key = Some(map_key(&lower).ok_or_else(|| format!("Unsupported key: {token}"))?);
            }
        }
    }
    let Some((vk, key_label)) = key else {
        return Err("Shortcut needs a key besides the modifiers".to_string());
    };
    if !(chord.ctrl || chord.shift || chord.alt || chord.super_key) {
        return Err("Shortcut needs at least one modifier (Ctrl, Shift, Alt or Super)".to_string());
    }
    chord.vk = vk;
    chord.key_label = key_label;
    Ok(chord)
}

/// Maps a lowercased key name to its virtual-key code. Letters and digits are
/// layout-independent; the punctuation keys assume the common US layout.
fn map_key(raw: &str) -> Option<(u16, String)> {
    if raw.len() == 1 {
        let character = raw.chars().next()?;
        if character.is_ascii_alphabetic() {
            let upper = character.to_ascii_uppercase();
            return Some((upper as u16, upper.to_string()));
        }
        if character.is_ascii_digit() {
            return Some((character as u16, character.to_string()));
        }
    }
    let mapped = match raw {
        "space" => (0x20, "Space"),
        "enter" | "return" => (0x0D, "Enter"),
        "tab" => (0x09, "Tab"),
        "backspace" => (0x08, "Backspace"),
        "-" | "minus" => (0xBD, "-"),
        "=" | "equal" | "equals" => (0xBB, "="),
        "[" | "bracketleft" => (0xDB, "["),
        "]" | "bracketright" => (0xDD, "]"),
        ";" | "semicolon" => (0xBA, ";"),
        "'" | "quote" => (0xDE, "'"),
        "," | "comma" => (0xBC, ","),
        "." | "period" => (0xBE, "."),
        "/" | "slash" => (0xBF, "/"),
        "\\" | "backslash" => (0xDC, "\\"),
        "`" | "backquote" => (0xC0, "`"),
        _ => {
            let number = raw.strip_prefix('f').and_then(|rest| rest.parse::<u8>().ok());
            match number {
                Some(number) if (1..=24).contains(&number) => {
                    return Some((0x70 + (number - 1) as u16, format!("F{number}")));
                }
                _ => return None,
            }
        }
    };
    Some((mapped.0, mapped.1.to_string()))
}

// ---------------------------------------------------------------------------
// Windows: low-level shortcut poll
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
mod shortcut_poller {
    use super::{emit_signal, parse_chord, ActivationMode, Chord, CONFIG, SESSION_ACTIVE};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};
    use tauri::AppHandle;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_CONTROL, VK_ESCAPE, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
    };

    const POLL_INTERVAL: Duration = Duration::from_millis(16);
    /// A release inside this window is a tap (toggle), anything longer is a hold.
    const TAP_THRESHOLD: Duration = Duration::from_millis(300);

    /// The high bit means "currently down". The low bit ("pressed since the last
    /// call") belongs to the foreground thread and is useless in a background loop.
    fn down(vk: u16) -> bool {
        unsafe { (GetAsyncKeyState(vk as i32) as u16 & 0x8000) != 0 }
    }

    impl Chord {
        fn is_down(&self) -> bool {
            (!self.ctrl || down(VK_CONTROL))
                && (!self.shift || down(VK_SHIFT))
                && (!self.alt || down(VK_MENU))
                && (!self.super_key || down(VK_LWIN) || down(VK_RWIN))
                && down(self.vk)
        }
    }

    pub fn spawn(app: AppHandle) {
        let spawned = std::thread::Builder::new()
            .name("voice-typing-shortcut".to_string())
            .spawn(move || poll(app));
        if let Err(error) = spawned {
            log::warn!("Voice typing shortcut thread could not start: {error}");
        }
    }

    fn poll(app: AppHandle) {
        let mut chord: Option<Chord> = None;
        let mut chord_source = String::new();
        let mut chord_held = false;
        let mut pressed_at: Option<Instant> = None;
        let mut press_started_session = false;
        let mut escape_held = false;

        loop {
            std::thread::sleep(POLL_INTERVAL);

            let (enabled, mode, source) = {
                let config = CONFIG.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                (config.enabled, config.mode, config.shortcut.clone())
            };
            if !enabled {
                chord = None;
                chord_source.clear();
                chord_held = false;
                pressed_at = None;
                escape_held = false;
                continue;
            }
            if source != chord_source {
                chord = parse_chord(&source).ok();
                chord_source = source;
                chord_held = false;
                pressed_at = None;
            }
            let Some(current) = chord.as_ref() else {
                continue;
            };

            let pressed = current.is_down();
            let now = Instant::now();
            if pressed && !chord_held {
                chord_held = true;
                pressed_at = Some(now);
                press_started_session = !SESSION_ACTIVE.load(Ordering::SeqCst);
                match mode {
                    // Recording starts on press so no speech is lost; whether the
                    // release ends it depends on the mode.
                    ActivationMode::Hold | ActivationMode::Hybrid => {
                        if press_started_session {
                            emit_signal(&app, "start");
                        }
                    }
                    ActivationMode::Toggle => {
                        emit_signal(
                            &app,
                            if press_started_session { "start" } else { "stop" },
                        );
                    }
                }
            } else if !pressed && chord_held {
                chord_held = false;
                let held_for = pressed_at
                    .map(|start| now.saturating_duration_since(start))
                    .unwrap_or_default();
                pressed_at = None;
                match mode {
                    ActivationMode::Hold => emit_signal(&app, "stop"),
                    ActivationMode::Toggle => {}
                    ActivationMode::Hybrid => {
                        if press_started_session {
                            // Push-to-talk release; a quick tap leaves the session on.
                            if held_for >= TAP_THRESHOLD {
                                emit_signal(&app, "stop");
                            }
                        } else {
                            // Pressing again while dictating always ends the session.
                            emit_signal(&app, "stop");
                        }
                    }
                }
            }

            // Escape ends an in-flight capture, matching the in-app shortcut.
            let escape = down(VK_ESCAPE);
            if escape && !escape_held && SESSION_ACTIVE.load(Ordering::SeqCst) {
                emit_signal(&app, "cancel");
            }
            escape_held = escape;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modifier_chords() {
        let chord = parse_chord("Ctrl+Shift+Space").expect("chord");
        assert!(chord.ctrl && chord.shift && !chord.alt && !chord.super_key);
        assert_eq!(chord.vk, 0x20);
        assert_eq!(chord.canonical(), "Ctrl+Shift+Space");
    }

    #[test]
    fn normalizes_aliases_and_order() {
        let chord = parse_chord("shift + control + a").expect("chord");
        assert_eq!(chord.canonical(), "Ctrl+Shift+A");
        assert_eq!(chord.vk, 0x41);
        let chord = parse_chord("Win+. ").expect("chord");
        assert_eq!(chord.canonical(), "Super+.");
        assert_eq!(chord.vk, 0xBE);
    }

    #[test]
    fn maps_digits_and_function_keys() {
        assert_eq!(parse_chord("Alt+1").expect("chord").vk, 0x31);
        assert_eq!(parse_chord("Alt+F4").expect("chord").vk, 0x73);
        assert_eq!(parse_chord("Alt+F4").expect("chord").canonical(), "Alt+F4");
    }

    #[test]
    fn rejects_bare_keys_escape_and_unknown() {
        assert!(parse_chord("Space").is_err());
        assert!(parse_chord("Ctrl+Escape").is_err());
        assert!(parse_chord("Ctrl+Nope").is_err());
        assert!(parse_chord("Ctrl+A+B").is_err());
        assert!(parse_chord("Ctrl+").is_err());
    }
}
