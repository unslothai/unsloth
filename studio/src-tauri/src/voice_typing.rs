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
//!
//! Every step logs at INFO level (never the transcript itself): a chord that produces
//! no visible reaction must still be traceable in the app log.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

mod hud;
pub(crate) mod paste;

/// A release inside this window is a tap (toggle), anything longer is a hold.
const TAP_THRESHOLD: Duration = Duration::from_millis(300);

/// True while a dictation is in flight: preparing, recording or transcribing. The
/// shortcut poller reads it to tell a press that opens a session from one that ends
/// it, and to route Escape to the frontend.
static SESSION_ACTIVE: AtomicBool = AtomicBool::new(false);

/// True while the user is recording a new shortcut chord in Settings, so pressing
/// the existing chord or Escape does not fire a live dictation session.
static SHORTCUT_SUSPENDED: AtomicBool = AtomicBool::new(false);

/// Lock-free packed representation of the active shortcut chord for the Windows
/// low-level keyboard hook (`WH_KEYBOARD_LL`), avoiding mutex locks and heap
/// allocations on every system keystroke.
/// Layout: bits 0..15 = vk, bit 16 = ctrl, bit 17 = shift, bit 18 = alt,
/// bit 19 = super_key, bit 20 = enabled.
static PACKED_CHORD: AtomicU32 = AtomicU32::new(0);

fn pack_chord(enabled: bool, chord: Option<&Chord>) -> u32 {
    if !enabled {
        return 0;
    }
    let Some(c) = chord else {
        return 0;
    };
    let mut bits = (c.vk as u32) & 0xFFFF;
    if c.ctrl {
        bits |= 1 << 16;
    }
    if c.shift {
        bits |= 1 << 17;
    }
    if c.alt {
        bits |= 1 << 18;
    }
    if c.super_key {
        bits |= 1 << 19;
    }
    bits |= 1 << 20;
    bits
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActivationMode {
    /// Hold the chord for the whole dictation (push-to-talk).
    Hold,
    /// Every press starts or ends the dictation.
    Toggle,
    /// Quick tap toggles the session on, holding talks and stops on release.
    Hybrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Edge {
    Press,
    Release,
}

/// Which signal a chord edge produces in a given mode. Pure so the mode matrix is
/// unit-tested without a keyboard; the poller supplies the state.
fn edge_signal(
    mode: ActivationMode,
    edge: Edge,
    started_session: bool,
    held_for: Duration,
) -> Option<&'static str> {
    match (mode, edge) {
        (ActivationMode::Hold | ActivationMode::Hybrid, Edge::Press) => {
            // Recording starts on press so no speech is lost; whether the release
            // ends it depends on the mode.
            started_session.then_some("start")
        }
        (ActivationMode::Toggle, Edge::Press) => {
            Some(if started_session { "start" } else { "stop" })
        }
        (ActivationMode::Hold, Edge::Release) => Some("stop"),
        (ActivationMode::Toggle, Edge::Release) => None,
        (ActivationMode::Hybrid, Edge::Release) => {
            if started_session {
                // Push-to-talk release; a quick tap leaves the session on.
                (held_for >= TAP_THRESHOLD).then_some("stop")
            } else {
                // Pressing again while dictating always ends the session.
                Some("stop")
            }
        }
    }
}

struct Config {
    enabled: bool,
    shortcut: String,
    parsed_chord: Option<Chord>,
    mode: ActivationMode,
    mute_while_recording: bool,
}

/// Starts disabled: the frontend pushes the persisted settings right after boot.
static CONFIG: Mutex<Config> = Mutex::new(Config {
    enabled: false,
    shortcut: String::new(),
    parsed_chord: None,
    mode: ActivationMode::Hybrid,
    mute_while_recording: false,
});

#[derive(Clone, Serialize)]
struct SignalPayload {
    action: &'static str,
}

fn emit_signal(app: &AppHandle, action: &'static str) {
    // The frontend logs how it handled the signal; this side proves the chord was
    // seen even when the window is hidden in the tray.
    log::info!("Voice typing shortcut signal: {action}");
    let _ = app.emit_to("main", "voice-typing:signal", SignalPayload { action });
}

/// Wire the shortcut poller. The overlay window is created on first use.
pub fn init(app: &AppHandle) {
    log::info!("Voice typing is waiting for the frontend to push its settings");
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
pub fn voice_typing_suspend_shortcut(suspended: bool) {
    SHORTCUT_SUSPENDED.store(suspended, Ordering::SeqCst);
    log::info!("Voice typing shortcut suspended={suspended}");
}

#[tauri::command]
pub fn voice_typing_configure(
    app: AppHandle,
    enabled: bool,
    shortcut: String,
    mode: String,
    mute_while_recording: Option<bool>,
    hud_position: Option<String>,
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
    let canonical = chord.canonical();
    if let Some(ref pos) = hud_position {
        hud::set_position_mode(&app, Some(pos.as_str()));
    }
    let packed = pack_chord(enabled, Some(&chord));
    PACKED_CHORD.store(packed, Ordering::SeqCst);
    let mute_pref = {
        let mut config = CONFIG.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        config.enabled = enabled;
        config.shortcut = canonical.clone();
        config.parsed_chord = Some(chord);
        config.mode = mode;
        if let Some(mute) = mute_while_recording {
            config.mute_while_recording = mute;
        }
        config.mute_while_recording
    };
    if !enabled || !mute_pref {
        audio_mute::set_recording_mute(false);
    }
    if !enabled {
        hud::hide(&app);
        SESSION_ACTIVE.store(false, Ordering::SeqCst);
    }
    log::info!(
        "Voice typing configured: enabled={enabled} shortcut={canonical} mode={mode:?} mute={mute_pref}"
    );
    Ok(())
}

/// The frontend reports every phase change here so the overlay and the shortcut
/// poller (which cancels on Escape) know what is happening.
///
/// Must stay `async`: synchronous commands run on the main thread, and `hud::show`
/// dispatches window creation through `run_on_main_thread`, which executes inline
/// when already on the main thread. Creating the overlay webview from inside a
/// webview IPC callback deadlocks it mid-creation, which then freezes the whole
/// command queue (the first HUD phase would silently block every later invoke,
/// including the transcript insertion). Async commands run on a worker thread, so
/// the creation closure is queued and handled by the event loop normally.
#[tauri::command]
pub async fn voice_typing_hud_state(
    app: AppHandle,
    state: String,
    label: Option<String>,
    sublabel: Option<String>,
    hud_position: Option<String>,
) {
    // Logged before anything else so every phase reaches the app log even when the
    // overlay window cannot be shown.
    log::info!("Voice typing phase update: {state}");
    if let Some(ref pos) = hud_position {
        hud::set_position_mode(&app, Some(pos.as_str()));
    }
    let mute_enabled = {
        let config = CONFIG.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        config.mute_while_recording
    };
    match state.as_str() {
        // The whole session lifecycle counts as active, including the wait for the
        // microphone or the model and the final transcription: a second press or
        // Escape during any of those must reach the frontend as stop/cancel.
        "preparing" | "recording" | "transcribing" | "fixing" | "refining" | "blocked" => {
            SESSION_ACTIVE.store(true, Ordering::SeqCst);
            if state == "recording" && mute_enabled {
                audio_mute::mute_after_cue();
            } else if state != "recording" {
                audio_mute::set_recording_mute(false);
            }
            hud::show(&app);
        }
        // The transcript is already inserted; the next press starts a new session.
        "done" => {
            SESSION_ACTIVE.store(false, Ordering::SeqCst);
            audio_mute::set_recording_mute(false);
        }
        "idle" => {
            SESSION_ACTIVE.store(false, Ordering::SeqCst);
            audio_mute::set_recording_mute(false);
            hud::hide(&app);
            return;
        }
        _ => return,
    }
    hud::emit_state(&app, &state, label, sublabel);
}

/// Async for the same reason as [`voice_typing_hud_state`]: these updates arrive
/// every few dozen milliseconds while recording and must not do main-thread work.
#[tauri::command]
pub async fn voice_typing_hud_level(app: AppHandle, level: f32) {
    hud::emit_level(&app, level.clamp(0.0, 1.0));
}

/// Build the overlay before the first press needs it.
///
/// The first [`voice_typing_hud_state`] of a session creates a webview, and that
/// cost lands between the shortcut and the recording indicator. Building it hidden
/// while the user is still in Settings moves it out of the press path. Async for
/// the same reason as the state command: creation is queued to the event loop.
#[tauri::command]
pub async fn voice_typing_hud_prewarm(app: AppHandle) {
    hud::prewarm(&app);
}

/// Immediately cancel an active dictation or transcription from the HUD overlay button.
#[tauri::command]
pub async fn voice_typing_cancel(app: AppHandle) {
    log::info!("Voice typing cancelled via HUD button");
    SESSION_ACTIVE.store(false, Ordering::SeqCst);
    audio_mute::set_recording_mute(false);
    hud::hide(&app);
    emit_signal(&app, "cancel");
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

    /// Strict modifier check used on the press edge so extra held modifiers (e.g.
    /// `Ctrl+Shift+Space` when the binding is `Ctrl+Space`) never false-trigger.
    fn matches_modifiers(&self, ctrl: bool, shift: bool, alt: bool, super_key: bool) -> bool {
        self.ctrl == ctrl
            && self.shift == shift
            && self.alt == alt
            && self.super_key == super_key
    }

    /// Relaxed modifier check used while a chord is already held, so brushing an
    /// unrelated key mid-hold does not prematurely release push-to-talk.
    fn required_modifiers_down(&self, ctrl: bool, shift: bool, alt: bool, super_key: bool) -> bool {
        (!self.ctrl || ctrl)
            && (!self.shift || shift)
            && (!self.alt || alt)
            && (!self.super_key || super_key)
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
        "insert" => (0x2D, "Insert"),
        "delete" => (0x2E, "Delete"),
        "home" => (0x24, "Home"),
        "end" => (0x23, "End"),
        "pageup" => (0x21, "PageUp"),
        "pagedown" => (0x22, "PageDown"),
        "capslock" => (0x14, "CapsLock"),
        "up" | "arrowup" => (0x26, "Up"),
        "down" | "arrowdown" => (0x28, "Down"),
        "left" | "arrowleft" => (0x25, "Left"),
        "right" | "arrowright" => (0x27, "Right"),
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
// System audio mute while dictating
// ---------------------------------------------------------------------------

mod audio_mute {
    use std::sync::atomic::{AtomicBool, Ordering};

    static WE_MUTED_SYSTEM_AUDIO: AtomicBool = AtomicBool::new(false);

    /// Generation counter: any later mute/unmute request cancels a pending delayed mute.
    static MUTE_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// How long the start cue needs to finish before the system is muted, so the
    /// interaction sound itself is never silenced.
    const START_CUE_GRACE: std::time::Duration = std::time::Duration::from_millis(450);

    pub fn mute_after_cue() {
        let generation = MUTE_GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
        std::thread::spawn(move || {
            std::thread::sleep(START_CUE_GRACE);
            if MUTE_GENERATION.load(Ordering::SeqCst) == generation {
                set_recording_mute_now(true);
            }
        });
    }

    pub fn set_recording_mute(mute: bool) {
        MUTE_GENERATION.fetch_add(1, Ordering::SeqCst);
        set_recording_mute_now(mute);
    }

    fn set_recording_mute_now(mute: bool) {
        if mute {
            if !WE_MUTED_SYSTEM_AUDIO.load(Ordering::SeqCst) && platform_set_mute(true) {
                WE_MUTED_SYSTEM_AUDIO.store(true, Ordering::SeqCst);
                log::info!("Voice typing muted system audio for recording");
            }
        } else if WE_MUTED_SYSTEM_AUDIO.swap(false, Ordering::SeqCst) {
            let _ = platform_set_mute(false);
            log::info!("Voice typing restored system audio after recording");
        }
    }

    #[cfg(target_os = "windows")]
    fn platform_set_mute(mute: bool) -> bool {
        use std::ffi::c_void;
        use std::ptr::null_mut;
        use windows_sys::core::GUID;
        use windows_sys::Win32::System::Com::{
            CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
        };

        #[repr(C)]
        struct IUnknownVtbl {
            query_interface:
                unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> i32,
            add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
            release: unsafe extern "system" fn(*mut c_void) -> u32,
        }

        #[repr(C)]
        struct IMMDeviceEnumeratorVtbl {
            base: IUnknownVtbl,
            enum_audio_endpoints: *const c_void,
            get_default_audio_endpoint:
                unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void) -> i32,
        }

        #[repr(C)]
        struct IMMDeviceVtbl {
            base: IUnknownVtbl,
            activate: unsafe extern "system" fn(
                *mut c_void,
                *const GUID,
                u32,
                *const c_void,
                *mut *mut c_void,
            ) -> i32,
        }

        #[repr(C)]
        struct IAudioEndpointVolumeVtbl {
            base: IUnknownVtbl,
            register_control_change_notify: *const c_void,
            unregister_control_change_notify: *const c_void,
            get_channel_count: *const c_void,
            set_master_volume_level: *const c_void,
            set_master_volume_level_scalar: *const c_void,
            get_master_volume_level: *const c_void,
            get_master_volume_level_scalar: *const c_void,
            set_channel_volume_level: *const c_void,
            set_channel_volume_level_scalar: *const c_void,
            get_channel_volume_level: *const c_void,
            get_channel_volume_level_scalar: *const c_void,
            set_mute: unsafe extern "system" fn(*mut c_void, i32, *const GUID) -> i32,
            get_mute: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32,
        }

        const CLSID_MMDEVICE_ENUMERATOR: GUID = GUID {
            data1: 0xBCDE0395,
            data2: 0xE52F,
            data3: 0x467C,
            data4: [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E],
        };
        const IID_IMMDEVICE_ENUMERATOR: GUID = GUID {
            data1: 0xA95664D2,
            data2: 0x9614,
            data3: 0x4F35,
            data4: [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6],
        };
        const IID_IAUDIO_ENDPOINT_VOLUME: GUID = GUID {
            data1: 0x5CDF2C82,
            data2: 0x841E,
            data3: 0x4546,
            data4: [0x97, 0x22, 0x0C, 0xF7, 0x40, 0x78, 0x22, 0x9A],
        };

        unsafe {
            let _ = CoInitializeEx(null_mut(), COINIT_MULTITHREADED as u32);
            let mut enumerator: *mut c_void = null_mut();
            if CoCreateInstance(
                &CLSID_MMDEVICE_ENUMERATOR,
                null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_IMMDEVICE_ENUMERATOR,
                &mut enumerator,
            ) < 0
                || enumerator.is_null()
            {
                return false;
            }
            let enum_vtbl = *(enumerator as *const *const IMMDeviceEnumeratorVtbl);
            let mut device: *mut c_void = null_mut();
            // eRender = 0, eMultimedia = 1
            let hr_dev = ((*enum_vtbl).get_default_audio_endpoint)(enumerator, 0, 1, &mut device);
            ((*enum_vtbl).base.release)(enumerator);
            if hr_dev < 0 || device.is_null() {
                return false;
            }

            let dev_vtbl = *(device as *const *const IMMDeviceVtbl);
            let mut endpoint: *mut c_void = null_mut();
            let hr_act = ((*dev_vtbl).activate)(
                device,
                &IID_IAUDIO_ENDPOINT_VOLUME,
                CLSCTX_INPROC_SERVER,
                null_mut(),
                &mut endpoint,
            );
            ((*dev_vtbl).base.release)(device);
            if hr_act < 0 || endpoint.is_null() {
                return false;
            }

            let ep_vtbl = *(endpoint as *const *const IAudioEndpointVolumeVtbl);
            let mut currently_muted: i32 = 0;
            let got = ((*ep_vtbl).get_mute)(endpoint, &mut currently_muted) >= 0;
            let changed = if mute {
                if got && currently_muted == 0 {
                    ((*ep_vtbl).set_mute)(endpoint, 1, null_mut()) >= 0
                } else {
                    false
                }
            } else {
                ((*ep_vtbl).set_mute)(endpoint, 0, null_mut()) >= 0
            };
            ((*ep_vtbl).base.release)(endpoint);
            changed
        }
    }

    #[cfg(target_os = "macos")]
    fn platform_set_mute(mute: bool) -> bool {
        let script = if mute {
            "if output muted of (get volume settings) is false then\nset volume with output muted\nreturn \"muted\"\nend if"
        } else {
            "set volume without output muted\nreturn \"unmuted\""
        };
        std::process::Command::new("osascript")
            .args(["-e", script])
            .output()
            .ok()
            .map(|out| {
                let s = String::from_utf8_lossy(&out.stdout);
                if mute {
                    s.trim() == "muted"
                } else {
                    out.status.success()
                }
            })
            .unwrap_or(false)
    }

    #[cfg(target_os = "linux")]
    fn platform_set_mute(mute: bool) -> bool {
        let arg = if mute { "1" } else { "0" };
        std::process::Command::new("wpctl")
            .args(["set-mute", "@DEFAULT_AUDIO_SINK@", arg])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    fn platform_set_mute(_mute: bool) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Windows: low-level keyboard hook + shortcut poller
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
mod shortcut_poller {
    use super::{
        edge_signal, emit_signal, parse_chord, Chord, Edge, CONFIG, PACKED_CHORD, SESSION_ACTIVE,
        SHORTCUT_SUSPENDED,
    };
    use std::ptr::null_mut;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    use tauri::AppHandle;
    use windows_sys::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_CONTROL, VK_ESCAPE, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, DispatchMessageW, GetMessageW, SetWindowsHookExW, TranslateMessage,
        KBDLLHOOKSTRUCT, LLKHF_INJECTED, MSG, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN,
        WM_SYSKEYUP,
    };

    const POLL_INTERVAL: Duration = Duration::from_millis(12);
    const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(120);

    /// Tracks whether the low-level keyboard hook swallowed the trigger keydown,
    /// because a swallowed `WM_KEYDOWN` does not always update `GetAsyncKeyState`.
    static HOOK_MAIN_KEY_DOWN: AtomicBool = AtomicBool::new(false);
    /// Latched on trigger keydown and consumed by the poller so a sub-12ms tap
    /// between poll ticks is never lost.
    static HOOK_MAIN_KEY_PRESSED: AtomicBool = AtomicBool::new(false);
    /// Tracks whether the low-level keyboard hook intercepted `Escape` during an
    /// active session so `Escape` cancels dictation without closing focused dialogs.
    static HOOK_ESCAPE_DOWN: AtomicBool = AtomicBool::new(false);

    /// The high bit means "currently down". The low bit ("pressed since the last
    /// call") belongs to the foreground thread and is useless in a background loop.
    fn down(vk: u16) -> bool {
        unsafe { (GetAsyncKeyState(vk as i32) as u16 & 0x8000) != 0 }
    }

    fn current_modifiers() -> (bool, bool, bool, bool) {
        (
            down(VK_CONTROL),
            down(VK_SHIFT),
            down(VK_MENU),
            down(VK_LWIN) || down(VK_RWIN),
        )
    }

    impl Chord {
        /// Strict check used when starting a chord press (`!chord_held`).
        fn is_fresh_press(&self) -> bool {
            if HOOK_MAIN_KEY_PRESSED.swap(false, Ordering::SeqCst) {
                return true;
            }
            let (ctrl, shift, alt, super_key) = current_modifiers();
            let key_down = down(self.vk) || HOOK_MAIN_KEY_DOWN.load(Ordering::SeqCst);
            self.matches_modifiers(ctrl, shift, alt, super_key) && key_down
        }

        /// Hold check used while `chord_held` is already true.
        fn is_held(&self) -> bool {
            let (ctrl, shift, alt, super_key) = current_modifiers();
            let mods_ok = self.required_modifiers_down(ctrl, shift, alt, super_key);
            if !mods_ok {
                // If the required modifiers were released (e.g. across a UAC prompt),
                // clear any latched hook key state so it can never stick.
                HOOK_MAIN_KEY_DOWN.store(false, Ordering::SeqCst);
                return false;
            }
            down(self.vk) || HOOK_MAIN_KEY_DOWN.load(Ordering::SeqCst)
        }
    }

    unsafe extern "system" fn low_level_keyboard_proc(
        code: i32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if code >= 0 && lparam != 0 && !SHORTCUT_SUSPENDED.load(Ordering::SeqCst) {
            let kb = &*(lparam as *const KBDLLHOOKSTRUCT);
            let injected = (kb.flags & LLKHF_INJECTED) != 0;
            if !injected {
                let msg = wparam as u32;
                let is_down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
                let is_up = msg == WM_KEYUP || msg == WM_SYSKEYUP;
                let vk = kb.vkCode as u16;

                // Swallow Escape during an active dictation session so cancelling
                // never closes a modal or clears an input in the target application.
                if vk == VK_ESCAPE && SESSION_ACTIVE.load(Ordering::SeqCst) {
                    if is_down {
                        HOOK_ESCAPE_DOWN.store(true, Ordering::SeqCst);
                        return 1;
                    }
                    if is_up {
                        return 1;
                    }
                }

                let packed = PACKED_CHORD.load(Ordering::Relaxed);
                if (packed & (1 << 20)) != 0 {
                    let chord_vk = (packed & 0xFFFF) as u16;
                    if vk == chord_vk {
                        if is_down {
                            let (ctrl, shift, alt, super_key) = current_modifiers();
                            let want_ctrl = (packed & (1 << 16)) != 0;
                            let want_shift = (packed & (1 << 17)) != 0;
                            let want_alt = (packed & (1 << 18)) != 0;
                            let want_super = (packed & (1 << 19)) != 0;
                            let matches = want_ctrl == ctrl
                                && want_shift == shift
                                && want_alt == alt
                                && want_super == super_key;
                            if matches || HOOK_MAIN_KEY_DOWN.load(Ordering::SeqCst) {
                                if !HOOK_MAIN_KEY_DOWN.swap(true, Ordering::SeqCst) {
                                    HOOK_MAIN_KEY_PRESSED.store(true, Ordering::SeqCst);
                                }
                                return 1;
                            }
                        } else if is_up && HOOK_MAIN_KEY_DOWN.swap(false, Ordering::SeqCst) {
                            return 1;
                        }
                    }
                }
            }
        }
        CallNextHookEx(null_mut(), code, wparam, lparam)
    }

    fn run_hook_loop() {
        unsafe {
            let hook = SetWindowsHookExW(WH_KEYBOARD_LL, Some(low_level_keyboard_proc), null_mut(), 0);
            if hook.is_null() {
                log::warn!("Voice typing low-level keyboard hook could not be installed; using poller fallback");
                return;
            }
            log::info!("Voice typing low-level keyboard hook installed");
            let mut msg = MSG {
                hwnd: null_mut(),
                message: 0,
                wParam: 0,
                lParam: 0,
                time: 0,
                pt: windows_sys::Win32::Foundation::POINT { x: 0, y: 0 },
            };
            while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    pub fn spawn(app: AppHandle) {
        let _ = std::thread::Builder::new()
            .name("voice-typing-llhook".to_string())
            .spawn(run_hook_loop);

        let spawned = std::thread::Builder::new()
            .name("voice-typing-shortcut".to_string())
            .spawn(move || poll(app));
        match spawned {
            Ok(_) => log::info!("Voice typing shortcut poller started"),
            Err(error) => {
                log::warn!("Voice typing shortcut thread could not start: {error}")
            }
        }
    }

    fn poll(app: AppHandle) {
        let mut chord: Option<Chord> = None;
        let mut chord_source = String::new();
        let mut armed_log = String::new();
        let mut chord_held = false;
        let mut pressed_at: Option<Instant> = None;
        let mut press_started_session = false;
        let mut escape_held = false;

        loop {
            let suspended = SHORTCUT_SUSPENDED.load(Ordering::SeqCst);
            let (enabled, mode, source) = {
                let config = CONFIG.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                (config.enabled, config.mode, config.shortcut.clone())
            };
            if !enabled || suspended {
                // Off or suspended means off: sleep longer to consume zero CPU and
                // clear any held state.
                chord = None;
                chord_source.clear();
                chord_held = false;
                pressed_at = None;
                HOOK_MAIN_KEY_DOWN.store(false, Ordering::SeqCst);
                HOOK_MAIN_KEY_PRESSED.store(false, Ordering::SeqCst);
                HOOK_ESCAPE_DOWN.store(false, Ordering::SeqCst);
                std::thread::sleep(IDLE_POLL_INTERVAL);
                continue;
            }

            std::thread::sleep(POLL_INTERVAL);

            if source != chord_source {
                chord_source = source;
                chord = if chord_source.is_empty() {
                    None
                } else {
                    match parse_chord(&chord_source) {
                        Ok(parsed) => {
                            if armed_log != chord_source {
                                log::info!(
                                    "Voice typing shortcut armed: {} ({mode:?}, enabled={enabled})",
                                    parsed.canonical()
                                );
                                armed_log = chord_source.clone();
                            }
                            Some(parsed)
                        }
                        Err(error) => {
                            log::warn!("Voice typing shortcut could not be armed: {error}");
                            None
                        }
                    }
                };
                chord_held = false;
                pressed_at = None;
                HOOK_MAIN_KEY_DOWN.store(false, Ordering::SeqCst);
                HOOK_MAIN_KEY_PRESSED.store(false, Ordering::SeqCst);
            }
            let Some(current) = chord.as_ref() else {
                watch_escape(&app, &mut escape_held);
                continue;
            };

            let pressed = if chord_held {
                current.is_held()
            } else {
                current.is_fresh_press()
            };
            let now = Instant::now();
            let press_edge = pressed && !chord_held;
            let release_edge = !pressed && chord_held;
            if press_edge {
                chord_held = true;
                pressed_at = Some(now);
                press_started_session = !SESSION_ACTIVE.load(Ordering::SeqCst);
            }

            if press_edge {
                if let Some(action) = edge_signal(mode, Edge::Press, press_started_session, Duration::ZERO)
                {
                    if action == "start" {
                        // Mark session active immediately so a fast release before JS
                        // finishes permission preflight still produces a "stop" signal.
                        SESSION_ACTIVE.store(true, Ordering::SeqCst);
                    }
                    emit_signal(&app, action);
                }
            } else if release_edge {
                chord_held = false;
                let held_for = pressed_at
                    .map(|start| now.saturating_duration_since(start))
                    .unwrap_or_default();
                pressed_at = None;
                if let Some(action) =
                    edge_signal(mode, Edge::Release, press_started_session, held_for)
                {
                    emit_signal(&app, action);
                }
            }

            // Escape ends an in-flight capture, matching the in-app shortcut.
            watch_escape(&app, &mut escape_held);
        }
    }

    /// Cancel needs the key to go from up to down between polls; holding it does
    /// not repeat. Only an in-flight session listens, so Escape elsewhere in the
    /// system stays untouched.
    fn watch_escape(app: &AppHandle, escape_held: &mut bool) {
        let held = down(VK_ESCAPE) || HOOK_ESCAPE_DOWN.swap(false, Ordering::SeqCst);
        if held && !*escape_held && SESSION_ACTIVE.swap(false, Ordering::SeqCst) {
            super::audio_mute::set_recording_mute(false);
            super::hud::hide(app);
            emit_signal(app, "cancel");
        }
        *escape_held = held;
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
    fn strict_modifier_matching_rejects_extra_held_modifiers() {
        let ctrl_space = parse_chord("Ctrl+Space").expect("chord");
        assert!(ctrl_space.matches_modifiers(true, false, false, false));
        // Holding Shift in addition to Ctrl+Space must NOT trigger a Ctrl+Space binding.
        assert!(!ctrl_space.matches_modifiers(true, true, false, false));
        assert!(!ctrl_space.matches_modifiers(true, false, true, false));
        assert!(!ctrl_space.matches_modifiers(false, false, false, false));
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
        assert_eq!(parse_chord("Ctrl+Insert").expect("chord").canonical(), "Ctrl+Insert");
    }

    #[test]
    fn rejects_bare_keys_escape_and_unknown() {
        assert!(parse_chord("Space").is_err());
        assert!(parse_chord("Ctrl+Escape").is_err());
        assert!(parse_chord("Ctrl+Nope").is_err());
        assert!(parse_chord("Ctrl+A+B").is_err());
        assert!(parse_chord("Ctrl+").is_err());
    }

    #[test]
    fn hybrid_taps_keep_the_session_running() {
        let tap = Duration::from_millis(120);
        let hold = Duration::from_millis(900);
        // Fresh press opens the session; a quick release leaves it on.
        assert_eq!(
            edge_signal(ActivationMode::Hybrid, Edge::Press, true, Duration::ZERO),
            Some("start")
        );
        assert_eq!(edge_signal(ActivationMode::Hybrid, Edge::Release, true, tap), None);
        // A long release was push-to-talk and stops.
        assert_eq!(
            edge_signal(ActivationMode::Hybrid, Edge::Release, true, hold),
            Some("stop")
        );
        // Pressing while a session is already running only acts on release.
        assert_eq!(
            edge_signal(ActivationMode::Hybrid, Edge::Press, false, Duration::ZERO),
            None
        );
        assert_eq!(
            edge_signal(ActivationMode::Hybrid, Edge::Release, false, tap),
            Some("stop")
        );
    }

    #[test]
    fn hold_and_toggle_edge_semantics() {
        // Push-to-talk always stops on release.
        assert_eq!(
            edge_signal(ActivationMode::Hold, Edge::Press, true, Duration::ZERO),
            Some("start")
        );
        assert_eq!(
            edge_signal(ActivationMode::Hold, Edge::Release, true, Duration::from_millis(10)),
            Some("stop")
        );
        // Toggle flips on every press; releases are silent.
        assert_eq!(
            edge_signal(ActivationMode::Toggle, Edge::Press, true, Duration::ZERO),
            Some("start")
        );
        assert_eq!(
            edge_signal(ActivationMode::Toggle, Edge::Press, false, Duration::ZERO),
            Some("stop")
        );
        assert_eq!(
            edge_signal(ActivationMode::Toggle, Edge::Release, true, Duration::from_millis(500)),
            None
        );
        // A press while already recording never restarts in any mode.
        assert_eq!(
            edge_signal(ActivationMode::Hold, Edge::Press, false, Duration::ZERO),
            None
        );
    }
}
