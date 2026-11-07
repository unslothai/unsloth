//! Cross-platform global chord listener for macOS and Linux.
//!
//! Windows uses the low-level keyboard hook in `voice_typing.rs`. The other
//! desktops use `rdev`, which reports press and release events so hold, toggle
//! and hybrid keep the same semantics. On macOS the process needs Accessibility
//! (Input Monitoring) permission; on Linux this uses X11, so Wayland sessions
//! depend on the compositor's XWayland compatibility.

use std::time::{Duration, Instant};

use rdev::{listen, Event, EventType, Key};
use tauri::AppHandle;

use super::{
    audio_mute, edge_signal, emit_signal, hud, main_window_focused, Edge, CONFIG, SESSION_ACTIVE,
    SHORTCUT_SUSPENDED,
};

#[derive(Clone, Copy)]
struct Chord {
    key: Key,
    ctrl: bool,
    shift: bool,
    alt: bool,
    super_key: bool,
}

fn key_from_name(raw: &str) -> Option<Key> {
    let name = raw.trim().to_ascii_lowercase();
    if name.len() == 1 {
        let character = name.chars().next()?;
        return match character {
            'a'..='z' => Some(match character {
                'a' => Key::KeyA,
                'b' => Key::KeyB,
                'c' => Key::KeyC,
                'd' => Key::KeyD,
                'e' => Key::KeyE,
                'f' => Key::KeyF,
                'g' => Key::KeyG,
                'h' => Key::KeyH,
                'i' => Key::KeyI,
                'j' => Key::KeyJ,
                'k' => Key::KeyK,
                'l' => Key::KeyL,
                'm' => Key::KeyM,
                'n' => Key::KeyN,
                'o' => Key::KeyO,
                'p' => Key::KeyP,
                'q' => Key::KeyQ,
                'r' => Key::KeyR,
                's' => Key::KeyS,
                't' => Key::KeyT,
                'u' => Key::KeyU,
                'v' => Key::KeyV,
                'w' => Key::KeyW,
                'x' => Key::KeyX,
                'y' => Key::KeyY,
                _ => Key::KeyZ,
            }),
            '0'..='9' => Some(match character {
                '0' => Key::Num0,
                '1' => Key::Num1,
                '2' => Key::Num2,
                '3' => Key::Num3,
                '4' => Key::Num4,
                '5' => Key::Num5,
                '6' => Key::Num6,
                '7' => Key::Num7,
                '8' => Key::Num8,
                _ => Key::Num9,
            }),
            _ => None,
        };
    }
    Some(match name.as_str() {
        "space" => Key::Space,
        "enter" | "return" => Key::Return,
        "tab" => Key::Tab,
        "backspace" => Key::Backspace,
        "insert" => Key::Insert,
        "delete" => Key::Delete,
        "home" => Key::Home,
        "end" => Key::End,
        "pageup" => Key::PageUp,
        "pagedown" => Key::PageDown,
        "capslock" => Key::CapsLock,
        "up" | "arrowup" => Key::UpArrow,
        "down" | "arrowdown" => Key::DownArrow,
        "left" | "arrowleft" => Key::LeftArrow,
        "right" | "arrowright" => Key::RightArrow,
        "-" | "minus" => Key::Minus,
        "=" | "equal" | "equals" => Key::Equal,
        "[" | "bracketleft" => Key::LeftBracket,
        "]" | "bracketright" => Key::RightBracket,
        ";" | "semicolon" => Key::SemiColon,
        "'" | "quote" => Key::Quote,
        "," | "comma" => Key::Comma,
        "." | "period" => Key::Dot,
        "/" | "slash" => Key::Slash,
        "\\" | "backslash" => Key::BackSlash,
        "`" | "backquote" => Key::BackQuote,
        _ => {
            let function = name
                .strip_prefix('f')
                .and_then(|rest| rest.parse::<u8>().ok());
            return match function {
                Some(number @ 1..=12) => Some(match number {
                    1 => Key::F1,
                    2 => Key::F2,
                    3 => Key::F3,
                    4 => Key::F4,
                    5 => Key::F5,
                    6 => Key::F6,
                    7 => Key::F7,
                    8 => Key::F8,
                    9 => Key::F9,
                    10 => Key::F10,
                    11 => Key::F11,
                    _ => Key::F12,
                }),
                // rdev has no F13-F24; Windows still supports them.
                _ => None,
            };
        }
    })
}

fn parse_chord(raw: &str) -> Option<Chord> {
    let mut chord = Chord {
        key: Key::Space,
        ctrl: false,
        shift: false,
        alt: false,
        super_key: false,
    };
    let mut key = None;
    for token in raw.split('+') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        match token.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => chord.ctrl = true,
            "shift" => chord.shift = true,
            "alt" | "option" => chord.alt = true,
            "super" | "meta" | "win" | "cmd" | "command" => chord.super_key = true,
            _ => {
                if key.is_some() {
                    return None;
                }
                key = key_from_name(token);
            }
        }
    }
    chord.key = key?;
    Some(chord)
}

fn modifier_key(key: Key) -> Option<u8> {
    Some(match key {
        Key::ControlLeft | Key::ControlRight => 0,
        Key::ShiftLeft | Key::ShiftRight => 1,
        Key::Alt | Key::AltGr => 2,
        Key::MetaLeft | Key::MetaRight => 3,
        _ => return None,
    })
}

fn matches_chord(chord: Chord, ctrl: bool, shift: bool, alt: bool, super_key: bool) -> bool {
    chord.ctrl == ctrl
        && chord.shift == shift
        && chord.alt == alt
        && chord.super_key == super_key
}

pub fn spawn(app: AppHandle) {
    let spawned = std::thread::Builder::new()
        .name("voice-typing-shortcut".to_string())
        .spawn(move || {
            let mut ctrl = false;
            let mut shift = false;
            let mut alt = false;
            let mut super_key = false;
            let mut chord: Option<Chord> = None;
            let mut chord_source = String::new();
            let mut main_down = false;
            let mut pressed_at: Option<Instant> = None;
            let mut press_started_session = false;
            let mut press_started_in_app = false;
            let mut press_cancelled_system = false;
            let mut escape_down = false;

            let result = listen(move |event: Event| {
                let (enabled, mode, source) = {
                    let config = CONFIG
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    (config.enabled, config.mode, config.shortcut.clone())
                };
                let suspended = SHORTCUT_SUSPENDED.load(std::sync::atomic::Ordering::SeqCst);
                if !enabled || suspended {
                    chord_source.clear();
                    chord = None;
                    main_down = false;
                    pressed_at = None;
                    press_started_in_app = false;
                    press_cancelled_system = false;
                    escape_down = false;
                    return;
                }
                if source != chord_source {
                    chord_source = source;
                    chord = parse_chord(&chord_source);
                    if chord.is_none() {
                        log::warn!(
                            "Voice typing shortcut cannot be armed on this platform: {chord_source}"
                        );
                    }
                    main_down = false;
                    pressed_at = None;
                    press_started_in_app = false;
                    press_cancelled_system = false;
                }

                match event.event_type {
                    EventType::KeyPress(key) => {
                        if let Some(bit) = modifier_key(key) {
                            match bit {
                                0 => ctrl = true,
                                1 => shift = true,
                                2 => alt = true,
                                _ => super_key = true,
                            }
                            return;
                        }
                        if key == Key::Escape {
                            if !escape_down
                                && SESSION_ACTIVE
                                    .swap(false, std::sync::atomic::Ordering::SeqCst)
                            {
                                audio_mute::set_recording_mute(false);
                                hud::hide(&app);
                                emit_signal(&app, "cancel", "system", 0);
                            }
                            escape_down = true;
                            return;
                        }
                        let Some(current) = chord else { return };
                        if key != current.key
                            || main_down
                            || !matches_chord(current, ctrl, shift, alt, super_key)
                        {
                            return;
                        }
                        main_down = true;
                        pressed_at = Some(Instant::now());
                        press_started_session = !SESSION_ACTIVE
                            .load(std::sync::atomic::Ordering::SeqCst);
                        press_started_in_app = main_window_focused(&app);
                        if press_started_in_app {
                            if SESSION_ACTIVE
                                .swap(false, std::sync::atomic::Ordering::SeqCst)
                            {
                                audio_mute::set_recording_mute(false);
                                hud::hide(&app);
                                emit_signal(&app, "cancel", "system", 0);
                                press_started_in_app = false;
                                press_cancelled_system = true;
                            } else {
                                emit_signal(&app, "press", "app", 0);
                            }
                        } else if let Some(action) =
                            edge_signal(mode, Edge::Press, press_started_session, Duration::ZERO)
                        {
                            if action == "start" {
                                SESSION_ACTIVE.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                            emit_signal(&app, action, "system", 0);
                        }
                    }
                    EventType::KeyRelease(key) => {
                        if let Some(bit) = modifier_key(key) {
                            match bit {
                                0 => ctrl = false,
                                1 => shift = false,
                                2 => alt = false,
                                _ => super_key = false,
                            }
                            return;
                        }
                        if key == Key::Escape {
                            escape_down = false;
                            return;
                        }
                        let Some(current) = chord else { return };
                        if key != current.key || !main_down {
                            return;
                        }
                        main_down = false;
                        let held_for = pressed_at
                            .map(|start| start.elapsed())
                            .unwrap_or_default();
                        pressed_at = None;
                        if press_cancelled_system {
                            press_cancelled_system = false;
                        } else if press_started_in_app {
                            press_started_in_app = false;
                            let held_ms =
                                u64::try_from(held_for.as_millis()).unwrap_or(u64::MAX);
                            emit_signal(&app, "release", "app", held_ms);
                        } else if let Some(action) =
                            edge_signal(mode, Edge::Release, press_started_session, held_for)
                        {
                            emit_signal(&app, action, "system", 0);
                        }
                    }
                    _ => {}
                }
            });

            if let Err(error) = result {
                log::warn!("Voice typing shortcut listener stopped: {error:?}");
            }
        });
    match spawned {
        Ok(_) => log::info!("Voice typing cross-platform shortcut listener started"),
        Err(error) => log::warn!("Voice typing shortcut thread could not start: {error}"),
    }
}
