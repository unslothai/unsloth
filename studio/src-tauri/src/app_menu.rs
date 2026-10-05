//! The macOS File, View, Go and Help menus. Their actions run in the renderer, which enables only
//! the ones it can handle right now, so they stay disabled through install, login and startup.
//! macOS searches every item from Help > Search, so Go lists each workspace and Settings page.

#[cfg(target_os = "macos")]
use tauri::{menu::MenuItem, Manager};

/// Event the renderer listens for; the payload is the action name.
#[cfg(target_os = "macos")]
pub const APP_MENU_ACTION_EVENT: &str = "app-menu-action";

/// A menu row: an action, a separator or a submenu of rows.
#[cfg(target_os = "macos")]
enum Row {
    /// (action sent to the renderer, label, accelerator; "" for none)
    Action(&'static str, &'static str, &'static str),
    Separator,
    Submenu(&'static str, &'static [Row]),
}

#[cfg(target_os = "macos")]
const FILE_ROWS: &[Row] = &[
    Row::Action("new-chat", "New Chat", "CmdOrCtrl+N"),
    Row::Action(
        "new-temporary-chat",
        "New Temporary Chat",
        "CmdOrCtrl+Shift+N",
    ),
    Row::Separator,
    Row::Action("open-folder", "Open Folder\u{2026}", "CmdOrCtrl+O"),
    Row::Separator,
];

#[cfg(target_os = "macos")]
const VIEW_ROWS: &[Row] = &[
    Row::Action("toggle-sidebar", "Toggle Sidebar", "CmdOrCtrl+B"),
    Row::Separator,
    Row::Action("find", "Find", "CmdOrCtrl+F"),
    Row::Separator,
    Row::Action("previous-chat", "Previous Chat", "CmdOrCtrl+Shift+["),
    Row::Action("next-chat", "Next Chat", "CmdOrCtrl+Shift+]"),
    Row::Action("back", "Back", "CmdOrCtrl+["),
    Row::Action("forward", "Forward", "CmdOrCtrl+]"),
    Row::Separator,
    Row::Action("zoom-in", "Zoom In", "CmdOrCtrl+="),
    Row::Action("zoom-out", "Zoom Out", "CmdOrCtrl+-"),
    Row::Action("actual-size", "Actual Size", "CmdOrCtrl+0"),
    Row::Separator,
];

#[cfg(target_os = "macos")]
const GO_ROWS: &[Row] = &[
    Row::Action("go-chat", "Chat", "Ctrl+1"),
    Row::Action("go-projects", "Projects", "Ctrl+2"),
    Row::Action("go-library", "Library", ""),
    Row::Action("go-hub", "Model Hub", "Ctrl+3"),
    Row::Action("go-train", "Train", "Ctrl+4"),
    Row::Action("go-recipes", "Recipes", "Ctrl+5"),
    Row::Action("go-images", "Images", "Ctrl+6"),
    Row::Action("go-video", "Video", "Ctrl+7"),
    Row::Action("go-audio", "Audio", "Ctrl+8"),
    Row::Action("go-export", "Export", "Ctrl+9"),
    Row::Separator,
    Row::Submenu("Settings", SETTINGS_ROWS),
];

/// The Settings pages, in the order and with the names of the dialog's tabs.
#[cfg(target_os = "macos")]
const SETTINGS_ROWS: &[Row] = &[
    Row::Action("settings-general", "General", ""),
    Row::Action("settings-profile", "Profile", ""),
    Row::Action("settings-appearance", "Appearance", ""),
    Row::Action("settings-resources", "System", ""),
    Row::Action("settings-chat", "Chat", ""),
    Row::Action("settings-api-keys", "API", ""),
    Row::Action("settings-remote-lan", "Remote & LAN", ""),
    Row::Action("settings-connections", "Connections", ""),
    Row::Action("settings-accounts", "Accounts", ""),
    Row::Action("settings-agents", "Agents", ""),
    Row::Action("settings-voice", "Voice", ""),
    Row::Action("settings-library", "Library", ""),
    Row::Action("settings-data", "Data", ""),
    Row::Action("settings-keyboard-shortcuts", "Shortcuts", ""),
    Row::Action("settings-debugging", "Logs", ""),
    Row::Action("settings-about", "About", ""),
];

#[cfg(target_os = "macos")]
const HELP_ROWS: &[Row] = &[
    Row::Action("help-documentation", "Documentation", ""),
    Row::Action(
        "help-keyboard-shortcuts",
        "Keyboard Shortcuts",
        "CmdOrCtrl+/",
    ),
    Row::Action("help-whats-new", "What's New", ""),
    Row::Separator,
    Row::Action("help-troubleshooting", "Troubleshooting", ""),
    Row::Action("help-system-status", "System Status", ""),
    Row::Action("help-send-feedback", "Send Feedback", ""),
];

/// Menu ids are the action names, prefixed so they cannot collide with other menu ids.
#[cfg(target_os = "macos")]
const ID_PREFIX: &str = "app-menu:";

#[cfg(target_os = "macos")]
struct ActionItem {
    action: &'static str,
    item: MenuItem<tauri::Wry>,
    submenu: tauri::menu::Submenu<tauri::Wry>,
    /// The chord it shows, which muda cannot report back.
    accelerator: Option<String>,
}

/// The action items, replaced in place when a chord is cleared.
#[cfg(target_os = "macos")]
pub struct AppMenuActions(std::sync::Mutex<Vec<ActionItem>>);

/// Put the Unsloth rows at the top of the File, View and Help menus, keeping each menu's native
/// items (Close, Enter Full Screen) below them, and add Go after View. Help keeps its role, so
/// macOS still adds Search.
#[cfg(target_os = "macos")]
pub fn setup_app_menus(
    app: &tauri::App,
    menu: &tauri::menu::Menu<tauri::Wry>,
) -> Result<(), Box<dyn std::error::Error>> {
    use tauri::menu::{PredefinedMenuItem, SubmenuBuilder};

    let find = |title: &str| -> tauri::Result<Option<(usize, tauri::menu::Submenu<tauri::Wry>)>> {
        Ok(menu
            .items()?
            .into_iter()
            .enumerate()
            .find_map(|(index, item)| {
                let submenu = item.as_submenu()?.clone();
                (submenu.text().ok()? == title).then_some((index, submenu))
            }))
    };
    if let Some((view, _)) = find("View")? {
        menu.insert(&SubmenuBuilder::new(app, "Go").build()?, view + 1)?;
    }

    let mut actions = Vec::new();
    for (title, rows) in [
        ("File", FILE_ROWS),
        ("View", VIEW_ROWS),
        ("Go", GO_ROWS),
        ("Help", HELP_ROWS),
    ] {
        let Some((_, submenu)) = find(title)? else {
            continue;
        };
        let native = submenu.items()?;
        for item in &native {
            submenu.remove(item)?;
        }
        append_rows(app, &submenu, rows, &mut actions)?;
        if title == "File" {
            // The native close, so Cmd+W still goes through the window's close handling.
            submenu.append(&PredefinedMenuItem::close_window(app, Some("Close"))?)?;
        } else {
            for item in &native {
                submenu.append(item)?;
            }
        }
    }
    app.manage(AppMenuActions(std::sync::Mutex::new(actions)));
    Ok(())
}

#[cfg(target_os = "macos")]
fn append_rows(
    app: &tauri::App,
    submenu: &tauri::menu::Submenu<tauri::Wry>,
    rows: &[Row],
    actions: &mut Vec<ActionItem>,
) -> tauri::Result<()> {
    use tauri::menu::{MenuItemBuilder, PredefinedMenuItem, SubmenuBuilder};

    for row in rows {
        match row {
            Row::Action(action, label, accelerator) => {
                let mut builder =
                    MenuItemBuilder::with_id(format!("{ID_PREFIX}{action}"), *label).enabled(false);
                if !accelerator.is_empty() {
                    builder = builder.accelerator(*accelerator);
                }
                let item = builder.build(app)?;
                submenu.append(&item)?;
                actions.push(ActionItem {
                    action,
                    item,
                    submenu: submenu.clone(),
                    accelerator: (!accelerator.is_empty()).then(|| accelerator.to_string()),
                });
            }
            Row::Separator => submenu.append(&PredefinedMenuItem::separator(app)?)?,
            Row::Submenu(label, rows) => {
                let child = SubmenuBuilder::new(app, *label).build()?;
                append_rows(app, &child, rows, actions)?;
                submenu.append(&child)?;
            }
        }
    }
    Ok(())
}

/// Forward a menu click to the renderer. Ignores ids this module does not own.
#[cfg(target_os = "macos")]
pub fn handle_menu_event(app: &tauri::AppHandle, id: &str) {
    use tauri::Emitter;

    let Some(action) = id.strip_prefix(ID_PREFIX) else {
        return;
    };
    // The window may be hidden after Close; bring it back for the action to show.
    crate::show_main_window(app);
    if let Err(error) = app.emit_to("main", APP_MENU_ACTION_EVENT, action) {
        log::warn!("Could not send menu action {action}: {error}");
    }
}

/// Enable exactly the listed actions, and show the chord each is bound to now (None for none).
/// A no-op where there are no app menus.
#[tauri::command]
pub fn set_app_menu_actions(
    app: tauri::AppHandle,
    enabled: Vec<String>,
    accelerators: std::collections::HashMap<String, Option<String>>,
) {
    #[cfg(target_os = "macos")]
    if let Some(actions) = app.try_state::<AppMenuActions>() {
        let Ok(mut actions) = actions.0.lock() else {
            return;
        };
        for entry in actions.iter_mut() {
            if let Some(wanted) = accelerators.get(entry.action) {
                if *wanted != entry.accelerator {
                    if let Err(error) = set_item_accelerator(&app, entry, wanted.as_deref()) {
                        log::warn!("Could not update the {} shortcut: {error}", entry.action);
                    }
                }
            }
            let _ = entry
                .item
                .set_enabled(enabled.iter().any(|name| name == entry.action));
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (app, enabled, accelerators);
}

/// muda's `set_accelerator(None)` leaves the native key equivalent in place, so clearing one
/// swaps in a fresh item without it. A chord muda cannot parse is cleared too.
#[cfg(target_os = "macos")]
fn set_item_accelerator(
    app: &tauri::AppHandle,
    entry: &mut ActionItem,
    accelerator: Option<&str>,
) -> tauri::Result<()> {
    use tauri::menu::MenuItemBuilder;

    if let Some(accelerator) = accelerator {
        if entry.item.set_accelerator(Some(accelerator)).is_ok() {
            entry.accelerator = Some(accelerator.to_string());
            return Ok(());
        }
    }
    if entry.accelerator.is_none() {
        return Ok(());
    }
    let position = entry
        .submenu
        .items()?
        .iter()
        .position(|item| item.id() == entry.item.id())
        .unwrap_or(0);
    let fresh = MenuItemBuilder::with_id(entry.item.id().clone(), entry.item.text()?)
        .enabled(entry.item.is_enabled()?)
        .build(app)?;
    entry.submenu.remove(&entry.item)?;
    entry.submenu.insert(&fresh, position)?;
    entry.item = fresh;
    entry.accelerator = None;
    Ok(())
}
