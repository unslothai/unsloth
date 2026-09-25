//! The macOS File and View menus. Their actions run in the renderer, which enables only the ones
//! it can handle right now, so they stay disabled through install, login and startup.

#[cfg(target_os = "macos")]
use tauri::{menu::MenuItem, Manager};

/// Event the renderer listens for; the payload is the action name.
#[cfg(target_os = "macos")]
pub const APP_MENU_ACTION_EVENT: &str = "app-menu-action";

/// A menu row: an action or a separator.
#[cfg(target_os = "macos")]
enum Row {
    /// (action sent to the renderer, label, accelerator)
    Action(&'static str, &'static str, &'static str),
    Separator,
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

/// Put the Unsloth rows at the top of the File and View menus, keeping each menu's native items
/// (Close, Enter Full Screen) below them.
#[cfg(target_os = "macos")]
pub fn setup_app_menus(
    app: &tauri::App,
    menu: &tauri::menu::Menu<tauri::Wry>,
) -> Result<(), Box<dyn std::error::Error>> {
    use tauri::menu::{MenuItemBuilder, PredefinedMenuItem};

    let mut actions = Vec::new();
    for (title, rows) in [("File", FILE_ROWS), ("View", VIEW_ROWS)] {
        let Some(submenu) = menu.items()?.into_iter().find_map(|item| {
            let submenu = item.as_submenu()?.clone();
            (submenu.text().ok()? == title).then_some(submenu)
        }) else {
            continue;
        };
        let native = submenu.items()?;
        for item in &native {
            submenu.remove(item)?;
        }
        for row in rows {
            match row {
                Row::Action(action, label, accelerator) => {
                    let item = MenuItemBuilder::with_id(format!("{ID_PREFIX}{action}"), *label)
                        .accelerator(*accelerator)
                        .enabled(false)
                        .build(app)?;
                    submenu.append(&item)?;
                    actions.push(ActionItem {
                        action,
                        item,
                        submenu: submenu.clone(),
                        accelerator: Some(accelerator.to_string()),
                    });
                }
                Row::Separator => submenu.append(&PredefinedMenuItem::separator(app)?)?,
            }
        }
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
