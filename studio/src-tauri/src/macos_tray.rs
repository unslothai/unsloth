use std::{
    cell::{Cell, RefCell},
    ffi::c_void,
    ptr::null_mut,
};

use objc2::{
    define_class, msg_send, rc::Retained, runtime::AnyObject, DefinedClass, MainThreadMarker,
    MainThreadOnly,
};
use objc2_app_kit::{
    NSAppearance, NSAppearanceCustomization, NSAppearanceNameAqua, NSAppearanceNameDarkAqua,
    NSStatusBarButton,
};
use objc2_foundation::{
    ns_string, NSArray, NSDictionary, NSKeyValueChangeKey, NSKeyValueObservingOptions, NSObject,
    NSObjectNSKeyValueObserverRegistration, NSString,
};
use tauri::tray::TrayIcon;

struct TrayAppearanceObserverIvars {
    button: Retained<NSStatusBarButton>,
    tray: TrayIcon,
    last_dark: Cell<Option<bool>>,
}

define_class!(
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[ivars = TrayAppearanceObserverIvars]
    struct TrayAppearanceObserver;

    impl TrayAppearanceObserver {
        #[unsafe(method(observeValueForKeyPath:ofObject:change:context:))]
        fn observe_appearance(
            &self,
            _key_path: Option<&NSString>,
            _object: Option<&AnyObject>,
            _change: Option<&NSDictionary<NSKeyValueChangeKey, AnyObject>>,
            _context: *mut c_void,
        ) {
            self.update_icon();
        }
    }
);

impl TrayAppearanceObserver {
    fn new(
        mtm: MainThreadMarker,
        button: Retained<NSStatusBarButton>,
        tray: TrayIcon,
    ) -> Retained<Self> {
        let observer = Self::alloc(mtm).set_ivars(TrayAppearanceObserverIvars {
            button,
            tray,
            last_dark: Cell::new(None),
        });
        // SAFETY: The object is allocated and all ivars are initialized.
        let observer: Retained<Self> = unsafe { msg_send![super(observer), init] };

        // SAFETY: The observer retains the button and unregisters in Drop on the main thread.
        unsafe {
            observer
                .ivars()
                .button
                .addObserver_forKeyPath_options_context(
                    &observer,
                    ns_string!("effectiveAppearance"),
                    NSKeyValueObservingOptions::Initial | NSKeyValueObservingOptions::New,
                    null_mut(),
                );
        }

        observer
    }

    fn update_icon(&self) {
        let is_dark = appearance_is_dark(&self.ivars().button.effectiveAppearance());
        if self.ivars().last_dark.get() == Some(is_dark) {
            return;
        }

        // Mark the classification before replacing the image so a synchronous duplicate KVO
        // delivery cannot recursively attempt the same update.
        self.ivars().last_dark.set(Some(is_dark));
        if let Err(error) = self
            .ivars()
            .tray
            .set_icon_with_as_template(Some(tray_icon(is_dark)), false)
        {
            self.ivars().last_dark.set(None);
            log::warn!("Failed to update the macOS tray appearance: {error}");
        }
    }
}

impl Drop for TrayAppearanceObserver {
    fn drop(&mut self) {
        // SAFETY: This exactly balances the registration in new while both objects are alive.
        unsafe {
            self.ivars()
                .button
                .removeObserver_forKeyPath(self, ns_string!("effectiveAppearance"));
        }
    }
}

thread_local! {
    // AppKit view objects are main-thread-only, so retain the observer in main-thread storage
    // instead of Tauri State (which requires Send + Sync).
    static TRAY_APPEARANCE_OBSERVER: RefCell<Option<Retained<TrayAppearanceObserver>>> =
        const { RefCell::new(None) };
}

pub fn install_appearance_observer(tray: &TrayIcon) -> Result<(), String> {
    let tray_for_observer = tray.clone();
    let installed = tray
        .with_inner_tray_icon(move |inner| {
            let Some(mtm) = MainThreadMarker::new() else {
                return false;
            };
            let Some(status_item) = inner.ns_status_item() else {
                return false;
            };
            let Some(button) = status_item.button(mtm) else {
                return false;
            };

            let observer = TrayAppearanceObserver::new(mtm, button, tray_for_observer);
            TRAY_APPEARANCE_OBSERVER.with(|slot| {
                *slot.borrow_mut() = Some(observer);
            });
            true
        })
        .map_err(|error| error.to_string())?;

    installed
        .then_some(())
        .ok_or_else(|| "AppKit did not expose the tray status button".to_string())
}

pub fn remove_appearance_observer() {
    if MainThreadMarker::new().is_none() {
        log::warn!("Could not remove the macOS tray appearance observer off the main thread");
        return;
    }
    TRAY_APPEARANCE_OBSERVER.with(|slot| {
        slot.borrow_mut().take();
    });
}

fn tray_icon(is_dark: bool) -> tauri::image::Image<'static> {
    if is_dark {
        // Light artwork contrasts with a dark menu-bar background.
        tauri::include_image!("./icons/tray-icon-dark.png")
    } else {
        // Dark artwork contrasts with a light menu-bar background.
        tauri::include_image!("./icons/tray-icon-light.png")
    }
}

fn appearance_is_dark(appearance: &NSAppearance) -> bool {
    // SAFETY: These are immutable AppKit constants available on every supported macOS version.
    let (aqua, dark_aqua) = unsafe { (NSAppearanceNameAqua, NSAppearanceNameDarkAqua) };
    let candidates = NSArray::from_slice(&[aqua, dark_aqua]);
    appearance
        .bestMatchFromAppearancesWithNames(&candidates)
        .is_some_and(|name| *name == *dark_aqua)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appearance_mapping_uses_contrasting_artwork() {
        // SAFETY: These are immutable AppKit constants available on every supported macOS version.
        let (aqua, dark_aqua) = unsafe { (NSAppearanceNameAqua, NSAppearanceNameDarkAqua) };
        let light = NSAppearance::appearanceNamed(aqua).expect("Aqua appearance should exist");
        let dark = NSAppearance::appearanceNamed(dark_aqua).expect("Dark Aqua should exist");

        assert!(!appearance_is_dark(&light));
        assert!(appearance_is_dark(&dark));

        for (is_dark, expected_rgb) in [(false, [0, 0, 0]), (true, [255, 255, 255])] {
            let icon = tray_icon(is_dark);
            assert_eq!((icon.width(), icon.height()), (36, 36));
            let visible: Vec<_> = icon
                .rgba()
                .chunks_exact(4)
                .filter(|pixel| pixel[3] != 0)
                .collect();
            assert!(!visible.is_empty());
            assert!(visible.iter().all(|pixel| pixel[..3] == expected_rgb));
        }
    }
}
