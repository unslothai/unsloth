// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

//! GTK's ordinary VBox ignores Wry child bounds. Own one overlay and its exact
//! allocation instead. All GTK objects below stay on GTK's thread. Tauri's
//! `unstable` feature creates WindowChild views, not the WindowContent view whose
//! resize handler assumes a WebView -> VBox -> Window ancestry.
use crate::browser_webview::BrowserRect;
use glib::signal::connect_raw;
use gtk::prelude::*;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use tauri::Manager;
use webkit2gtk::WebView;

struct Host {
    overlay: gtk::Overlay,
    page: Option<WebView>,
    allocation: Rc<Cell<BrowserRect>>,
    handler: Option<glib::SignalHandlerId>,
}
thread_local! {
    static HOST: RefCell<Option<Host>> = const { RefCell::new(None) };
}

pub fn install(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let main = app.get_webview("main").ok_or("main webview missing")?;
    main.with_webview(|platform| {
        let main = platform.inner();
        let Some(parent) = main.parent().and_then(|p| p.downcast::<gtk::Box>().ok()) else {
            log::error!("Cannot install browser overlay: main webview has no GTK box parent");
            return;
        };
        let overlay = gtk::Overlay::new();
        overlay.set_hexpand(true);
        overlay.set_vexpand(true);
        parent.remove(&main);
        overlay.add(&main);
        parent.pack_start(&overlay, true, true, 0);
        overlay.show_all();
        HOST.with(|host| {
            *host.borrow_mut() = Some(Host {
                overlay,
                page: None,
                allocation: Rc::new(Cell::new(BrowserRect::default())),
                handler: None,
            })
        });
    })?;
    Ok(())
}

struct AllocationData {
    widget: WebView,
    rect: Rc<Cell<BrowserRect>>,
}

// GTK passes a writable allocation, not a boxed copy. gtk-rs 0.18 does not
// generate this signal. connect_raw owns and drops AllocationData on disconnect.
unsafe extern "C" fn position_child(
    _overlay: *mut gtk::ffi::GtkOverlay,
    widget: *mut gtk::ffi::GtkWidget,
    allocation: *mut gdk::ffi::GdkRectangle,
    data: glib::ffi::gpointer,
) -> glib::ffi::gboolean {
    let data = &*(data as *const AllocationData);
    if widget != data.widget.as_ptr() as *mut gtk::ffi::GtkWidget || allocation.is_null() {
        return 0;
    }
    let rect = data.rect.get();
    *allocation = gdk::ffi::GdkRectangle {
        x: rect.x.round() as i32,
        y: rect.y.round() as i32,
        width: rect.width.round().max(1.0) as i32,
        height: rect.height.round().max(1.0) as i32,
    };
    1
}

pub fn attach(page: WebView, bounds: BrowserRect) -> Result<(), String> {
    HOST.with(|host| {
        let mut host = host.borrow_mut();
        let host = host.as_mut().ok_or("GTK browser host not initialized")?;
        if host.page.is_some() {
            return Err("GTK browser already attached".into());
        }
        page.hide();
        let parent = page
            .parent()
            .and_then(|p| p.downcast::<gtk::Container>().ok())
            .ok_or("browser has no GTK parent")?;
        parent.remove(&page);
        host.allocation.set(bounds);
        let data = Box::new(AllocationData {
            widget: page.clone(),
            rect: host.allocation.clone(),
        });
        host.handler = Some(unsafe {
            connect_raw(
                host.overlay.as_ptr() as *mut _,
                c"get-child-position".as_ptr(),
                Some(std::mem::transmute::<*const (), unsafe extern "C" fn()>(
                    position_child as *const (),
                )),
                Box::into_raw(data),
            )
        });
        page.set_hexpand(false);
        page.set_vexpand(false);
        host.overlay.add_overlay(&page);
        host.overlay.set_overlay_pass_through(&page, false);
        host.page = Some(page);
        apply(host, bounds, true);
        Ok(())
    })
}

fn apply(host: &Host, rect: BrowserRect, visible: bool) {
    host.allocation.set(rect);
    if let Some(page) = &host.page {
        page.set_size_request(
            rect.width.round().max(1.0) as i32,
            rect.height.round().max(1.0) as i32,
        );
        if visible && rect.width >= 1.0 && rect.height >= 1.0 {
            page.show();
        } else {
            page.hide();
        }
        host.overlay.queue_resize();
    }
}

pub fn set_bounds(expected: &WebView, rect: BrowserRect, visible: bool) -> Result<(), String> {
    HOST.with(|host| {
        let host = host.borrow();
        let host = host.as_ref().ok_or("GTK browser host not initialized")?;
        if host.page.as_ref() != Some(expected) {
            return Err("GTK browser not attached".into());
        }
        apply(host, rect, visible);
        Ok(())
    })
}

pub fn detach(expected: &WebView) {
    HOST.with(|host| {
        if let Some(host) = host.borrow_mut().as_mut() {
            if host.page.as_ref() != Some(expected) {
                return;
            }
            if let Some(handler) = host.handler.take() {
                host.overlay.disconnect(handler);
            }
            if let Some(page) = host.page.take() {
                page.hide();
                host.overlay.remove(&page);
            }
        }
    });
}

pub fn allocated_bounds(page: &WebView) -> BrowserRect {
    let rect = page.allocation();
    // WebKit owns a GdkWindow; its allocation origin is local (0,0), even
    // when GTK correctly positions that window within the overlay.
    let (x, y) = HOST.with(|host| {
        host.borrow()
            .as_ref()
            .and_then(|host| page.translate_coordinates(&host.overlay, 0, 0))
            .unwrap_or((rect.x(), rect.y()))
    });
    BrowserRect {
        x: x as f64,
        y: y as f64,
        width: rect.width() as f64,
        height: rect.height() as f64,
    }
}
