// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! GtkBox ignores child webview bounds. An overlay owns each page's allocation instead.
//! Tauri's `unstable` WindowChild path lets the main view be reparented without the
//! WindowContent resize handler's WebView -> VBox -> Window ancestry assumption.

use glib::signal::connect_raw;
use gtk::prelude::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Duration;
use tauri::{LogicalPosition, LogicalSize, Runtime, Webview};
use webkit2gtk::WebView;

#[derive(Clone, Copy)]
struct Bounds {
    position: LogicalPosition<f64>,
    size: LogicalSize<f64>,
}

struct Page {
    view: WebView,
    bounds: Rc<Cell<Bounds>>,
    handler: glib::SignalHandlerId,
}

struct Host {
    overlay: gtk::Overlay,
    pages: HashMap<String, Page>,
}

thread_local! {
    static HOST: RefCell<Option<Host>> = const { RefCell::new(None) };
}

async fn on_view<R: Runtime>(
    view: &Webview<R>,
    apply: impl FnOnce(WebView) -> Result<(), String> + Send + 'static,
) -> Result<(), String> {
    let (send, receive) = tokio::sync::oneshot::channel();
    view.with_webview(move |platform| {
        let _ = send.send(apply(platform.inner()));
    })
    .map_err(|error| error.to_string())?;
    tokio::time::timeout(Duration::from_secs(5), receive)
        .await
        .map_err(|_| "GTK browser update timed out".to_string())?
        .map_err(|_| "GTK browser view closed".to_string())?
}

pub async fn install<R: Runtime>(main: &Webview<R>) -> Result<(), String> {
    on_view(main, |main| {
        HOST.with(|host| {
            let mut host = host.borrow_mut();
            if host.is_some() {
                return Ok(());
            }
            let parent = main
                .parent()
                .and_then(|parent| parent.downcast::<gtk::Box>().ok())
                .ok_or("main webview has no GTK box parent")?;
            let overlay = gtk::Overlay::new();
            overlay.set_hexpand(true);
            overlay.set_vexpand(true);
            parent.remove(&main);
            overlay.add(&main);
            parent.pack_start(&overlay, true, true, 0);
            overlay.show_all();
            *host = Some(Host {
                overlay,
                pages: HashMap::new(),
            });
            Ok(())
        })
    })
    .await
}

struct AllocationData {
    view: WebView,
    bounds: Rc<Cell<Bounds>>,
}

// GTK passes a writable allocation; gtk-rs 0.18 does not generate this signal.
// connect_raw owns AllocationData and drops it when its handler is disconnected.
unsafe extern "C" fn position_child(
    _overlay: *mut gtk::ffi::GtkOverlay,
    widget: *mut gtk::ffi::GtkWidget,
    allocation: *mut gdk::ffi::GdkRectangle,
    data: glib::ffi::gpointer,
) -> glib::ffi::gboolean {
    let data = &*(data as *const AllocationData);
    if widget != data.view.as_ptr() as *mut gtk::ffi::GtkWidget || allocation.is_null() {
        return 0;
    }
    let Bounds { position, size } = data.bounds.get();
    *allocation = gdk::ffi::GdkRectangle {
        x: position.x.round() as i32,
        y: position.y.round() as i32,
        width: size.width.round().max(1.0) as i32,
        height: size.height.round().max(1.0) as i32,
    };
    1
}

pub async fn attach<R: Runtime>(
    page: &Webview<R>,
    position: LogicalPosition<f64>,
    size: LogicalSize<f64>,
) -> Result<(), String> {
    let label = page.label().to_string();
    on_view(page, move |view| {
        HOST.with(|host| {
            let mut host = host.borrow_mut();
            let host = host.as_mut().ok_or("GTK browser host not initialized")?;
            if host.pages.contains_key(&label) {
                return Err("GTK browser already attached".into());
            }
            let parent = view
                .parent()
                .and_then(|parent| parent.downcast::<gtk::Container>().ok())
                .ok_or("browser has no GTK parent")?;
            view.hide();
            parent.remove(&view);
            let bounds = Rc::new(Cell::new(Bounds { position, size }));
            let data = Box::new(AllocationData {
                view: view.clone(),
                bounds: bounds.clone(),
            });
            let handler = unsafe {
                connect_raw(
                    host.overlay.as_ptr() as *mut _,
                    c"get-child-position".as_ptr(),
                    Some(std::mem::transmute::<*const (), unsafe extern "C" fn()>(
                        position_child as *const (),
                    )),
                    Box::into_raw(data),
                )
            };
            view.set_hexpand(false);
            view.set_vexpand(false);
            host.overlay.add_overlay(&view);
            host.overlay.set_overlay_pass_through(&view, false);
            host.pages.insert(
                label,
                Page {
                    view,
                    bounds,
                    handler,
                },
            );
            host.overlay.queue_resize();
            Ok(())
        })
    })
    .await
}

pub async fn set_bounds<R: Runtime>(
    page: &Webview<R>,
    position: LogicalPosition<f64>,
    size: LogicalSize<f64>,
) -> Result<(), String> {
    let label = page.label().to_string();
    on_view(page, move |view| {
        HOST.with(|host| {
            let host = host.borrow();
            let host = host.as_ref().ok_or("GTK browser host not initialized")?;
            let page = host
                .pages
                .get(&label)
                .filter(|page| page.view == view)
                .ok_or("GTK browser not attached")?;
            page.bounds.set(Bounds { position, size });
            host.overlay.queue_resize();
            Ok(())
        })
    })
    .await
}

pub fn detach<R: Runtime>(page: &Webview<R>) -> tauri::Result<()> {
    let label = page.label().to_string();
    page.run_on_main_thread(move || {
        HOST.with(|host| {
            if let Some(host) = host.borrow_mut().as_mut() {
                if let Some(page) = host.pages.remove(&label) {
                    host.overlay.disconnect(page.handler);
                    page.view.hide();
                    if page.view.parent().as_ref() == Some(host.overlay.upcast_ref()) {
                        host.overlay.remove(&page.view);
                    }
                }
            }
        });
    })
}
