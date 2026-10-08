//! macOS context menu downloads (Download Image, Download Linked File, Download Video). WebKit
//! hands them to a navigation delegate callback wry lacks, so they never started. Route them
//! through the panel's download path as a save-as.

use std::{cell::RefCell, collections::HashMap, path::PathBuf, ptr, rc::Rc};

use objc2::{
    define_class, ffi, msg_send,
    rc::Retained,
    runtime::{AnyObject, Imp, NSObject, ProtocolObject, Sel},
    sel, MainThreadOnly,
};
use objc2_foundation::{
    MainThreadMarker, NSData, NSError, NSObjectProtocol, NSString, NSURLResponse, NSURL,
};
use objc2_web_kit::{WKDownload, WKDownloadDelegate, WKWebView};
use tauri::{Manager, Runtime, Webview};
use url::Url;

use crate::browser_webview::{download_finished, download_requested};

/// Stages a tab's download; false refuses it.
type Request = Rc<dyn Fn(Url, &mut PathBuf, bool) -> bool>;
/// Settles a finished download. Holds no view, which may close first.
type Finish = Rc<dyn Fn(Url, PathBuf, bool)>;
type Hook = unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut WKWebView, *mut WKDownload);

thread_local! {
    /// Tab views by WKWebView address.
    static VIEWS: RefCell<HashMap<usize, (Request, Finish)>> = RefCell::new(HashMap::new());
    /// Downloads in flight: how to settle them, their address and destination.
    static ACTIVE: RefCell<HashMap<usize, (Finish, Url, PathBuf)>> = RefCell::new(HashMap::new());
    /// A download keeps only a weak reference to its delegate.
    static DELEGATE: RefCell<Option<Retained<ContextDownloadDelegate>>> = const { RefCell::new(None) };
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    struct ContextDownloadDelegate;

    unsafe impl NSObjectProtocol for ContextDownloadDelegate {}

    unsafe impl WKDownloadDelegate for ContextDownloadDelegate {
        #[unsafe(method(download:decideDestinationUsingResponse:suggestedFilename:completionHandler:))]
        fn decide(
            &self,
            download: &WKDownload,
            _response: &NSURLResponse,
            suggested: &NSString,
            done: &block2::DynBlock<dyn Fn(*mut NSURL)>,
        ) {
            match start(download, suggested) {
                Some(path) => {
                    let path = NSString::from_str(&path.to_string_lossy());
                    let url = NSURL::fileURLWithPath_isDirectory(&path, false);
                    done.call((Retained::as_ptr(&url).cast_mut(),));
                }
                None => done.call((ptr::null_mut(),)),
            }
        }

        #[unsafe(method(downloadDidFinish:))]
        fn did_finish(&self, download: &WKDownload) {
            finish(download, true);
        }

        #[unsafe(method(download:didFailWithError:resumeData:))]
        fn did_fail(&self, download: &WKDownload, _error: &NSError, _resume: Option<&NSData>) {
            finish(download, false);
        }
    }
);

/// Send this tab's context menu downloads down the panel's download path.
pub(crate) fn watch<R: Runtime>(webview: &Webview<R>, tab: &str) {
    let app = webview.app_handle().clone();
    let (label, tab) = (webview.label().to_string(), tab.to_string());
    let _ = webview.with_webview(move |platform| {
        // Safety: wry's live view, on the main thread.
        let Some(view) = (unsafe { (platform.inner() as *const WKWebView).as_ref() }) else {
            return;
        };
        let Some(navigation) = (unsafe { view.navigationDelegate() }) else {
            return;
        };
        add_hook((*navigation).as_ref());
        // WebKit caches which callbacks a delegate has when it is set: set it again.
        unsafe { view.setNavigationDelegate(Some(&navigation)) };
        let request_app = app.clone();
        let request: Request = Rc::new(move |url, destination, save_as| {
            let Some(page) = request_app.get_webview(&label) else {
                return false;
            };
            download_requested(&page, &tab, url, destination, save_as)
        });
        let finish: Finish =
            Rc::new(move |url, path, success| download_finished(&app, url, Some(path), success));
        VIEWS.with(|views| views.borrow_mut().insert(key(view), (request, finish)));
    });
}

fn key<T>(object: &T) -> usize {
    object as *const T as usize
}

/// Give wry's navigation delegate class the private callback WebKit calls for context menu downloads.
fn add_hook(ui: &AnyObject) {
    let class = ui.class();
    let name = sel!(_webView:contextMenuDidCreateDownload:);
    if class.responds_to(name) {
        return;
    }
    let hook: Hook = context_menu_did_create_download;
    // Safety: the signature matches the "v@:@@" encoding.
    unsafe {
        ffi::class_addMethod(
            (class as *const objc2::runtime::AnyClass).cast_mut(),
            name,
            std::mem::transmute::<Hook, Imp>(hook),
            c"v@:@@".as_ptr(),
        );
    }
}

unsafe extern "C-unwind" fn context_menu_did_create_download(
    _this: *mut AnyObject,
    _cmd: Sel,
    view: *mut WKWebView,
    download: *mut WKDownload,
) {
    // Safety: WebKit passes live objects on the main thread.
    let (Some(view), Some(download)) = (unsafe { view.as_ref() }, unsafe { download.as_ref() })
    else {
        return;
    };
    if !VIEWS.with(|views| views.borrow().contains_key(&key(view))) {
        return;
    }
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    DELEGATE.with(|cell| {
        let delegate = cell
            .borrow_mut()
            .get_or_insert_with(|| unsafe {
                msg_send![mtm.alloc::<ContextDownloadDelegate>(), init]
            })
            .clone();
        unsafe { download.setDelegate(Some(ProtocolObject::from_ref(&*delegate))) };
    });
}

/// Stage the download as a save-as. None refuses it.
fn start(download: &WKDownload, suggested: &NSString) -> Option<PathBuf> {
    let view = unsafe { download.webView() }?;
    let (request, finish) = VIEWS.with(|views| views.borrow().get(&key(&*view)).cloned())?;
    let address = unsafe { download.originalRequest() }?
        .URL()?
        .absoluteString()?;
    let url = Url::parse(&address.to_string()).ok()?;
    let mut destination = PathBuf::from(suggested.to_string());
    if !request(url.clone(), &mut destination, true) {
        return None;
    }
    ACTIVE.with(|active| {
        active
            .borrow_mut()
            .insert(key(download), (finish, url, destination.clone()))
    });
    Some(destination)
}

fn finish(download: &WKDownload, success: bool) {
    let Some((finish, url, path)) =
        ACTIVE.with(|active| active.borrow_mut().remove(&key(download)))
    else {
        return;
    };
    finish(url, path, success);
}
