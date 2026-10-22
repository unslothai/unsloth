// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Loopback proxy for browser-panel pages: resolves hosts itself, refuses private answers and
//! connects to the checked address, defeating DNS rebinding. Hosts the system proxy covers go to it
//! by name instead, after the same check, so a poisoned public answer or none at all doesn't matter.
//! Local tabs use a separate, short-lived proxy; the public proxy never admits local hosts.

use crate::browser_webview::{host_is_private, ip_is_private};
use hyper_util::client::proxy::matcher::Matcher;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use url::{Host, Position, Url};

const MAX_HEAD: usize = 16 * 1024;
const TIMEOUT: Duration = Duration::from_secs(30);
// One address's share of it, so a dead first answer leaves time for the next.
const ADDRESS_TIMEOUT: Duration = Duration::from_secs(10);
const ESTABLISHED: &[u8] = b"HTTP/1.1 200 Connection Established\r\n\r\n";

static ADDRESS: OnceLock<Result<SocketAddr, String>> = OnceLock::new();

pub fn address() -> Result<SocketAddr, String> {
    ADDRESS
        .get_or_init(|| {
            let listener =
                std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
            listener.set_nonblocking(true).map_err(|e| e.to_string())?;
            let address = listener.local_addr().map_err(|e| e.to_string())?;
            tauri::async_runtime::spawn(async move {
                let Ok(listener) = TcpListener::from_std(listener) else {
                    return;
                };
                loop {
                    match listener.accept().await {
                        Ok((client, _)) => {
                            // Read per connection, so a proxy switched on or off applies at once.
                            tauri::async_runtime::spawn(async move {
                                serve(client, system_proxies()).await
                            });
                        }
                        // Out of descriptors, say: don't spin.
                        Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
                    }
                }
            });
            Ok(address)
        })
        .clone()
}

/// Dropping a local view closes its listener and every connection, including WebSockets.
pub(crate) struct LocalProxy {
    pub address: SocketAddr,
    _stop: tokio::sync::watch::Sender<()>,
}

impl LocalProxy {
    pub fn new(origin: url::Origin) -> Result<Self, String> {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let (stop, mut stopped) = tokio::sync::watch::channel(());
        tauri::async_runtime::spawn(async move {
            let Ok(listener) = TcpListener::from_std(listener) else {
                return;
            };
            let mut connections = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    biased;
                    _ = stopped.changed() => break,
                    accepted = listener.accept() => {
                        let Ok((client, _)) = accepted else { break };
                        let origin = origin.clone();
                        connections.spawn(async move {
                            serve_with_local(client, system_proxies(), Some(&origin)).await;
                        });
                    }
                    _ = connections.join_next(), if !connections.is_empty() => {}
                }
            }
            // JoinSet aborts the open tunnels on drop.
        });
        Ok(Self {
            address,
            _stop: stop,
        })
    }
}

#[derive(Debug, PartialEq)]
struct Request {
    host: String,
    port: u16,
    forward: Option<Forward>,
    /// A tunnel seen to carry cleartext (a `ws://` socket), not TLS.
    cleartext: bool,
}

/// A plain-HTTP request, without the proxy's own headers.
#[derive(Debug, PartialEq)]
struct Forward {
    method: String,
    url: Url,
    version: String,
    headers: String,
    upgrade: bool,
}

impl Forward {
    /// One request per connection: absolute-form to an HTTP proxy, origin-form otherwise.
    fn head(&self, upstream: Option<&Upstream>) -> String {
        let (target, auth) = match upstream {
            Some(Upstream::Http { auth, .. }) => (self.url.as_str(), auth.as_deref()),
            _ => (&self.url[Position::BeforePath..], None),
        };
        let mut head = format!(
            "{} {target} {}\r\n{}",
            self.method, self.version, self.headers
        );
        if let Some(auth) = auth {
            head.push_str(&format!("Proxy-Authorization: {auth}\r\n"));
        }
        head.push_str(if self.upgrade {
            "Connection: Upgrade\r\n\r\n"
        } else {
            "Connection: close\r\n\r\n"
        });
        head
    }
}

/// The system proxy a request goes through.
#[derive(Debug, PartialEq)]
enum Upstream {
    Http {
        address: String,
        auth: Option<String>,
    },
    Socks {
        address: String,
        auth: Option<(String, String)>,
    },
    /// What a Windows PAC script's `SOCKS` names, as WinINet and Chromium read it, or a Linux
    /// desktop's `socks4(a)` proxy.
    #[cfg(any(windows, target_os = "linux", test))]
    Socks4 { address: String },
}

#[derive(Debug, PartialEq)]
enum Refusal {
    Private,
    Unreachable,
    /// The proxy itself can't be reached, so the next route may be tried.
    ProxyDown,
}

fn parse(head: &str) -> Option<Request> {
    let mut lines = head.split("\r\n");
    let mut parts = lines.next()?.split_ascii_whitespace();
    let (method, target, version) = (parts.next()?, parts.next()?, parts.next()?);
    if method.eq_ignore_ascii_case("CONNECT") {
        let url = Url::parse(&format!("http://{target}/")).ok()?;
        return Some(Request {
            host: url.host_str()?.to_string(),
            port: url.port_or_known_default()?,
            forward: None,
            cleartext: false,
        });
    }
    let mut url = Url::parse(target).ok()?;
    match url.scheme() {
        "http" => {}
        // WebKitGTK sends a `ws://` socket to an HTTP proxy in absolute form, not as a tunnel.
        "ws" => url.set_scheme("http").ok()?,
        _ => return None,
    }
    let (mut headers, mut upgrade) = (String::new(), false);
    for line in lines.filter(|line| !line.is_empty()) {
        let name = line
            .split(':')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        upgrade |= name == "upgrade";
        if !matches!(
            name.as_str(),
            "connection" | "keep-alive" | "proxy-connection" | "proxy-authorization"
        ) {
            headers.push_str(line);
            headers.push_str("\r\n");
        }
    }
    Some(Request {
        host: url.host_str()?.to_string(),
        port: url.port_or_known_default()?,
        forward: Some(Forward {
            method: method.to_string(),
            url,
            version: version.to_string(),
            headers,
            upgrade,
        }),
        cleartext: false,
    })
}

/// How the system routes pages.
#[derive(Default)]
struct Proxies {
    /// A PAC script (on Linux, the desktop's settings), which decides before `fixed` unless it
    /// can't be found or run.
    #[cfg(any(target_os = "macos", windows, target_os = "linux"))]
    pac: Option<pac::Pac>,
    /// Fixed settings, in order: proxy variables, else macOS or Windows settings.
    fixed: Vec<Matcher>,
}

fn system_proxies() -> Proxies {
    #[cfg(target_os = "macos")]
    let proxies = macos_proxies(|name| std::env::var(name).ok());
    #[cfg(windows)]
    let proxies = Proxies {
        pac: pac::configured()
            .map(|source| pac::Pac::new(source, &|name: &str| std::env::var(name).ok())),
        fixed: vec![windows_per_protocol().unwrap_or_else(Matcher::from_system)],
    };
    #[cfg(target_os = "linux")]
    let proxies = linux_proxies(|name| std::env::var(name).ok());
    #[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
    let proxies = Proxies {
        fixed: vec![Matcher::from_system()],
    };
    proxies
}

/// Proxy settings by protocol, as macOS and Windows store them.
#[cfg(any(target_os = "macos", windows, test))]
#[derive(Default)]
struct Settings {
    http: String,
    https: String,
    bypass: String,
}

/// Proxies for `settings`. As in hyper-util, each proxy variable that is set and not empty
/// overrides its own setting.
#[cfg(any(target_os = "macos", windows, test))]
fn with_variables(settings: Settings, env: &impl Fn(&str) -> Option<String>) -> Matcher {
    let variable = |names| variable(env, names);
    Matcher::builder()
        .all(variable(["ALL_PROXY", "all_proxy"]).unwrap_or_default())
        .http(variable(["HTTP_PROXY", "http_proxy"]).unwrap_or(settings.http))
        .https(variable(["HTTPS_PROXY", "https_proxy"]).unwrap_or(settings.https))
        .no(variable(["NO_PROXY", "no_proxy"]).unwrap_or(settings.bypass))
        .build()
}

#[cfg(any(target_os = "macos", windows, target_os = "linux", test))]
fn variable(env: &impl Fn(&str) -> Option<String>, names: [&str; 2]) -> Option<String> {
    names
        .into_iter()
        .filter_map(env)
        .find(|value| !value.is_empty())
}

/// The desktop's settings, unless proxy variables (Linux's own way to set a proxy) are.
#[cfg(target_os = "linux")]
fn linux_proxies(env: impl Fn(&str) -> Option<String>) -> Proxies {
    let names = [
        ["ALL_PROXY", "all_proxy"],
        ["HTTP_PROXY", "http_proxy"],
        ["HTTPS_PROXY", "https_proxy"],
    ];
    let variables = names
        .into_iter()
        .any(|names| variable(&env, names).is_some());
    Proxies {
        pac: (!variables).then(|| pac::Pac::new(pac::Source::Desktop, &env)),
        fixed: vec![Matcher::from_system()],
    }
}

/// macOS' proxy settings, read here rather than by hyper-util, which skips PAC, the SOCKS proxy and
/// the exceptions and can't parse an IPv6 host. The SOCKS proxy serves what HTTP(S) ones don't cover.
#[cfg(target_os = "macos")]
fn macos_proxies(env: impl Fn(&str) -> Option<String>) -> Proxies {
    use system_configuration::core_foundation::array::CFArray;
    use system_configuration::core_foundation::base::{CFType, TCFType};
    use system_configuration::core_foundation::number::CFNumber;
    use system_configuration::core_foundation::string::{CFString, CFStringRef};
    use system_configuration::dynamic_store::SCDynamicStoreBuilder;
    use system_configuration::sys::schema_definitions::*;

    let Some(proxies) = SCDynamicStoreBuilder::new("unsloth-browser")
        .build()
        .and_then(|store| store.get_proxies())
    else {
        return Proxies {
            pac: None,
            fixed: vec![with_variables(Settings::default(), &env)],
        };
    };
    let number = |key| {
        proxies
            .find(key)
            .and_then(|value| value.downcast::<CFNumber>())
            .and_then(|value| value.to_i32())
    };
    let text = |key| {
        proxies
            .find(key)
            .and_then(|value| value.downcast::<CFString>())
            .map(|value| value.to_string())
    };
    let proxy = |enable: CFStringRef, host: CFStringRef, port: CFStringRef| {
        if number(enable) != Some(1) {
            return None;
        }
        Some(authority(&text(host)?, number(port)))
    };
    let pac = unsafe {
        if number(kSCPropNetProxiesProxyAutoConfigEnable) != Some(1) {
            None
        } else {
            text(kSCPropNetProxiesProxyAutoConfigJavaScript)
                .map(|script| pac::Source::Script(script.into()))
                .or_else(|| text(kSCPropNetProxiesProxyAutoConfigURLString).map(pac::Source::Url))
        }
    };
    let pac = pac.map(|source| pac::Pac::new(source, &env));
    let exceptions = proxies
        .find(unsafe { kSCPropNetProxiesExceptionsList })
        .and_then(|value| value.downcast::<CFArray>())
        .map(|list| {
            list.iter()
                .filter_map(|item| {
                    unsafe { CFType::wrap_under_get_rule(*item as _) }.downcast::<CFString>()
                })
                .map(|entry| entry.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let settings = unsafe {
        Settings {
            http: proxy(
                kSCPropNetProxiesHTTPEnable,
                kSCPropNetProxiesHTTPProxy,
                kSCPropNetProxiesHTTPPort,
            )
            .unwrap_or_default(),
            https: proxy(
                kSCPropNetProxiesHTTPSEnable,
                kSCPropNetProxiesHTTPSProxy,
                kSCPropNetProxiesHTTPSPort,
            )
            .unwrap_or_default(),
            bypass: bypass_list(exceptions.iter().map(String::as_str)),
        }
    };
    let socks = unsafe {
        proxy(
            kSCPropNetProxiesSOCKSEnable,
            kSCPropNetProxiesSOCKSProxy,
            kSCPropNetProxiesSOCKSPort,
        )
    };
    let no = variable(&env, ["NO_PROXY", "no_proxy"]).unwrap_or_else(|| settings.bypass.clone());
    let mut fixed = vec![with_variables(settings, &env)];
    fixed.extend(socks.map(|address| socks_fallback(&address, &no)));
    Proxies { pac, fixed }
}

/// `host[:port]`, with an IPv6 host bracketed as a URI needs.
#[cfg(any(target_os = "macos", windows, target_os = "linux", test))]
fn authority(host: &str, port: Option<i32>) -> String {
    let host = match host.parse::<std::net::Ipv6Addr>() {
        Ok(_) => format!("[{host}]"),
        Err(_) => host.to_string(),
    };
    match port {
        Some(port) => format!("{host}:{port}"),
        None => host,
    }
}

/// A NO_PROXY list from bypass entries, `*.` wildcards as suffixes as hyper-util reads them.
#[cfg(any(target_os = "macos", windows, test))]
fn bypass_list<'a>(entries: impl Iterator<Item = &'a str>) -> String {
    entries
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(",")
        .replace("*.", "")
}

/// Every destination through the SOCKS proxy at `address`, minus the `no` list.
#[cfg(any(target_os = "macos", test))]
fn socks_fallback(address: &str, no: &str) -> Matcher {
    Matcher::builder()
        .all(format!("socks5h://{address}"))
        .no(no)
        .build()
}

/// hyper-util reads Windows' `ProxyServer` as one address, so its per-protocol form
/// (`http=a:80;https=b:443`) is split here.
#[cfg(windows)]
fn windows_per_protocol() -> Option<Matcher> {
    use winreg::enums::{HKEY_CURRENT_USER, KEY_QUERY_VALUE};
    use winreg::RegKey;

    let settings = RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey_with_flags(
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            KEY_QUERY_VALUE,
        )
        .ok()?;
    if settings.get_value::<u32, _>("ProxyEnable").unwrap_or(0) == 0 {
        return None;
    }
    let server = settings.get_value::<String, _>("ProxyServer").ok()?;
    let proxy_override = settings
        .get_value::<String, _>("ProxyOverride")
        .unwrap_or_default();
    per_protocol(&server, &proxy_override, |name| std::env::var(name).ok())
}

/// Proxies for a per-protocol `ProxyServer` value (`None` for a single address).
#[cfg(any(windows, test))]
fn per_protocol(
    server: &str,
    proxy_override: &str,
    env: impl Fn(&str) -> Option<String>,
) -> Option<Matcher> {
    if !server.contains('=') {
        return None;
    }
    let setting = |wanted: &str| {
        server
            .split(';')
            .filter_map(|entry| entry.split_once('='))
            .find(|(scheme, _)| scheme.trim().eq_ignore_ascii_case(wanted))
            .map(|(_, address)| address.trim().to_string())
            .unwrap_or_default()
    };
    let settings = Settings {
        http: setting("http"),
        https: setting("https"),
        bypass: bypass_list(proxy_override.split(';')),
    };
    Some(with_variables(settings, &env))
}

/// Where `request` may go, in order: through a system proxy, or direct (`None`).
async fn routes(proxies: &Proxies, request: &Request) -> Vec<Option<Upstream>> {
    let destination: Option<http::Uri> = match &request.forward {
        Some(forward) => forward.url.as_str().parse().ok(),
        None if request.cleartext => format!("http://{}:{}/", request.host, request.port)
            .parse()
            .ok(),
        // Tunnels carry TLS, so they take the HTTPS proxy.
        None => format!("https://{}:{}/", request.host, request.port)
            .parse()
            .ok(),
    };
    let Some(destination) = destination else {
        return vec![None];
    };
    #[cfg(any(target_os = "macos", windows, target_os = "linux"))]
    if let Some(pac) = proxies.pac.as_ref().filter(|pac| pac.decides(&destination)) {
        // As browsers ask, without a default port.
        let url = Url::parse(&destination.to_string())
            .map_or_else(|_| destination.to_string(), String::from);
        if let Some(routes) = pac::routes(pac, &url).await {
            return routes;
        }
    }
    vec![system_proxy(&proxies.fixed, &destination)]
}

/// The first fixed setting that covers `destination`. HTTP and SOCKS5 proxies are chained.
fn system_proxy(proxies: &[Matcher], destination: &http::Uri) -> Option<Upstream> {
    let intercept = proxies
        .iter()
        .find_map(|proxies| proxies.intercept(destination))?;
    let uri = intercept.uri();
    let address = |default| {
        format!(
            "{}:{}",
            uri.host().unwrap_or(""),
            uri.port_u16().unwrap_or(default)
        )
    };
    match uri.scheme_str() {
        Some("http") => Some(Upstream::Http {
            address: address(80),
            auth: intercept
                .basic_auth()
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned),
        }),
        // Either way the proxy gets the name, as with HTTP proxies.
        Some("socks5" | "socks5h") => Some(Upstream::Socks {
            address: address(1080),
            auth: intercept
                .raw_auth()
                .map(|(user, pass)| (user.to_owned(), pass.to_owned())),
        }),
        scheme => {
            log::info!("browser proxy can't chain to a {scheme:?} proxy");
            None
        }
    }
}

/// Parse `host`, refusing it if it names a private address.
fn public_host(host: &str) -> Result<Host<String>, Refusal> {
    let parsed = Host::parse(host).map_err(|_| Refusal::Unreachable)?;
    let borrowed = match &parsed {
        Host::Domain(name) => Host::Domain(name.as_str()),
        Host::Ipv4(ip) => Host::Ipv4(*ip),
        Host::Ipv6(ip) => Host::Ipv6(*ip),
    };
    if host_is_private(&borrowed) {
        return Err(Refusal::Private);
    }
    Ok(parsed)
}

/// A local origin the user granted, dialled as written: never looked up, never via a system proxy.
fn granted_addresses(
    origin: Option<&url::Origin>,
    host: &str,
    port: u16,
) -> Option<Vec<SocketAddr>> {
    if !matches!(origin, Some(url::Origin::Tuple(_, granted, granted_port))
        if *granted_port == port && granted.to_string() == host)
    {
        return None;
    }
    Some(match Host::parse(host).ok()? {
        Host::Domain(_) => vec![
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port),
            SocketAddr::new(Ipv6Addr::LOCALHOST.into(), port),
        ],
        Host::Ipv4(ip) => vec![SocketAddr::new(ip.into(), port)],
        Host::Ipv6(ip) => vec![SocketAddr::new(ip.into(), port)],
    })
}

/// Connect to a public address of `host`, refusing it if any answer is private. One deadline covers
/// the lookup and every address, so a host with many dead answers can't hold a task for long.
async fn connect(host: &str, port: u16) -> Result<TcpStream, Refusal> {
    tokio::time::timeout(TIMEOUT, connect_within(host, port))
        .await
        .unwrap_or(Err(Refusal::Unreachable))
}

async fn connect_within(host: &str, port: u16) -> Result<TcpStream, Refusal> {
    let addresses: Vec<SocketAddr> = match public_host(host)? {
        Host::Ipv4(ip) => vec![SocketAddr::new(IpAddr::V4(ip), port)],
        Host::Ipv6(ip) => vec![SocketAddr::new(IpAddr::V6(ip), port)],
        Host::Domain(name) => tokio::net::lookup_host((name.as_str(), port))
            .await
            .map_err(|_| Refusal::Unreachable)?
            .collect(),
    };
    let answers: Vec<IpAddr> = addresses.iter().map(SocketAddr::ip).collect();
    if answers_are_private(&answers).await {
        return Err(Refusal::Private);
    }
    dial(addresses).await
}

/// Whether any DNS answer is private. Fake-IP answers (198.18.0.0/15) pass only once the resolver
/// probe confirms fake-IP DNS; literals are refused earlier by `public_host`.
async fn answers_are_private(answers: &[IpAddr]) -> bool {
    if answers
        .iter()
        .any(|&ip| ip_is_private(ip) && !is_fake_ip(ip))
    {
        return true;
    }
    answers.iter().any(|&ip| is_fake_ip(ip)) && !resolver_is_fake_ip().await
}

fn is_fake_ip(ip: IpAddr) -> bool {
    matches!(ip, IpAddr::V4(v4) if u32::from(v4) >> 17 == 0xc612_0000 >> 17)
}

const FAKE_IP_PROBE_TTL: Duration = Duration::from_secs(60);
static FAKE_IP_RESOLVER: Mutex<Option<(Instant, bool)>> = Mutex::new(None);

/// Whether an unregistered name resolves into the fake-IP range. Without this, any public domain could point into it.
async fn resolver_is_fake_ip() -> bool {
    let cached = *FAKE_IP_RESOLVER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((checked, answer)) = cached {
        if checked.elapsed() < FAKE_IP_PROBE_TTL {
            return answer;
        }
    }
    let label = RandomState::new().build_hasher().finish();
    let probe = tokio::time::timeout(
        ADDRESS_TIMEOUT,
        tokio::net::lookup_host((format!("unsloth-{label:016x}.com"), 443)),
    );
    let answer = match probe.await {
        Ok(Ok(addresses)) => {
            let answers: Vec<IpAddr> = addresses.map(|address| address.ip()).collect();
            !answers.is_empty() && answers.into_iter().all(is_fake_ip)
        }
        _ => false,
    };
    *FAKE_IP_RESOLVER.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), answer));
    answer
}

async fn dial(addresses: impl IntoIterator<Item = SocketAddr>) -> Result<TcpStream, Refusal> {
    for address in addresses {
        if let Ok(Ok(stream)) =
            tokio::time::timeout(ADDRESS_TIMEOUT, TcpStream::connect(address)).await
        {
            return Ok(stream);
        }
    }
    Err(Refusal::Unreachable)
}

/// Open `request` through the system proxy, which gets the name. Never falls back to a direct
/// connection. Returns what the proxy sent past its answer.
async fn connect_via(
    upstream: &Upstream,
    request: &Request,
) -> Result<(TcpStream, Vec<u8>), Refusal> {
    let host = public_host(&request.host)?;
    let (address, http_auth) = match upstream {
        Upstream::Http { address, auth } => (address, auth),
        Upstream::Socks { address, .. } => (address, &None),
        #[cfg(any(windows, target_os = "linux", test))]
        Upstream::Socks4 { address } => (address, &None),
    };
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    // Running out of time before the proxy answers the dial means it's down: the next route may go.
    let mut stream = tokio::time::timeout_at(deadline, async {
        refuse_private_answers(&host, request.port).await?;
        let addresses = tokio::net::lookup_host(address.as_str())
            .await
            .map_err(|_| Refusal::ProxyDown)?;
        dial(addresses).await.map_err(|_| Refusal::ProxyDown)
    })
    .await
    .unwrap_or(Err(Refusal::ProxyDown))?;
    tokio::time::timeout_at(deadline, async {
        let socks = match upstream {
            Upstream::Http { .. } => None,
            Upstream::Socks { auth, .. } => {
                Some(socks_connect(&mut stream, &host, request.port, auth.as_ref()).await)
            }
            #[cfg(any(windows, target_os = "linux", test))]
            Upstream::Socks4 { .. } => Some(socks4_connect(&mut stream, &host, request.port).await),
        };
        if let Some(connected) = socks {
            connected.ok_or(Refusal::Unreachable)?;
            return Ok((stream, Vec::new()));
        }
        if request.forward.is_some() {
            return Ok((stream, Vec::new()));
        }
        let target = format!("{}:{}", request.host, request.port);
        let mut head = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n");
        if let Some(auth) = http_auth {
            head.push_str(&format!("Proxy-Authorization: {auth}\r\n"));
        }
        head.push_str("\r\n");
        stream
            .write_all(head.as_bytes())
            .await
            .map_err(|_| Refusal::Unreachable)?;
        let (answer, early) = read_head(&mut stream).await.ok_or(Refusal::Unreachable)?;
        let status = answer.split_ascii_whitespace().nth(1).unwrap_or("");
        if !status.starts_with('2') {
            log::info!("system proxy answered {status} for {target}");
            return Err(Refusal::Unreachable);
        }
        Ok((stream, early))
    })
    .await
    .unwrap_or(Err(Refusal::Unreachable))
}

/// Refuse a name local DNS answers privately, as the direct path does. One it can't answer in time
/// (a proxy-only network) is left to the proxy.
async fn refuse_private_answers(host: &Host<String>, port: u16) -> Result<(), Refusal> {
    let Host::Domain(name) = host else {
        return Ok(());
    };
    // Its own share of the deadline, so a resolver dropping queries leaves time for the proxy.
    let answers = tokio::time::timeout(
        ADDRESS_TIMEOUT,
        tokio::net::lookup_host((name.as_str(), port)),
    );
    let Ok(Ok(answers)) = answers.await else {
        return Ok(());
    };
    let answers: Vec<IpAddr> = answers.map(|answer| answer.ip()).collect();
    if answers_are_private(&answers).await {
        return Err(Refusal::Private);
    }
    Ok(())
}

/// A SOCKS5 CONNECT (RFC 1928, password auth RFC 1929) that names `host`, so the proxy resolves it.
async fn socks_connect(
    stream: &mut TcpStream,
    host: &Host<String>,
    port: u16,
    auth: Option<&(String, String)>,
) -> Option<()> {
    let methods: &[u8] = if auth.is_some() { &[0, 2] } else { &[0] };
    let mut greeting = vec![5, methods.len() as u8];
    greeting.extend_from_slice(methods);
    stream.write_all(&greeting).await.ok()?;
    let mut chosen = [0u8; 2];
    stream.read_exact(&mut chosen).await.ok()?;
    match (chosen, auth) {
        ([5, 0], _) => {}
        ([5, 2], Some((user, pass))) => {
            let (user, pass) = (user.as_bytes(), pass.as_bytes());
            let mut login = vec![1, u8::try_from(user.len()).ok()?];
            login.extend_from_slice(user);
            login.push(u8::try_from(pass.len()).ok()?);
            login.extend_from_slice(pass);
            stream.write_all(&login).await.ok()?;
            let mut status = [0u8; 2];
            stream.read_exact(&mut status).await.ok()?;
            if status[1] != 0 {
                return None;
            }
        }
        _ => return None,
    }
    let mut connect = vec![5, 1, 0];
    match host {
        Host::Domain(name) => {
            connect.push(3);
            connect.push(u8::try_from(name.len()).ok()?);
            connect.extend_from_slice(name.as_bytes());
        }
        Host::Ipv4(ip) => {
            connect.push(1);
            connect.extend_from_slice(&ip.octets());
        }
        Host::Ipv6(ip) => {
            connect.push(4);
            connect.extend_from_slice(&ip.octets());
        }
    }
    connect.extend_from_slice(&port.to_be_bytes());
    stream.write_all(&connect).await.ok()?;
    let mut reply = [0u8; 4];
    stream.read_exact(&mut reply).await.ok()?;
    if reply[..2] != [5, 0] {
        log::info!("SOCKS proxy answered {} for {host}:{port}", reply[1]);
        return None;
    }
    // Skip the bound address the reply carries.
    let bound = match reply[3] {
        1 => 4,
        4 => 16,
        3 => {
            let mut length = [0u8; 1];
            stream.read_exact(&mut length).await.ok()?;
            usize::from(length[0])
        }
        _ => return None,
    };
    let mut rest = vec![0u8; bound + 2];
    stream.read_exact(&mut rest).await.ok()?;
    Some(())
}

/// A SOCKS4a CONNECT that names `host`, so the proxy resolves it. SOCKS4 has no IPv6.
#[cfg(any(windows, target_os = "linux", test))]
async fn socks4_connect(stream: &mut TcpStream, host: &Host<String>, port: u16) -> Option<()> {
    let mut connect = vec![4, 1];
    connect.extend_from_slice(&port.to_be_bytes());
    match host {
        Host::Ipv4(ip) => {
            connect.extend_from_slice(&ip.octets());
            connect.push(0);
        }
        Host::Domain(name) => {
            connect.extend_from_slice(&[0, 0, 0, 1, 0]);
            connect.extend_from_slice(name.as_bytes());
            connect.push(0);
        }
        Host::Ipv6(_) => return None,
    }
    stream.write_all(&connect).await.ok()?;
    let mut reply = [0u8; 8];
    stream.read_exact(&mut reply).await.ok()?;
    if reply[1] != 0x5a {
        log::info!("SOCKS4 proxy answered {} for {host}:{port}", reply[1]);
        return None;
    }
    Some(())
}

/// The plain-HTTP request a tunnel carries, made absolute so it is routed and forwarded as one,
/// with the body read past its head; or no request (a WebSocket upgrade, TLS), with every byte read.
async fn untunnel(client: &mut TcpStream, tunnel: &Request) -> Option<(Option<Request>, Vec<u8>)> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = client.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        let method = buffer
            .iter()
            .take_while(|byte| byte.is_ascii_uppercase())
            .count();
        if method == buffer.len() && method < 16 {
            continue;
        }
        if method == 0 || buffer.get(method..method + 2) != Some(b" /") {
            return Some((None, buffer));
        }
        let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") else {
            if buffer.len() > MAX_HEAD {
                return None;
            }
            continue;
        };
        let request = std::str::from_utf8(&buffer[..end + 4])
            .ok()
            .filter(|head| {
                !head
                    .split("\r\n")
                    .any(|line| line.to_ascii_lowercase().starts_with("upgrade:"))
            })
            .and_then(|head| {
                let (method, rest) = head.split_once(' ')?;
                parse(&format!(
                    "{method} http://{}:{}{rest}",
                    tunnel.host, tunnel.port
                ))
            })
            .filter(|request| request.host == tunnel.host && request.port == tunnel.port);
        return Some(match request {
            Some(request) => (Some(request), buffer.split_off(end + 4)),
            None => (None, buffer),
        });
    }
}

async fn read_head(client: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = client.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            let rest = buffer.split_off(end + 4);
            return Some((String::from_utf8(buffer).ok()?, rest));
        }
        if buffer.len() > MAX_HEAD {
            return None;
        }
    }
}

async fn serve(client: TcpStream, proxies: Proxies) {
    serve_with_local(client, proxies, None).await;
}

async fn serve_with_local(mut client: TcpStream, proxies: Proxies, local: Option<&url::Origin>) {
    let Ok(Some((head, rest))) = tokio::time::timeout(TIMEOUT, read_head(&mut client)).await else {
        return;
    };
    let Some(mut request) = parse(&head) else {
        let _ = reply(&mut client, "400 Bad Request").await;
        return;
    };
    let (mut rest, mut established) = (rest, false);
    // WebKit on macOS 14+ tunnels plain-HTTP pages too; routed as tunnels they'd count as HTTPS,
    // and a nested CONNECT to port 80 is refused by many HTTP proxies.
    if request.forward.is_none()
        && request.port == 80
        && rest.is_empty()
        && public_host(&request.host).is_ok()
    {
        if client.write_all(ESTABLISHED).await.is_err() {
            return;
        }
        established = true;
        let Ok(Some((inner, read))) =
            tokio::time::timeout(TIMEOUT, untunnel(&mut client, &request)).await
        else {
            return;
        };
        match inner {
            Some(inner) => request = inner,
            // A TLS handshake starts with a 0x16 record.
            None => request.cleartext = read.first() != Some(&0x16),
        }
        rest = read;
    }
    let (mut via, mut opened) = (None, Err(Refusal::Unreachable));
    if let Some(addresses) = granted_addresses(local, &request.host, request.port) {
        opened = dial(addresses).await.map(|stream| (stream, Vec::new()));
    } else {
        for route in routes(&proxies, &request).await {
            opened = match &route {
                Some(route) => connect_via(route, &request).await,
                None => connect(&request.host, request.port)
                    .await
                    .map(|stream| (stream, Vec::new())),
            };
            via = route;
            if !matches!(opened, Err(Refusal::ProxyDown)) {
                break;
            }
        }
    }
    let (mut upstream, early) = match opened {
        Ok(opened) => opened,
        Err(Refusal::Private) => {
            log::info!("browser proxy refused private host {}", request.host);
            let _ = reply(&mut client, "403 Forbidden").await;
            return;
        }
        Err(Refusal::Unreachable | Refusal::ProxyDown) => {
            let _ = reply(&mut client, "502 Bad Gateway").await;
            return;
        }
    };
    let opened = match &request.forward {
        Some(forward) => {
            upstream
                .write_all(forward.head(via.as_ref()).as_bytes())
                .await
        }
        None if established => Ok(()),
        None => client.write_all(ESTABLISHED).await,
    };
    if opened.is_ok()
        && client.write_all(&early).await.is_ok()
        && upstream.write_all(&rest).await.is_ok()
    {
        let _ = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
    }
}

async fn reply(client: &mut TcpStream, status: &str) -> std::io::Result<()> {
    let response = format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    client.write_all(response.as_bytes()).await
}

/// PAC scripts, run as the system's own browser runs them: on macOS fetched here (directly, since a
/// local proxy app or the intranet often serves them) and run by CFNetwork; on Windows found and
/// run by WinHTTP, which also discovers them (WPAD). On Linux GIO's resolver, which WebKitGTK uses,
/// answers from the desktop's settings: GNOME's, else libproxy's (KDE, PAC, WPAD).
#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
mod pac {
    use super::{variable, Upstream, ADDRESS_TIMEOUT};
    use hyper_util::client::proxy::matcher::Matcher;
    #[cfg(windows)]
    pub(super) use winhttp::configured;
    #[cfg(target_os = "macos")]
    use {
        super::authority,
        std::sync::Arc,
        std::time::{Duration, Instant},
        system_configuration::core_foundation::array::{CFArray, CFArrayRef},
        system_configuration::core_foundation::base::{CFType, TCFType},
        system_configuration::core_foundation::dictionary::CFDictionary,
        system_configuration::core_foundation::error::{CFError, CFErrorRef},
        system_configuration::core_foundation::number::CFNumber,
        system_configuration::core_foundation::string::{CFString, CFStringRef},
        system_configuration::core_foundation::url::{CFURLCreateWithString, CFURLRef, CFURL},
        tokio::sync::Mutex,
    };

    #[cfg(target_os = "macos")]
    const SCRIPT_TTL: Duration = Duration::from_secs(60);

    #[cfg(target_os = "macos")]
    #[link(name = "CFNetwork", kind = "framework")]
    extern "C" {
        fn CFNetworkCopyProxiesForAutoConfigurationScript(
            script: CFStringRef,
            url: CFURLRef,
            error: *mut CFErrorRef,
        ) -> CFArrayRef;
        static kCFProxyTypeKey: CFStringRef;
        static kCFProxyHostNameKey: CFStringRef;
        static kCFProxyPortNumberKey: CFStringRef;
        static kCFProxyTypeNone: CFStringRef;
        static kCFProxyTypeHTTP: CFStringRef;
        static kCFProxyTypeHTTPS: CFStringRef;
        static kCFProxyTypeSOCKS: CFStringRef;
    }

    pub(super) enum Source {
        #[cfg(target_os = "macos")]
        Url(String),
        #[cfg(target_os = "macos")]
        Script(Arc<str>),
        /// Windows' automatic configuration: discovery when `detect`, then the script at `url`.
        #[cfg(windows)]
        Automatic { detect: bool, url: Option<String> },
        #[cfg(target_os = "linux")]
        Desktop,
    }

    pub(super) struct Pac {
        source: Source,
        /// Proxy variables, which override the script for their protocols.
        variables: Matcher,
        /// NO_PROXY, which keeps its hosts from the script: a stand-in proxy minus the list.
        bypass: Matcher,
    }

    impl Pac {
        pub(super) fn new(source: Source, env: &impl Fn(&str) -> Option<String>) -> Pac {
            let variable = |names| variable(env, names).unwrap_or_default();
            Pac {
                source,
                variables: Matcher::builder()
                    .all(variable(["ALL_PROXY", "all_proxy"]))
                    .http(variable(["HTTP_PROXY", "http_proxy"]))
                    .https(variable(["HTTPS_PROXY", "https_proxy"]))
                    .build(),
                bypass: Matcher::builder()
                    .all("http://pac.invalid")
                    .no(variable(["NO_PROXY", "no_proxy"]))
                    .build(),
            }
        }

        /// Whether the script routes `destination`, which the fixed settings route otherwise.
        pub(super) fn decides(&self, destination: &http::Uri) -> bool {
            self.variables.intercept(destination).is_none()
                && self.bypass.intercept(destination).is_some()
        }
    }

    /// The script's routes for `url`, `None` meaning direct; nothing if it can't be had or run.
    pub(super) async fn routes(pac: &Pac, url: &str) -> Option<Vec<Option<Upstream>>> {
        let url = url.to_string();
        #[cfg(target_os = "macos")]
        let run = {
            let script = match &pac.source {
                Source::Url(location) => script(location).await?,
                Source::Script(script) => script.clone(),
            };
            move || evaluate(&script, &url)
        };
        #[cfg(windows)]
        let run = {
            let Source::Automatic {
                detect,
                url: script,
            } = &pac.source;
            let (detect, script) = (*detect, script.clone());
            move || winhttp::evaluate(detect, script.as_deref(), &url)
        };
        #[cfg(target_os = "linux")]
        let run = {
            let Source::Desktop = pac.source;
            move || desktop::evaluate(&gio::ProxyResolver::default(), &url)
        };
        tokio::time::timeout(ADDRESS_TIMEOUT, tokio::task::spawn_blocking(run))
            .await
            .ok()?
            .ok()?
    }

    #[cfg(target_os = "macos")]
    struct Fetched {
        location: String,
        at: Instant,
        script: Option<Arc<str>>,
    }

    #[cfg(target_os = "macos")]
    async fn script(location: &str) -> Option<Arc<str>> {
        // Held across the fetch, so a page's many connections fetch the script once.
        static CACHE: Mutex<Option<Fetched>> = Mutex::const_new(None);
        let mut cache = CACHE.lock().await;
        if let Some(fetched) = &*cache {
            if fetched.location == location && fetched.at.elapsed() < SCRIPT_TTL {
                return fetched.script.clone();
            }
        }
        let script = fetch(location).await.map(Arc::from);
        if script.is_none() {
            log::info!("browser proxy couldn't fetch the PAC script at {location}");
        }
        *cache = Some(Fetched {
            location: location.to_string(),
            at: Instant::now(),
            script: script.clone(),
        });
        script
    }

    #[cfg(target_os = "macos")]
    async fn fetch(location: &str) -> Option<String> {
        let client = crate::loopback_http::client(ADDRESS_TIMEOUT).ok()?;
        let response = client.get(location).send().await.ok()?;
        response.error_for_status().ok()?.text().await.ok()
    }

    /// The entries that are DIRECT or a proxy this one can chain to, in order.
    #[cfg(target_os = "macos")]
    fn evaluate(script: &str, url: &str) -> Option<Vec<Option<Upstream>>> {
        let (script, url) = (CFString::new(script), CFString::new(url));
        unsafe {
            let url = CFURLCreateWithString(
                std::ptr::null(),
                url.as_concrete_TypeRef(),
                std::ptr::null(),
            );
            if url.is_null() {
                return None;
            }
            let url = CFURL::wrap_under_create_rule(url);
            let mut error: CFErrorRef = std::ptr::null_mut();
            let proxies = CFNetworkCopyProxiesForAutoConfigurationScript(
                script.as_concrete_TypeRef(),
                url.as_concrete_TypeRef(),
                &mut error,
            );
            if proxies.is_null() {
                if !error.is_null() {
                    let error = CFError::wrap_under_create_rule(error);
                    log::info!("browser proxy couldn't run the PAC script: {error}");
                }
                return None;
            }
            let proxies: CFArray<CFDictionary<CFString, CFType>> =
                CFArray::wrap_under_create_rule(proxies);
            let mut routes = Vec::new();
            for proxy in proxies.iter() {
                let text = |key| proxy.find(key).and_then(|v| v.downcast::<CFString>());
                let Some(kind) = text(kCFProxyTypeKey) else {
                    continue;
                };
                let is = |other| kind == CFString::wrap_under_get_rule(other);
                if is(kCFProxyTypeNone) {
                    routes.push(None);
                    continue;
                }
                let (socks, default_port) = if is(kCFProxyTypeSOCKS) {
                    (true, 1080)
                } else if is(kCFProxyTypeHTTP) || is(kCFProxyTypeHTTPS) {
                    (false, 80)
                } else {
                    continue;
                };
                let Some(host) = text(kCFProxyHostNameKey) else {
                    continue;
                };
                let port = proxy
                    .find(kCFProxyPortNumberKey)
                    .and_then(|v| v.downcast::<CFNumber>())
                    .and_then(|v| v.to_i32());
                let address = authority(&host.to_string(), Some(port.unwrap_or(default_port)));
                routes.push(Some(match socks {
                    true => Upstream::Socks {
                        address,
                        auth: None,
                    },
                    false => Upstream::Http {
                        address,
                        auth: None,
                    },
                }));
            }
            // An empty answer means direct, as browsers read one.
            if routes.is_empty() {
                routes.push(None);
            }
            Some(routes)
        }
    }

    #[cfg(target_os = "linux")]
    pub(super) mod desktop {
        use super::super::{authority, system_proxy, Upstream};
        use gio::prelude::*;
        use hyper_util::client::proxy::matcher::Matcher;
        use url::Url;

        pub(in crate::browser_proxy) fn evaluate(
            resolver: &gio::ProxyResolver,
            url: &str,
        ) -> Option<Vec<Option<Upstream>>> {
            let destination: http::Uri = url.parse().ok()?;
            let entries = resolver
                .lookup(url, gio::Cancellable::NONE)
                .map_err(|error| log::info!("browser proxy couldn't ask the desktop: {error}"))
                .ok()?;
            Some(from_entries(&entries, &destination))
        }

        /// The entries that are DIRECT or a proxy this one can chain to, in order.
        pub(in crate::browser_proxy) fn from_entries(
            entries: &[impl AsRef<str>],
            destination: &http::Uri,
        ) -> Vec<Option<Upstream>> {
            let mut routes: Vec<_> = entries
                .iter()
                .filter_map(|entry| route(entry.as_ref(), destination))
                .collect();
            // GLib lists a `socks` proxy once per version; the first that connects is the one used.
            routes.dedup_by(|later, earlier| match (earlier, later) {
                (
                    Some(Upstream::Socks { address, .. }),
                    Some(Upstream::Socks4 { address: same }),
                ) => address == same,
                (earlier, later) => earlier == later,
            });
            if routes.is_empty() {
                routes.push(None);
            }
            routes
        }

        fn route(entry: &str, destination: &http::Uri) -> Option<Option<Upstream>> {
            let proxy = Url::parse(entry).ok()?;
            let chained = |entry: String| {
                system_proxy(&[Matcher::builder().all(entry).build()], destination).map(Some)
            };
            match proxy.scheme() {
                "direct" => Some(None),
                "http" => chained(entry.to_string()),
                // GLib tries SOCKS5 first for a plain `socks` proxy.
                "socks" | "socks5" => chained(entry.replacen(proxy.scheme(), "socks5", 1)),
                "socks4" | "socks4a" => Some(Some(Upstream::Socks4 {
                    address: authority(
                        proxy.host_str()?,
                        Some(i32::from(proxy.port().unwrap_or(1080))),
                    ),
                })),
                _ => None,
            }
        }
    }

    #[cfg(windows)]
    mod winhttp {
        use super::super::{authority, Upstream, ADDRESS_TIMEOUT};
        use super::Source;
        use std::ffi::c_void;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc::{sync_channel, SyncSender};
        use std::sync::{Mutex, OnceLock};
        use std::time::{Duration, Instant};
        use windows_sys::Win32::Foundation::GlobalFree;
        use windows_sys::Win32::Networking::WinHttp::*;

        // Discovery that found nothing is remembered, so pages don't each wait on it again.
        const FAILURE_TTL: Duration = Duration::from_secs(300);
        const ERROR_IO_PENDING: u32 = 997;
        const ERROR_TIMEOUT: u32 = 1460;
        const ERROR_WINHTTP_UNABLE_TO_DOWNLOAD_SCRIPT: u32 = 12167;

        /// Lookups in flight, by the context their callback gets; one that timed out is gone.
        static PENDING: Mutex<Vec<(usize, SyncSender<u32>)>> = Mutex::new(Vec::new());

        struct Handle(*mut c_void);
        // WinHTTP handles may be used from any thread.
        unsafe impl Send for Handle {}
        unsafe impl Sync for Handle {}

        impl Drop for Handle {
            fn drop(&mut self) {
                unsafe { WinHttpCloseHandle(self.0) };
            }
        }

        /// Internet Options' automatic configuration for this user, if any is on.
        pub(in crate::browser_proxy) fn configured() -> Option<Source> {
            let mut config: WINHTTP_CURRENT_USER_IE_PROXY_CONFIG = unsafe { std::mem::zeroed() };
            if unsafe { WinHttpGetIEProxyConfigForCurrentUser(&mut config) } == 0 {
                return None;
            }
            let url = unsafe { text(config.lpszAutoConfigUrl) };
            for owned in [
                config.lpszAutoConfigUrl,
                config.lpszProxy,
                config.lpszProxyBypass,
            ] {
                if !owned.is_null() {
                    unsafe { GlobalFree(owned.cast()) };
                }
            }
            let detect = config.fAutoDetect != 0;
            (detect || url.is_some()).then_some(Source::Automatic { detect, url })
        }

        /// One asynchronous session, fetching scripts directly; WinHTTP caches what it finds in it.
        fn session() -> Option<&'static Handle> {
            static SESSION: OnceLock<Option<Handle>> = OnceLock::new();
            SESSION
                .get_or_init(|| unsafe {
                    let agent = wide("Unsloth");
                    let flags = WINHTTP_FLAG_ASYNC;
                    let handle = WinHttpOpen(
                        agent.as_ptr(),
                        WINHTTP_ACCESS_TYPE_NO_PROXY,
                        std::ptr::null(),
                        std::ptr::null(),
                        flags,
                    );
                    if handle.is_null() {
                        return None;
                    }
                    let events = WINHTTP_CALLBACK_FLAG_GETPROXYFORURL_COMPLETE
                        | WINHTTP_CALLBACK_FLAG_REQUEST_ERROR;
                    WinHttpSetStatusCallback(handle, Some(completed), events, 0);
                    Some(Handle(handle))
                })
                .as_ref()
        }

        unsafe extern "system" fn completed(
            _handle: *mut c_void,
            context: usize,
            status: u32,
            info: *mut c_void,
            _length: u32,
        ) {
            let error = match status {
                WINHTTP_CALLBACK_STATUS_GETPROXYFORURL_COMPLETE => 0,
                WINHTTP_CALLBACK_STATUS_REQUEST_ERROR if !info.is_null() => unsafe {
                    (*info.cast::<WINHTTP_ASYNC_RESULT>()).dwError
                },
                _ => return,
            };
            let Ok(mut pending) = PENDING.lock() else {
                return;
            };
            if let Some(index) = pending.iter().position(|(id, _)| *id == context) {
                let _ = pending.swap_remove(index).1.send(error);
            }
        }

        /// The entries that are DIRECT or a proxy this one can chain to, in order.
        pub(super) fn evaluate(
            detect: bool,
            script: Option<&str>,
            url: &str,
        ) -> Option<Vec<Option<Upstream>>> {
            static FAILED: Mutex<Option<(String, Instant)>> = Mutex::new(None);
            let key = format!("{detect} {script:?}");
            let failed = FAILED.lock().ok()?;
            if failed
                .as_ref()
                .is_some_and(|(failed, at)| *failed == key && at.elapsed() < FAILURE_TTL)
            {
                return None;
            }
            drop(failed);
            match resolve(detect, script, url) {
                Ok(routes) => Some(routes),
                Err(error) => {
                    log::info!("browser proxy couldn't use the automatic proxy setup: {error}");
                    let missing = [
                        ERROR_WINHTTP_AUTODETECTION_FAILED,
                        ERROR_WINHTTP_UNABLE_TO_DOWNLOAD_SCRIPT,
                    ];
                    if missing.contains(&error) {
                        *FAILED.lock().ok()? = Some((key, Instant::now()));
                    }
                    None
                }
            }
        }

        fn resolve(
            detect: bool,
            script: Option<&str>,
            url: &str,
        ) -> Result<Vec<Option<Upstream>>, u32> {
            let (url, script) = (wide(url), script.map(wide));
            let session = session().ok_or(0u32)?;
            let mut resolver = std::ptr::null_mut();
            let error = unsafe { WinHttpCreateProxyResolver(session.0, &mut resolver) };
            if error != 0 {
                return Err(error);
            }
            let resolver = Handle(resolver);
            let mut flags = 0;
            if detect {
                flags |= WINHTTP_AUTOPROXY_AUTO_DETECT;
            }
            if script.is_some() {
                flags |= WINHTTP_AUTOPROXY_CONFIG_URL;
            }
            let options = WINHTTP_AUTOPROXY_OPTIONS {
                dwFlags: flags,
                dwAutoDetectFlags: if detect {
                    WINHTTP_AUTO_DETECT_TYPE_DHCP | WINHTTP_AUTO_DETECT_TYPE_DNS_A
                } else {
                    0
                },
                lpszAutoConfigUrl: script
                    .as_ref()
                    .map_or(std::ptr::null(), |script| script.as_ptr()),
                lpvReserved: std::ptr::null_mut(),
                dwReserved: 0,
                fAutoLogonIfChallenged: 1,
            };
            static NEXT: AtomicUsize = AtomicUsize::new(1);
            let context = NEXT.fetch_add(1, Ordering::Relaxed);
            let (sender, done) = sync_channel(1);
            PENDING.lock().map_err(|_| 0u32)?.push((context, sender));
            let started =
                unsafe { WinHttpGetProxyForUrlEx(resolver.0, url.as_ptr(), &options, context) };
            let error = match started {
                ERROR_IO_PENDING => done.recv_timeout(ADDRESS_TIMEOUT).unwrap_or(ERROR_TIMEOUT),
                error => error,
            };
            if let Ok(mut pending) = PENDING.lock() {
                pending.retain(|(id, _)| *id != context);
            }
            if error != 0 {
                return Err(error);
            }
            let mut result = WINHTTP_PROXY_RESULT {
                cEntries: 0,
                pEntries: std::ptr::null_mut(),
            };
            let error = unsafe { WinHttpGetProxyResult(resolver.0, &mut result) };
            if error != 0 {
                return Err(error);
            }
            let entries = match result.pEntries.is_null() {
                true => &[][..],
                false => unsafe {
                    std::slice::from_raw_parts(result.pEntries, result.cEntries as usize)
                },
            };
            let mut routes: Vec<_> = entries.iter().filter_map(route).collect();
            unsafe { WinHttpFreeProxyResult(&mut result) };
            // An empty answer means direct, as browsers read one.
            if routes.is_empty() {
                routes.push(None);
            }
            Ok(routes)
        }

        fn route(entry: &WINHTTP_PROXY_RESULT_ENTRY) -> Option<Option<Upstream>> {
            if entry.fProxy == 0 {
                return Some(None);
            }
            let host = unsafe { text(entry.pwszProxy) }?;
            let address = authority(&host, Some(i32::from(entry.ProxyPort)));
            match entry.ProxyScheme {
                WINHTTP_INTERNET_SCHEME_HTTP => Some(Some(Upstream::Http {
                    address,
                    auth: None,
                })),
                WINHTTP_INTERNET_SCHEME_SOCKS => Some(Some(Upstream::Socks4 { address })),
                // An HTTPS proxy wants TLS to the proxy itself, which this one doesn't speak.
                _ => None,
            }
        }

        fn wide(text: &str) -> Vec<u16> {
            text.encode_utf16().chain(Some(0)).collect()
        }

        unsafe fn text(pointer: *const u16) -> Option<String> {
            if pointer.is_null() {
                return None;
            }
            let length = (0..)
                .take_while(|&i| unsafe { *pointer.add(i) } != 0)
                .count();
            String::from_utf16(unsafe { std::slice::from_raw_parts(pointer, length) }).ok()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::task::JoinHandle;

    #[test]
    fn parses_tunnels_and_plain_requests() {
        assert_eq!(
            parse("CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n"),
            Some(Request {
                host: "example.com".into(),
                port: 443,
                forward: None,
                cleartext: false,
            })
        );
        assert_eq!(
            parse("CONNECT [::1]:8888 HTTP/1.1\r\n\r\n").unwrap().host,
            "[::1]"
        );
        let plain = parse(
            "GET http://example.com/a?b=1 HTTP/1.1\r\nHost: example.com\r\nProxy-Connection: keep-alive\r\nConnection: keep-alive\r\nAccept: */*\r\n\r\n",
        )
        .unwrap();
        assert_eq!((plain.host.as_str(), plain.port), ("example.com", 80));
        let forward = plain.forward.unwrap();
        assert_eq!(
            forward.head(None),
            "GET /a?b=1 HTTP/1.1\r\nHost: example.com\r\nAccept: */*\r\nConnection: close\r\n\r\n"
        );
        let proxy = Upstream::Http {
            address: "127.0.0.1:9".into(),
            auth: Some("Basic dTpw".into()),
        };
        assert_eq!(
            forward.head(Some(&proxy)),
            "GET http://example.com/a?b=1 HTTP/1.1\r\nHost: example.com\r\nAccept: */*\r\nProxy-Authorization: Basic dTpw\r\nConnection: close\r\n\r\n"
        );
        assert_eq!(parse("GET /relative HTTP/1.1\r\n\r\n"), None);
        assert_eq!(parse("GET https://example.com/ HTTP/1.1\r\n\r\n"), None);
    }

    #[tokio::test]
    async fn private_hosts_are_refused_however_written() {
        for host in [
            "localhost",
            "127.0.0.1",
            "127.1",
            "[::1]",
            "[::ffff:7f00:1]",
            "10.0.0.1",
            "printer.local",
            "wiki.corp",
            "router.home",
            "198.18.0.1",
            "3323068417",
            "[::ffff:c612:1]",
        ] {
            assert_eq!(
                connect(host, 80).await.err(),
                Some(Refusal::Private),
                "{host}"
            );
        }
    }

    #[tokio::test]
    async fn fake_ip_answers_pass_only_behind_a_fake_ip_resolver() {
        let ips =
            |list: &[&str]| -> Vec<IpAddr> { list.iter().map(|ip| ip.parse().unwrap()).collect() };
        let fake = ips(&["198.18.0.24"]);
        let edges = ips(&["198.19.255.254", "198.17.255.255", "198.20.0.1"]);
        for (resolver, fake_refused) in [(true, false), (false, true)] {
            *FAKE_IP_RESOLVER.lock().unwrap() = Some((Instant::now(), resolver));
            assert_eq!(answers_are_private(&fake).await, fake_refused, "{resolver}");
            assert_eq!(
                answers_are_private(&edges).await,
                fake_refused,
                "{resolver}"
            );
            for private in ["10.0.0.1", "192.168.1.1", "127.0.0.1", "::ffff:198.18.0.1"] {
                assert!(
                    answers_are_private(&ips(&["198.18.0.7", private])).await,
                    "{private}"
                );
            }
        }
        *FAKE_IP_RESOLVER.lock().unwrap() = None;
    }

    #[tokio::test]
    async fn the_proxy_answers_a_private_tunnel_with_403() {
        let address = tokio::task::spawn_blocking(address).await.unwrap().unwrap();
        let mut client = TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"CONNECT 127.0.0.1:22 HTTP/1.1\r\n\r\n")
            .await
            .unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 403"), "{response}");
    }

    /// The port of the proxy `per_protocol` picks for `url`, given proxy variables `env`.
    fn port_for(server: &str, bypass: &str, env: &[(&str, &str)], url: &str) -> Option<u16> {
        let env = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, v)| v.to_string())
        };
        let intercept = per_protocol(server, bypass, env)?.intercept(&url.parse().unwrap())?;
        intercept.uri().port_u16()
    }

    #[test]
    fn windows_per_protocol_proxies_are_split() {
        let both = "http=127.0.0.1:1; HTTPS=127.0.0.1:2;socks=127.0.0.1:3";
        let corp = "*.corp.example;<local>";
        let (http, https) = ("http://a.example/", "https://a.example/");
        assert!(per_protocol("127.0.0.1:7890", "", |_| None).is_none());
        let (wiki, http_wiki) = ("https://wiki.corp.example/", "http://wiki.corp.example/");
        for (url, port) in [(http, Some(1)), (https, Some(2)), (wiki, None)] {
            assert_eq!(port_for(both, corp, &[], url), port, "{url}");
        }
        // Each variable overrides only its own setting, and only when not empty.
        let http_var = [("HTTP_PROXY", "http://127.0.0.1:9"), ("HTTPS_PROXY", "")];
        assert_eq!(port_for(both, "", &http_var, http), Some(9));
        assert_eq!(port_for(both, "", &http_var, https), Some(2));
        let https_var = [
            ("HTTPS_PROXY", "http://127.0.0.1:8"),
            ("NO_PROXY", "a.example"),
        ];
        for (url, port) in [(https, None), (wiki, Some(8)), (http_wiki, Some(1))] {
            assert_eq!(port_for(both, corp, &https_var, url), port, "{url}");
        }
        let all_var = [("ALL_PROXY", "http://127.0.0.1:7")];
        assert_eq!(port_for("http=127.0.0.1:1", "", &all_var, https), Some(7));
    }

    // A name local DNS can't answer, so only a request sent on by name gets through.
    const UNRESOLVABLE: &str = "studio-proxy-test.invalid";

    fn tunnel(target: &str) -> String {
        format!("CONNECT {target} HTTP/1.1\r\n\r\n")
    }

    /// A system proxy that records the request head it gets and sends `answer`.
    async fn system_proxy_answering(answer: &'static str) -> (Matcher, JoinHandle<Option<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://u:p@{}", listener.local_addr().unwrap());
        let seen = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.ok()?;
            let (head, _) = read_head(&mut stream).await?;
            stream.write_all(answer.as_bytes()).await.ok()?;
            Some(head)
        });
        let proxies = Matcher::builder().http(&address).https(&address).build();
        (proxies, seen)
    }

    /// As `system_proxy_answering`, with a proxy set for `scheme` only, as with System Settings'
    /// "Web Proxy (HTTP)" or "Secure Web Proxy (HTTPS)" alone.
    async fn proxy_answering_for(
        scheme: &str,
        answer: &'static str,
    ) -> (Matcher, JoinHandle<Option<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://u:p@{}", listener.local_addr().unwrap());
        let seen = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.ok()?;
            let (head, _) = read_head(&mut stream).await?;
            stream.write_all(answer.as_bytes()).await.ok()?;
            Some(head)
        });
        let proxies = match scheme {
            "http" => Matcher::builder().http(&address),
            _ => Matcher::builder().https(&address),
        };
        (proxies.build(), seen)
    }

    async fn exchange(proxies: Matcher, request: &str) -> String {
        exchange_all(vec![proxies], request).await
    }

    async fn exchange_all(fixed: Vec<Matcher>, request: &str) -> String {
        let proxies = Proxies {
            fixed,
            ..Default::default()
        };
        exchange_via(proxies, request).await
    }

    async fn exchange_via(proxies: Proxies, request: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        tokio::spawn(serve(listener.accept().await.unwrap().0, proxies));
        client.write_all(request.as_bytes()).await.unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn requests_go_through_the_system_proxy_by_name() {
        let target = format!("{UNRESOLVABLE}:443");
        let auth = "Proxy-Authorization: Basic dTpw";
        let (proxies, seen) = system_proxy_answering("HTTP/1.1 200 OK\r\n\r\nhello").await;
        let response = exchange(proxies, &tunnel(&target)).await;
        assert_eq!(response, "HTTP/1.1 200 Connection Established\r\n\r\nhello");
        let expected = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n{auth}\r\n\r\n");
        assert_eq!(seen.await.unwrap().as_deref(), Some(expected.as_str()));
        let (proxies, seen) = system_proxy_answering("HTTP/1.1 200 OK\r\n\r\nhello").await;
        let url = format!("http://{UNRESOLVABLE}/a");
        let response = exchange(proxies, &format!("GET {url} HTTP/1.1\r\n\r\n")).await;
        assert_eq!(response, "HTTP/1.1 200 OK\r\n\r\nhello");
        let expected = format!("GET {url} HTTP/1.1\r\n{auth}\r\nConnection: close\r\n\r\n");
        assert_eq!(seen.await.unwrap().as_deref(), Some(expected.as_str()));
    }

    /// Opens a tunnel to `target`, then sends `inner` through it once the tunnel is up.
    async fn exchange_in_tunnel(proxies: Matcher, target: &str, inner: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let proxies = Proxies {
            fixed: vec![proxies],
            ..Default::default()
        };
        tokio::spawn(serve(listener.accept().await.unwrap().0, proxies));
        client.write_all(tunnel(target).as_bytes()).await.unwrap();
        let mut established = vec![0u8; ESTABLISHED.len()];
        client.read_exact(&mut established).await.unwrap();
        assert_eq!(established, ESTABLISHED);
        client.write_all(inner.as_bytes()).await.unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn plain_http_tunnels_are_forwarded_as_http_requests() {
        let (proxies, seen) = proxy_answering_for("http", "HTTP/1.1 200 OK\r\n\r\nhello").await;
        let host = format!("Host: {UNRESOLVABLE}\r\n");
        let inner = format!("GET /a?q=1 HTTP/1.1\r\n{host}Connection: keep-alive\r\n\r\n");
        let response = exchange_in_tunnel(proxies, &format!("{UNRESOLVABLE}:80"), &inner).await;
        assert_eq!(response, "HTTP/1.1 200 OK\r\n\r\nhello");
        let auth = "Proxy-Authorization: Basic dTpw";
        let expected = format!(
            "GET http://{UNRESOLVABLE}/a?q=1 HTTP/1.1\r\n{host}{auth}\r\nConnection: close\r\n\r\n"
        );
        assert_eq!(seen.await.unwrap().as_deref(), Some(expected.as_str()));
    }

    #[tokio::test]
    async fn websocket_tunnels_to_port_80_stay_tunnels() {
        // A `ws://` socket takes the HTTP proxy, which is all that is set here.
        let (proxies, seen) = proxy_answering_for("http", "HTTP/1.1 200 OK\r\n\r\n").await;
        let inner = "GET /ws HTTP/1.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n";
        let target = format!("{UNRESOLVABLE}:80");
        // The proxy took the tunnel, so no refusal comes back.
        assert_eq!(exchange_in_tunnel(proxies, &target, inner).await, "");
        let auth = "Proxy-Authorization: Basic dTpw";
        let expected = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n{auth}\r\n\r\n");
        assert_eq!(seen.await.unwrap().as_deref(), Some(expected.as_str()));
    }

    #[tokio::test]
    async fn tls_tunnels_to_port_80_take_the_https_proxy() {
        let (proxies, seen) = proxy_answering_for("https", "HTTP/1.1 200 OK\r\n\r\n").await;
        let target = format!("{UNRESOLVABLE}:80");
        let hello = "\u{16}\u{3}\u{1}\u{0}\u{5}hello";
        assert_eq!(exchange_in_tunnel(proxies, &target, hello).await, "");
        let auth = "Proxy-Authorization: Basic dTpw";
        let expected = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n{auth}\r\n\r\n");
        assert_eq!(seen.await.unwrap().as_deref(), Some(expected.as_str()));
    }

    #[tokio::test]
    async fn private_hosts_and_refused_tunnels_fail_through_the_system_proxy() {
        let (proxies, _) = system_proxy_answering("HTTP/1.1 200 OK\r\n\r\n").await;
        let response = exchange(proxies, "CONNECT printer.local:80 HTTP/1.1\r\n\r\n").await;
        assert!(response.starts_with("HTTP/1.1 403"), "{response}");
        let (proxies, _) = system_proxy_answering("HTTP/1.1 407 Proxy Auth\r\n\r\n").await;
        let response = exchange(proxies, &tunnel(&format!("{UNRESOLVABLE}:443"))).await;
        assert!(response.starts_with("HTTP/1.1 502"), "{response}");
    }

    async fn dev_server(body: &'static str) -> (u16, JoinHandle<Option<String>>) {
        let listener = (20_000..32_000)
            .find_map(|port| std::net::TcpListener::bind(("127.0.0.1", port)).ok())
            .unwrap();
        listener.set_nonblocking(true).unwrap();
        let listener = TcpListener::from_std(listener).unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.ok()?;
            let (head, _) = read_head(&mut stream).await?;
            let answer = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(answer.as_bytes()).await.ok()?;
            Some(head)
        });
        (port, seen)
    }

    #[tokio::test]
    async fn a_granted_local_origin_is_dialled_as_written_and_no_other_local_address_is() {
        let (port, server) = dev_server("dev app").await;
        let target = format!("127.0.0.1:{port}");
        let request = format!("{}GET /data.json HTTP/1.1\r\n\r\n", tunnel(&target));
        let response = exchange_via(Proxies::default(), &request).await;
        assert!(response.starts_with("HTTP/1.1 403"), "{response}");

        let origin = Url::parse(&format!("http://{target}/")).unwrap().origin();
        let local = LocalProxy::new(origin).unwrap();
        let response = exchange_via(Proxies::default(), &request).await;
        assert!(response.starts_with("HTTP/1.1 403"), "{response}");
        let mut client = TcpStream::connect(local.address).await.unwrap();
        client.write_all(request.as_bytes()).await.unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        assert!(
            response.starts_with("HTTP/1.1 200 Connection Established"),
            "{response}"
        );
        assert!(response.ends_with("\r\n\r\ndev app"), "{response}");
        assert!(server.await.unwrap().unwrap().starts_with("GET /data.json"));

        // Only that host as written and that port: `localhost` resolves to it, and stays refused.
        for other in [
            format!("localhost:{port}"),
            format!("localhost.:{port}"),
            format!("[::1]:{port}"),
            format!("127.0.0.2:{port}"),
            format!("127.0.0.1:{}", port + 1),
        ] {
            let mut client = TcpStream::connect(local.address).await.unwrap();
            client.write_all(tunnel(&other).as_bytes()).await.unwrap();
            let mut response = String::new();
            client.read_to_string(&mut response).await.unwrap();
            assert!(response.starts_with("HTTP/1.1 403"), "{other}: {response}");
        }
    }

    #[tokio::test]
    async fn a_granted_origins_websocket_sent_in_absolute_form_is_upgraded() {
        let listener = (20_000..32_000)
            .find_map(|port| std::net::TcpListener::bind(("127.0.0.1", port)).ok())
            .unwrap();
        listener.set_nonblocking(true).unwrap();
        let listener = TcpListener::from_std(listener).unwrap();
        let port = listener.local_addr().unwrap().port();
        let origin = Url::parse(&format!("http://127.0.0.1:{port}/"))
            .unwrap()
            .origin();
        let local = LocalProxy::new(origin).unwrap();
        let mut client = TcpStream::connect(local.address).await.unwrap();
        client
            .write_all(
                format!(
                    "GET ws://127.0.0.1:{port}/hmr HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
                     Connection: Upgrade\r\nUpgrade: websocket\r\n\r\n"
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let (mut server, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
            .await
            .expect("the handshake never reached the server")
            .unwrap();
        let (head, _) = read_head(&mut server).await.unwrap();
        assert!(head.starts_with("GET /hmr HTTP/1.1\r\n"), "{head}");
        assert!(head.contains("Upgrade: websocket\r\n"), "{head}");
        assert!(head.contains("Connection: Upgrade\r\n"), "{head}");
        assert!(!head.contains("Connection: close"), "{head}");
        let switched = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\nfrom server";
        server.write_all(switched).await.unwrap();
        let mut answer = vec![0; switched.len()];
        client.read_exact(&mut answer).await.unwrap();
        assert_eq!(answer, switched);
        client.write_all(b"from page").await.unwrap();
        let mut data = [0; 9];
        server.read_exact(&mut data).await.unwrap();
        assert_eq!(&data, b"from page");
    }

    #[tokio::test]
    async fn local_proxies_are_isolated_and_closing_one_cancels_live_tunnels() {
        let listener = (20_000..32_000)
            .find_map(|port| std::net::TcpListener::bind(("127.0.0.1", port)).ok())
            .unwrap();
        listener.set_nonblocking(true).unwrap();
        let listener = TcpListener::from_std(listener).unwrap();
        let port = listener.local_addr().unwrap().port();
        let origin = Url::parse(&format!("http://127.0.0.1:{port}/"))
            .unwrap()
            .origin();
        let owner = LocalProxy::new(origin).unwrap();
        let other =
            LocalProxy::new(Url::parse("http://localhost:17871/").unwrap().origin()).unwrap();
        let target = format!("127.0.0.1:{port}");
        let mut denied = TcpStream::connect(other.address).await.unwrap();
        denied.write_all(tunnel(&target).as_bytes()).await.unwrap();
        let mut response = String::new();
        denied.read_to_string(&mut response).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 403"));

        let address = owner.address;
        let mut client = TcpStream::connect(address).await.unwrap();
        client.write_all(tunnel(&target).as_bytes()).await.unwrap();
        let (mut server, _) = listener.accept().await.unwrap();
        let mut established = vec![0; ESTABLISHED.len()];
        client.read_exact(&mut established).await.unwrap();
        assert_eq!(established, ESTABLISHED);
        client.write_all(b"before close").await.unwrap();
        let mut data = [0; 12];
        server.read_exact(&mut data).await.unwrap();
        assert_eq!(&data, b"before close");
        drop(owner);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), server.read(&mut data))
                .await
                .unwrap()
                .unwrap(),
            0
        );
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), client.read(&mut data))
                .await
                .unwrap()
                .unwrap(),
            0
        );
        assert!(TcpStream::connect(address).await.is_err());
    }

    /// A SOCKS5 proxy that wants `u`/`p`, records the CONNECT it gets, answers with a domain-typed
    /// bound address, sends "hello", then (with `request`) records the request head sent next.
    async fn socks_proxy(request: bool) -> (String, JoinHandle<(String, u16, String)>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let seen = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut greeting = [0u8; 4];
            stream.read_exact(&mut greeting).await.unwrap();
            assert_eq!(greeting, [5, 2, 0, 2]);
            stream.write_all(&[5, 2]).await.unwrap();
            let mut login = [0u8; 5];
            stream.read_exact(&mut login).await.unwrap();
            assert_eq!(login, [1, 1, b'u', 1, b'p']);
            stream.write_all(&[1, 0]).await.unwrap();
            let mut connect = [0u8; 5];
            stream.read_exact(&mut connect).await.unwrap();
            assert_eq!(connect[..4], [5, 1, 0, 3]);
            let mut name = vec![0u8; usize::from(connect[4])];
            stream.read_exact(&mut name).await.unwrap();
            let mut port = [0u8; 2];
            stream.read_exact(&mut port).await.unwrap();
            stream
                .write_all(&[5, 0, 0, 3, 4, b'b', b'o', b'u', b'n', 0, 80])
                .await
                .unwrap();
            stream.write_all(b"hello").await.unwrap();
            let sent = if request {
                tokio::time::timeout(Duration::from_secs(5), read_head(&mut stream))
                    .await
                    .unwrap()
                    .unwrap()
                    .0
            } else {
                String::new()
            };
            (
                String::from_utf8(name).unwrap(),
                u16::from_be_bytes(port),
                sent,
            )
        });
        (address, seen)
    }

    #[test]
    fn proxy_settings_bracket_ipv6_hosts() {
        assert_eq!(authority("::1", Some(1080)), "[::1]:1080");
        assert_eq!(authority("::1", None), "[::1]");
        assert_eq!(authority("proxy.example", Some(1080)), "proxy.example:1080");
        let proxies = socks_fallback(&authority("::1", Some(1080)), "");
        assert!(proxies
            .intercept(&"https://example.com/".parse().unwrap())
            .is_some());
    }

    #[tokio::test]
    async fn socks4_tunnels_name_the_host() {
        for (answer, opened) in [(0x5a, true), (0x5b, false)] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap().to_string();
            let seen = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut connect = vec![0u8; 10 + UNRESOLVABLE.len()];
                stream.read_exact(&mut connect).await.unwrap();
                stream
                    .write_all(&[0, answer, 0, 0, 0, 0, 0, 0])
                    .await
                    .unwrap();
                connect
            });
            let request = parse(&tunnel(&format!("{UNRESOLVABLE}:443"))).unwrap();
            let result = connect_via(&Upstream::Socks4 { address }, &request).await;
            assert_eq!(result.is_ok(), opened, "{answer}");
            let mut expected = vec![4, 1, 1, 187, 0, 0, 0, 1, 0];
            expected.extend_from_slice(UNRESOLVABLE.as_bytes());
            expected.push(0);
            assert_eq!(seen.await.unwrap(), expected);
        }
    }

    #[tokio::test]
    async fn tunnels_go_through_a_socks_proxy_by_name() {
        let (address, seen) = socks_proxy(false).await;
        let proxies = Matcher::builder()
            .all(format!("socks5://u:p@{address}"))
            .build();
        let response = exchange(proxies, &tunnel(&format!("{UNRESOLVABLE}:443"))).await;
        assert_eq!(response, "HTTP/1.1 200 Connection Established\r\n\r\nhello");
        let (name, port, _) = seen.await.unwrap();
        assert_eq!((name.as_str(), port), (UNRESOLVABLE, 443));
    }

    #[tokio::test]
    async fn plain_requests_through_socks_stay_origin_form() {
        let (address, seen) = socks_proxy(true).await;
        // The SOCKS fallback covers what the first settings leave out.
        let proxies = vec![
            Matcher::builder().build(),
            socks_fallback(&format!("u:p@{address}"), ""),
        ];
        let response = exchange_all(
            proxies,
            &format!("GET http://{UNRESOLVABLE}/a HTTP/1.1\r\nHost: {UNRESOLVABLE}\r\n\r\n"),
        )
        .await;
        assert_eq!(response, "hello");
        let (name, port, sent) = seen.await.unwrap();
        assert_eq!((name.as_str(), port), (UNRESOLVABLE, 80));
        assert!(sent.starts_with("GET /a HTTP/1.1\r\n"), "{sent}");
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn pac_scripts_decide_before_fixed_settings() {
        let script = r#"function FindProxyForURL(url, host) {
            if (host == "h.example") return "PROXY 127.0.0.1:1; DIRECT";
            if (host == "s.example") return "SOCKS 127.0.0.1:2";
            return "DIRECT";
        }"#;
        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let location = format!("http://{}/proxy.pac", server.local_addr().unwrap());
        // Answers once, so only a cached script serves the later pages.
        tokio::spawn(async move {
            let (mut stream, _) = server.accept().await.unwrap();
            read_head(&mut stream).await;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                script.len()
            );
            stream.write_all((head + script).as_bytes()).await.unwrap();
        });
        let env = |name: &str| match name {
            "HTTP_PROXY" => Some("http://127.0.0.1:8".to_string()),
            "NO_PROXY" => Some("n.example".to_string()),
            _ => None,
        };
        let pac = |source| Some(pac::Pac::new(source, &env));
        let fixed = || vec![Matcher::builder().all("http://127.0.0.1:9").build()];
        let proxies = Proxies {
            pac: pac(pac::Source::Url(location)),
            fixed: fixed(),
        };
        let to = |host| parse(&tunnel(&format!("{host}:443"))).unwrap();
        let http = |address: &str| {
            Some(Upstream::Http {
                address: address.into(),
                auth: None,
            })
        };
        assert_eq!(
            routes(&proxies, &to("h.example")).await,
            [http("127.0.0.1:1"), None]
        );
        let socks = Upstream::Socks {
            address: "127.0.0.1:2".into(),
            auth: None,
        };
        assert_eq!(routes(&proxies, &to("s.example")).await, [Some(socks)]);
        assert_eq!(routes(&proxies, &to("d.example")).await, [None]);
        // Proxy variables and NO_PROXY take precedence, as over the other settings.
        let plain = parse("GET http://h.example/ HTTP/1.1\r\n\r\n").unwrap();
        assert_eq!(routes(&proxies, &plain).await, [http("127.0.0.1:9")]);
        assert_eq!(
            routes(&proxies, &to("n.example")).await,
            [http("127.0.0.1:9")]
        );
        let broken = Proxies {
            pac: pac(pac::Source::Script("not a script(".into())),
            fixed: fixed(),
        };
        assert_eq!(
            routes(&broken, &to("h.example")).await,
            [http("127.0.0.1:9")]
        );
        // The script sees the URL Safari would give it.
        let by_url = "function FindProxyForURL(url) { return url == 'https://h.example/' ? 'PROXY 127.0.0.1:3' : 'DIRECT'; }";
        let no_env = |_: &str| None;
        let proxies = Proxies {
            pac: Some(pac::Pac::new(pac::Source::Script(by_url.into()), &no_env)),
            fixed: vec![],
        };
        assert_eq!(
            routes(&proxies, &to("h.example")).await,
            [http("127.0.0.1:3")]
        );
        // A proxy that can't be reached gives way to the next route; one that refuses doesn't.
        let address = |(proxies, _): (Matcher, JoinHandle<_>)| {
            let proxy = proxies
                .intercept(&"https://a.example/".parse().unwrap())
                .unwrap();
            format!(
                "{}:{}",
                proxy.uri().host().unwrap(),
                proxy.uri().port_u16().unwrap()
            )
        };
        let working = address(system_proxy_answering("HTTP/1.1 200 OK\r\n\r\nhello").await);
        let refusing = address(system_proxy_answering("HTTP/1.1 407 Proxy Auth\r\n\r\n").await);
        for (first, expected) in [
            (refusing.as_str(), "HTTP/1.1 502"),
            ("127.0.0.1:1", "HTTP/1.1 200"),
        ] {
            let script = format!(
                "function FindProxyForURL() {{ return 'PROXY {first}; PROXY {working}'; }}"
            );
            let proxies = Proxies {
                pac: pac(pac::Source::Script(script.into())),
                fixed: vec![],
            };
            let response = exchange_via(proxies, &tunnel(&format!("{UNRESOLVABLE}:443"))).await;
            assert!(response.starts_with(expected), "{first}: {response}");
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_automatic_setup_decides_before_fixed_settings() {
        let script = r#"function FindProxyForURL(url, host) {
            if (host == "h.example") return "PROXY 127.0.0.1:1; DIRECT";
            if (host == "s.example") return "SOCKS 127.0.0.1:2";
            if (url == "https://u.example/") return "PROXY 127.0.0.1:3";
            return "DIRECT";
        }"#;
        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let location = format!("http://{}/proxy.pac", server.local_addr().unwrap());
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = server.accept().await {
                read_head(&mut stream).await;
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                    script.len()
                );
                let _ = stream.write_all((head + script).as_bytes()).await;
            }
        });
        let proxies = |url: &str| Proxies {
            pac: Some(pac::Pac::new(
                pac::Source::Automatic {
                    detect: false,
                    url: Some(url.into()),
                },
                &|_: &str| None,
            )),
            fixed: vec![Matcher::builder().all("http://127.0.0.1:9").build()],
        };
        let to = |host| parse(&tunnel(&format!("{host}:443"))).unwrap();
        let http = |address: &str| {
            Some(Upstream::Http {
                address: address.into(),
                auth: None,
            })
        };
        let socks = Upstream::Socks4 {
            address: "127.0.0.1:2".into(),
        };
        let found = proxies(&location);
        for (host, expected) in [
            ("h.example", vec![http("127.0.0.1:1"), None]),
            ("s.example", vec![Some(socks)]),
            ("u.example", vec![http("127.0.0.1:3")]),
            ("d.example", vec![None]),
        ] {
            assert_eq!(routes(&found, &to(host)).await, expected, "{host}");
        }
        let missing = proxies("http://127.0.0.1:1/proxy.pac");
        assert_eq!(
            routes(&missing, &to("h.example")).await,
            [http("127.0.0.1:9")]
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_desktop_proxies_map_to_routes() {
        let routes = |proxy: &str, host: &str| {
            let resolver = gio::SimpleProxyResolver::new(Some(proxy), &["n.example"][..]);
            pac::desktop::evaluate(&resolver, &format!("https://{host}/")).unwrap()
        };
        assert_eq!(routes("http://127.0.0.1:1", "n.example"), [None]);
        let socks4 = Upstream::Socks4 {
            address: "127.0.0.1:3".into(),
        };
        assert_eq!(routes("socks4a://127.0.0.1:3", "h.example"), [Some(socks4)]);
        // In order: GLib's expanded `socks`, libproxy's unexpanded one, then a DIRECT fallback.
        let entries = [
            "http://u:p@127.0.0.1:1",
            "socks5://127.0.0.1:2",
            "socks4a://127.0.0.1:2",
            "socks://127.0.0.1:4",
            "direct://",
        ];
        let http = Upstream::Http {
            address: "127.0.0.1:1".into(),
            auth: Some("Basic dTpw".into()),
        };
        let socks = |address: &str| {
            Some(Upstream::Socks {
                address: address.into(),
                auth: None,
            })
        };
        let to = "https://h.example/".parse().unwrap();
        assert_eq!(
            pac::desktop::from_entries(&entries, &to),
            [Some(http), socks("127.0.0.1:2"), socks("127.0.0.1:4"), None]
        );
        // Proxy variables replace the desktop's settings rather than mixing with them.
        let env = |name: &str| (name == "https_proxy").then(|| "http://127.0.0.1:8".to_string());
        assert!(linux_proxies(env).pac.is_none());
        assert!(linux_proxies(|_: &str| None).pac.is_some());
    }
}
