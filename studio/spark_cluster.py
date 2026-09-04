# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA DGX Spark cluster detection and setup.

Two DGX Sparks cabled together over their ConnectX-7 QSFP ports are two
*independent hosts*, not one two-GPU machine: NVLink-C2C never leaves the
package, so ``nvidia-smi`` shows one GB10 on each and ``torch.cuda.device_count()``
is always 1. What the cable buys is a ~200 Gb/s RoCEv2 link between them, which
llama.cpp's RPC backend and any NCCL job can use.

Nothing here runs off a DGX Spark. ``is_dgx_spark()`` is the gate every entry
point calls first, and on a non-Spark host it answers from two string compares
with no I/O at all -- an x86 laptop, a Mac, a Windows box or an AMD host pays
nothing for this module existing. The shell installer has a byte-for-byte twin of
that gate (``_unsloth_is_dgx_spark`` in install.sh) so a piped install does not
even source Python to find out.

Layout of a Spark's ConnectX-7, which drives the whole design:

    rocep1s0f0   -> enp1s0f0np0    physical QSFP port 0, PCIe fn 1
    roceP2p1s0f0 -> enP2p1s0f0np0  SAME physical port 0, PCIe fn 2
    rocep1s0f1   -> enp1s0f1np1    physical QSFP port 1, PCIe fn 1
    roceP2p1s0f1 -> enP2p1s0f1np1  SAME physical port 1, PCIe fn 2

The NIC hangs off GB10 by two independent PCIe Gen5 x4 links, ~100 Gb/s each, so
NVIDIA exposes each physical port as two PCIe functions. One cable carries the
full ~200 Gb/s, but only if both functions are used, and a single TCP flow over
one function tops out near 100 Gb/s. That is why each rail needs its own /24:
one subnet cannot drive both functions.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Identity gate ────────────────────────────────────────────────────────────
# Ordered cheapest-first. Linux+aarch64 is two comparisons against values Python
# already has, so every non-Spark host returns before touching the filesystem.

_DGX_RELEASE = "/etc/dgx-release"
_DMI_PRODUCT = "/sys/class/dmi/id/product_name"
_SPARK_RE = re.compile(r"dgx[_ -]*spark", re.IGNORECASE)

_IS_SPARK_CACHE: Optional[bool] = None


def is_dgx_spark() -> bool:
    """True only on an NVIDIA DGX Spark. Cached; safe to call in hot paths.

    A DGX Spark is always Linux on aarch64, so the two cheap checks come first
    and short-circuit every other platform -- Windows, macOS, WSL on x86, and any
    x86_64 Linux box, NVIDIA or AMD or CPU-only -- before a single file is opened.
    Only an aarch64 Linux host pays for the two small reads below, and a Grace
    Hopper or Jetson box fails them too (neither advertises "DGX Spark").
    """
    global _IS_SPARK_CACHE
    if _IS_SPARK_CACHE is not None:
        return _IS_SPARK_CACHE

    result = False
    if platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"):
        for path in (_DGX_RELEASE, _DMI_PRODUCT):
            try:
                # Both files are a few hundred bytes; cap anyway so a bad mount
                # cannot make the gate expensive.
                with open(path, "r", errors = "replace") as handle:
                    if _SPARK_RE.search(handle.read(4096)):
                        result = True
                        break
            except OSError:
                continue

    _IS_SPARK_CACHE = result
    return result


# What every entry point says off a Spark. One string, so the answer cannot drift
# between commands, and so a caller can match on it.
NOT_A_SPARK = "This machine is not a DGX Spark; nothing to do."


# ── Rail discovery (pure sysfs, no subprocesses) ─────────────────────────────

_IB_ROOT = Path("/sys/class/infiniband")
_NET_ROOT = Path("/sys/class/net")


def _int_or_none(text: str) -> Optional[int]:
    return int(text) if text.isdigit() else None


def _read(path: Path, limit: int = 256) -> str:
    try:
        with open(path, "r", errors = "replace") as handle:
            return handle.read(limit).strip()
    except OSError:
        return ""


def _rail_sort_key(name: str) -> Tuple[int, int, str]:
    """Order rails by physical port, then PCIe function -- not alphabetically.

    Plain sorting puts ``roceP2p1s0f0`` (function 2) ahead of ``rocep1s0f0``
    (function 1) because uppercase P sorts first, which would make the *second*
    function the one every caller treats as primary. Ordering by (port, function)
    keeps the choice stable and matches how NVIDIA's docs name these, so
    ``NCCL_SOCKET_IFNAME`` lands on enp1s0f0np0 on every Spark.
    """
    match = re.search(r"s0f(\d+)$", name)
    port = int(match.group(1)) if match else 9
    function = 2 if name.startswith("roceP2p") else 1
    return (port, function, name)


def local_rails() -> List[Dict[str, Any]]:
    """Every RoCE device on this Spark, with its netdev, link state and IPv4s.

    Reads sysfs only -- no ``ibdev2netdev``, no ``ip``, no fork. A rail is
    "usable" when the IB port is ACTIVE *and* the Ethernet netdev reports
    carrier, which is exactly the pair of facts that says a cable is seated and
    trained at the far end.
    """
    rails: List[Dict[str, Any]] = []
    if not _IB_ROOT.is_dir():
        return rails

    for dev in sorted(_IB_ROOT.iterdir(), key = lambda p: _rail_sort_key(p.name)):
        port = dev / "ports" / "1"
        state = _read(port / "state")  # e.g. "4: ACTIVE"
        phys = _read(port / "phys_state")  # e.g. "5: LinkUp"
        netdev = ""
        # ConnectX exposes the owning netdev under the device's own tree.
        gid_attr = dev / "ports" / "1" / "gid_attrs" / "ndevs" / "0"
        netdev = _read(gid_attr)
        if not netdev:
            # Fall back to matching by parent PCI device.
            try:
                dev_pci = (dev / "device").resolve()
                for candidate in _NET_ROOT.iterdir():
                    try:
                        if (candidate / "device").resolve() == dev_pci:
                            netdev = candidate.name
                            break
                    except OSError:
                        continue
            except OSError:
                pass

        carrier = _read(_NET_ROOT / netdev / "carrier") if netdev else ""
        mtu = _read(_NET_ROOT / netdev / "mtu") if netdev else ""
        rails.append(
            {
                "ib_device": dev.name,
                "netdev": netdev,
                "carrier_up_count": _int_or_none(_read(_NET_ROOT / netdev / "carrier_up_count"))
                if netdev
                else None,
                "ib_active": "ACTIVE" in state.upper(),
                "link_up": "LINKUP" in phys.upper().replace(" ", ""),
                "carrier": carrier == "1",
                "mtu": int(mtu) if mtu.isdigit() else None,
                "ipv4": _netdev_ipv4(netdev) if netdev else [],
            }
        )
    return rails


def _netdev_ipv4(netdev: str) -> List[str]:
    """IPv4 addresses on a netdev. Uses `ip` when present, else returns []."""
    ip_bin = shutil.which("ip")
    if not ip_bin:
        return []
    try:
        out = subprocess.run(
            [ip_bin, "-4", "-o", "addr", "show", "dev", netdev],
            capture_output = True,
            text = True,
            timeout = 5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return re.findall(r"inet\s+(\d+\.\d+\.\d+\.\d+)/", out)


def cabled_rails() -> List[Dict[str, Any]]:
    """Rails with a live cable -- the evidence another Spark is attached."""
    return [r for r in local_rails() if r["ib_active"] and r["carrier"]]


# ── Peer discovery: RoCE probe + mDNS ────────────────────────────────────────

_MDNS_TIMEOUT = 3.0
# A peer probe must be bounded and cheap: this runs in `status`, which people run
# while a job is starting. One TCP SYN to sshd per peer, sub-second, no fork.
_PEER_PROBE_PORT = 22
_PEER_PROBE_TIMEOUT = 0.75


def _ipv4_sort_key(address: str) -> Tuple[int, int, int, int, int]:
    """Numeric ordering for an IPv4 string, so .9 sorts before .10.

    Lexical ordering of addresses is the classic way a "deterministic" peer list
    silently reorders itself once a cluster grows past ten nodes, which would
    reshuffle rank assignment between runs. Non-IPv4 (a link-local IPv6 from
    avahi) sorts last rather than being dropped.
    """
    parts = address.split(".")
    if len(parts) == 4 and all(p.isdigit() and len(p) <= 3 for p in parts):
        a, b, c, d = (int(p) for p in parts)
        if max(a, b, c, d) <= 255:
            return (0, a, b, c, d)
    return (1, 0, 0, 0, 0)


def peer_reachable(
    address: str,
    port: int = _PEER_PROBE_PORT,
    timeout: float = _PEER_PROBE_TIMEOUT,
) -> Optional[bool]:
    """Is this peer answering on the network? ``None`` when we could not tell.

    A refused connection still proves the host is up and routable, which is the
    question being asked -- so it counts as reachable. Only a timeout or a routing
    failure counts as unreachable, and anything unexpected answers ``None`` rather
    than claiming a healthy peer is down.
    """
    if not address:
        return None
    try:
        conn = socket.create_connection((address, port), timeout = timeout)
    except (socket.timeout, TimeoutError):
        return False
    except ConnectionRefusedError:
        return True  # host is up; sshd merely is not listening
    except OSError:
        return False
    except Exception:
        return None
    try:
        conn.close()
    except OSError:
        pass
    return True


def discover_peers(timeout: float = _MDNS_TIMEOUT, check_reachable: bool = False) -> Dict[str, Any]:
    """Look for other Sparks: a cabled RoCE rail, plus mDNS hostnames.

    Both halves are bounded and best-effort. The RoCE probe is authoritative
    about a *cable* (it cannot name the peer); mDNS is authoritative about a
    *name* (it cannot prove the peer is cabled to us rather than merely on the
    same Wi-Fi). Reporting both lets the caller say "a cable is live and I can
    see spark-82be" without pretending either fact implies the other.

    Off a DGX Spark this returns immediately with ``is_spark: False`` and touches
    nothing -- no sysfs walk, and in particular no avahi-browse, which does exist
    on an ordinary Linux laptop and would otherwise put this module on the network
    on a machine that has no Spark at all.

    ``peers`` generalises past a single pair: every discovered Spark, plus any
    peer pinned in the saved config, deduplicated and ordered deterministically by
    address so node index N means the same host on every node of the cluster.
    ``check_reachable`` adds a bounded TCP probe per peer.
    """
    if not is_dgx_spark():
        return {
            "is_spark": False,
            "cabled_rails": [],
            "cable_present": False,
            "configured": [],
            "mdns_peers": [],
            "peers": [],
            "n_peers": 0,
            "n_nodes": 1,
            "note": "not a DGX Spark; no discovery attempted",
        }
    rails = cabled_rails()
    result: Dict[str, Any] = {
        "is_spark": True,
        "cabled_rails": rails,
        "cable_present": bool(rails),
        "configured": [r for r in rails if r["ipv4"]],
        "mdns_peers": [],
    }
    if timeout > 0:
        result["mdns_peers"] = _mdns_spark_peers(timeout)
    peers = merge_peers(result["mdns_peers"], check_reachable = check_reachable)
    result["peers"] = peers
    result["n_peers"] = len(peers)
    # This node plus its peers. The planner counts nodes, not peers, and getting
    # that off by one is how a 3-Spark cluster ends up planned as a 2-Spark one.
    result["n_nodes"] = len(peers) + 1
    return result


def configured_peers() -> List[Dict[str, str]]:
    """Peers pinned in the saved config, if any.

    mDNS is not reliable past a direct cable -- a switched fabric may not carry it,
    and a peer that is up but not advertising is invisible. A cluster larger than a
    pair is expected to be written down, so the config is a first-class source and
    not a fallback. Malformed entries are skipped rather than raising.
    """
    out: List[Dict[str, str]] = []
    raw = load_config().get("peers")
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if isinstance(entry, str):
            out.append({"hostname": entry, "address": entry, "source": "config"})
        elif isinstance(entry, dict):
            address = str(entry.get("address") or entry.get("ip") or "")
            hostname = str(entry.get("hostname") or address)
            if address or hostname:
                out.append(
                    {"hostname": hostname, "address": address or hostname, "source": "config"}
                )
    return out


def merge_peers(
    mdns: Optional[List[Dict[str, str]]] = None, check_reachable: bool = False
) -> List[Dict[str, Any]]:
    """Every known peer, deduplicated, deterministically ordered, index-stamped.

    Ordering is by address (numerically, see ``_ipv4_sort_key``) then hostname, so
    the same list comes out on every node of the cluster in the same order. That is
    what makes ``index`` usable as a node rank: an ordering that depends on which
    node ran the discovery would hand two nodes the same rank.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in list(mdns or []) + configured_peers():
        hostname = str(entry.get("hostname", ""))
        address = str(entry.get("address", ""))
        key = hostname.split(".")[0].lower() or address
        if not key:
            continue
        prev = merged.get(key)
        if prev is None:
            merged[key] = {
                "hostname": hostname or address,
                "short": key,
                "address": address,
                "source": entry.get("source", "mdns"),
            }
        elif not prev["address"] or (":" in prev["address"] and ":" not in address):
            prev["address"] = address or prev["address"]
    peers = sorted(merged.values(), key = lambda d: (_ipv4_sort_key(d["address"]), d["short"]))
    for index, peer in enumerate(peers):
        # index 0 is the FIRST PEER, not this node; this node is always node 0 of
        # the cluster and peers occupy 1..N-1.
        peer["index"] = index + 1
        peer["reachable"] = peer_reachable(peer["address"]) if check_reachable else None
    return peers


def _mdns_spark_peers(timeout: float) -> List[Dict[str, str]]:
    """`spark-*.local` hosts other than ourselves, via avahi. [] if unavailable.

    Returns every Spark it can see, not just one: a three-node cluster on a switch
    advertises three names and the caller needs all of them. Ordering is by address
    so the list is stable across nodes and across runs.
    """
    browse = shutil.which("avahi-browse")
    if not browse:
        return []
    me = socket.gethostname().split(".")[0].lower()
    try:
        proc = subprocess.run(
            [browse, "-a", "-t", "-r", "-p", "-k"],
            capture_output = True,
            text = True,
            timeout = timeout + 2,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    seen: Dict[str, Dict[str, str]] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("="):
            continue
        parts = line.split(";")
        if len(parts) < 8:
            continue
        host, addr = parts[6], parts[7]
        short = host.split(".")[0].lower()
        if not short.startswith("spark-") or short == me:
            continue
        # Prefer a routable IPv4 over a link-local IPv6 for the same host.
        prev = seen.get(short)
        if prev is None or (":" in prev["address"] and ":" not in addr):
            seen[short] = {"hostname": host, "address": addr, "source": "mdns"}
    return sorted(seen.values(), key = lambda d: (_ipv4_sort_key(d["address"]), d["hostname"]))


# ── QSFP hot-plug throttle ───────────────────────────────────────────────────
# The usual reason a two-Spark link runs at ~13-14 Gb/s instead of ~98 per rail:
# when the QSFP cable is connected after boot, the ConnectX-7 can come up with both
# PCIe domains throttled. The fix is procedural -- reboot with the cabling already
# in place -- and rebooting *either* end can clear it.
#
# It cannot be inferred from sysfs. `carrier_up_count` looks like a tempting signal
# (a link trained at boot reads 1) but it is not one: measured on a GB10 pair, a node
# sitting at carrier_up_count=7, whose link had flapped repeatedly, ran at a full
# 97.97 Gb/s per rail once the *peer* was rebooted. Counting link events says nothing
# about whether this NIC is throttled, so anything built on that counter reports a
# throttled link on a healthy machine. Only a measurement settles it.
HOTPLUG_NOTE = (
    "If the link measures far below ~98 Gb/s per rail, the usual cause is the QSFP "
    "cable having been connected after boot, which can leave the ConnectX-7 throttled. "
    "Reboot both Sparks with the cable already plugged in, then leave the cabling alone."
)


def link_carrier_events(rails: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Optional[int]]:
    """Per-rail carrier_up_count, reported as a FACT and never as a verdict.

    Useful context when reading a slow measurement (a link that has flapped a lot may
    have been recabled), but see HOTPLUG_NOTE: it does not imply the link is throttled.
    """
    rails = rails if rails is not None else cabled_rails()
    return {r["ib_device"]: r.get("carrier_up_count") for r in rails}


# ── Persisted state (idempotency) ────────────────────────────────────────────


def _studio_root() -> Path:
    for var in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser()
    return Path.home() / ".unsloth" / "studio"


def config_path() -> Path:
    return _studio_root() / "spark_cluster.json"


def load_config() -> Dict[str, Any]:
    try:
        with open(config_path(), "r") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(config: Dict[str, Any]) -> None:
    path = config_path()
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as handle:
            json.dump(config, handle, indent = 2, sort_keys = True)
        os.replace(tmp, path)
        # Peer credentials never land here, but the file names hosts and subnets.
        os.chmod(path, 0o600)
    except OSError:
        pass


def cluster_state() -> str:
    """One of: ``not_spark``, ``no_cable``, ``unconfigured``, ``configured``.

    This is what makes a re-run safe. ``configured`` means a previous setup wrote
    a config *and* the rails it named still carry IPv4, so an install that runs
    again skips straight past the prompt instead of asking a settled question.
    """
    if not is_dgx_spark():
        return "not_spark"
    rails = cabled_rails()
    if not rails:
        return "no_cable"
    config = load_config()
    if config.get("enabled") and any(r["ipv4"] for r in rails):
        return "configured"
    return "unconfigured"


# ── Recommended tuning ───────────────────────────────────────────────────────
# Defaults, applied automatically, so a user never has to know any of this.

# NVIDIA's addressing for the port-0 rail pair. Each PCIe function of the one
# physical port gets its own /24, because a single subnet can only drive one
# function and would leave half the ~200 Gb/s on the floor.
DEFAULT_SUBNETS = ("192.168.200", "192.168.201")
DEFAULT_MTU = 9000  # lifts the RoCE path MTU from 1024 to 4096


def nccl_env(rails: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """NCCL settings that are correct for GB10. Every one of these is load-bearing.

    ``NCCL_NET_GDR_LEVEL=0`` is not a tuning knob: Grace Blackwell on Spark has no
    GPUDirect RDMA, so NIC<->GPU traffic must stage through system memory. Leaving
    it unset is a documented hang at ``init_process_group``, as is
    ``NCCL_IB_GID_INDEX=0`` (that is the RoCEv1 GID; RoCEv2's IPv4 GID is index 3).
    ``NCCL_IB_MERGE_NICS=1`` is what lets NCCL drive both PCIe functions of the one
    physical port -- without it a job sits on a single rail at roughly half the link.
    """
    rails = rails if rails is not None else cabled_rails()
    hcas = ",".join(r["ib_device"] for r in rails) or "rocep1s0f0,roceP2p1s0f0"
    primary = next((r["netdev"] for r in rails if r["netdev"]), "enp1s0f0np0")
    return {
        "NCCL_SOCKET_IFNAME": primary,
        "GLOO_SOCKET_IFNAME": primary,
        "NCCL_IB_HCA": hcas,
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_MERGE_NICS": "1",
        "NCCL_CROSS_NIC": "1",
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_IB_DISABLE": "0",
    }


def apply_nccl_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Put the GB10 NCCL defaults into ``os.environ`` without overriding a user.

    Only ever fills in names the caller has not already set, so an explicit
    export from the user or a launcher always wins.
    """
    target = env if env is not None else os.environ
    applied: Dict[str, str] = {}
    if not is_dgx_spark():
        return applied
    for key, value in nccl_env().items():
        if not target.get(key):
            target[key] = value
            applied[key] = value
    return applied


# ── Health / verification ────────────────────────────────────────────────────

# NVIDIA's published dual-Spark reference is ~92-97 Gb/s per rail (~190 aggregate).
# Anything near 13-16 Gb/s is the documented kernel-6.17/ConnectX-7 firmware fault,
# not a cable problem, and system updates are the published remedy.
DEGRADED_GBPS = 40.0
EXPECTED_GBPS = 90.0


def link_health(
    peer_ip: str,
    ib_device: str,
    local_ip: str,
    seconds: int = 5,
    port: int = 18999,
) -> Dict[str, Any]:
    """Measure one rail end to end. Returns {} when it cannot be measured.

    ib_write_bw needs a server on one node and a client on the other, so this starts
    the server locally and drives the client on the peer over SSH. That means it only
    works once passwordless SSH is set up -- which the pairing step arranges anyway.
    A measurement is the only thing that can tell a throttled link from a healthy one.
    """
    if not shutil.which("ib_write_bw") or not shutil.which("ssh"):
        return {}
    server = None
    try:
        server = subprocess.Popen(
            [
                "ib_write_bw",
                "-d",
                ib_device,
                "-F",
                "-x",
                "3",
                "--report_gbits",
                "-D",
                str(seconds),
                "-s",
                "1048576",
                "-q",
                "4",
                "-p",
                str(port),
            ],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        )
        # Give the server its listening socket before the client dials in.
        import time

        time.sleep(3)
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=8",
                f"{os.environ.get('USER', 'nvidia')}@{peer_ip}",
                f"ib_write_bw -d {ib_device} -F -x 3 --report_gbits -D {seconds} "
                f"-s 1048576 -q 4 -p {port} {local_ip}",
            ],
            capture_output = True,
            text = True,
            timeout = seconds + 60,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    finally:
        if server is not None:
            try:
                server.wait(timeout = seconds + 20)
            except Exception:
                server.kill()

    gbps = None
    for line in proc.stdout.splitlines():
        fields = line.split()
        # The results row starts with the message size in bytes.
        if len(fields) >= 4 and fields[0].isdigit():
            try:
                gbps = float(fields[3])
            except ValueError:
                continue
    if gbps is None:
        return {}
    return {
        "gbps": gbps,
        "degraded": gbps < DEGRADED_GBPS,
        "expected_gbps": EXPECTED_GBPS,
    }


_WC_MARKER = "Write combining is not supported"


def write_combining_broken() -> Optional[bool]:
    """Whether mlx5 reported that ARM64 write-combining failed its boot test.

    ``True`` broken, ``False`` healthy, ``None`` *could not tell* -- and the
    difference matters. Ubuntu ships ``kernel.dmesg_restrict=1``, so an
    unprivileged read fails; answering ``False`` there would report a healthy
    link on the very machines that have this fault. Callers must treat ``None``
    as unknown and stay quiet rather than claim the link is fine.

    This is the fingerprint of the ~13 Gb/s ceiling on kernel 6.17 Sparks: the
    mlx5 write-combining probe is unreliable on Grace-class ARM64 cores, and when
    it fails the driver gives up BlueFlame doorbell batching. It is a platform
    fault -- Unsloth cannot patch around it, only name it.
    """
    dmesg = shutil.which("dmesg")
    if dmesg:
        try:
            proc = subprocess.run([dmesg], capture_output = True, text = True, timeout = 10)
            # dmesg exits 0 even when the buffer read is denied, so test output.
            if proc.stdout.strip():
                return _WC_MARKER in proc.stdout and "mlx5" in proc.stdout
        except (OSError, subprocess.SubprocessError):
            pass
    # Boot messages survive in the journal/kern.log even when dmesg is restricted.
    for log in ("/var/log/kern.log", "/var/log/dmesg"):
        try:
            with open(log, "r", errors = "replace") as handle:
                text = handle.read()
        except OSError:
            continue
        if "mlx5" in text:
            return _WC_MARKER in text
    return None


def pending_system_updates() -> List[str]:
    """DGX/ConnectX-relevant packages with an upgrade waiting.

    Unsloth never installs these itself. A degraded link is a platform fault with
    a published NVIDIA remedy (dist-upgrade + fwupdmgr + reboot), so the most
    useful thing to do is name the packages and hand the user the commands --
    a reboot is not Unsloth's to take.
    """
    apt = shutil.which("apt")
    if not apt:
        return []
    try:
        out = subprocess.run(
            [apt, "list", "--upgradable"],
            capture_output = True,
            text = True,
            timeout = 60,
            env = {**os.environ, "LC_ALL": "C"},
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    wanted = ("dgx", "mlnx", "nvidia", "linux-image", "linux-nvidia", "firmware")
    held = _held_packages()
    names = []
    for line in out.splitlines():
        pkg = line.split("/", 1)[0].strip()
        # A held package is a decision already taken, not an outstanding update.
        if pkg and pkg not in held and any(token in pkg for token in wanted):
            names.append(pkg)
    return sorted(set(names))


def _held_packages() -> set:
    """Packages pinned with `apt-mark hold`; empty set when apt-mark is unavailable."""
    apt_mark = shutil.which("apt-mark")
    if not apt_mark:
        return set()
    try:
        out = subprocess.run(
            [apt_mark, "showhold"], capture_output = True, text = True, timeout = 30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def ota_status() -> Dict[str, Any]:
    """What NVIDIA's own OTA checker says about this Spark, when it is installed.

    ``nvidia-spark-ota-check`` (from dgx-spark-ota-update-meta) is authoritative about
    whether a newer validated release exists, and it is the only thing that can tell a
    genuinely out-of-date box from one that is already fully converged. Returns {} when
    the tool is absent -- callers must not assume "no OTA" from that.
    """
    tool = shutil.which("nvidia-spark-ota-check")
    if not tool:
        return {}
    try:
        proc = subprocess.run(
            [tool, "is-ota-available"],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        return json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def update_instructions() -> List[str]:
    """What to actually do about a degraded link -- OTA state decides.

    The widely repeated advice ("dist-upgrade + fwupdmgr + reboot") only helps a Spark
    that is genuinely behind. On a box already converged with the newest OTA it changes
    nothing: measured on a GB10 running OTA2607, the upgrade carried no kernel, no GPU
    driver and no ConnectX firmware, and the link stayed at ~14 Gb/s afterwards. Telling
    that user to dist-upgrade is a reboot spent for no reason, so check first.
    """
    status = ota_status()
    if status.get("available"):
        name = status.get("name") or "a newer release"
        return [
            f"A newer DGX Spark OTA is available ({name}).",
            "Apply it with the DGX Dashboard -- NVIDIA's supported path, which keeps the",
            "firmware and driver paired. Raw `apt upgrade` can install a driver the on-box",
            "firmware rejects, which is a documented way to lose the GPU entirely.",
        ]
    if status:
        return [
            "This Spark is already on the newest validated OTA, so no update fixes this.",
            "The ~13 Gb/s ceiling is a platform-level mlx5 write-combining limitation on",
            "ARM64 Grace cores; the kernel fix is not in a released Spark kernel yet.",
            "Re-check later with: nvidia-spark-ota-check is-ota-available",
        ]
    return [
        "Check for a validated update first:  nvidia-spark-ota-check is-ota-available",
        "Apply any update through the DGX Dashboard rather than raw apt, so firmware and",
        "driver stay paired.",
    ]


# ── Distributed GGUF inference across both Sparks (llama.cpp RPC) ────────────
# This is the one thing two Sparks genuinely buy you that one cannot: llama.cpp's
# RPC backend places layers on a remote device, so a model too large for a single
# 121 GiB Spark can be split across the pair. Training cannot do this -- see
# unsloth/unsloth#4858 -- because each Spark is its own host with its own single
# GPU; torch.cuda.device_count() is 1 on both, so `device_map="balanced"` has
# nothing to balance across.
#
# The RPC transport auto-selects RDMA over RoCE when llama.cpp was built with
# libibverbs present (set GGML_RPC_NO_RDMA=1 to force TCP).

RPC_DEFAULT_PORT = 50052

# The executable and the library, under every name the bundles have used. The
# legacy `rpc-server` name predates the ggml- prefix; Windows bundles carry .exe
# and ggml-rpc.dll; macOS carries a versioned dylib.
_RPC_SERVER_NAMES = ("ggml-rpc-server", "rpc-server", "ggml-rpc-server.exe", "rpc-server.exe")
_RPC_LIB_NAMES = ("libggml-rpc.so", "libggml-rpc.dylib", "libggml-rpc.0.dylib", "ggml-rpc.dll")
# Where inside a bundle the payload lives: the Linux/macOS layout, the Windows
# layout, the raw tarball layout, and a flat directory, in that order.
_BUNDLE_SUBDIRS = (("build", "bin"), ("build", "bin", "Release"), ("bin",), ())


def llama_bundle_dir() -> Path:
    """The managed llama.cpp prebuilt bundle, resolved the way the installer resolves it.

    Mirrors ``default_managed_llama_dir()`` in studio/install_llama_prebuilt.py and the
    ``_css_llama_path`` logic in setup.sh rather than importing either: this module has
    to stay stdlib-only and cheap. The rule is ``UNSLOTH_LLAMA_CPP_PATH`` if set, else
    ``<UNSLOTH_STUDIO_HOME>/llama.cpp`` for a custom studio home, else the legacy
    ``~/.unsloth/llama.cpp``. The default is deliberately NOT under the studio root:
    the venv lives at ``~/.unsloth/studio/unsloth_studio`` while the bundle lives one
    level up, so ``_studio_root() / "llama.cpp"`` would name a directory that does not
    exist on a default install and provision would silently skip the bundle.
    """
    override = (os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    root = _studio_root()
    try:
        is_default = root.resolve() == _DEFAULT_STUDIO_ROOT.resolve()
    except (OSError, ValueError, RuntimeError):
        is_default = root == _DEFAULT_STUDIO_ROOT
    if not is_default:
        return root / "llama.cpp"
    return Path.home() / ".unsloth" / "llama.cpp"


def _find_in_bundle(root: Path, names: Tuple[str, ...], executable: bool = False) -> Optional[Path]:
    """The first of ``names`` present in any known bundle layout under ``root``."""
    for parts in _BUNDLE_SUBDIRS:
        base = root.joinpath(*parts) if parts else root
        for name in names:
            candidate = base / name
            try:
                if not candidate.is_file():
                    continue
                if executable and not os.access(candidate, os.X_OK):
                    continue
            except OSError:
                continue
            return candidate
    return None


def rpc_server_binary() -> Optional[str]:
    """Path to ggml-rpc-server, or None.

    Unsloth's llama.cpp prebuilt ships the executable next to llama-server, together
    with libggml-rpc.so, from release b10796-mix-659e406 of unslothai/llama.cpp
    onward; earlier bundles shipped only the RPC client backend. So the managed bundle
    (``llama_bundle_dir()``, which honours UNSLOTH_STUDIO_HOME and
    UNSLOTH_LLAMA_CPP_PATH) is searched first, under the current name and the legacy
    ``rpc-server`` name, and a source build or a binary on PATH is the fallback for
    an older bundle.
    """
    roots = [llama_bundle_dir(), Path.home() / "src" / "llamacpp-rpc"]
    for root in roots:
        found = _find_in_bundle(root, _RPC_SERVER_NAMES, executable = True)
        if found is not None:
            return str(found)
    return shutil.which("ggml-rpc-server") or shutil.which("rpc-server")


def _bundle_version(root: Path) -> str:
    """The release tag from the bundle's BUILD_INFO.txt, or ``"unknown"``.

    The first line reads ``llama.cpp version: b10796-mix-659e406``. Bundles older
    than the file, and source builds, have no BUILD_INFO.txt at all, and that must
    read as unknown rather than fail: an unknown is compared by library hash instead.
    """
    for parts in ((), ("build", "bin"), ("bin",)):
        base = root.joinpath(*parts) if parts else root
        text = _read(base / "BUILD_INFO.txt", 4096)
        if not text:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                if "version" in key.lower() and value.strip():
                    return value.strip()
            return line
    return "unknown"


def _file_md5(path: Path) -> Optional[str]:
    import hashlib

    digest = hashlib.md5()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def llama_bundle_identity(root: Optional[Path] = None) -> Dict[str, Any]:
    """What llama.cpp this node would run: the bundle tag and the RPC library's hash.

    Two independent signals because each fails alone. BUILD_INFO.txt is absent from
    older bundles and from source builds (``version`` is then ``"unknown"``); the md5
    of libggml-rpc catches a bundle that was patched in place under the same tag.
    Never raises: a missing bundle answers ``present: False``.
    """
    root = Path(root) if root is not None else llama_bundle_dir()
    out: Dict[str, Any] = {
        "root": str(root),
        "present": False,
        "version": "unknown",
        "rpc_lib": None,
        "rpc_lib_md5": None,
        "rpc_server": None,
    }
    try:
        out["present"] = root.is_dir()
    except OSError:
        return out
    if not out["present"]:
        return out
    out["version"] = _bundle_version(root)
    lib = _find_in_bundle(root, _RPC_LIB_NAMES)
    if lib is not None:
        out["rpc_lib"] = str(lib)
        out["rpc_lib_md5"] = _file_md5(lib)
    server = _find_in_bundle(root, _RPC_SERVER_NAMES, executable = True)
    if server is not None:
        out["rpc_server"] = str(server)
    return out


def _peer_relative_path(path: Path) -> str:
    """``path`` as the peer should see it: ``~/...`` when it sits under our home.

    Provision copies to the same path on the peer, but the peer's home directory may
    differ from ours (different username), so a path under our home is sent home
    relative and expanded THERE. A custom absolute location is sent as is.
    """
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return path.as_posix()


# Runs on the PEER under its own python3, so it must be self-contained: the peer may
# have no Unsloth checkout at all. It mirrors llama_bundle_identity() field for field.
_BUNDLE_PROBE = """\
import hashlib, json, os
root = os.path.expanduser(os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or {root!r})
subs = (("build", "bin"), ("build", "bin", "Release"), ("bin",), ())
libs = {libs!r}
servers = {servers!r}
out = {{"root": root, "present": os.path.isdir(root), "version": "unknown",
       "rpc_lib": None, "rpc_lib_md5": None, "rpc_server": None}}
def find(names, executable):
    for parts in subs:
        base = os.path.join(root, *parts) if parts else root
        for name in names:
            c = os.path.join(base, name)
            if os.path.isfile(c) and (not executable or os.access(c, os.X_OK)):
                return c
    return None
if out["present"]:
    for parts in ((), ("build", "bin"), ("bin",)):
        p = os.path.join(root, *parts, "BUILD_INFO.txt")
        try:
            with open(p, "r", errors="replace") as fh:
                text = fh.read(4096)
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                if "version" in k.lower() and v.strip():
                    out["version"] = v.strip()
                    break
            out["version"] = line
            break
        break
    lib = find(libs, False)
    if lib:
        out["rpc_lib"] = lib
        h = hashlib.md5()
        with open(lib, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out["rpc_lib_md5"] = h.hexdigest()
    out["rpc_server"] = find(servers, True)
print("UNSLOTH_BUNDLE " + json.dumps(out))
"""


def peer_llama_bundle_identity(peer_ip: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """``llama_bundle_identity()`` as evaluated ON THE PEER, or None if it cannot be.

    Runs over non-interactive ssh with the same base64 transport the other peer probes
    use. None means "could not check", which callers must report as unverified rather
    than as matching.
    """
    if not peer_ip or not shutil.which("ssh"):
        return None
    import base64

    source = _BUNDLE_PROBE.format(
        root = _peer_relative_path(llama_bundle_dir()),
        libs = _RPC_LIB_NAMES,
        servers = _RPC_SERVER_NAMES,
    )
    blob = base64.b64encode(source.encode()).decode()
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "nvidia"
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-n",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=8",
                f"{user}@{peer_ip}",
                f"echo {blob} | base64 -d | python3 -",
            ],
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except Exception:
        return None
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("UNSLOTH_BUNDLE "):
            try:
                data = json.loads(line[len("UNSLOTH_BUNDLE ") :])
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


PROVISION_FIX = "run `unsloth spark provision` to copy this node's llama.cpp bundle to the peer"


def compare_llama_bundles(
    local: Dict[str, Any], peer: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Will llama-server here and ggml-rpc-server there speak the same RPC protocol?

    The protocol is pinned by the build, so two nodes on the same bundle tag with the
    same libggml-rpc are safe, and two nodes on different tags are not: b10796 speaks
    RPC 6.0 where the bundles before it spoke 5.1, and llama-server then fails at load
    with "RPC server version mismatch". ``ok`` is True, False, or None for
    "could not verify". Pure: it compares two dicts and touches nothing.
    """
    out: Dict[str, Any] = {"ok": None, "problems": [], "notes": [], "local": local, "peer": peer}
    lv = str(local.get("version") or "unknown")
    if peer is None:
        out["notes"].append(
            "could not read the peer's llama.cpp bundle (ssh unavailable or the probe "
            "failed), so RPC protocol parity is UNVERIFIED. If llama-server fails at "
            f"load with 'RPC server version mismatch', {PROVISION_FIX}."
        )
        return out
    if not peer.get("present"):
        out["problems"].append(
            f"the peer has no llama.cpp bundle at {peer.get('root')} while this Spark "
            f"runs {lv}. Nothing there can answer an RPC connection. Fix: {PROVISION_FIX}."
        )
        out["ok"] = False
        return out
    pv = str(peer.get("version") or "unknown")
    lm, pm = local.get("rpc_lib_md5"), peer.get("rpc_lib_md5")
    if lv != pv and not (lv == "unknown" and pv == "unknown"):
        out["problems"].append(
            f"llama.cpp bundle mismatch: this Spark has {lv}, the peer has {pv}. The RPC "
            f"protocol is pinned by the build (b10796 speaks 6.0, earlier bundles 5.1), so "
            f"llama-server fails at load with 'RPC server version mismatch'. "
            f"Fix: {PROVISION_FIX}."
        )
    elif lm and pm and lm != pm:
        out["problems"].append(
            f"libggml-rpc differs between the nodes (md5 {lm[:12]} here, {pm[:12]} on the "
            f"peer) although both report {lv}; one of them was rebuilt or patched in "
            f"place. Fix: {PROVISION_FIX}."
        )
    elif lv == "unknown" and not (lm and pm):
        out["notes"].append(
            "neither bundle carries BUILD_INFO.txt and libggml-rpc could not be hashed "
            "on both nodes, so RPC protocol parity is UNVERIFIED."
        )
        return out
    out["ok"] = not out["problems"]
    if out["ok"]:
        out["notes"].append(
            f"llama.cpp bundles match on both nodes ({lv}"
            + (f", libggml-rpc md5 {lm[:12]}" if lm else "")
            + ")."
        )
    return out


# ── Live RPC HELLO probe ─────────────────────────────────────────────────────
# Wire format of the handshake, from ggml/src/ggml-rpc/ggml-rpc.cpp and transport.h
# at ggml-org/llama.cpp tag b10796 (RPC protocol 6.0.0):
#
#   client -> one byte RPC_CMD_HELLO (14), a little-endian uint64 payload length,
#             then rpc_msg_hello_req: uint8 conn_caps[RPC_CONN_CAPS_SIZE], all zero
#   server -> a little-endian uint64 body length, then rpc_msg_hello_rsp:
#             uint8 major, minor, patch, padding, then conn_caps[RPC_CONN_CAPS_SIZE]
#
# A 6.0 server checks the request length first and, if it is not exactly
# sizeof(rpc_msg_hello_req), logs "HELLO request size mismatch" and closes the socket
# without replying. So EOF before any reply means "something is listening but it is
# not a 6.0 server", which is a different finding from a refused connection.
RPC_CMD_HELLO = 14
RPC_CONN_CAPS_SIZE = 24
RPC_HELLO_MAX_BODY = 4096


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def rpc_hello_probe_detail(
    host: str,
    port: int = RPC_DEFAULT_PORT,
    timeout: float = 2.0,
    read_timeout: float = 3.0,
) -> Dict[str, Any]:
    """Send one HELLO and classify what came back. Never raises, always bounded.

    ``state`` is one of ``ok`` (``version`` holds (major, minor, patch)), ``refused``
    (nothing listening), ``closed`` (a listener hung up without replying, which is
    what a 6.0 server does to a request it does not recognise and what an older
    server may do to a 6.0 request), ``timeout``, or ``garbled`` (a reply too short
    or too long to be a HELLO response).
    """
    import struct

    out: Dict[str, Any] = {"host": host, "port": port, "state": "refused", "version": None}
    try:
        sock = socket.create_connection((host, port), timeout = timeout)
    except (socket.timeout, TimeoutError):
        out["state"] = "timeout"
        return out
    except Exception:
        return out
    try:
        sock.settimeout(read_timeout)
        payload = bytes(RPC_CONN_CAPS_SIZE)
        sock.sendall(bytes([RPC_CMD_HELLO]) + struct.pack("<Q", len(payload)) + payload)
        header = _recv_exact(sock, 8)
        if header is None:
            out["state"] = "closed"
            return out
        (length,) = struct.unpack("<Q", header)
        if length < 3 or length > RPC_HELLO_MAX_BODY:
            out["state"] = "garbled"
            return out
        body = _recv_exact(sock, length)
        if body is None:
            out["state"] = "closed"
            return out
        out["state"] = "ok"
        out["version"] = (body[0], body[1], body[2])
        return out
    except (socket.timeout, TimeoutError):
        out["state"] = "timeout"
        return out
    except Exception:
        out["state"] = "garbled"
        return out
    finally:
        try:
            sock.close()
        except OSError:
            pass


def rpc_hello_probe(
    host: str,
    port: int = RPC_DEFAULT_PORT,
    timeout: float = 2.0,
) -> Optional[Tuple[int, int, int]]:
    """The (major, minor, patch) RPC protocol of a running ggml-rpc-server, or None."""
    return rpc_hello_probe_detail(host, port, timeout = timeout)["version"]


def rpc_protocol_preflight(peer_ip: str, port: int = RPC_DEFAULT_PORT) -> Dict[str, Any]:
    """Both signals, before a two-node layer split is launched.

    (a) the bundle identity on both nodes, from BUILD_INFO.txt and the libggml-rpc
    hash, which works before anything is running; (b) a live HELLO against whatever
    already listens on the peer's RPC port, and on ours, which catches a stale server
    left over from an older bundle. A refused connection is the normal state before
    launch and is only a note. ``ok`` False means a mismatch was CONFIRMED; None means
    it could not be verified and the caller should say so and carry on.
    """
    result = compare_llama_bundles(llama_bundle_identity(), peer_llama_bundle_identity(peer_ip))
    result["peer_rpc"] = peer_live = rpc_hello_probe_detail(peer_ip, port)
    result["local_rpc"] = local_live = rpc_hello_probe_detail("127.0.0.1", port)
    seen: Dict[str, Tuple[int, int, int]] = {}
    for where, live in (("the peer", peer_live), ("this Spark", local_live)):
        state, version = live["state"], live["version"]
        if state == "ok":
            seen[where] = version
            result["notes"].append(
                f"a ggml-rpc-server on {where} ({live['host']}:{port}) answers HELLO with "
                f"RPC protocol {version[0]}.{version[1]}.{version[2]}."
            )
        elif state == "closed":
            result["problems"].append(
                f"something listens on {where} at {live['host']}:{port} but closed the "
                f"connection on an RPC 6.0 HELLO without answering. That is what an older "
                f"(5.x) ggml-rpc-server does, and llama-server would report 'RPC server "
                f"version mismatch'. Stop it, then {PROVISION_FIX} and start the one from "
                f"the current bundle."
            )
        elif state == "garbled":
            result["problems"].append(
                f"the listener on {where} at {live['host']}:{port} is not a ggml-rpc-server "
                f"(its HELLO reply was malformed). Free the port or pick another with "
                f"--rpc-port."
            )
    if len(seen) == 2 and seen["the peer"] != seen["this Spark"]:
        a, b = seen["this Spark"], seen["the peer"]
        result["problems"].append(
            f"RPC protocol mismatch between the running servers: this Spark speaks "
            f"{a[0]}.{a[1]}.{a[2]}, the peer {b[0]}.{b[1]}.{b[2]}. Fix: {PROVISION_FIX}, "
            f"then restart both servers."
        )
    if result["problems"]:
        result["ok"] = False
    return result


def peer_ip_for(rails: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """The peer's address on the first configured rail (ours is .12, peer .13)."""
    rails = rails if rails is not None else cabled_rails()
    for rail in rails:
        for addr in rail.get("ipv4", []):
            head, _, last = addr.rpartition(".")
            try:
                return f"{head}.{int(last) + 1}"
            except ValueError:
                continue
    return None


def rpc_cluster_plan(port: int = RPC_DEFAULT_PORT) -> Dict[str, Any]:
    """Everything needed to run one model across both Sparks, or the reason we cannot."""
    if not is_dgx_spark():
        # Answer before the sysfs walk and the `ip` fork that rail discovery would do:
        # the answer cannot change, so paying for it would be pure waste off-box.
        return {
            "ok": False,
            "problems": ["not a DGX Spark"],
            "rpc_server": None,
            "local_ip": None,
            "peer_ip": None,
            "port": port,
            "rpc_arg": None,
        }
    binary = rpc_server_binary()
    peer = peer_ip_for()
    local = None
    for rail in cabled_rails():
        if rail.get("ipv4"):
            local = rail["ipv4"][0]
            break
    problems = []
    if not is_dgx_spark():
        problems.append("not a DGX Spark")
    if binary is None:
        problems.append(
            f"no ggml-rpc-server binary in {llama_bundle_dir()} (bundles from "
            f"b10796-mix-659e406 onward ship it; update the llama.cpp prebuilt, or build "
            f"llama.cpp with -DGGML_RPC=ON)"
        )
    if peer is None or local is None:
        problems.append("no configured peer rail (run `unsloth spark setup`)")
    return {
        "ok": not problems,
        "problems": problems,
        "rpc_server": binary,
        "local_ip": local,
        "peer_ip": peer,
        "port": port,
        # Remote first: llama.cpp fills RPC devices in the order given, and putting
        # the remote ahead of the local device keeps the split from starving it.
        "rpc_arg": f"{peer}:{port},127.0.0.1:{port}" if peer else None,
    }


# ── Peer setup ───────────────────────────────────────────────────────────────

# NVIDIA's numbering puts the first node at .12 of each rail subnet, and each
# further node one above it. A /24 leaves room for far more Sparks than anyone
# has, so the cap below is about honesty, not address space.
NODE_BASE_OCTET = 12
MAX_PLANNABLE_NODES = 240


def rail_plan_report(
    rails: Optional[List[Dict[str, Any]]] = None,
    node_index: int = 0,
    n_nodes: int = 2,
    switched: bool = False,
) -> Dict[str, Any]:
    """The addressing plan, or an explicit refusal -- never a wrong plan.

    Two Sparks are cabled QSFP-to-QSFP: one point-to-point link, so a flat /24 per
    PCIe function is exactly right and needs no switch. THREE OR MORE Sparks cannot
    be cabled that way. Either they hang off a switch (in which case the same flat
    /24 per rail is right, and ``switched=True`` says so), or they are cabled in a
    chain or ring, in which case each *link* needs its own subnet and this flat plan
    is simply wrong -- it would give two nodes that share no cable addresses on the
    same subnet, and every route between them would black-hole.

    A netplan that looks plausible and does not work is worse than a refusal,
    because the user applies it, reboots, and then debugs the wrong layer. So for
    N>2 without ``switched=True`` this returns ``ok: False`` and says why.
    """
    rails = rails if rails is not None else cabled_rails()
    problems: List[str] = []
    notes: List[str] = []
    n_nodes = max(1, int(n_nodes))
    if node_index < 0 or node_index >= n_nodes:
        problems.append(
            f"node_index {node_index} is outside 0..{n_nodes - 1} for a {n_nodes}-node cluster"
        )
    if n_nodes > MAX_PLANNABLE_NODES:
        problems.append(
            f"{n_nodes} nodes does not fit one /24 starting at .{NODE_BASE_OCTET}; "
            f"Unsloth will not plan more than {MAX_PLANNABLE_NODES}"
        )
    if n_nodes > 2 and not switched:
        problems.append(
            f"{n_nodes} Sparks cannot be cabled point-to-point the way two are. This flat "
            f"one-/24-per-rail plan is only correct if all {n_nodes} nodes share a "
            f"switched RoCE fabric. Re-run with switched=True (`--switched`) if they do. "
            f"If instead they are daisy-chained, each cable needs its own subnet and "
            f"Unsloth will not guess your cabling -- a wrong netplan is worse than none."
        )
    if not rails:
        problems.append("no cabled ConnectX rail found on this node")
    if n_nodes > 2 and switched:
        notes.append(
            "Assuming a switched RoCE fabric with all rails in the same broadcast domain, "
            "jumbo frames enabled on every switch port (MTU 9000), and PFC/ECN configured. "
            "None of that is verified from here."
        )
    plan: List[Dict[str, str]] = []
    if not problems:
        for slot, rail in enumerate(rails[: len(DEFAULT_SUBNETS)]):
            plan.append(
                {
                    "ib_device": rail["ib_device"],
                    "netdev": rail["netdev"],
                    "address": f"{DEFAULT_SUBNETS[slot]}.{NODE_BASE_OCTET + node_index}",
                    "prefix": "24",
                    "mtu": str(DEFAULT_MTU),
                }
            )
    return {
        "ok": not problems,
        "problems": problems,
        "notes": notes,
        "plan": plan,
        "node_index": node_index,
        "n_nodes": n_nodes,
        "switched": switched,
    }


def rail_plan(
    rails: Optional[List[Dict[str, Any]]] = None,
    node_index: int = 0,
    n_nodes: int = 2,
    switched: bool = False,
) -> List[Dict[str, str]]:
    """Addressing plan for the cabled rails: one /24 per PCIe function.

    ``node_index`` 0 is this Spark, 1 the next, so the hosts land on .12, .13, ...
    of each subnet (NVIDIA's own numbering). Two subnets are not redundancy: a
    single subnet can only drive one PCIe function, which would cap a pair near
    100 Gb/s instead of ~190.

    Returns ``[]`` -- never a half-right plan -- when the request cannot be
    honoured; ``rail_plan_report`` carries the reason. See its docstring for why
    N>2 needs ``switched=True``.
    """
    return rail_plan_report(rails, node_index, n_nodes, switched)["plan"]


def netplan_yaml(plan: List[Dict[str, str]]) -> str:
    """A netplan drop-in that makes the rail addressing survive a reboot.

    An empty plan renders as a comment block, not as an empty ``ethernets:`` map:
    the latter is valid YAML that netplan accepts and that quietly configures
    nothing, which is exactly the silent-wrong-answer this module exists to avoid.
    """
    if not plan:
        return (
            "# No addressing plan was produced -- nothing to apply.\n"
            "# Run `unsloth spark setup` and read the refusal it prints; writing this\n"
            "# file with no `ethernets:` entries would configure nothing while looking\n"
            "# like it had.\n"
        )
    lines = ["network:", "  version: 2", "  renderer: NetworkManager", "  ethernets:"]
    for entry in plan:
        lines += [
            f"    {entry['netdev']}:",
            f"      addresses: [{entry['address']}/{entry['prefix']}]",
            "      dhcp4: no",
            f"      mtu: {entry['mtu']}",
        ]
    return "\n".join(lines) + "\n"


def _print_manual_steps(
    plan: List[Dict[str, str]],
    peer_plan: List[Dict[str, str]],
    *,
    extra_plans: Optional[List[List[Dict[str, str]]]] = None,
) -> None:
    """Print the netplan drop-in for this node and for each peer.

    ``extra_plans`` carries nodes 2..N-1 for a cluster larger than a pair; the
    two positional arguments keep the pair case calling exactly as it did.
    """

    def emit(where: str, entries: List[Dict[str, str]]) -> None:
        print(f"\n  Run these on {where}:")
        print("    sudo tee /etc/netplan/40-unsloth-cx7.yaml >/dev/null <<'EOF'")
        print(netplan_yaml(entries), end = "")
        print("    EOF")
        print("    sudo chmod 600 /etc/netplan/40-unsloth-cx7.yaml && sudo netplan apply")

    emit("THIS Spark", plan)
    others = [peer_plan] + list(extra_plans or [])
    for index, entries in enumerate(others, start = 1):
        label = "the PEER Spark" if len(others) == 1 else f"Spark node {index}"
        emit(label, entries)


# Below this, the link is degraded rather than merely imperfect. Healthy on this hardware
# measures ~21.6 GB/s (88% of the 24.5 GB/s raw RDMA ceiling); the fault state measures ~3.0.
# The gap is so wide that a single threshold is safe, and 8.0 sits far from both.
NCCL_DEGRADED_GBPS = 8.0
NCCL_HEALTHY_GBPS = 15.0

POWER_CYCLE_ADVICE = """\
  This is almost always the DGX Spark power-delivery fault. Fix it in this order:

    1. FULLY POWER CYCLE BOTH SPARKS -- shut down, unplug the power for ~30 seconds,
       then boot. A REBOOT IS NOT ENOUGH; the fault survives one.
    2. Leave the QSFP cable connected the whole time. Hot-plugging it after boot is a
       separate fault that also produces a slow link and needs another reboot.

  Measured here before and after a power cycle: 3.0 GB/s -> 21.6 GB/s, a 7x difference,
  with every setting identical. Raw RDMA looked healthy (24.5 GB/s) in BOTH states, which
  is why only a real NCCL collective detects this."""


def nccl_bandwidth(
    peer_ip: str,
    local_ip: str,
    mb: int = 1024,
    timeout: int = 90,
) -> Optional[float]:
    """Real NCCL all-reduce bus bandwidth in GB/s, or None if it cannot be measured.

    Shells out to torchrun on both nodes rather than importing torch here, so that merely
    importing this module stays free on every other platform.
    """
    if not shutil.which("ssh"):
        return None
    # A fixed port collides with a previous run that died at the rendezvous and left
    # the socket held, which then fails as an unexplained 'could not measure'.
    import random

    port = random.randint(29600, 29999)
    # Copy the probe to the peer and run BOTH sides by absolute path. Running it as
    # `-m studio.spark_nccl_probe` would require Unsloth to be installed at the same
    # importable path on both nodes, which is not guaranteed -- the peer may have a
    # different venv, or none.
    local_probe = osp.join(osp.dirname(osp.abspath(__file__)), "spark_nccl_probe.py")
    if not osp.exists(local_probe):
        return None
    remote_probe = "/tmp/spark_nccl_probe.py"
    user = os.environ.get("USER", "nvidianew")
    ssh_opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
    try:
        subprocess.run(
            ["scp", *ssh_opts, local_probe, f"{user}@{peer_ip}:{remote_probe}"],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = 30,
            check = True,
        )
    except Exception:
        return None

    env = " ".join(f"{k}={v}" for k, v in nccl_env().items())
    common = (
        f"{env} SPARK_PROBE_MB={mb} torchrun --nnodes=2 --nproc_per_node=1 "
        f"--master_addr={local_ip} --master_port={port}"
    )
    # A non-interactive ssh shell does not have the Unsloth venv on PATH, so `torchrun`
    # is simply missing on the peer -- the peer side never starts, and the local side sits
    # at the rendezvous until it times out. Source the venv when it is there.
    activate = venv_activate()
    peer_cmd = (
        f"setsid nohup bash -c '[ -f {activate} ] && . {activate}; "
        f"exec env {common} --node_rank=1 {remote_probe}' "
        f"> /tmp/spark_nccl_probe.log 2>&1 < /dev/null &"
    )
    try:
        subprocess.run(
            ["ssh", *ssh_opts, f"{user}@{peer_ip}", peer_cmd],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = 30,
        )
    except Exception:
        return None
    import time

    time.sleep(4)  # let the peer's rendezvous come up before we dial in
    try:
        out = subprocess.run(
            f"env {common} --node_rank=0 {local_probe}",
            shell = True,
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except Exception:
        return None
    for line in (out.stdout or "").splitlines():
        if line.startswith("SPARK_NCCL_BUSBW"):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def diagnose_link(busbw: Optional[float]) -> Dict[str, Any]:
    """Turn a measured bandwidth into a verdict and, when needed, the fix."""
    if busbw is None:
        return {
            "verdict": "unknown",
            "busbw": None,
            "summary": "could not measure NCCL bandwidth (needs a peer and torchrun)",
            "advice": "",
        }
    if busbw < NCCL_DEGRADED_GBPS:
        return {
            "verdict": "degraded",
            "busbw": busbw,
            "summary": (
                f"NCCL all-reduce is {busbw:.1f} GB/s -- far below the "
                f"~21 GB/s this link should reach. TRAINING WILL BE SLOW."
            ),
            "advice": POWER_CYCLE_ADVICE,
        }
    if busbw < NCCL_HEALTHY_GBPS:
        return {
            "verdict": "suspect",
            "busbw": busbw,
            "summary": (
                f"NCCL all-reduce is {busbw:.1f} GB/s. Healthy is ~21 GB/s, so "
                f"this is workable but below par."
            ),
            "advice": POWER_CYCLE_ADVICE,
        }
    return {
        "verdict": "healthy",
        "busbw": busbw,
        "summary": f"NCCL all-reduce is {busbw:.1f} GB/s -- healthy.",
        "advice": "",
    }


def python_dev_headers(peer_ip: Optional[str] = None) -> Dict[str, Any]:
    """Check for CPython dev headers locally and on the peer.

    Worth a dedicated check because the failure is invisible. Triton JIT-compiles a small
    `cuda_utils.c` shim at model-inspection time, so without `Python.h` it cannot build --
    and neither vLLM nor torch says so. Locally it surfaces as
    `Model architectures [...] failed to be inspected` (the gcc error is swallowed in a
    subprocess). On a WORKER node it is worse: the head rank simply blocks for 601 s and
    dies with `DistStoreError: Timed out ... 1/2 clients joined`, never reporting that its
    peer already exited. An apparently slow multi-node launch is often this.

    rsyncing a venv between nodes -- which is the fast way to set up the second Spark,
    since the RoCE link does ~444 MB/s while the internet may not -- copies the Python
    packages but NOT the system headers they shell out to, so the peer is the likely
    offender.
    """
    import sysconfig

    out: Dict[str, Any] = {"local": False, "peer": None, "include": ""}
    inc = sysconfig.get_paths().get("include", "")
    out["include"] = inc
    out["local"] = bool(inc) and osp.exists(osp.join(inc, "Python.h"))
    if peer_ip and shutil.which("ssh"):
        user = os.environ.get("USER", "nvidianew")
        # Ask the PEER's own interpreter where its headers live rather than assuming it
        # matches ours. Triton builds against whichever Python that node's venv runs, so
        # checking our include path over there answers the wrong question -- and the whole
        # point of this check is a node whose environment differs from the head's.
        # base64 the probe: it has to survive ssh -> bash -lc -> python -c, and every
        # layer of nested quoting is a chance to mangle it (it did, twice).
        import base64

        probe = (
            "import sysconfig, os\n"
            "p = os.path.join(sysconfig.get_paths()['include'], 'Python.h')\n"
            "print('yes' if os.path.isfile(p) else 'no')\n"
        )
        b64 = base64.b64encode(probe.encode()).decode()
        activate = venv_activate()
        remote = f"[ -f {activate} ] && . {activate}; " f"echo {b64} | base64 -d | python3 -"
        try:
            r = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"{user}@{peer_ip}",
                    remote,
                ],
                capture_output = True,
                text = True,
                timeout = 25,
            )
            answer = [l for l in (r.stdout or "").strip().splitlines() if l in ("yes", "no")]
            out["peer"] = (answer[-1] == "yes") if answer else None
        except Exception:
            out["peer"] = None
    return out


# Caches that must exist on BOTH nodes. A node missing one does not fail -- it silently
# rebuilds from scratch, which presents as a HANG rather than an error. Observed: a peer
# without `flashinfer_autotune_cache` sat 17+ minutes in CUDA-graph capture at 191% CPU
# while the head logged only "No available shared memory broadcast block found in 60
# seconds" once a minute. Nothing named the cause.
SHARED_CACHES = (
    "~/.cache/flashinfer",
    "~/.cache/vllm/flashinfer_autotune_cache",
    "~/.cache/vllm/torch_compile_cache",
)


def cache_symmetry(peer_ip: str) -> Dict[str, Optional[bool]]:
    """For each shared cache present locally, is it also present on the peer?

    Only flags caches we HAVE locally: one neither node has is simply cold, and both will
    build it. The dangerous case is asymmetry, because then one node is fast, the other
    silently spends many minutes rebuilding, and the job looks stuck.
    """
    out: Dict[str, Optional[bool]] = {}
    if not shutil.which("ssh"):
        return out
    user = os.environ.get("USER", "nvidianew")
    for c in SHARED_CACHES:
        local = osp.expanduser(c)
        if not osp.isdir(local):
            continue
        try:
            r = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"{user}@{peer_ip}",
                    f"test -d {c} && echo yes || echo no",
                ],
                capture_output = True,
                text = True,
                timeout = 20,
            )
            out[c] = (r.stdout or "").strip() == "yes"
        except Exception:
            out[c] = None
    return out


def cuda_health(peer_ip: Optional[str] = None) -> Dict[str, Any]:
    """Detect a GPU that enumerates but whose compute engine is dead.

    Observed on a DGX Spark after a power cycle, and it is genuinely confusing because every
    obvious check passes:

        nvidia-smi                -> works, reports NVIDIA GB10
        driver modules            -> all loaded (nvidia, nvidia_uvm, nvidia_drm, ...)
        /dev/nvidia*              -> present, correct permissions
        CUDA_VISIBLE_DEVICES      -> unset
        torch.cuda.is_available() -> False
        cuInit(0)                 -> 100  (CUDA_ERROR_NO_DEVICE)
        dmesg                     -> NVRM: ... Possible bad register read ... 0xbadf5600

    `0xbadf5600` is NVIDIA's sentinel for a failed register read: the GPU is on the bus and
    enumerable, but not responding. **A reboot clears it; a module reload does not, and a
    power cycle is not required** -- which is worth stating because the natural response to
    "the GPU is dead after a power cycle" is another power cycle, and that is the slower fix.

    Distinct from the power-delivery fault, which leaves CUDA perfectly healthy and instead
    caps NCCL bandwidth -- see `diagnose_link`. Same symptom class, opposite remedy.
    """
    probe = (
        "import ctypes\n"
        "try:\n"
        "    r = ctypes.CDLL('libcuda.so.1').cuInit(0)\n"
        "except Exception:\n"
        "    r = -1\n"
        "print(r)\n"
    )
    out: Dict[str, Any] = {"local": None, "peer": None}

    def _classify(cuinit: Optional[int], smi_ok: bool) -> str:
        if cuinit == 0:
            return "ok"
        if smi_ok and cuinit is not None:
            return "dead-engine"  # enumerates but will not initialise
        return "unknown"

    smi = shutil.which("nvidia-smi") is not None
    if smi:
        try:
            r = subprocess.run(
                [sys.executable, "-c", probe], capture_output = True, text = True, timeout = 60
            )
            code = int((r.stdout or "-1").strip().splitlines()[-1])
        except Exception:
            code = None
        smi_ok = (
            subprocess.run(["nvidia-smi", "-L"], capture_output = True, timeout = 30).returncode == 0
        )
        out["local"] = {"cuinit": code, "state": _classify(code, smi_ok)}

    if peer_ip and shutil.which("ssh"):
        import base64

        b64 = base64.b64encode(probe.encode()).decode()
        act = venv_activate()
        cmd = (
            f"[ -f {act} ] && . {act}; nvidia-smi -L >/dev/null 2>&1 && echo SMI_OK || echo SMI_BAD; "
            f"echo {b64} | base64 -d | python3 -"
        )
        try:
            r = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"{os.environ.get('USER', 'nvidianew')}@{peer_ip}",
                    cmd,
                ],
                capture_output = True,
                text = True,
                timeout = 90,
            )
            lines = [l.strip() for l in (r.stdout or "").splitlines() if l.strip()]
            smi_ok = "SMI_OK" in lines
            code = next((int(l) for l in reversed(lines) if l.lstrip("-").isdigit()), None)
            out["peer"] = {"cuinit": code, "state": _classify(code, smi_ok)}
        except Exception:
            out["peer"] = {"cuinit": None, "state": "unknown"}
    return out


def _cmd_doctor() -> int:
    """Measure the link the only way that detects the power-delivery fault."""
    if not is_dgx_spark():
        print("This machine is not a DGX Spark; nothing to check.")
        return 0
    info = discover_peers()
    if not info["cable_present"]:
        print("No QSFP cable detected. Nothing to measure.")
        return 0
    peer = peer_ip_for()
    local = None
    for rail in cabled_rails():
        if rail.get("ipv4"):
            local = rail["ipv4"][0]
            break
    if not peer or not local:
        print("No configured peer rail yet -- run `unsloth spark setup` first.")
        return 1

    # Cheap preflight first: a missing header wastes 10 minutes on a silent hang later.
    hdr = python_dev_headers(peer)
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}-dev"
    for where, ok in (("this Spark", hdr["local"]), ("the peer", hdr["peer"])):
        if ok is False:
            print(f"  MISSING Python.h on {where}.")
            print("    Without it Triton cannot build its cuda_utils shim. Locally that")
            print("    reads as 'Model architectures [...] failed to be inspected'; on a")
            print("    worker it makes the head rank hang for 601s and report only")
            print("    'DistStoreError: 1/2 clients joined'. Neither names the header.")
            print("")
            print("    Fix, preferred -- a uv-managed CPython ships its own headers, so")
            print("    this class of failure cannot recur:")
            print("        uv python install 3.12")
            print(f"        uv venv --python 3.12 && . .venv/bin/activate")
            print(f"    Or, if you must use the system interpreter: apt install {ver}")
            print("")
        elif ok is None:
            print(f"  could not check Python.h on {where} (ssh unavailable)\n")

    # A dead compute engine makes every downstream measurement fail in confusing ways, so
    # check it before anything expensive.
    for where, info in cuda_health(peer).items():
        if not info:
            continue
        if info["state"] == "dead-engine":
            print(
                f"  GPU NOT USABLE on {where}: nvidia-smi works but cuInit returns "
                f"{info['cuinit']} (CUDA_ERROR_NO_DEVICE)."
            )
            print("    The GPU enumerates but its compute engine is not responding. Check")
            print("    `dmesg | grep NVRM` for 'Possible bad register read ... 0xbadf5600'.")
            print("    FIX: REBOOT that node. A module reload does not clear it, and a")
            print("    power cycle is not required -- a plain reboot is faster and works.")
            print("    (This is NOT the power-delivery fault, which leaves CUDA healthy and")
            print("     instead caps NCCL bandwidth. Same symptom class, opposite remedy.)")
            print("")
        elif info["state"] == "unknown":
            print(f"  could not determine GPU health on {where}\n")

    for cache, present in cache_symmetry(peer).items():
        if present is False:
            print(f"  CACHE ASYMMETRY: {cache} exists here but NOT on the peer.")
            print("    The peer will rebuild it from scratch on first run. That is not an")
            print("    error and produces no message -- it looks like a hang, for many")
            print("    minutes. Copy it over the fast link instead:")
            print(f"      rsync -a {cache}/ {peer}:{cache}/")
            print("")

    # The llama.cpp bundle must match too, or a two-node layer split dies at load
    # with "RPC server version mismatch". Only worth asking when we have a bundle.
    bundle_bad = False
    local_bundle = llama_bundle_identity()
    if local_bundle["present"]:
        bundles = compare_llama_bundles(local_bundle, peer_llama_bundle_identity(peer))
        for problem in bundles["problems"]:
            bundle_bad = True
            print(f"  LLAMA.CPP BUNDLE MISMATCH: {problem}")
            print("")
        for note in bundles["notes"]:
            print(f"  llama.cpp: {note}")
            print("")

    print(f"Measuring NCCL all-reduce {local} <-> {peer} (takes ~30s)...")
    result = diagnose_link(nccl_bandwidth(peer, local))
    print("")
    print(f"  {result['summary']}")
    if result["advice"]:
        print("")
        print(result["advice"])
    if result["verdict"] not in ("healthy", "unknown"):
        return 1
    return 1 if bundle_bad else 0


def _cmd_status(benchmark: bool = False) -> int:
    if not is_dgx_spark():
        # Explicit, and first: `cluster_state()` happens to answer "not_spark" here too,
        # but relying on that made the guard a property of another function's ordering
        # rather than a stated rule of this one.
        print("state: not_spark")
        print(NOT_A_SPARK)
        return 0
    state = cluster_state()
    print(f"state: {state}")
    if state == "not_spark":
        print(NOT_A_SPARK)
        return 0
    info = discover_peers()
    print(f"cable present: {info['cable_present']}")
    for rail in info["cabled_rails"]:
        ips = ", ".join(rail["ipv4"]) or "no IPv4"
        print(f"  {rail['ib_device']:<14} {rail['netdev']:<16} mtu={rail['mtu']}  {ips}")
    for peer in info["mdns_peers"]:
        print(f"  peer seen: {peer['hostname']} at {peer['address']}")
    events = link_carrier_events(info["cabled_rails"])
    shown = ", ".join(f"{k}={v}" for k, v in events.items() if v is not None)
    if shown:
        print(f"\n  link events: carrier_up_count {shown}")

    # Only a measurement can say whether the link is throttled -- carrier counters
    # cannot (see HOTPLUG_NOTE). Measure when perftest and a peer are both available.
    peer_ip = None
    for rail in info["configured"]:
        for addr in rail["ipv4"]:
            octets = addr.rsplit(".", 1)
            peer_ip = f"{octets[0]}.{int(octets[1]) + 1}"
            break
        if peer_ip:
            break
    if benchmark and peer_ip:
        rails = info["cabled_rails"]
        if rails:
            print(f"  measuring {rails[0]['ib_device']} against {peer_ip} ...")
            local_ip = info["configured"][0]["ipv4"][0]
            health = link_health(peer_ip, rails[0]["ib_device"], local_ip)
            if not health:
                print("  (could not measure; is ib_write_bw running on the peer?)")
            elif health["degraded"]:
                print(
                    f"  MEASURED {health['gbps']:.2f} Gb/s -- well below "
                    f"~{health['expected_gbps']:.0f} Gb/s per rail."
                )
                print(f"  {HOTPLUG_NOTE}")
            else:
                print(f"  MEASURED {health['gbps']:.2f} Gb/s per rail -- healthy.")
    elif peer_ip:
        print("  (run `unsloth spark status --benchmark` to measure the link;")
        print("   carrier counters alone cannot tell a throttled link from a healthy one)")
    return 0


def _cmd_env() -> int:
    """Print the GB10 NCCL settings as shell exports."""
    if not is_dgx_spark():
        return 0
    for key, value in nccl_env().items():
        print(f"export {key}={value}")
    return 0


# What must exist identically on both nodes for a two-Spark job to work, and which the
# internet cannot supply here: HuggingFace measures ~20 KB/s from these boxes while the
# RoCE link does ~444 MB/s. Installing on the peer is therefore both slower and a source of
# resolver drift; copying is faster and bit-identical.
# The venv path is resolved, not hardcoded, because UNSLOTH_STUDIO_HOME moves it. Copying
# `~/.unsloth/studio/unsloth_studio` from a machine whose venv lives somewhere else copies
# a stale venv or nothing at all, and then reports "Peer now matches this node" -- which
# hands the user the exact 601 s `DistStoreError: 1/2 clients joined` this command exists
# to prevent, while telling them everything is fine.
_DEFAULT_STUDIO_ROOT = Path.home() / ".unsloth" / "studio"


def provision_paths() -> Tuple[Tuple[str, str], ...]:
    # The llama.cpp bundle is NOT inside the venv: it sits beside the studio root
    # (see llama_bundle_dir). Without it a paired peer keeps whatever llama-server it
    # had, and two bundles a release apart speak different RPC protocols, which
    # llama-server reports at load as "RPC server version mismatch".
    return (
        (str(_studio_root() / "unsloth_studio"), "Unsloth venv"),
        (str(llama_bundle_dir()), "llama.cpp prebuilt"),
        ("~/.cache/flashinfer", "FlashInfer JIT cache"),
        ("~/.cache/vllm/flashinfer_autotune_cache", "vLLM FlashInfer autotune cache"),
        ("~/.cache/vllm/torch_compile_cache", "vLLM torch.compile cache"),
    )


def venv_activate() -> str:
    """The peer's `activate`, for a non-interactive ssh that has no venv on PATH.

    Left as the literal `$HOME/...` in the default case so it expands on the PEER, which
    stays correct even if the two nodes have different usernames or home directories.
    Only a custom UNSLOTH_STUDIO_HOME forces an absolute path, and that is right because
    `provision` copies to the same absolute path on the peer.
    """
    root = _studio_root()
    if root == _DEFAULT_STUDIO_ROOT:
        return "$HOME/.unsloth/studio/unsloth_studio/bin/activate"
    return str(root / "unsloth_studio" / "bin" / "activate")


# A process holding less than this is a CUDA context and scratch, not a job. Anything
# at or above it means someone's work is resident on that GPU right now.
PEER_BUSY_MIB = 96


def peer_gpu_busy(peer_ip: str, timeout: int = 25) -> Dict[str, Any]:
    """Is the peer's GPU holding someone's work? FAILS CLOSED, by design.

    This gates a destructive `rsync --delete` onto a machine that may be running a
    job out of the very directory being overwritten. The asymmetry is deliberate and
    is the whole point: "I could not tell" must read as BUSY, never as free. Probing
    a peer is unreliable in exactly the situations where a running job is most likely
    (the box is loaded, ssh is slow, nvidia-smi is queued behind a driver call), so a
    probe that fails to answer is evidence of nothing and must not be spent as
    permission. Fail-open here has cost this project twice.

    Returns ``{"busy": bool, "known": bool, "processes": [...], "reason": str}``.
    """
    out: Dict[str, Any] = {"busy": True, "known": False, "processes": [], "reason": ""}
    if not peer_ip:
        out["reason"] = "no peer address"
        return out
    if not shutil.which("ssh"):
        out["reason"] = "ssh unavailable, so the peer's GPU state cannot be checked"
        return out
    user = os.environ.get("USER", "nvidianew")
    # The RC marker separates "nvidia-smi ran and listed nothing" (idle) from
    # "nvidia-smi did not run" (unknown). Without it both are an empty string, and
    # reading the second as idle is exactly the fail-open mistake.
    remote = (
        "nvidia-smi --query-compute-apps=pid,used_gpu_memory "
        "--format=csv,noheader,nounits; echo RC=$?"
    )
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=8",
                f"{user}@{peer_ip}",
                remote,
            ],
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except Exception as exc:
        out["reason"] = f"could not reach the peer to check its GPU ({str(exc)[:80]})"
        return out
    lines = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip()]
    rc = next((l[3:] for l in lines if l.startswith("RC=")), None)
    if rc is None or rc != "0":
        out["reason"] = f"nvidia-smi did not run on the peer (rc={rc!r}); treating the GPU as BUSY"
        return out
    for line in lines:
        if line.startswith("RC=") or line.lower().startswith("pid"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        mem = parts[1].replace("MiB", "").replace("MB", "").strip()
        try:
            mib = int(float(mem))
        except ValueError:
            continue
        if mib >= PEER_BUSY_MIB:
            out["processes"].append({"pid": int(parts[0]), "used_mib": mib})
    out["known"] = True
    out["busy"] = bool(out["processes"])
    out["reason"] = (
        f"{len(out['processes'])} compute process(es) resident"
        if out["busy"]
        else "no compute processes on the peer GPU"
    )
    return out


def provision_peer(
    peer_ip: str,
    dry_run: bool = False,
    delete: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Copy the environment and warm caches to the peer over the fast link.

    Two failures this prevents, both of which cost hours in testing and neither of which
    reports itself:

    * a peer missing the venv's system headers, or the venv entirely -- the head then blocks
      601 s and dies with `DistStoreError: 1/2 clients joined`, never surfacing the worker's
      real error;
    * a peer missing a warm cache -- it silently rebuilds from scratch, which presents as a
      17-minute hang in CUDA graph capture with no message beyond
      "No available shared memory broadcast block found in 60 seconds".

    rsync rather than reinstall: identical bytes, no dependency-resolution drift between the
    two nodes, and it runs at link speed rather than internet speed.
    """
    results: Dict[str, Any] = {
        "copied": [],
        "skipped": [],
        "failed": [],
        "dry_run": dry_run,
        "refused": "",
        "peer_gpu": None,
        "delete": delete,
    }
    if not shutil.which("rsync") or not shutil.which("ssh"):
        results["failed"].append(("rsync/ssh", "not installed"))
        return results
    # A dry run reads nothing on the peer and writes nothing, so it needs no gate.
    if not dry_run:
        gpu = peer_gpu_busy(peer_ip)
        results["peer_gpu"] = gpu
        if gpu["busy"] and not force:
            held = (
                ", ".join(f"pid {p['pid']} ({p['used_mib']} MiB)" for p in gpu["processes"])
                or "state unknown"
            )
            results["refused"] = (
                f"peer {peer_ip} is BUSY or unverifiable ({gpu['reason']}: {held}). "
                f"Refusing to overwrite a venv a running job may be executing out of. "
                f"Wait for the job, or pass force=True if you are certain."
            )
            return results
    user = os.environ.get("USER", "nvidianew")
    for path, label in provision_paths():
        local = osp.expanduser(path)
        if not osp.isdir(local):
            results["skipped"].append((label, "not present locally"))
            continue
        # rsync creates only the LAST component of the destination, so a peer that does
        # not yet have the parent fails with `mkdir "<dest>" failed: No such file or
        # directory`. That is the normal case for the thing this command is for: pairing
        # a brand-new second Spark, which has no ~/.unsloth/studio yet. Create the parent
        # remotely first. `~` is rewritten to "$HOME" because a quoted ~ does not expand,
        # and it must expand on the PEER, whose home may differ from ours.
        remote_parent = osp.dirname(path)
        if remote_parent == "~":
            remote_parent = "$HOME"
        elif remote_parent.startswith("~/"):
            remote_parent = "$HOME/" + remote_parent[2:]
        # Trailing slashes matter: copy the CONTENTS into the same path on the peer.
        # --delete is OFF by default: a stale extra file on the peer costs disk, while a
        # deleted live one takes a running interpreter out from under a job mid-flight.
        cmd = [
            "rsync",
            "-a",
            "--rsync-path",
            f'mkdir -p "{remote_parent}" && rsync',
            "-e",
            "ssh -o BatchMode=yes -o StrictHostKeyChecking=no",
            local + "/",
            f"{user}@{peer_ip}:{path}/",
        ]
        if delete:
            cmd.insert(2, "--delete")
        if dry_run:
            cmd.insert(1, "--dry-run")
        try:
            r = subprocess.run(cmd, capture_output = True, text = True, timeout = 3600)
            if r.returncode == 0:
                results["copied"].append((label, path))
            else:
                results["failed"].append((label, (r.stderr or "").strip()[:200]))
        except Exception as exc:
            results["failed"].append((label, str(exc)[:200]))
    return results


def _cmd_provision(
    dry_run: bool = False,
    delete: bool = False,
    force: bool = False,
) -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; nothing to provision.")
        return 0
    peer = peer_ip_for()
    if not peer:
        print("No configured peer. Run `unsloth spark setup` first.")
        return 1
    print(
        f"Provisioning peer {peer} over the ConnectX link" f"{' (dry run)' if dry_run else ''}..."
    )
    print("  Copying rather than installing: HuggingFace measures ~20 KB/s from these")
    print("  boxes while this link does ~444 MB/s, and copying cannot drift.")
    res = provision_peer(peer, dry_run = dry_run, delete = delete, force = force)
    if res["refused"]:
        print(f"  REFUSED: {res['refused']}")
        print("  (nothing was copied, nothing was deleted)")
        return 1
    for label, path in res["copied"]:
        print(f"  ok      {label} ({path})")
    for label, why in res["skipped"]:
        print(f"  skip    {label}: {why}")
    for label, why in res["failed"]:
        print(f"  FAILED  {label}: {why}")
    if res["failed"]:
        return 1
    print("\n  Peer now matches this node. Verify with: unsloth doctor")
    return 0


# Measured on this hardware; see DGX_SPARK_DETAILS.md. A Spark reports 121.69 GiB usable
# (130.66 GB decimal) -- "128GB" is 128 GiB physical with ~6.3 GiB firmware-reserved.
SPARK_USABLE_GIB = 121.69
# Room a served model needs beyond its weights: KV cache, compute buffers, fragmentation.
# The 235B measured 736+768 MiB of KV and 208 MiB/node of compute buffers at modest context,
# but context dominates and grows fast, so this is deliberately not tight.
SERVE_OVERHEAD_GIB = 8.0


def model_size_gib(target: str) -> Optional[float]:
    """Best-effort size of a model, from a file, a directory, or the HF cache.

    Returns None rather than guessing when it cannot tell -- a wrong size here would
    produce confidently wrong deployment advice, which is worse than no advice.
    """
    path = osp.expanduser(target)
    if osp.isfile(path):
        return osp.getsize(path) / 2**30
    if osp.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".safetensors", ".gguf", ".bin")):
                    try:
                        total += osp.getsize(osp.join(root, f))
                    except OSError:
                        pass
        return total / 2**30 if total else None
    # A repo id: look for it in the HF cache rather than hitting the network, which is
    # unusable from these machines anyway (~20 KB/s).
    cache = osp.expanduser("~/.cache/huggingface/hub")
    slug = "models--" + target.replace("/", "--")
    root = osp.join(cache, slug)
    if osp.isdir(root):
        total, seen = 0, set()
        for dirpath, _, files in os.walk(root):
            for f in files:
                full = osp.join(dirpath, f)
                try:
                    real = osp.realpath(full)
                    if real in seen:
                        continue
                    seen.add(real)
                    if f.endswith((".safetensors", ".gguf", ".bin")):
                        total += osp.getsize(real)
                except OSError:
                    pass
        return total / 2**30 if total else None
    return None


# ── What each parallelism axis actually buys, measured ───────────────────────
# Llama-3.3-70B fp8 served across TWO Sparks, against the same model on ONE Spark.
# Concurrency is requests in flight; the number is end-to-end speedup.
#
#   axis                       c=1    c=2    c=4    c=8   median TPOT
#   tensor-parallel (TP=2)    2.09x  2.13x  2.10x  1.97x  332.7ms -> 162.4ms
#   pipeline-parallel (PP=2)  1.08x  1.11x  1.09x  1.07x  ~320ms  -> ~320ms  (FLAT)
#   replicas (2 copies)       1.00x per request; AGGREGATE only (measured below)
#   layer-split a model that already fits on one node   decode 0.85x to 1.01x (never a win)
#
# The whole planner follows from those four rows:
#   * TP is the ONLY axis that makes a single request faster.
#   * PP moves tokens through more silicon but does not shorten the critical path
#     of one token, so its TPOT is flat -- PP is for CAPACITY, never for latency.
#   * replicas raise aggregate throughput and change per-request latency not at all.
#   * splitting a model that fits never speeds up decode; it is a capacity feature
#     and a prefill feature (see REPLICAS_DECODE_SPEEDUP and its neighbours).
TP_SPEEDUP_2 = {1: 2.09, 2: 2.13, 4: 2.10, 8: 1.97}
PP_SPEEDUP_2 = {1: 1.08, 2: 1.11, 4: 1.09, 8: 1.07}
TP_TPOT_MS_2 = (332.7, 162.4)
PP_TPOT_MS_2 = (320.0, 320.0)
# Splitting a model that ALREADY FITS on one node. This was a flat 0.92x loss, and that is
# still what you get from a llama.cpp whose RPC backend predates ggml-org/llama.cpp#18626
# ("rpc: implement event and async backend APIs", merged 2026-08-26). Without it the RPC
# backend advertises neither async nor events, ggml_backend_sched therefore refuses to
# pipeline across RPC devices, and the two halves run strictly one after the other.
LAYER_SPLIT_FITTING_SPEEDUP = 0.92

# WITH that commit the answer stops being a constant and becomes a function of prompt length,
# because what overlaps is prefill. Measured end to end on two Sparks, Qwen3-27B Q4_K_XL,
# same binary both arms, so the only variable is whether the model is split:
#
#   prompt tokens |  c=1    c=4    c=8
#           128   | 0.94x  0.95x  0.95x
#           256   | 0.98x  1.00x  1.00x     <- break-even
#           512   | 0.96x  1.05x  1.07x
#          1024   | 1.02x  1.12x  1.17x
#          2048   | 1.07x  1.23x  1.29x
#          4096   | 1.11x  1.35x  1.45x
#
# Decode is 0.93-0.98x throughout and cannot be otherwise: a layer split moves the same
# weight bytes per token, so the whole gain is prefill and the whole question is prompt
# length. Below ~256 tokens splitting costs 2-6%; above ~1024 it wins, and the win grows with
# both prompt length and concurrency.
LAYER_SPLIT_ASYNC_RPC_SPEEDUP = {
    128:  {1: 0.94, 4: 0.95, 8: 0.95},
    256:  {1: 0.98, 4: 1.00, 8: 1.00},
    512:  {1: 0.96, 4: 1.05, 8: 1.07},
    1024: {1: 1.02, 4: 1.12, 8: 1.17},
    2048: {1: 1.07, 4: 1.23, 8: 1.29},
    4096: {1: 1.11, 4: 1.35, 8: 1.45},
}
LAYER_SPLIT_BREAK_EVEN_TOKENS = 256


def layer_split_speedup(prompt_tokens = None, concurrency = 1, async_rpc = False):
    """End-to-end speedup from splitting a model that already fits on one node.

    `async_rpc=False` is the conservative default and reports the flat 0.92x, because that is
    what a stock fork build still does. Pass True only for a build carrying #18626. Returns
    the nearest measured row at or below `prompt_tokens` rather than interpolating: these are
    six measured points, not a fitted curve, and pretending otherwise would invent precision.
    """
    if not async_rpc:
        return LAYER_SPLIT_FITTING_SPEEDUP
    if prompt_tokens is None:
        return None  # genuinely unknown; the caller must not guess
    rows = sorted(LAYER_SPLIT_ASYNC_RPC_SPEEDUP)
    key = rows[0]
    for r in rows:
        if prompt_tokens >= r:
            key = r
    by_c = LAYER_SPLIT_ASYNC_RPC_SPEEDUP[key]
    near = min(by_c, key = lambda c: abs(c - max(1, int(concurrency))))
    return by_c[near]


# ── Replicas versus layer split for a model that FITS, measured ──────────────
# Qwen3.8-27B-UD-Q4_K_XL (16.4 GiB) served by llama.cpp b10796 on two DGX Sparks
# (GB10, aarch64, 121 GiB each, cabled over ConnectX-7), measured 2026-09-04 with
# UNCAPPED clocks, closed-loop concurrent clients and 128 generated tokens per
# request. Every ratio is aggregate DECODE tok/s against the same model on ONE
# Spark. That is a different question from LAYER_SPLIT_ASYNC_RPC_SPEEDUP above,
# which is end to end request throughput with prefill included; both tables are
# right about what they measure.
#
#   prompt 512   users |  1 Spark | 2 replicas | layer split || replicas | split
#                    1 |   12.0   |    12.0    |    11.3     ||  1.00x   | 0.95x
#                    2 |   21.2   |    23.8    |    21.0     ||  1.13x   | 0.99x
#                    4 |   39.0   |    44.0    |    35.8     ||  1.13x   | 0.92x
#                    8 |   59.7   |    77.8    |    51.0     ||  1.30x   | 0.85x
#                   16 |   68.1   |   119.4    |    64.2     ||  1.75x   | 0.94x
#                   32 |   70.9   |   135.7    |    71.7     ||  1.91x   | 1.01x
#
#   prompt 2048  users |    1      2      4      8     16     32
#           replicas   |  1.01   1.38   1.30   1.81   1.99   2.38
#           split      |  0.94   0.96   0.88   0.96   1.06   1.12
#
# The two split figures above 1.0 at prompt 2048 are not decode wins: the single
# node control was prefill-contended there. Measured decode-only, the split is
# 0.95x. Layer split PREFILL is 1.7x to 1.85x. Memory per node at 16 users, prompt
# 512: single 22.6 GiB; replicas 19.3 GiB on each node; split 10.9 + 11.8 GiB.
#
# What follows from the table:
#   * a layer split never speeds up decode at any user count. It is a capacity
#     feature (model larger than one node) and a prefill feature;
#   * two replicas are the throughput winner whenever model plus KV fits on one
#     node and 8 or more users are concurrent;
#   * below 8 users a second copy buys 1.00x to 1.13x, and at 1 user nothing helps
#     except vLLM tensor parallel (TP_SPEEDUP_2).
TOPOLOGY_MEASUREMENT = (
    "Qwen3.8-27B-UD-Q4_K_XL on llama.cpp b10796, two DGX Sparks, 2026-09-04, uncapped clocks"
)
REPLICAS_DECODE_SPEEDUP = {
    512: {1: 1.00, 2: 1.13, 4: 1.13, 8: 1.30, 16: 1.75, 32: 1.91},
    2048: {1: 1.01, 2: 1.38, 4: 1.30, 8: 1.81, 16: 1.99, 32: 2.38},
}
LAYER_SPLIT_DECODE_SPEEDUP = {
    512: {1: 0.95, 2: 0.99, 4: 0.92, 8: 0.85, 16: 0.94, 32: 1.01},
    2048: {1: 0.94, 2: 0.96, 4: 0.88, 8: 0.96, 16: 1.06, 32: 1.12},
}
LAYER_SPLIT_DECODE_ONLY_SPEEDUP = 0.95
LAYER_SPLIT_PREFILL_SPEEDUP = (1.7, 1.85)
REPLICAS_MIN_USERS = 8
REPLICAS_FEW_USERS_SPEEDUP = 1.13  # 2 to 4 users, prompt 512
TOPOLOGIES = ("single", "replicas", "layer_split")


def _measured_cell(table: Dict[int, Dict[int, float]], prompt_tokens: int, users: int) -> float:
    """Nearest measured row at or below ``prompt_tokens``, nearest user count. No fitting."""
    rows = sorted(table)
    key = rows[0]
    for row in rows:
        if prompt_tokens >= row:
            key = row
    by_users = table[key]
    near = min(by_users, key = lambda u: (abs(u - max(1, int(users))), u))
    return by_users[near]


def replicas_speedup(prompt_tokens: int = 512, users: int = 1) -> float:
    """Aggregate decode gain of two replicas over one Spark, at the nearest measured point."""
    return _measured_cell(REPLICAS_DECODE_SPEEDUP, prompt_tokens, users)


def layer_split_decode_speedup(prompt_tokens: int = 512, users: int = 1) -> float:
    """Aggregate decode ratio of a layer split over one Spark, for a model that fits."""
    return _measured_cell(LAYER_SPLIT_DECODE_SPEEDUP, prompt_tokens, users)


def recommend_topology(
    model_bytes: float,
    kv_bytes_per_user: float,
    users: int,
    prompt_tokens: int,
    per_node_free_bytes: float,
    prefill_heavy: bool = False,
) -> Dict[str, Any]:
    """Which of single / replicas / layer_split to serve a GGUF with, and why. Pure.

    The rules, from the measurements above:

    * a model that does not fit on one node is a ``layer_split``: the only option;
    * a model that fits, with 8 or more concurrent users, is ``replicas``;
    * a model that fits, with fewer users, is ``single``: leave the second node idle,
      because a second copy buys 1.00x to 1.13x and nothing helps one user except
      tensor parallel, which llama.cpp does not do;
    * ``layer_split`` is never recommended for a model that fits UNLESS the caller
      says the work is prefill-heavy long-prompt work, where the split's 1.7x to
      1.85x prefill outweighs its 0.95x decode. Even then, at 8 or more users the
      replicas win end to end (1.81x against 0.96x at prompt 2048, 8 users).

    Memory is checked with the KV of every concurrent user included, so a model that
    fits alone but not with its users' KV is routed to replicas (each node carries
    half the users) or, failing that, to a layer split. Returns a dict with
    ``topology``, a one-paragraph ``reason``, the measured ``speedup`` where one
    exists, and the byte counts it decided on.
    """
    users = max(1, int(users or 1))
    prompt_tokens = max(1, int(prompt_tokens or 512))
    model_bytes = max(0.0, float(model_bytes or 0))
    kv_each = max(0.0, float(kv_bytes_per_user or 0))
    free = float(per_node_free_bytes or 0)
    single_need = model_bytes + kv_each * users
    replica_need = model_bytes + kv_each * ((users + 1) // 2)
    fits_model = model_bytes <= free
    out: Dict[str, Any] = {
        "topology": "single",
        "reason": "",
        "speedup": None,
        "prefill_speedup": None,
        "fits_one_node": fits_model,
        "users": users,
        "prompt_tokens": prompt_tokens,
        "single_node_bytes": single_need,
        "replica_node_bytes": replica_need,
        "per_node_free_bytes": free,
        "measured_on": TOPOLOGY_MEASUREMENT,
    }
    gib = 2**30
    if not fits_model:
        out.update(
            topology = "layer_split",
            prefill_speedup = LAYER_SPLIT_PREFILL_SPEEDUP,
            reason = (
                f"the model ({model_bytes / gib:.1f} GiB) does not fit in one node's "
                f"{free / gib:.1f} GiB, so a layer split across both Sparks is the only way "
                f"to run it. That is a capacity feature: expect decode about "
                f"{LAYER_SPLIT_DECODE_ONLY_SPEEDUP:.2f}x of what one node would do if it "
                f"could, and prefill {LAYER_SPLIT_PREFILL_SPEEDUP[0]:.1f}x to "
                f"{LAYER_SPLIT_PREFILL_SPEEDUP[1]:.2f}x."
            ),
        )
        return out
    if single_need > free:
        if replica_need <= free:
            out.update(
                topology = "replicas",
                speedup = replicas_speedup(prompt_tokens, users),
                reason = (
                    f"the model fits, but with KV for {users} users it needs "
                    f"{single_need / gib:.1f} GiB against {free / gib:.1f} GiB free. Two "
                    f"replicas carry half the users each ({replica_need / gib:.1f} GiB per "
                    f"node) and measured {replicas_speedup(prompt_tokens, users):.2f}x "
                    f"aggregate decode at {users} users."
                ),
            )
        else:
            out.update(
                topology = "layer_split",
                prefill_speedup = LAYER_SPLIT_PREFILL_SPEEDUP,
                reason = (
                    f"the model fits, but model plus KV for {users} users "
                    f"({single_need / gib:.1f} GiB) exceeds one node even when halved "
                    f"across replicas ({replica_need / gib:.1f} GiB against "
                    f"{free / gib:.1f} GiB free), so only a layer split, which spreads the KV "
                    f"with the layers, has the room. Capacity, not speed: decode about "
                    f"{LAYER_SPLIT_DECODE_ONLY_SPEEDUP:.2f}x."
                ),
            )
        return out
    if prefill_heavy and users < REPLICAS_MIN_USERS:
        out.update(
            topology = "layer_split",
            speedup = layer_split_decode_speedup(prompt_tokens, users),
            prefill_speedup = LAYER_SPLIT_PREFILL_SPEEDUP,
            reason = (
                f"you asked for prefill-heavy long-prompt work at {users} users. A layer "
                f"split measured {LAYER_SPLIT_PREFILL_SPEEDUP[0]:.1f}x to "
                f"{LAYER_SPLIT_PREFILL_SPEEDUP[1]:.2f}x on prefill, which is the only "
                f"reason to split a model that fits; its decode is "
                f"{layer_split_decode_speedup(prompt_tokens, users):.2f}x, so time to first "
                f"token improves and tokens per second do not. Chat-shaped traffic should "
                f"stay on one node."
            ),
        )
        return out
    fit_note = (
        f"the model fits on one node with KV for {users} users "
        f"({single_need / gib:.1f} of {free / gib:.1f} GiB)"
        if kv_each
        else f"the model fits on one node ({single_need / gib:.1f} of {free / gib:.1f} GiB, "
        f"KV not counted)"
    )
    if users >= REPLICAS_MIN_USERS:
        gain = replicas_speedup(prompt_tokens, users)
        out.update(
            topology = "replicas",
            speedup = gain,
            reason = (
                f"{fit_note}, and at {users} concurrent "
                f"users two replicas measured {gain:.2f}x aggregate decode "
                f"(1.30x at 8, 1.75x at 16, 1.91x at 32 users, prompt 512). A layer split "
                f"measured {layer_split_decode_speedup(prompt_tokens, users):.2f}x here, so "
                f"never split a model that fits for throughput."
                + (
                    " Prefill-heavy work does not change this at this many users: the "
                    "replicas still win end to end (1.81x against 0.96x at prompt 2048, "
                    "8 users)."
                    if prefill_heavy
                    else ""
                )
            ),
        )
        return out
    few = replicas_speedup(prompt_tokens, users)
    out.update(
        topology = "single",
        speedup = 1.0,
        reason = (
            f"{fit_note}, "
            f"and {users} concurrent user{'s' if users != 1 else ''} cannot use a second "
            f"one: two replicas measured {few:.2f}x at this concurrency (1.00x at 1 user, "
            f"{REPLICAS_FEW_USERS_SPEEDUP:.2f}x at 2 to 4) for the cost of a full second "
            f"copy, and a layer split measured "
            f"{layer_split_decode_speedup(prompt_tokens, users):.2f}x. Leave the second "
            f"Spark idle, or use it for something else. Replicas start paying at "
            f"{REPLICAS_MIN_USERS} users."
            + (
                " At 1 user the only measured win is vLLM tensor parallel "
                f"({TP_SPEEDUP_2[1]:.2f}x), which llama.cpp cannot do."
                if users == 1
                else ""
            )
        ),
    )
    return out


REPLICA_AGGREGATE_PER_NODE = 1.0  # n replicas -> ~n x aggregate, 1.0x per request
# Training, GPipe pipeline parallel with M=4 microbatches, 2 nodes:
# 3024 tok/s against a 2032 tok/s single-node control.
TRAIN_PP_SPEEDUP_2 = 1.49

INTENTS = ("latency", "throughput", "capacity")


def _nearest_concurrency(table: Dict[int, float], concurrency: int) -> Tuple[int, float]:
    key = min(table, key = lambda c: (abs(c - concurrency), c))
    return key, table[key]


def expected_gain(
    axis: str,
    n_nodes: int,
    concurrency: int = 1,
    prompt_tokens: int = 512,
) -> Dict[str, Any]:
    """What to expect from an axis at N nodes -- measured where measured, honest elsewhere.

    Everything in the table above was measured at exactly TWO Sparks. Reporting a
    scaled number for N>2 as though it were measured would be the same class of
    mistake this module refuses everywhere else, so ``measured`` is part of the
    answer and the note says plainly that three Sparks were never benchmarked here.
    """
    out: Dict[str, Any] = {
        "axis": axis,
        "n_nodes": n_nodes,
        "concurrency": concurrency,
        "speedup": None,
        "measured": False,
        "note": "",
    }
    if n_nodes <= 1 or axis in ("none", "single", None):
        out["speedup"] = 1.0
        out["measured"] = True
        out["note"] = "one node: nothing is distributed, so there is nothing to gain."
        return out
    if axis == "tensor-parallel":
        c, value = _nearest_concurrency(TP_SPEEDUP_2, concurrency)
        if n_nodes == 2:
            out.update(
                speedup = value,
                measured = True,
                note = (
                    f"measured {value:.2f}x at concurrency {c} on 2 Sparks; "
                    f"median TPOT {TP_TPOT_MS_2[0]:.1f}ms -> {TP_TPOT_MS_2[1]:.1f}ms."
                ),
            )
        else:
            out.update(
                speedup = None,
                measured = False,
                note = (
                    f"TP=2 measured {value:.2f}x; TP={n_nodes} across Sparks is "
                    f"NOT measured here. Every token needs an all-reduce over the "
                    f"RoCE link, and that cost grows with the node count while the "
                    f"per-node work shrinks, so expect clearly sublinear scaling and "
                    f"benchmark before believing a number."
                ),
            )
        return out
    if axis in ("pipeline-parallel", "layer-split"):
        c, value = _nearest_concurrency(PP_SPEEDUP_2, concurrency)
        out.update(
            speedup = value if n_nodes == 2 else None,
            measured = n_nodes == 2,
            note = (
                f"PP=2 measured {value:.2f}x end-to-end with median TPOT FLAT at "
                f"~{PP_TPOT_MS_2[0]:.0f}ms -- i.e. NO latency benefit. Pipelining "
                f"buys capacity: it is how a model too large for one Spark runs at "
                f"all, not how a request gets faster."
            ),
        )
        return out
    if axis == "replicas":
        if n_nodes == 2:
            gain = replicas_speedup(prompt_tokens or 512, concurrency)
            out.update(
                speedup = 1.0,
                measured = True,
                aggregate = gain,
                note = (
                    f"{gain:.2f}x AGGREGATE decode at {concurrency} concurrent, "
                    f"{prompt_tokens or 512} prompt tokens (measured at prompt 512: 1.00x "
                    f"at 1, 1.13x at 2 to 4, 1.30x at 8, 1.75x at 16, 1.91x at 32 users), "
                    f"1.00x per request. Two copies pay from {REPLICAS_MIN_USERS} users "
                    f"up; below that the second Spark is idle money."
                ),
            )
        else:
            out.update(
                speedup = 1.0,
                measured = False,
                aggregate = float(n_nodes) * REPLICA_AGGREGATE_PER_NODE,
                note = (
                    f"up to ~{n_nodes}x AGGREGATE throughput, 1.00x per request; only two "
                    f"replicas were measured here (1.91x at 32 users). Independent copies "
                    f"never make one request faster; they let you serve more at once."
                ),
            )
        return out
    if axis == "layer-split-fitting":
        out.update(
            speedup = LAYER_SPLIT_DECODE_ONLY_SPEEDUP,
            measured = True,
            note = (
                f"decode measured {LAYER_SPLIT_DECODE_ONLY_SPEEDUP:.2f}x, and 0.85x to "
                f"1.01x across 1 to 32 users: a layer split never speeds up decode for a "
                f"model that fits. It is a capacity feature and a prefill feature "
                f"({LAYER_SPLIT_PREFILL_SPEEDUP[0]:.1f}x to "
                f"{LAYER_SPLIT_PREFILL_SPEEDUP[1]:.2f}x prefill), so it pays only for "
                f"prefill-heavy long-prompt work at few users. For {REPLICAS_MIN_USERS} or "
                f"more users two replicas measured 1.30x to 1.91x instead."
            ),
        )
        return out
    if axis == "training-pipeline":
        out.update(
            speedup = TRAIN_PP_SPEEDUP_2 if n_nodes == 2 else None,
            measured = n_nodes == 2,
            note = (
                f"GPipe with M=4 microbatches measured {TRAIN_PP_SPEEDUP_2:.2f}x on 2 "
                f"Sparks (3024 vs 2032 tok/s). Bubbles, not bandwidth, are the ceiling."
            ),
        )
        return out
    out["note"] = f"unknown axis {axis!r}; no measurement to report."
    return out


def _nodes_needed(size_gib: float, budget: float) -> int:
    """Fewest nodes whose combined budget holds the model."""
    if budget <= 0:
        return 1
    count = int(size_gib / budget)
    if count * budget < size_gib - 1e-9:
        count += 1
    return max(1, count)


def _serve_commands(
    axis: str,
    n_nodes: int,
    model: str = "<model>",
) -> List[str]:
    """The concrete command for an axis. Names a model so it can be pasted."""
    env = 'eval "$(unsloth spark env)"   # GB10 NCCL settings; NCCL_NET_GDR_LEVEL=0 is mandatory'
    if axis == "tensor-parallel":
        return [
            env,
            "ray start --head --port=6379            # on THIS Spark",
            "ray start --address=<this-spark>:6379   # on each of the other Sparks",
            f"vllm serve {model} --tensor-parallel-size {n_nodes} "
            f"--distributed-executor-backend ray",
        ]
    if axis == "replicas":
        backends = ",".join(
            f"{DEFAULT_SUBNETS[0]}.{NODE_BASE_OCTET + i}:8080" for i in range(n_nodes)
        )
        return [
            env,
            f"unsloth spark serve --model {model} --engines 1     # run on EACH Spark",
            f"python -m studio.spark_lb --backends {backends}     # one front door",
        ]
    if axis in ("pipeline-parallel", "layer-split"):
        return [
            env,
            f"unsloth spark serve --model {model} --engines 1   # llama.cpp RPC layer split",
            f"# or, with vLLM:  vllm serve {model} --pipeline-parallel-size {n_nodes} "
            f"--distributed-executor-backend ray",
        ]
    if axis == "single":
        return [f"unsloth serve --model {model}"]
    return []


def plan_deployment(
    size_gib: Optional[float],
    two_sparks: Optional[bool] = None,
    *,
    n_nodes: Optional[int] = None,
    intent: str = "throughput",
    concurrency: int = 1,
    model: str = "<model>",
    prompt_tokens: Optional[int] = None,
    prefill_heavy: bool = False,
    kv_gib_per_user: float = 0.0,
) -> Dict[str, Any]:
    """Recommend a topology AND an axis from model size, node count and intent.

    Measured behaviour, not theory. Two facts drive everything:

    * a model that FITS on one Spark never decodes faster layer-split across two:
      0.85x to 1.01x measured from 1 to 32 users (LAYER_SPLIT_DECODE_SPEEDUP). A split
      buys capacity and prefill (1.7x to 1.85x), never decode. For 8 or more users two
      replicas measured 1.30x to 1.91x aggregate instead (REPLICAS_DECODE_SPEEDUP).
    * TP is the only axis that shortens a single request (2.09x on two Sparks);
      PP's median TPOT is flat, and replicas raise aggregate throughput only.

    ``serving`` carries the llama.cpp specific answer from ``recommend_topology()``
    for a model that fits across the cluster: ``prompt_tokens`` (default 512),
    ``prefill_heavy`` and ``kv_gib_per_user`` feed it and change nothing else.

    ``topology`` is a MEMORY-FIT class and keeps its historical vocabulary --
    ``replicas`` / ``single-or-replicas`` / ``layer-split`` / ``too-large`` /
    ``single`` / ``unknown``. ``axis`` is the new, orthogonal answer: which
    parallelism to actually use. They are different questions: a 70B fp8 that fits
    on one node is topology ``single-or-replicas``, and its axis is
    ``tensor-parallel`` if you want latency and ``replicas`` if you want throughput.

    Node count comes from ``n_nodes`` when given, else from the legacy
    ``two_sparks`` bool, which keeps working exactly as before.
    """
    budget = SPARK_USABLE_GIB - SERVE_OVERHEAD_GIB
    if n_nodes is None:
        nodes = 1 if two_sparks is None else (2 if two_sparks else 1)
    else:
        try:
            nodes = max(1, int(n_nodes))
        except (TypeError, ValueError):
            nodes = 1
    if intent not in INTENTS:
        intent = "throughput"

    out: Dict[str, Any] = {
        "size_gib": size_gib,
        "budget_gib": budget,
        "n_nodes": nodes,
        "cluster_gib": nodes * budget,
        "intent": intent,
        "concurrency": concurrency,
    }

    # Never guess. A wrong size produces confidently wrong deployment advice, and a
    # user cannot tell that apart from right advice until the run fails.
    if size_gib is None:
        out.update(
            topology = "unknown",
            axis = None,
            fits = None,
            commands = [],
            command = "",
            expected = expected_gain("none", 1, concurrency),
            recommendation = "",
            summary = "could not determine model size; not guessing",
        )
        return out

    fits_one = size_gib <= budget
    min_nodes = _nodes_needed(size_gib, budget)
    out["min_nodes"] = min_nodes
    out["fits_one_node"] = fits_one

    # ── One node ─────────────────────────────────────────────────────────────
    if nodes < 2:
        out.update(
            topology = "single",
            fits = fits_one,
            axis = "single" if fits_one else "none",
            summary = (
                f"{size_gib:.1f} GiB fits on this Spark ({budget:.0f} GiB budget)"
                if fits_one
                else f"{size_gib:.1f} GiB does NOT fit on one Spark ({budget:.0f} GiB "
                f"budget) -- pair a second one, or use a smaller quant"
            ),
            expected = expected_gain("none", 1, concurrency),
            commands = _serve_commands("single" if fits_one else "none", 1, model),
        )
        out["recommendation"] = (
            f"Serve it on this Spark. With one node there is no axis to choose."
            if fits_one
            else f"This needs at least {min_nodes} Sparks at {budget:.0f} GiB each, or a "
            f"smaller quant. One Spark cannot run it at any speed."
        )
        out["command"] = "\n".join(out["commands"])
        return out

    # ── Two or more nodes: memory-fit class first ────────────────────────────
    if size_gib > nodes * budget:
        topology = "too-large"
    elif not fits_one:
        topology = "layer-split"
    elif size_gib * 2 <= budget:
        # Historical meaning, kept: two engines fit side by side on ONE node.
        topology = "replicas"
    else:
        topology = "single-or-replicas"
    out["topology"] = topology
    out["fits"] = topology != "too-large"
    if topology != "too-large":
        out["serving"] = recommend_topology(
            size_gib * 2**30,
            max(0.0, float(kv_gib_per_user or 0)) * 2**30,
            concurrency,
            prompt_tokens or 512,
            budget * 2**30,
            prefill_heavy = prefill_heavy,
        )

    # `summary` answers ONLY "what fits where". Every statement about which axis to
    # use lives in `recommendation`. They used to overlap, and the overlap read as the
    # tool contradicting itself: for a 70B the summary named the llama.cpp layer split
    # while the recommendation named tensor parallel (2.09x), both true of different
    # axes but printed as though they were one answer. A caller can now print both,
    # in either order, and get one coherent paragraph.
    copies = int(budget // size_gib) if size_gib > 0 else 0
    if topology == "replicas":
        out["summary"] = (
            f"{size_gib:.1f} GiB against a {budget:.0f} GiB budget per node: TWO copies "
            f"fit side by side on a single Spark, so all {nodes} nodes have room to spare."
            if copies < 3
            else f"{size_gib:.1f} GiB against a {budget:.0f} GiB budget per node: {copies} "
            f"copies fit on each of your {nodes} Sparks."
        )
    elif topology == "single-or-replicas":
        out["summary"] = (
            f"{size_gib:.1f} GiB fits on ONE Spark ({budget:.0f} GiB budget), but a second "
            f"copy does not fit beside it on the same node. Each of your {nodes} Sparks "
            f"can hold exactly one copy."
        )
    elif topology == "layer-split":
        out["summary"] = (
            f"{size_gib:.1f} GiB exceeds one Spark's {budget:.0f} GiB, so it cannot run on "
            f"a single node. It fits across {min_nodes} of your {nodes} "
            f"({nodes * budget:.0f} GiB total), which means it has to be sharded somehow."
        )
    else:
        out["summary"] = (
            f"{size_gib:.1f} GiB exceeds all {nodes} Sparks together ({nodes * budget:.0f} "
            f"GiB usable). At least {min_nodes} nodes would be needed, or a smaller quant."
        )

    # ── Then the axis, which is what the intent actually decides ─────────────
    if topology == "too-large":
        out.update(axis = "none", expected = expected_gain("none", 1, concurrency), commands = [])
        out["recommendation"] = (
            f"No topology helps: {size_gib:.1f} GiB does not fit in {nodes} x "
            f"{budget:.0f} GiB. Add nodes until you have {min_nodes}, or quantise smaller."
        )
    elif not fits_one:
        # It must be sharded to run at all. TP is the axis that is both possible
        # and fast; PP/layer-split is the fallback when the engine cannot TP.
        shard_nodes = min(nodes, max(2, min_nodes))
        out.update(
            axis = "tensor-parallel",
            axis_nodes = shard_nodes,
            expected = expected_gain("tensor-parallel", shard_nodes, concurrency),
            commands = _serve_commands("tensor-parallel", shard_nodes, model),
        )
        out["fallback_axis"] = "pipeline-parallel"
        out["fallback_expected"] = expected_gain("pipeline-parallel", shard_nodes, concurrency)
        out["recommendation"] = (
            f"This model cannot run on one Spark, so it must be sharded. Use TENSOR "
            f"parallel across {shard_nodes} nodes: at TP=2 that measured 2.09x a single "
            f"Spark and halved median TPOT (332.7ms -> 162.4ms), so it is both the way to "
            f"run this at all and the fastest way. Pipeline/layer split is the fallback "
            f"when your engine cannot TP across hosts (llama.cpp RPC): it measured 1.08x "
            f"with FLAT TPOT -- capacity only."
        )
    elif intent == "latency":
        out.update(
            axis = "tensor-parallel",
            axis_nodes = nodes,
            expected = expected_gain("tensor-parallel", nodes, concurrency),
            commands = _serve_commands("tensor-parallel", nodes, model),
        )
        out["recommendation"] = (
            f"TENSOR parallel across {nodes} Sparks. It is the only axis that makes a "
            f"single request faster: TP=2 measured 2.09x with median TPOT 332.7ms -> "
            f"162.4ms. Do NOT use pipeline parallel for this -- its TPOT is flat at "
            f"~320ms -- and do NOT layer-split a model that fits: its decode measured "
            f"0.85x to 1.01x across 1 to 32 users, never a win."
        )
    elif intent == "throughput":
        gain = replicas_speedup(prompt_tokens or 512, concurrency)
        out.update(
            axis = "replicas",
            axis_nodes = nodes,
            expected = expected_gain("replicas", nodes, concurrency, prompt_tokens or 512),
            commands = _serve_commands("replicas", nodes, model),
        )
        out["recommendation"] = (
            f"REPLICAS: one independent server per Spark, {nodes} in total, behind "
            f"`python -m studio.spark_lb`. Two replicas measured {gain:.2f}x aggregate "
            f"decode at {concurrency} concurrent (1.30x at 8, 1.75x at 16, 1.91x at 32 "
            f"users; only 1.00x to 1.13x below {REPLICAS_MIN_USERS}, where one Spark is as "
            f"good and the second copy is wasted). It does not make any single request "
            f"faster -- if that is what you want, ask for intent=latency and use tensor "
            f"parallel instead. Never layer-split a model that fits for throughput: "
            f"decode measured 0.85x to 1.01x."
        )
    else:  # capacity, and it already fits
        out.update(
            axis = "none",
            axis_nodes = 1,
            expected = expected_gain("none", 1, concurrency),
            commands = _serve_commands("single", 1, model),
        )
        out["recommendation"] = (
            f"For capacity, MORE SPARKS WILL NOT HELP YOU HERE: {size_gib:.1f} GiB already "
            f"fits in one node's {budget:.0f} GiB. Serve it on a single Spark. The extra "
            f"nodes are worth using only for throughput (replicas, 1.30x to 1.91x "
            f"aggregate decode at 8 to 32 users) or for latency (tensor parallel, 2.09x "
            f"measured at 2 nodes)."
        )
    out["command"] = "\n".join(out.get("commands") or [])
    return out


def _cmd_plan(
    model: str,
    intent: str = "throughput",
    nodes: Optional[int] = None,
    concurrency: int = 1,
    prompt_tokens: int = 512,
    prefill_heavy: bool = False,
) -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; nothing to plan.")
        return 0
    if nodes is None:
        info = discover_peers(timeout = 0.0)
        nodes = max(info.get("n_nodes", 1), 2 if peer_ip_for() else 1)
    size = model_size_gib(model)
    plan = plan_deployment(
        size,
        n_nodes = nodes,
        intent = intent,
        concurrency = concurrency,
        model = model,
        prompt_tokens = prompt_tokens,
        prefill_heavy = prefill_heavy,
    )
    print(f"  model     : {model}")
    print(f"  size      : " + (f"{size:.1f} GiB" if size else "unknown (not cached locally)"))
    print(f"  Sparks    : {nodes}")
    print(f"  intent    : {intent}")
    if concurrency != 1 or prompt_tokens != 512:
        print(f"  traffic   : {concurrency} concurrent, {prompt_tokens} prompt tokens")
    print(f"  topology  : {plan['topology']}")
    if plan.get("axis"):
        print(f"  axis      : {plan['axis']}")
    print("")
    print(f"  {plan['summary']}")
    if plan.get("recommendation"):
        print("")
        print(f"  {plan['recommendation']}")
    serving = plan.get("serving")
    if serving:
        print("")
        print(f"  llama.cpp : {serving['topology']}")
        print(f"              {serving['reason']}")
        print(f"              (measured on {serving['measured_on']})")
    exp = plan.get("expected") or {}
    if exp.get("note"):
        print("")
        label = "measured" if exp.get("measured") else "NOT measured at this node count"
        speed = f"{exp['speedup']:.2f}x" if exp.get("speedup") is not None else "unknown"
        print(f"  expected  : {speed} ({label})")
        print(f"              {exp['note']}")
    if plan.get("commands"):
        print("")
        print("  Run:")
        for line in plan["commands"]:
            print(f"    {line}")
    return 0


def _cmd_peers(check: bool = True) -> int:
    """List every Spark we can see, in the order the planner will rank them."""
    if not is_dgx_spark():
        print("Not a DGX Spark; no peers to look for.")
        return 0
    info = discover_peers(check_reachable = check)
    print(f"  cable present : {info['cable_present']}")
    print(f"  nodes         : {info['n_nodes']} (this Spark + {info['n_peers']} peer(s))")
    if not info["peers"]:
        print("")
        print("  No peer Sparks discovered. mDNS only sees peers that advertise, so a")
        print('  switched cluster may need them written down: add a "peers" list of')
        print(f'  {{"hostname": ..., "address": ...}} to {config_path()}.')
        return 0
    for peer in info["peers"]:
        state = {True: "reachable", False: "UNREACHABLE", None: "not probed"}[peer["reachable"]]
        print(
            f"    node {peer['index']}  {peer['short']:<16} {peer['address']:<18} "
            f"{state}  ({peer['source']})"
        )
    # mDNS answers with whatever interface advertised, which is usually Wi-Fi, not the
    # 200 Gb/s rail. Say so rather than letting someone paste a 1 Gb/s address into a
    # distributed launch and wonder why NCCL is slow.
    rail_peer = peer_ip_for()
    if rail_peer:
        print("")
        print(f"  rail peer     : {rail_peer} (from the addressing plan, not from mDNS)")
        print("    Use THIS address for NCCL/ray/rsync. An address discovered over mDNS is")
        print("    typically the Wi-Fi one and would not touch the ConnectX link at all.")
    if any(p["reachable"] is False for p in info["peers"]):
        print("")
        print("  An unreachable peer will not fail loudly in a distributed launch: the head")
        print("  rank blocks at the rendezvous for 601s and then reports only")
        print("  'DistStoreError: N/M clients joined'. Fix reachability first.")
    return 0


# Measured on GB10, identical weights and one GEMM shape, only the kernel changing. The
# spread is far larger than any checkpoint difference, and it inverts with batch size -- so a
# single "best kernel" does not exist, and picking by workload is worth up to 6.2x.
#
#   kernel        acts    M=1        M=4096
#   marlin        A16     429 us     29257 us /  50 TF
#   fi_cutlass    A4      447 us      4727 us / 309 TF
#   vllm_cutlass  A4      486 us      4511 us / 324 TF
#   fi_b12x       A4      484 us      4548 us / 321 TF
#   bf16          A16    1544 us     15339 us /  95 TF
#
# Crossover sits between M=32 and M=256, matching the memory/compute roofline knee at M~436.
NVFP4_KERNELS = {
    "decode": {
        "backend": "marlin",
        "flag": "--linear-backend marlin --moe-backend marlin",
        "why": (
            "fastest measured at M=1 (429 us). At decode batch sizes every format runs at "
            "94-106% of achievable memory bandwidth, so 4-bit activations cannot help -- "
            "and Marlin's 16-bit compute costs nothing it was not already paying in stalls."
        ),
    },
    "prefill": {
        "backend": "flashinfer_cutlass",
        "flag": "--linear-backend flashinfer_cutlass",
        "why": (
            "309 TF/s against Marlin's 50 TF/s at M=4096 -- a 6.2x difference. Prefill is "
            "compute-bound, which is the one regime where FP4's 3.3x arithmetic advantage "
            "over BF16 is reachable."
        ),
    },
}


def recommend_kernels(workload: str = "mixed") -> Dict[str, Any]:
    """Kernel choice for an NVFP4 model on GB10, by workload.

    There is no single right answer: the fastest decode kernel is the slowest prefill kernel
    by 6.5x. vLLM's auto-selection picks reasonably for decode, so the actionable case is a
    prefill-heavy or long-prompt workload, where an explicit flag is worth multiples.
    """
    out: Dict[str, Any] = {"workload": workload}
    if workload in NVFP4_KERNELS:
        out.update(NVFP4_KERNELS[workload])
        return out
    out.update(
        backend = "marlin for decode, flashinfer_cutlass for prefill",
        flag = "(choose per workload; see `unsloth spark kernels --workload prefill`)",
        why = (
            "The crossover is at roughly 256 tokens per forward pass. Chat-style decode "
            "wants Marlin; RAG, summarisation and long-prompt workloads want CUTLASS."
        ),
    )
    return out


def _cmd_kernels(workload: str = "mixed") -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; these measurements do not apply.")
        return 0
    rec = recommend_kernels(workload)
    print(f"  workload : {rec['workload']}")
    print(f"  kernel   : {rec['backend']}")
    print(f"  flag     : {rec['flag']}")
    print("")
    print(f"  {rec['why']}")
    print("")
    print("  Measured on this hardware (same weights, same shape, kernel varied):")
    print("    kernel        acts   M=1        M=4096")
    print("    marlin        A16    429 us     29257 us /  50 TF")
    print("    fi_cutlass    A4     447 us      4727 us / 309 TF")
    print("    vllm_cutlass  A4     486 us      4511 us / 324 TF")
    print("    bf16          A16   1544 us     15339 us /  95 TF")
    print("")
    print("  Also on GB10: pin `nvidia-cutlass-dsl==4.6.2` -- 4.7.0 fails b12x with an")
    print("  internal DSL compiler error, disabling the kernel family built for this GPU.")
    return 0


def training_memory_estimate(
    size_gib: float,
    world: int,
    batch: int,
    microbatches: int,
    seq: int,
    hidden: int = 8192,
    layers: int = 80,
    vocab: int = 128256,
    checkpointed: bool = True,
) -> Dict[str, Any]:
    """Per-node memory for a layer-split training step, before it is attempted.

    This exists because the failure it prevents is severe and slow: a 70B arm loaded 66 GiB
    of weights, spent an hour materialising, then exhausted the node's memory on activations
    and left it **unreachable over ssh** -- the kernel and NIC stayed healthy while userspace
    could no longer fork, which needs a power cycle to clear. Discovering "this does not fit"
    an hour in, by taking a machine down, is the worst possible way to find out.

    Deliberately rough and deliberately pessimistic. The point is to refuse the obviously
    impossible, not to predict the last gigabyte.
    """
    weights = size_gib / world
    # Adam: fp32 master + two moments. LoRA trains a tiny fraction, so this is bounded by
    # trainable params rather than total -- but a full finetune pays all of it.
    optimizer_full = weights * 6.0
    optimizer_lora = weights * 0.02
    mb_rows = max(batch // max(microbatches, 1), 1)
    own_layers = max(layers // world, 1)
    # Residual stream per layer per in-flight microbatch, bf16. Checkpointing keeps one
    # layer's internals live instead of all of them, but still stores every boundary.
    per_layer = mb_rows * seq * hidden * 2 / 2**30
    live_microbatches = microbatches if not checkpointed else min(microbatches, 2)
    activations = per_layer * own_layers * live_microbatches
    if not checkpointed:
        activations *= 4.0  # attention + MLP intermediates kept for the backward pass
    # The LAST stage additionally holds logits, and for a large vocabulary that single tensor
    # can dominate everything else: cross-entropy is computed in fp32, so a 128k-vocab model
    # at 1024 tokens per microbatch is 0.5 GiB per microbatch in logits alone, plus the same
    # again for its gradient. Ignoring this is why a naive estimate says a configuration fits
    # when the final stage is the one that runs out.
    logits_gib = (mb_rows * seq * vocab * 4 / 2**30) * live_microbatches * 2
    activations_last_stage = activations + logits_gib
    budget = SPARK_USABLE_GIB - 6.0  # driver, CUDA context, fragmentation
    # Size the answer on the WORST stage, which is the last one.
    worst_activations = max(activations, activations_last_stage)
    total_lora = weights + optimizer_lora + worst_activations
    total_full = weights + optimizer_full + worst_activations
    return {
        "weights_gib": weights,
        "activations_gib": worst_activations,
        "logits_gib": logits_gib,
        "optimizer_lora_gib": optimizer_lora,
        "optimizer_full_gib": optimizer_full,
        "total_lora_gib": total_lora,
        "total_full_gib": total_full,
        "budget_gib": budget,
        "fits_lora": total_lora <= budget,
        "fits_full": total_full <= budget,
        "tokens_per_microbatch": mb_rows * seq,
    }


def _cmd_estimate(
    model: str, batch: int, microbatches: int, seq: int, full_finetune: bool, checkpointed: bool
) -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; nothing to estimate.")
        return 0
    size = model_size_gib(model)
    if size is None:
        print(f"Cannot size {model} (not cached locally); refusing to guess.")
        return 1
    world = 2 if peer_ip_for() else 1
    est = training_memory_estimate(size, world, batch, microbatches, seq, checkpointed = checkpointed)
    fits = est["fits_full"] if full_finetune else est["fits_lora"]
    total = est["total_full_gib"] if full_finetune else est["total_lora_gib"]
    mode = "full finetune" if full_finetune else "LoRA"

    print(f"  model      : {model}  ({size:.1f} GiB)")
    print(f"  stages     : {world}")
    print(
        f"  per node   : weights {est['weights_gib']:.1f} + "
        f"activations {est['activations_gib']:.1f} + optimizer "
        f"{(est['optimizer_full_gib'] if full_finetune else est['optimizer_lora_gib']):.1f}"
        f" = {total:.1f} GiB"
    )
    print(f"  budget     : {est['budget_gib']:.1f} GiB per node")
    print(
        f"  microbatch : {est['tokens_per_microbatch']} tokens"
        + (
            ""
            if est["tokens_per_microbatch"] >= 436
            else "  <-- BELOW the ~436-token crossover; the split cannot speed this up"
        )
    )
    print("")
    if fits:
        print(f"  OK: {mode} should fit, with " f"{est['budget_gib'] - total:.1f} GiB headroom.")
        return 0
    print(
        f"  WILL NOT FIT: {mode} needs {total:.1f} GiB against a "
        f"{est['budget_gib']:.1f} GiB budget."
    )
    print("  Reduce --batch or --seq, raise --microbatches, or add --grad-checkpoint.")
    if not checkpointed:
        print("  Gradient checkpointing alone would cut activations roughly 8x here.")
    return 1


def _consented(assume_yes: bool, prompt: str) -> bool:
    """Explicit yes, an interactive yes, or no. Never a default yes.

    Anything that writes to another machine or rewrites saved state has to be asked
    for. A command someone runs to SEE what it would do must not do it: the failure
    that motivated this rule was an `rsync --delete` of the studio venv onto a peer
    that was at that moment running a job out of it, reached by simply calling the
    setup entry point. Without a TTY and without an explicit flag the answer is no,
    so no automation, CI job or agent can trip it by accident.
    """
    if assume_yes:
        return True
    try:
        watching = sys.stdout.isatty()
    except (AttributeError, ValueError):
        watching = False
    if not watching:
        return False  # nobody is there to answer; the answer is no
    try:
        if sys.stdin.isatty():
            return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (AttributeError, ValueError, EOFError, KeyboardInterrupt, OSError):
        return False
    # stdout is a terminal but stdin is not: this is `curl ... | sh`, where the shell
    # script itself occupies stdin. install.sh handles the same situation by reading
    # /dev/tty (its `_can_read_tty`), so do exactly that rather than declining a
    # question the user is sitting in front of. If /dev/tty cannot be opened -- a
    # container, cron, CI -- that is a real "no terminal" and the answer stays no.
    try:
        with open("/dev/tty", "r") as tty:
            print(f"{prompt} [y/N] ", end = "", flush = True)
            return (tty.readline() or "").strip().lower() in ("y", "yes")
    except (OSError, EOFError, KeyboardInterrupt):
        # No prompt was printed if /dev/tty could not be opened, so there is no
        # dangling line to close -- and stdout may not be writable either.
        return False


def _cmd_setup(
    assume_yes: bool = False,
    n_nodes: int = 2,
    switched: bool = False,
    dry_run: bool = False,
) -> int:
    if not is_dgx_spark():
        print("Not a DGX Spark; nothing to do.")
        return 0
    rails = cabled_rails()
    if not rails:
        print("No cabled ConnectX rail found. Connect the QSFP cable between the two")
        print("Sparks (use the SAME physical port on both), then re-run:")
        print("    unsloth spark setup")
        return 1

    n_nodes = max(2, int(n_nodes))
    report = rail_plan_report(rails, node_index = 0, n_nodes = n_nodes, switched = switched)
    if not report["ok"]:
        # Refuse rather than emit a netplan that looks right and routes nowhere.
        print(f"Cannot plan addressing for {n_nodes} Sparks:")
        for problem in report["problems"]:
            print(f"  - {problem}")
        return 1
    plan = report["plan"]
    peer_plan = rail_plan(rails, node_index = 1, n_nodes = n_nodes, switched = switched)
    extra_plans = [
        rail_plan(rails, node_index = i, n_nodes = n_nodes, switched = switched)
        for i in range(2, n_nodes)
    ]
    for note in report["notes"]:
        print(f"  NOTE: {note}")
    print(
        "Detected a cabled second Spark on:" if n_nodes == 2 else f"Planning {n_nodes} Sparks on:"
    )
    for entry in plan:
        print(
            f"  {entry['ib_device']:<14} {entry['netdev']:<16} -> {entry['address']}/24 mtu {entry['mtu']}"
        )

    _print_manual_steps(plan, peer_plan, extra_plans = extra_plans)
    print("\n  NOTE: " + HOTPLUG_NOTE)

    print("\n  Then verify with:")
    print(f"    ping -c3 {peer_plan[0]['address']}")
    print("    unsloth spark status")

    # Provisioning prevents two failures that both present as hangs with no diagnostic
    # (601 s DistStoreError from a missing venv, 17-minute graph-capture hang from a cold
    # cache) -- but it WRITES TO ANOTHER MACHINE, so it is never automatic. Printing the
    # plan is free; performing it needs a yes.
    peer_now = peer_ip_for()
    changes = [f"rewrite {config_path()} with the plan above"]
    if peer_now:
        for path, label in provision_paths():
            if osp.isdir(osp.expanduser(path)):
                changes.append(f"rsync {path}/ -> {peer_now}:{path}/  ({label})")
    print("\n  This command would then:")
    for change in changes:
        print(f"    - {change}")
    if peer_now:
        print(f"    (the peer's GPU is checked first; a busy or unverifiable {peer_now}")
        print("     is refused, because a job may be running out of that venv)")

    if dry_run:
        print("\n  --dry-run: nothing was written here and nothing was sent to the peer.")
        if peer_now:
            print("  Preview the file list with: unsloth spark provision --dry-run")
        return 0
    if not _consented(assume_yes, "\n  Apply this plan and provision the peer?"):
        print("\n  Not applied. Nothing was written here and nothing was sent to the peer.")
        print("  Re-run with --yes to apply, or --dry-run to see it again.")
        return 0

    if peer_now:
        print(f"\n  Peer {peer_now} -- copying environment and caches over the ConnectX link:")
        res = provision_peer(peer_now)
        if res["refused"]:
            print(f"    REFUSED: {res['refused']}")
            print("    Nothing was copied. Re-run `unsloth spark provision` when it is idle.")
        for label, _ in res["copied"]:
            print(f"    ok      {label}")
        for label, why in res["failed"]:
            print(f"    FAILED  {label}: {why}")
        if res["failed"]:
            print("    Re-run later with: unsloth spark provision")
    else:
        print("\n  Once the peer is reachable, run: unsloth spark provision")

    save_config(
        {
            "enabled": True,
            "planned": True,
            "n_nodes": n_nodes,
            "switched": switched,
            "rails": plan,
            "peer_rails": peer_plan,
            "other_rails": extra_plans,
            "nccl_env": nccl_env(rails),
        }
    )
    print(f"\nSaved plan to {config_path()}")
    return 0


def _cmd_serve(
    model: str,
    port: int = 8080,
    rpc_port: int = RPC_DEFAULT_PORT,
    ctx: int = 8192,
    engines: int = 2,
    slots: int = 16,
) -> int:
    """Serve a GGUF across BOTH Sparks, using the layout that actually wins.

    Measured on this hardware, and the reason this prints what it prints:

      one engine, layer-split, a model that fits   decode 0.85x to 1.01x (1 to 32 users)
      TWO engines, each split, requests alternated 1.35x a single Spark
      two independent replicas, one per Spark      1.30x / 1.75x / 1.91x at 8 / 16 / 32 users
      prefill, `-ub 512` + CUDA_SCALE_LAUNCH_QUEUES=4x   1.51x

    A single split engine never decodes faster than one Spark: the split moves the
    same weight bytes per token and the nodes take turns on the graph. Two independent
    engines give the pair data-independent work, which is the same structure vLLM and
    SGLang require to fill a pipeline. A single autoregressive stream can never be
    pipelined -- token t+1 depends on token t.

    So use one split engine only when the model does not fit on one Spark (121.69
    GiB); for anything that fits, replicas win from 8 users up and a single Spark is
    as good below that. Before a split is printed, both nodes' llama.cpp bundles are
    compared and any running RPC server is asked its protocol version, because a
    peer on a different bundle fails at load with "RPC server version mismatch".
    """
    if not is_dgx_spark():
        print(NOT_A_SPARK)
        return 0
    plan = rpc_cluster_plan(rpc_port)
    if not plan["ok"]:
        for problem in plan["problems"]:
            print(f"  cannot serve across both Sparks: {problem}")
        return 1

    binary = plan["rpc_server"]
    peer_ip = plan["peer_ip"]
    engines = max(1, engines)

    # Decide the topology from the model rather than making the user know the rule. The
    # rule is not guessable: a model that FITS on one Spark never decodes faster
    # layer-split across two (0.85x to 1.01x measured), so splitting is for capacity
    # and prefill only, and the right answer flips at the point where two copies stop
    # fitting.
    size = model_size_gib(model)
    advice = plan_deployment(size, two_sparks = True, concurrency = slots)
    bin_dir = Path(binary).parent
    peer_bin_dir = _peer_relative_path(bin_dir)
    if advice["topology"] == "replicas":
        # Emit the layout that actually wins rather than advising and then printing a
        # worse one. Independent replicas never touch the wire during decode: each Spark
        # runs at full local memory bandwidth, and the only coordination is a
        # request-level round-robin on the CPU.
        local_port, peer_port = port + 1, port + 2
        print(f"  model    : {model}  ({size:.1f} GiB)")
        print("  topology : INDEPENDENT REPLICAS -- one full model per Spark, no RPC")
        print("")
        print("  Two copies fit, and for a model that fits this beats every split layout:")
        print("  a layer split never speeds up decode (0.85x to 1.01x measured from 1 to 32")
        print("  users), while two replicas measured 1.30x at 8 users, 1.75x at 16 and")
        print("  1.91x at 32. Below 8 concurrent users one Spark is as good as two.")
        print("")
        print("  1. This Spark:")
        print(f"     {bin_dir}/llama-server -m {model} \\")
        print(f"         -ngl 999 --ctx-size {ctx} -np {slots} -cb -ub 512 \\")
        print(f"         --host 0.0.0.0 --port {local_port}")
        print("")
        print(f"  2. The peer ({peer_ip}) -- the model must exist there; copy it over the")
        print("     ConnectX link rather than downloading (444 MB/s vs ~20 KB/s internet):")
        print(f"     rsync -a <model.gguf> {peer_ip}:<path>")
        print(f"     ssh {peer_ip} '{peer_bin_dir}/llama-server -m <path> \\")
        print(f"         -ngl 999 --ctx-size {ctx} -np {slots} -cb -ub 512 \\")
        print(f"         --host 0.0.0.0 --port {peer_port}'")
        print("")
        print("  3. Round-robin front end:")
        print(
            f"     python -m studio.spark_lb --port {port} "
            f"127.0.0.1:{local_port} {peer_ip}:{peer_port}"
        )
        print("")
        print(f"  Clients talk to port {port}. Nothing crosses the wire during decode.")
        return 0
    elif advice["topology"] in ("layer-split", "single-or-replicas") and engines > 1:
        # Two split engines need two full copies of the weights; that is exactly what
        # does not fit here, so silently obeying --engines 2 would OOM mid-load.
        print(f"  {model} is {size:.1f} GiB -- two copies do not fit across the pair.")
        print("  Forcing --engines 1 (layer split). This buys capacity, not speed.")
        print("")
        engines = 1
    elif advice["topology"] == "too-large":
        print(f"  {advice['summary']}")
        return 1

    # Both nodes must speak the same RPC protocol, and that is pinned by the build:
    # b10796 speaks 6.0, the bundles before it 5.1. Check the bundles and ask any
    # server that is already listening, before printing a launch that would fail at
    # load with "RPC server version mismatch".
    preflight = rpc_protocol_preflight(peer_ip, rpc_port)
    for note in preflight["notes"]:
        print(f"  note: {note}")
    if preflight["problems"]:
        for problem in preflight["problems"]:
            print(f"  RPC PROTOCOL: {problem}")
        print("")
        print("  Not printing a launch that would fail at load. Fix the above and re-run.")
        return 1
    print("")

    print(f"  model   : {model}")
    print(f"  engines : {engines} (each layer-split across both Sparks)")
    print(f"  peer    : {peer_ip}")
    print("")
    print(f"  1. Start {engines} rpc-server(s) on the peer, one per engine:")
    for i in range(engines):
        print(
            f"     ssh {peer_ip} '{peer_bin_dir}/{Path(binary).name} "
            f"-H 0.0.0.0 -p {rpc_port + i} -c'"
        )
    print("")
    print(f"  2. Start {engines} llama-server(s) on this Spark:")
    for i in range(engines):
        print(f"     {bin_dir}/llama-server -m {model} \\")
        print(f"         --rpc {peer_ip}:{rpc_port + i} -ngl 999 --ctx-size {ctx} \\")
        print(f"         -np {slots} -cb -ub 512 --host 127.0.0.1 --port {port + 1 + i}")
    print("")
    if engines > 1:
        ends = " ".join(f"127.0.0.1:{port + 1 + i}" for i in range(engines))
        print("  3. Put the round-robin front end in front of them:")
        print(f"     python -m studio.spark_lb --port {port} {ends}")
        print("")
        print(f"  Clients then talk to a single endpoint on port {port}.")
    print("")
    print("  Notes that change the numbers materially:")
    print("    * `-ub 512` -- the ubatch optimum INVERTS under a split (512 beats 1024).")
    print("    * CUDA_SCALE_LAUNCH_QUEUES=4x -- noise on one node, +4.2% when split.")
    print("    * Do NOT set GGML_CUDA_ENABLE_UNIFIED_MEMORY=1: measured -22% decode on")
    print("      GB10. It swaps cudaMalloc for cudaMallocManaged, and the memory-sizing")
    print("      benefit it is usually wanted for is already active via prop.integrated.")
    print("    * A model that FITS on one Spark is faster served by two independent")
    print("      single-node servers than by any cross-node split.")
    print("    * Restart engines between benchmark runs: llama-server at -np 32 degrades")
    print("      2.6x across successive load bursts against the same instance.")
    return 0


# ── Distributed TRAINING across both Sparks (torchrun + DDP) ─────────────────
# What is and is not possible, because the distinction trips people up:
#
#   device_map="balanced"  -- NO. That splits layers across GPUs *inside one
#                             process*. Each Spark is a separate host with a
#                             single GB10, so there is nothing to split across.
#   DDP (torchrun)         -- YES. Replicate the model on both, all-reduce the
#                             gradients. Buys THROUGHPUT, not capacity: the model
#                             must still fit on one Spark. Measured 1.71x.
#   FSDP (accelerate)      -- would buy capacity by sharding parameters, but is
#                             not something Unsloth supports today (see #4858),
#                             and GB10's lack of GPUDirect RDMA makes the
#                             per-step shard traffic expensive (~2.8 GB/s).
#
# For capacity, use llama.cpp RPC for inference (see rpc_cluster_plan) rather than
# expecting training to shard.


def _not_a_spark_plan(what: str) -> Dict[str, Any]:
    """The refusal every *_launch_plan returns off a Spark.

    Shaped exactly like a failed plan so callers need no special case: `ok` False and
    a `problems` list they already print. Guarding here rather than only in the CLI
    matters because these are importable functions -- `studio.spark_cluster` is called
    directly by the installer and by `unsloth run`, not only through `main()`.
    """
    return {
        "ok": False,
        "problems": [f"not a DGX Spark, so there is no peer to {what}"],
        "env": {},
        "node0": None,
        "node1": None,
        "peer_ip": None,
        "local_ip": None,
    }


def train_launch_plan(script: str, port: int = 29500) -> Dict[str, Any]:
    """torchrun commands for a two-Spark DDP run, plus the env both nodes need."""
    if not is_dgx_spark():
        return _not_a_spark_plan("train against")
    peer = peer_ip_for()
    local = None
    for rail in cabled_rails():
        if rail.get("ipv4"):
            local = rail["ipv4"][0]
            break
    if not peer or not local:
        return {"ok": False, "problems": ["no configured peer rail (run `unsloth spark setup`)"]}
    base = f"torchrun --nnodes=2 --nproc_per_node=1 --master_addr={local} " f"--master_port={port}"
    return {
        "ok": True,
        "problems": [],
        "env": nccl_env(),
        "node0": f"{base} --node_rank=0 {script}",
        "node1": f"{base} --node_rank=1 {script}",
        "peer_ip": peer,
        "local_ip": local,
    }


def pipeline_launch_plan(
    model: str,
    port: int = 29500,
    *,
    extra: str = "",
) -> Dict[str, Any]:
    """torchrun commands for a layer-split (pipeline-parallel) run across the Sparks.

    Distinct from `train_launch_plan`, which is DDP: DDP replicates the model and buys
    throughput, so the model must still fit on ONE Spark. This splits the decoder stack
    across the nodes, which buys *capacity* -- it is the only way to train a model larger
    than a single Spark's ~117 GiB.
    """
    if not is_dgx_spark():
        return _not_a_spark_plan("split a model across")
    peer = peer_ip_for()
    local = None
    for rail in cabled_rails():
        if rail.get("ipv4"):
            local = rail["ipv4"][0]
            break
    if not peer or not local:
        return {"ok": False, "problems": ["no configured peer rail (run `unsloth spark setup`)"]}
    base = f"torchrun --nnodes=2 --nproc_per_node=1 --master_addr={local} " f"--master_port={port}"
    target = f"-m studio.spark_pipeline --model {model}"
    if extra:
        target = f"{target} {extra}"
    return {
        "ok": True,
        "problems": [],
        "env": nccl_env(),
        "node0": f"{base} --node_rank=0 {target}",
        "node1": f"{base} --node_rank=1 {target}",
        "peer_ip": peer,
        "local_ip": local,
    }


def run_pipeline(plan: Dict[str, Any], log_peer: str = "/tmp/unsloth_pp_stage1.log") -> int:
    """Actually launch a layer-split run on both Sparks.

    The peer is started with `ssh -f` and its own log file. Both details matter: without
    `-f`, ssh holds the launcher open and the head rank never starts; and without a log
    redirect the peer's errors are lost entirely, because the head only ever reports
    `DistStoreError: Timed out ... 1/2 clients joined` and never says why its peer left.
    """
    user = os.environ.get("USER", "nvidianew")
    activate = venv_activate()
    env = "; ".join(f"export {k}={v}" for k, v in plan["env"].items())
    remote = (
        f"cd {os.getcwd()} && setsid nohup bash -c '[ -f {activate} ] && . {activate}; "
        f"{env}; exec {plan['node1']}' > {log_peer} 2>&1 < /dev/null &"
    )
    try:
        subprocess.run(
            [
                "ssh",
                "-f",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                f"{user}@{plan['peer_ip']}",
                remote,
            ],
            timeout = 60,
            check = False,
        )
    except Exception as exc:
        print(f"  could not start the peer stage: {exc}")
        return 1
    print(f"  peer stage started; its log is {log_peer} on {plan['peer_ip']}")
    import time

    time.sleep(6)  # let the peer reach the rendezvous first
    child_env = dict(os.environ)
    child_env.update({k: str(v) for k, v in plan["env"].items()})
    return subprocess.run(plan["node0"], shell = True, env = child_env).returncode


def _cmd_pipeline(
    model: str,
    port: int = 29500,
    extra: str = "",
    run: bool = False,
) -> int:
    if not is_dgx_spark():
        print(NOT_A_SPARK)
        return 0
    plan = pipeline_launch_plan(model, port, extra = extra)
    if not plan["ok"]:
        for problem in plan["problems"]:
            print(f"  cannot launch: {problem}")
        return 1

    # A layer split is for capacity. Warn when it is not needed, because a model that fits
    # on one Spark trains faster there than split across two.
    size = model_size_gib(model)
    if size is not None:
        budget = SPARK_USABLE_GIB - SERVE_OVERHEAD_GIB
        if size <= budget:
            print(
                f"  NOTE: {model} is {size:.1f} GiB and fits on ONE Spark "
                f"({budget:.0f} GiB budget)."
            )
            print("        A layer split buys capacity, not speed. Consider DDP")
            print("        (`--script`) for throughput instead.")
            print("")

    if run:
        return run_pipeline(plan)
    print("  Two-Spark layer-split training (capacity, not throughput -- this is how a")
    print("  model too large for one Spark gets trained; add --shard-load for those).")
    print("")
    print("  Export on BOTH nodes:")
    for key, value in plan["env"].items():
        print(f"    export {key}={value}")
    print("")
    print(f"  On this Spark ({plan['local_ip']}):")
    print(f"    {plan['node0']}")
    print(f"  On the peer ({plan['peer_ip']}):")
    print(f"    {plan['node1']}")
    print("")
    return 0


def _cmd_train(script: str, port: int = 29500) -> int:
    if not is_dgx_spark():
        print(NOT_A_SPARK)
        return 0
    plan = train_launch_plan(script, port)
    if not plan["ok"]:
        for problem in plan["problems"]:
            print(f"  cannot launch: {problem}")
        return 1
    print("  Two-Spark DDP training (throughput, not capacity -- the model must")
    print("  still fit on ONE Spark; use `unsloth spark serve` to split a model).")
    print("")
    print("  Export on BOTH nodes:")
    for key, value in plan["env"].items():
        print(f"    export {key}={value}")
    print("")
    print(f"  On this Spark ({plan['local_ip']}):")
    print(f"    {plan['node0']}")
    print(f"  On the peer ({plan['peer_ip']}):")
    print(f"    {plan['node1']}")
    print("")
    print("  Both nodes must have the SAME Unsloth/torch versions and the same")
    print("  optional kernels installed, or the ranks disagree on which path to run.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog = "unsloth spark", description = __doc__)
    parser.add_argument(
        "command",
        nargs = "?",
        default = "status",
        choices = (
            "status",
            "setup",
            "env",
            "detect",
            "serve",
            "train",
            "doctor",
            "provision",
            "plan",
            "kernels",
            "estimate",
            "peers",
        ),
    )
    parser.add_argument("--model", default = "", help = "GGUF path for `serve`")
    parser.add_argument("--script", default = "", help = "training script for `train`")
    parser.add_argument("--port", type = int, default = 8080)
    parser.add_argument("--ctx", type = int, default = 8192)
    parser.add_argument("--yes", "-y", action = "store_true")
    parser.add_argument(
        "--run",
        action = "store_true",
        help = "launch the layer-split run on both Sparks, not just print it",
    )
    parser.add_argument("--batch", type = int, default = 8)
    parser.add_argument("--microbatches", type = int, default = 4)
    parser.add_argument("--seq", type = int, default = 512)
    parser.add_argument("--full-finetune", action = "store_true")
    parser.add_argument("--grad-checkpoint", action = "store_true")
    parser.add_argument(
        "--workload",
        default = "mixed",
        choices = ("decode", "prefill", "mixed"),
        help = "which regime to optimise the kernel choice for",
    )
    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help = "show exactly what would be written or copied, and do none "
        "of it (works for `setup` and `provision`)",
    )
    parser.add_argument(
        "--rsync-delete",
        action = "store_true",
        help = "also delete files on the peer that are absent here. OFF by "
        "default: a stale extra file is far cheaper than deleting one "
        "a running job is executing.",
    )
    parser.add_argument(
        "--force",
        action = "store_true",
        help = "provision even though the peer GPU looks busy (or could not "
        "be checked). You are asserting no job is running there.",
    )
    parser.add_argument(
        "--engines",
        type = int,
        default = 2,
        help = "independent engines to run; >1 is what beats a single Spark",
    )
    parser.add_argument("--slots", type = int, default = 16, help = "server slots per engine")
    parser.add_argument(
        "--layer-split",
        default = "",
        help = "model to train split across the Sparks (capacity, not speed)",
    )
    parser.add_argument(
        "--pipeline-args", default = "", help = "extra flags passed through to studio.spark_pipeline"
    )
    parser.add_argument("--master-port", type = int, default = 29500)
    parser.add_argument(
        "--benchmark",
        action = "store_true",
        help = "measure the link with ib_write_bw (needs perftest on both nodes)",
    )
    parser.add_argument(
        "--intent",
        default = "throughput",
        choices = INTENTS,
        help = "what you want from the cluster; it decides the axis, "
        "not just whether the model fits",
    )
    parser.add_argument(
        "--nodes", type = int, default = None, help = "how many Sparks to plan for (default: discovered)"
    )
    parser.add_argument(
        "--concurrency",
        type = int,
        default = 1,
        help = "requests in flight, for the expected-speedup number",
    )
    parser.add_argument(
        "--prompt-tokens",
        type = int,
        default = 512,
        help = "typical prompt length, for the replicas/layer-split decision in `plan`",
    )
    parser.add_argument(
        "--prefill-heavy",
        action = "store_true",
        help = "the work is prefill-heavy long-prompt work (RAG, documents); the only "
        "case where `plan` will layer-split a model that fits",
    )
    parser.add_argument(
        "--switched",
        action = "store_true",
        help = "all Sparks share a switched RoCE fabric; required to plan "
        "addressing for more than two",
    )
    parser.add_argument("--no-probe", action = "store_true", help = "do not TCP-probe peers in `peers`")
    args = parser.parse_args(argv)

    if args.command == "detect":
        # Machine-readable, for install.sh: exit 0 only when a peer is cabled.
        print(json.dumps({"is_spark": is_dgx_spark(), "state": cluster_state()}))
        return 0 if cluster_state() in ("unconfigured", "configured") else 1
    if args.command == "doctor":
        return _cmd_doctor()
    if args.command == "estimate":
        if not args.model:
            print("estimate needs --model")
            return 2
        return _cmd_estimate(
            args.model,
            args.batch,
            args.microbatches,
            args.seq,
            args.full_finetune,
            args.grad_checkpoint,
        )
    if args.command == "kernels":
        return _cmd_kernels(args.workload)
    if args.command == "plan":
        if not args.model:
            print("plan needs --model <path-or-repo-id>")
            return 2
        return _cmd_plan(
            args.model,
            intent = args.intent,
            nodes = args.nodes,
            concurrency = args.concurrency,
            prompt_tokens = args.prompt_tokens,
            prefill_heavy = args.prefill_heavy,
        )
    if args.command == "peers":
        return _cmd_peers(check = not args.no_probe)
    if args.command == "provision":
        return _cmd_provision(dry_run = args.dry_run, delete = args.rsync_delete, force = args.force)
    if args.command == "env":
        return _cmd_env()
    if args.command == "serve":
        if not args.model:
            print("serve needs --model <path-to.gguf>")
            return 2
        return _cmd_serve(
            args.model, port = args.port, ctx = args.ctx, engines = args.engines, slots = args.slots
        )
    if args.command == "train":
        if args.layer_split:
            return _cmd_pipeline(
                args.layer_split, port = args.master_port, extra = args.pipeline_args, run = args.run
            )
        if not args.script:
            print("train needs --script <train.py>, or --layer-split <model>")
            return 2
        return _cmd_train(args.script)
    if args.command == "setup":
        return _cmd_setup(
            assume_yes = args.yes,
            n_nodes = args.nodes if args.nodes else 2,
            switched = args.switched,
            dry_run = args.dry_run,
        )
    return _cmd_status(benchmark = args.benchmark)


if __name__ == "__main__":
    raise SystemExit(main())
