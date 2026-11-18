// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// unsloth-link carries Unsloth cluster traffic over tailcat tunnels. Secrets travel only on
// stdin/stdout (never argv, env or logs), and the process exits when stdin closes.
package main

import (
	"bufio"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/tailscale/tailcat"
	"tailscale.com/derp/derpserver"
	"tailscale.com/net/stun"
	"tailscale.com/tailcfg"
	"tailscale.com/types/key"
	"tailscale.com/types/logger"
	"tailscale.com/wgengine/filter"
)

const protocolVersion = 1

const dialTimeout = 15 * time.Second

var tailcatVersion = "dev"

var (
	logf       logger.Logf = logger.Discard
	derpMapURL             = tailcat.DefaultDERPMapURL
	devDERP    *tailcfg.DERPRegion
)

type request struct {
	ID         int64    `json:"id"`
	Op         string   `json:"op"`
	Ports      []uint16 `json:"ports,omitempty"`
	KeyPath    string   `json:"key_path,omitempty"`
	Allow      []string `json:"allow,omitempty"`
	AllowAll   bool     `json:"allow_all,omitempty"`
	Pubkey     string   `json:"pubkey,omitempty"`
	Address    string   `json:"address,omitempty"`
	RemotePort uint16   `json:"remote_port,omitempty"`
	TimeoutS   float64  `json:"timeout_s,omitempty"`
}

type response map[string]any

func main() {
	if len(os.Args) > 1 && os.Args[1] == "genkey" {
		os.Exit(runGenkey(os.Args[2:]))
	}
	if len(os.Args) > 1 && os.Args[1] == "devderp" {
		os.Exit(runDevDERPMap())
	}
	if len(os.Args) < 2 || os.Args[1] != "stdio" {
		fmt.Fprintln(os.Stderr, "usage: unsloth-link stdio [--derpmap-url URL] | genkey --kind server|client --path P [--fixed-region]")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("stdio", flag.ExitOnError)
	url := fs.String("derpmap-url", "", "DERP map URL")
	verbose := fs.Bool("verbose", false, "tailcat debug logs on stderr")
	parent := fs.Int("parent-pid", 0, "exit once this process is no longer the parent")
	fs.Parse(os.Args[2:])
	setup(*url, *verbose)
	h := &helper{clients: map[string]*client{}}
	if *parent > 0 {
		// A forked grandchild can keep stdin open after the parent dies: exit on reparenting too.
		go func() {
			for os.Getppid() == *parent {
				time.Sleep(time.Second)
			}
			h.close()
			os.Exit(0)
		}()
	}
	out := json.NewEncoder(os.Stdout)
	var outMu sync.Mutex
	send := func(r response) {
		outMu.Lock()
		defer outMu.Unlock()
		out.Encode(r)
	}
	send(response{"event": "ready", "version": protocolVersion, "tailcat": tailcatVersion})
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 64<<10), 1<<20)
	var wg sync.WaitGroup
	for in.Scan() {
		var req request
		if err := json.Unmarshal(in.Bytes(), &req); err != nil {
			send(response{"id": nil, "ok": false, "error": "bad request line"})
			continue
		}
		if req.Op == "shutdown" {
			send(response{"id": req.ID, "ok": true})
			break
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := h.handle(req)
			if res == nil {
				res = response{}
			}
			res["id"] = req.ID
			res["ok"] = err == nil
			if err != nil {
				res["error"] = err.Error()
			}
			send(res)
		}()
	}
	h.close()
	os.Exit(0)
}

func setup(url string, verbose bool) {
	if url == "" {
		url = os.Getenv("TAILCAT_DERPMAP_URL")
	}
	if url != "" {
		derpMapURL = url
	}
	if verbose {
		logf = log.New(os.Stderr, "", log.LstdFlags).Printf
	}
	if os.Getenv("TS_DEBUG_TAILCAT_LOCAL_DERP") == "1" {
		devDERP = runDevDERP()
	}
}

type helper struct {
	mu      sync.Mutex
	srv     *server
	clients map[string]*client // by address
}

func (h *helper) handle(req request) (response, error) {
	switch req.Op {
	case "serve":
		return h.serve(req)
	case "allow", "revoke":
		h.mu.Lock()
		s := h.srv
		h.mu.Unlock()
		if s == nil {
			return nil, errors.New("not serving")
		}
		var k key.NodePublic
		if err := k.UnmarshalText([]byte(req.Pubkey)); err != nil {
			return nil, fmt.Errorf("invalid public key: %v", err)
		}
		if req.Op == "allow" {
			s.allowKey(k)
			return nil, nil
		}
		return response{"was_connected": s.revoke(k)}, nil
	case "forward":
		c, err := h.client(req.Address, req.KeyPath)
		if err != nil {
			return nil, err
		}
		port, err := c.forward(req.RemotePort)
		if err != nil {
			return nil, err
		}
		return response{"local_port": port}, nil
	case "unforward":
		h.mu.Lock()
		c := h.clients[req.Address]
		if c != nil && req.RemotePort == 0 {
			delete(h.clients, req.Address)
		}
		h.mu.Unlock()
		if c != nil {
			if req.RemotePort == 0 {
				c.close()
			} else {
				c.unforward(req.RemotePort)
			}
		}
		return nil, nil
	case "probe":
		c, err := h.client(req.Address, req.KeyPath)
		if err != nil {
			return nil, err
		}
		return c.probe(req.TimeoutS)
	case "pubkey":
		if req.KeyPath == "" {
			return nil, errors.New("key_path required")
		}
		pk, err := readKey(req.KeyPath)
		if err != nil {
			return nil, err
		}
		return response{"pubkey": pk.Private.Public().String()}, nil
	}
	return nil, fmt.Errorf("unknown op %q", req.Op)
}

func (h *helper) close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, c := range h.clients {
		c.close()
	}
	if h.srv != nil {
		h.srv.ts.Close()
	}
}

type server struct {
	ts      *tailcat.Server
	address string
	keyPath string

	mu       sync.Mutex
	allowAll bool
	allowed  tailcat.KeySet
	revoked  tailcat.KeySet
	ports    map[uint16]net.Listener
	conns    map[key.NodePublic]map[net.Conn]struct{}
}

func (s *server) admit(k key.NodePublic) bool {
	if s.revoked.Contains(k) {
		return false
	}
	s.mu.Lock()
	all := s.allowAll
	s.mu.Unlock()
	return all || s.allowed.Contains(k)
}

func (s *server) allowKey(k key.NodePublic) {
	s.revoked.Remove(k)
	s.allowed.Add(k)
}

func (s *server) revoke(k key.NodePublic) bool {
	s.allowed.Remove(k)
	s.revoked.Add(k)
	s.mu.Lock()
	conns := s.conns[k]
	delete(s.conns, k)
	s.mu.Unlock()
	// Close before disconnecting, so the resets still reach the client instead of its forwards stalling.
	for c := range conns {
		c.Close()
	}
	if len(conns) > 0 {
		time.Sleep(250 * time.Millisecond)
	}
	return s.ts.DisconnectClient(k) || len(conns) > 0
}

func (s *server) track(k key.NodePublic, cs ...net.Conn) (untrack func()) {
	s.mu.Lock()
	m := s.conns[k]
	if m == nil {
		m = map[net.Conn]struct{}{}
		s.conns[k] = m
	}
	for _, c := range cs {
		m[c] = struct{}{}
	}
	s.mu.Unlock()
	return func() {
		s.mu.Lock()
		for _, c := range cs {
			delete(s.conns[k], c)
		}
		s.mu.Unlock()
	}
}

func (s *server) listen(port uint16) error {
	s.mu.Lock()
	_, dup := s.ports[port]
	s.mu.Unlock()
	if dup {
		return nil
	}
	ln, err := s.ts.Listen(context.Background(), "tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.ports[port] = ln
	s.mu.Unlock()
	target := net.JoinHostPort("127.0.0.1", fmt.Sprint(port))
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				k, ok := s.ts.PeerKey(c.RemoteAddr())
				if !ok || !s.admit(k) {
					c.Close()
					return
				}
				local, err := net.DialTimeout("tcp", target, 10*time.Second)
				if err != nil {
					logf("dial local port %d: %v", port, err)
					c.Close()
					return
				}
				untrack := s.track(k, c, local)
				defer untrack()
				tailcat.ProxyConns(c, local)
			}()
		}
	}()
	return nil
}

func (h *helper) serve(req request) (response, error) {
	if len(req.Ports) == 0 {
		return nil, errors.New("ports required")
	}
	h.mu.Lock()
	s := h.srv
	h.mu.Unlock()
	if s != nil {
		if req.KeyPath != s.keyPath {
			return nil, errors.New("already serving with a different key")
		}
		for _, k := range req.Allow {
			var pk key.NodePublic
			if err := pk.UnmarshalText([]byte(k)); err != nil {
				return nil, fmt.Errorf("invalid allowed key: %v", err)
			}
			s.allowKey(pk)
		}
		for _, p := range req.Ports {
			if err := s.listen(p); err != nil {
				return nil, err
			}
		}
		return response{"address": s.address}, nil
	}
	s = &server{keyPath: req.KeyPath, allowAll: req.AllowAll, ports: map[uint16]net.Listener{}, conns: map[key.NodePublic]map[net.Conn]struct{}{}}
	for _, k := range req.Allow {
		var pk key.NodePublic
		if err := pk.UnmarshalText([]byte(k)); err != nil {
			return nil, fmt.Errorf("invalid allowed key: %v", err)
		}
		s.allowed.Add(pk)
	}
	var priv *tailcat.PrivateKey
	if req.KeyPath == "" {
		priv = tailcat.NewPrivateKey()
		priv.Public.RegionID = 0
	} else {
		var err error
		if priv, err = readKey(req.KeyPath); err != nil {
			return nil, err
		}
	}
	ts := &tailcat.Server{
		Key:            priv.Private,
		PresharedKey:   priv.Public.PresharedKey,
		Logf:           logf,
		DERPMapURL:     derpMapURL,
		AllowClient:    s.admit,
		ServedTCPPorts: []filter.PortRange{}, // listener ports only
	}
	if ts.PresharedKey.IsZero() {
		return nil, errors.New("key has no pre-shared key")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// A fixed region keeps the short address stable across restarts.
	short := ""
	switch {
	case devDERP != nil:
		ts.Region = devDERP
	case len(priv.Public.Region) > 0:
		ts.Region = priv.Public.Region[0]
	case priv.Public.RegionID > 0:
		ci := priv.Public
		ci.Region = nil
		if err := ci.Expand(ctx, tailcat.ExpandForServer, tailcat.DERPMapURL(derpMapURL)); err != nil {
			return nil, fmt.Errorf("DERP region %d: %v", priv.Public.RegionID, err)
		}
		ts.Region = ci.Region[0]
		pub := tailcat.ConnInfo{
			ServerPublic:      tailcat.NodePublic{NodePublic: priv.Private.Public()},
			ServerDiscoPublic: tailcat.DiscoPublicForNode(priv.Private),
			PresharedKey:      priv.Public.PresharedKey,
			RegionID:          priv.Public.RegionID,
		}
		short = string(pub.Addr())
	}
	if err := ts.StartContext(ctx); err != nil {
		return nil, fmt.Errorf("start: %v", err)
	}
	s.ts = ts
	s.address = short
	if s.address == "" {
		s.address = string(ts.TailcatAddr())
	}
	for _, p := range req.Ports {
		if err := s.listen(p); err != nil {
			ts.Close()
			return nil, err
		}
	}
	h.mu.Lock()
	if h.srv != nil {
		h.mu.Unlock()
		ts.Close()
		return nil, errors.New("already serving")
	}
	h.srv = s
	h.mu.Unlock()
	return response{"address": s.address, "pubkey": priv.Private.Public().String()}, nil
}

type client struct {
	tc      *tailcat.Client
	keyPath string

	mu       sync.Mutex
	closed   bool
	forwards map[uint16]net.Listener
}

func (h *helper) client(address, keyPath string) (*client, error) {
	if address == "" {
		return nil, errors.New("address required")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if c := h.clients[address]; c != nil {
		if keyPath != "" && keyPath != c.keyPath {
			return nil, errors.New("already connected to this address with a different key")
		}
		return c, nil
	}
	if _, err := tailcat.ParseAddr(tailcat.Addr(address)); err != nil {
		return nil, errors.New("not a tailcat address")
	}
	tc := &tailcat.Client{Server: tailcat.Addr(address), Logf: logf, DERPMapURL: derpMapURL}
	if keyPath != "" {
		pk, err := readKey(keyPath)
		if err != nil {
			return nil, err
		}
		tc.Key = pk.Private
	}
	c := &client{tc: tc, keyPath: keyPath, forwards: map[uint16]net.Listener{}}
	h.clients[address] = c
	return c, nil
}

func (c *client) forward(port uint16) (int, error) {
	if port == 0 {
		return 0, errors.New("remote_port required")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0, errors.New("forward was removed")
	}
	if ln := c.forwards[port]; ln != nil {
		return ln.Addr().(*net.TCPAddr).Port, nil
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	c.forwards[port] = ln
	go func() {
		for {
			local, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				// tailcat drops a refused client silently: give up soon so callers see EOF, not a hang.
				ctx, cancel := context.WithTimeout(context.Background(), dialTimeout)
				remote, err := c.tc.DialTCPPort(ctx, port)
				cancel()
				if err != nil {
					logf("dial remote port %d: %v", port, err)
					local.Close()
					return
				}
				tailcat.ProxyConns(local, remote)
			}()
		}
	}()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

func (c *client) unforward(port uint16) {
	c.mu.Lock()
	ln := c.forwards[port]
	delete(c.forwards, port)
	c.mu.Unlock()
	if ln != nil {
		ln.Close()
	}
}

func (c *client) close() {
	c.mu.Lock()
	c.closed = true
	for p, ln := range c.forwards {
		ln.Close()
		delete(c.forwards, p)
	}
	c.mu.Unlock()
	c.tc.Close()
}

func (c *client) probe(timeoutS float64) (response, error) {
	if timeoutS <= 0 {
		timeoutS = 10
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutS*float64(time.Second)))
	defer cancel()
	var last response
	var lastErr error
	for {
		pctx, pcancel := context.WithTimeout(ctx, 3*time.Second)
		r, err := c.tc.DiscoPing(pctx)
		pcancel()
		if err == nil {
			last = response{"rtt_ms": r.LatencySeconds * 1000}
			if r.Endpoint != "" {
				last["direct"] = true
				last["via"] = "direct " + r.Endpoint
				return last, nil
			}
			last["direct"] = false
			last["via"] = "relay " + r.DERPRegionCode
		} else {
			lastErr = err
		}
		select {
		case <-ctx.Done():
			if last != nil {
				return last, nil
			}
			if lastErr == nil {
				lastErr = ctx.Err()
			}
			return nil, fmt.Errorf("no answer: %v", lastErr)
		case <-time.After(250 * time.Millisecond):
		}
	}
}

func readKey(path string) (*tailcat.PrivateKey, error) {
	j, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var pk tailcat.PrivateKey
	if err := json.Unmarshal(j, &pk); err != nil {
		return nil, fmt.Errorf("key file %s is not a tailcat key", path)
	}
	if pk.Private.IsZero() {
		return nil, fmt.Errorf("key file %s has no private key", path)
	}
	return &pk, nil
}

func runGenkey(args []string) int {
	fs := flag.NewFlagSet("genkey", flag.ExitOnError)
	kind := fs.String("kind", "", "server or client")
	path := fs.String("path", "", "key file to write")
	fixed := fs.Bool("fixed-region", false, "pick the nearest DERP region now and bake it into the key")
	force := fs.Bool("force", false, "overwrite an existing file")
	url := fs.String("derpmap-url", "", "DERP map URL")
	fs.Parse(args)
	if (*kind != "server" && *kind != "client") || *path == "" {
		fmt.Fprintln(os.Stderr, "genkey: --kind server|client and --path are required")
		return 2
	}
	if !*force {
		// A key left by an interrupted first run is kept: print its public key again.
		if pk, err := readKey(*path); err == nil {
			fmt.Println(pk.Private.Public().String())
			return 0
		} else if !errors.Is(err, os.ErrNotExist) {
			fmt.Fprintln(os.Stderr, "genkey:", err)
			return 1
		}
	}
	setup(*url, false)
	priv := tailcat.NewPrivateKey()
	if *kind == "client" {
		priv.Public = tailcat.ConnInfo{}
	} else if *fixed && devDERP == nil {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		dm, err := tailcat.FetchDERPMap(ctx, tailcat.DERPMapURL(derpMapURL), tailcat.ExpandForServer)
		if err != nil {
			fmt.Fprintf(os.Stderr, "genkey: DERP map: %v\n", err)
			return 1
		}
		id, err := tailcat.PickBestRegion(ctx, dm)
		if err != nil || id == 0 {
			fmt.Fprintf(os.Stderr, "genkey: no reachable DERP region: %v\n", err)
			return 1
		}
		priv.Public.RegionID = id
	}
	j, err := json.MarshalIndent(priv, "", "\t")
	if err != nil {
		fmt.Fprintln(os.Stderr, "genkey:", err)
		return 1
	}
	if err := writeKey(*path, j, *force); err != nil {
		if errors.Is(err, os.ErrExist) {
			if pk, rerr := readKey(*path); rerr == nil {
				fmt.Println(pk.Private.Public().String())
				return 0
			}
		}
		fmt.Fprintln(os.Stderr, "genkey:", err)
		return 1
	}
	fmt.Println(priv.Private.Public().String())
	return 0
}

// writeKey publishes a complete key (CreateTemp makes it 0600) or nothing; without force an existing key is never replaced.
func writeKey(path string, data []byte, force bool) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if force {
		return os.Rename(name, path)
	}
	if err := os.Link(name, path); err != nil {
		if _, serr := os.Lstat(path); serr == nil {
			return os.ErrExist
		}
		return os.Rename(name, path)
	}
	return nil
}

// Loopback DERP for tests (TS_DEBUG_TAILCAT_LOCAL_DERP=1), as in cmd/tailcat.

func runDevDERP() *tailcfg.DERPRegion {
	d := derpserver.New(key.NewNode(), logf)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	srv := httptest.NewUnstartedServer(derpserver.Handler(d))
	srv.Listener = ln
	srv.Config.ErrorLog = log.New(io.Discard, "", 0)
	srv.Config.TLSNextProto = make(map[string]func(*http.Server, *tls.Conn, http.Handler))
	srv.StartTLS()
	uln, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		panic(err)
	}
	go func() {
		var buf [1500]byte
		for {
			n, src, err := uln.ReadFromUDPAddrPort(buf[:])
			if err != nil {
				return
			}
			txid, err := stun.ParseBindingRequest(buf[:n])
			if err != nil {
				continue
			}
			uln.WriteToUDPAddrPort(stun.Response(txid, src), src)
		}
	}()
	return &tailcfg.DERPRegion{
		RegionID:   1,
		RegionCode: "D",
		Nodes: []*tailcfg.DERPNode{{
			Name:             "t1",
			RegionID:         1,
			HostName:         "T",
			IPv4:             "127.0.0.1",
			IPv6:             "none",
			STUNPort:         uln.LocalAddr().(*net.UDPAddr).Port,
			DERPPort:         ln.Addr().(*net.TCPAddr).Port,
			InsecureForTests: true,
		}},
	}
}

// One loopback DERP shared by test processes through a map URL, so fixed-region keys behave as with the public map.
func runDevDERPMap() int {
	region := runDevDERP()
	region.RegionID, region.Nodes[0].RegionID = 900, 900
	body, err := json.Marshal(&tailcfg.DERPMap{Regions: map[tailcfg.DERPRegionID]*tailcfg.DERPRegion{900: region}})
	if err != nil {
		fmt.Fprintln(os.Stderr, "devderp:", err)
		return 1
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Fprintln(os.Stderr, "devderp:", err)
		return 1
	}
	go http.Serve(ln, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	}))
	fmt.Printf("http://%s/derpmap.json\n", ln.Addr())
	io.Copy(io.Discard, os.Stdin)
	return 0
}
