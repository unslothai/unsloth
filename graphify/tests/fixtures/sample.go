package main

import (
    "fmt"
    "net/http"
)

type Server struct {
    port int
}

func NewServer(port int) *Server {
    return &Server{port: port}
}

func (s *Server) Start() error {
    return http.ListenAndServe(fmt.Sprintf(":%d", s.port), nil)
}

func (s *Server) Stop() {
    fmt.Println("stopped")
}

func main() {
    s := NewServer(8080)
    s.Start()
}
