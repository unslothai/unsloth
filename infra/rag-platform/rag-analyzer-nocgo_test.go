//go:build !cgo

package rag_analyzer

import "testing"

func TestPureGoAnalyzerContract(t *testing.T) {
	analyzer, err := NewAnalyzer("unused")
	if err != nil {
		t.Fatalf("NewAnalyzer: %v", err)
	}
	if err := analyzer.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}

	got, err := analyzer.Tokenize("Hello, 世界!")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if want := "Hello , 世 界 !"; got != want {
		t.Fatalf("Tokenize = %q, want %q", got, want)
	}

	positioned, err := analyzer.TokenizeWithPosition("A 世")
	if err != nil {
		t.Fatalf("TokenizeWithPosition: %v", err)
	}
	if len(positioned) != 2 || positioned[1].Offset != 2 || positioned[1].EndOffset != 5 {
		t.Fatalf("unexpected positioned tokens: %#v", positioned)
	}
	if analyzer.Copy() == nil {
		t.Fatal("Copy returned nil")
	}
}
