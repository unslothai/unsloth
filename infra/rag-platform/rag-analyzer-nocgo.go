//go:build !cgo

// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rag_analyzer

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

type Token struct {
	Text      string
	Offset    uint32
	EndOffset uint32
}

type TokenWithPosition struct {
	Text      string
	Offset    uint32
	EndOffset uint32
}

// Analyzer is the deterministic pure-Go fallback used when native tokenizer
// bindings are unavailable. It keeps the server and route surface operational;
// the Python service remains the authoritative deep-document parser.
type Analyzer struct {
	fineGrained    bool
	enablePosition bool
}

func NewAnalyzer(string) (*Analyzer, error) {
	return &Analyzer{}, nil
}

func (*Analyzer) Load() error { return nil }

func (a *Analyzer) SetFineGrained(value bool) { a.fineGrained = value }

func (a *Analyzer) SetEnablePosition(value bool) { a.enablePosition = value }

func (a *Analyzer) Analyze(text string) ([]Token, error) {
	positioned := a.tokens(text)
	tokens := make([]Token, len(positioned))
	for index, token := range positioned {
		tokens[index] = Token(token)
	}
	return tokens, nil
}

func (a *Analyzer) Tokenize(text string) (string, error) {
	positioned := a.tokens(text)
	parts := make([]string, len(positioned))
	for index, token := range positioned {
		parts[index] = token.Text
	}
	return strings.Join(parts, " "), nil
}

func (a *Analyzer) TokenizeWithPosition(text string) ([]TokenWithPosition, error) {
	return a.tokens(text), nil
}

func (*Analyzer) Close() {}

func (a *Analyzer) FineGrainedTokenize(tokens string) (string, error) {
	copy := *a
	copy.fineGrained = true
	return copy.Tokenize(tokens)
}

func (*Analyzer) GetTermFreq(string) int32 { return 0 }

func (*Analyzer) GetTermTag(string) string { return "" }

func (a *Analyzer) Copy() *Analyzer {
	if a == nil {
		return nil
	}
	copy := *a
	return &copy
}

func (a *Analyzer) tokens(text string) []TokenWithPosition {
	tokens := make([]TokenWithPosition, 0)
	wordStart := -1

	flushWord := func(end int) {
		if wordStart < 0 {
			return
		}
		tokens = append(tokens, TokenWithPosition{
			Text:      text[wordStart:end],
			Offset:    uint32(wordStart),
			EndOffset: uint32(end),
		})
		wordStart = -1
	}

	for offset, current := range text {
		end := offset + utf8.RuneLen(current)
		switch {
		case unicode.IsSpace(current):
			flushWord(offset)
		case isCJK(current):
			flushWord(offset)
			tokens = append(tokens, TokenWithPosition{
				Text:      text[offset:end],
				Offset:    uint32(offset),
				EndOffset: uint32(end),
			})
		case unicode.IsLetter(current), unicode.IsDigit(current),
			unicode.IsMark(current), current == '_':
			if wordStart < 0 {
				wordStart = offset
			}
		case a.fineGrained && current == '-':
			flushWord(offset)
			tokens = append(tokens, TokenWithPosition{
				Text:      text[offset:end],
				Offset:    uint32(offset),
				EndOffset: uint32(end),
			})
		default:
			flushWord(offset)
			tokens = append(tokens, TokenWithPosition{
				Text:      text[offset:end],
				Offset:    uint32(offset),
				EndOffset: uint32(end),
			})
		}
	}
	flushWord(len(text))
	return tokens
}

func isCJK(current rune) bool {
	return unicode.Is(unicode.Han, current) ||
		unicode.Is(unicode.Hiragana, current) ||
		unicode.Is(unicode.Katakana, current) ||
		unicode.Is(unicode.Hangul, current)
}
