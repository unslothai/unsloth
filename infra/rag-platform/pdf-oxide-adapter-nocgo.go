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

package pdfoxide

import "errors"

var errNativePDFOxideUnavailable = errors.New("pdf_oxide is unavailable in the pure-Go server profile")

// Document keeps the no-CGO API surface compatible with the native adapter.
// The Python service remains responsible for PDF parsing in this profile.
type Document struct{}

// OpenBytes reports that the optional native PDF page-count fallback is not
// present. Callers retain their non-native fallback behavior.
func OpenBytes([]byte) (*Document, error) {
	return nil, errNativePDFOxideUnavailable
}

func (*Document) Close() {}

func (*Document) PageCount() (int, error) {
	return 0, errNativePDFOxideUnavailable
}
