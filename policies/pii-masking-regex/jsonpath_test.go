/*
 *  Copyright (c) 2026, WSO2 LLC. (http://www.wso2.org) All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package piimaskingregex

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	utils "github.com/wso2/api-platform/sdk/core/utils"
)

func TestParseJSONPath_Valid(t *testing.T) {
	tests := []struct {
		path string
		want []jsonPathStep
	}{
		{"", nil},
		{"$", nil},
		{"  $  ", nil},
		{"$.content", []jsonPathStep{{kind: stepKey, key: "content"}}},
		{"content", []jsonPathStep{{kind: stepKey, key: "content"}}},
		{"$.messages[-1].content", []jsonPathStep{
			{kind: stepKey, key: "messages"},
			{kind: stepIndex, key: "messages", index: -1, raw: "-1"},
			{kind: stepKey, key: "content"},
		}},
		{"$.messages[0].content[0].text", []jsonPathStep{
			{kind: stepKey, key: "messages"},
			{kind: stepIndex, key: "messages", index: 0, raw: "0"},
			{kind: stepKey, key: "content"},
			{kind: stepIndex, key: "content", index: 0, raw: "0"},
			{kind: stepKey, key: "text"},
		}},
		{"$.messages[*].content", []jsonPathStep{
			{kind: stepKey, key: "messages"},
			{kind: stepWildcard, key: "messages"},
			{kind: stepKey, key: "content"},
		}},
		{"$.messages.*.content", []jsonPathStep{
			{kind: stepKey, key: "messages"},
			{kind: stepWildcard},
			{kind: stepKey, key: "content"},
		}},
		{"$.grid[1][-2]", []jsonPathStep{
			{kind: stepKey, key: "grid"},
			{kind: stepIndex, key: "grid", index: 1, raw: "1"},
			{kind: stepIndex, key: "grid", index: -2, raw: "-2"},
		}},
		{"$[0].text", []jsonPathStep{
			{kind: stepIndex, index: 0, raw: "0"},
			{kind: stepKey, key: "text"},
		}},
		{"$.user-data.e-mail", []jsonPathStep{
			{kind: stepKey, key: "user-data"},
			{kind: stepKey, key: "e-mail"},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got, err := parseJSONPath(tt.path)
			if err != nil {
				t.Fatalf("parseJSONPath(%q) error: %v", tt.path, err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("parseJSONPath(%q)\n got  %+v\n want %+v", tt.path, got, tt.want)
			}
		})
	}
}

func TestParseJSONPath_Invalid(t *testing.T) {
	for _, path := range []string{
		"$.",
		"$.a..b",
		"$.a.",
		"$.a[",
		"$.a[x]",
		"$.a[1.5]",
		"$.a]b",
		"$.a[0]x",
		"$.a*[0]",
	} {
		t.Run(path, func(t *testing.T) {
			if _, err := parseJSONPath(path); err == nil {
				t.Fatalf("parseJSONPath(%q): expected an error", path)
			}
		})
	}
}

// TestRewriteAtPath_ErrorsMatchSDK pins the wording of a path that does not resolve to the SDK's,
// word for word. The message reaches the client in the mediation-failure body, so a change here is
// a client-visible change.
func TestRewriteAtPath_ErrorsMatchSDK(t *testing.T) {
	const payload = `{"messages":[{"role":"system","content":"be helpful"},{"role":"user","content":"hi"}],"n":1}`
	for _, path := range []string{
		"$.nonexistent.field",
		"$.messages[0].content[0].text", // the path from the reported issue: content is a string
		"$.messages[5].content",
		"$.messages[-9].content",
		"$.n.value",
		"$.n[0]",
		"$.missing[0]",
	} {
		t.Run(path, func(t *testing.T) {
			var sdkDoc map[string]interface{}
			if err := json.Unmarshal([]byte(payload), &sdkDoc); err != nil {
				t.Fatal(err)
			}
			_, sdkErr := utils.ExtractValueFromJsonpath(sdkDoc, path)
			if sdkErr == nil {
				t.Fatalf("test setup: the SDK resolves %q", path)
			}

			doc, err := decodeJSONDocument([]byte(payload))
			if err != nil {
				t.Fatal(err)
			}
			steps, err := parseJSONPath(path)
			if err != nil {
				t.Fatal(err)
			}
			_, _, gotErr := rewriteAtPath(doc, steps, func(v interface{}) (interface{}, bool) { return v, false })
			if gotErr == nil {
				t.Fatalf("expected %q not to resolve", path)
			}
			if gotErr.Error() != sdkErr.Error() {
				t.Fatalf("error wording differs from the SDK:\n got  %q\n want %q", gotErr.Error(), sdkErr.Error())
			}
		})
	}
}

// TestRewriteAtPath_SelectsWhatTheSDKSelects checks that every path the SDK could read selects the
// same value here, so no configuration that worked before changes meaning.
func TestRewriteAtPath_SelectsWhatTheSDKSelects(t *testing.T) {
	const payload = `{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"u1"},{"role":"user","content":"u2"}],"meta":{"owner":{"email":"x"}}}`
	for _, path := range []string{
		"$.messages[-1].content",
		"$.messages[0].content",
		"$.messages[1].role",
		"$.meta.owner.email",
		"meta.owner.email",
	} {
		t.Run(path, func(t *testing.T) {
			var sdkDoc map[string]interface{}
			if err := json.Unmarshal([]byte(payload), &sdkDoc); err != nil {
				t.Fatal(err)
			}
			want, err := utils.ExtractValueFromJsonpath(sdkDoc, path)
			if err != nil {
				t.Fatalf("test setup: the SDK cannot resolve %q: %v", path, err)
			}

			doc, _ := decodeJSONDocument([]byte(payload))
			steps, err := parseJSONPath(path)
			if err != nil {
				t.Fatal(err)
			}
			var selected []interface{}
			if _, _, err := rewriteAtPath(doc, steps, func(v interface{}) (interface{}, bool) {
				selected = append(selected, v)
				return v, false
			}); err != nil {
				t.Fatalf("rewriteAtPath(%q): %v", path, err)
			}
			if len(selected) != 1 || !reflect.DeepEqual(selected[0], want) {
				t.Fatalf("selected %v, the SDK selects %v", selected, want)
			}
		})
	}
}

func TestRewriteAtPath_WildcardSkipsMembersWithoutThePath(t *testing.T) {
	// An assistant turn that only calls tools has no content key, and must not fail the request.
	doc, _ := decodeJSONDocument([]byte(`{"messages":[
		{"role":"user","content":"a"},
		{"role":"assistant","tool_calls":[{"id":"1"}]},
		{"role":"user","content":"b"}]}`))
	steps, _ := parseJSONPath("$.messages[*].content")
	var seen []string
	_, changed, err := rewriteAtPath(doc, steps, func(v interface{}) (interface{}, bool) {
		seen = append(seen, v.(string))
		return strings.ToUpper(v.(string)), true
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !changed || !reflect.DeepEqual(seen, []string{"a", "b"}) {
		t.Fatalf("expected both contents to be selected, got changed=%v seen=%v", changed, seen)
	}
	out, _ := encodeJSONDocument(doc)
	if !strings.Contains(string(out), `"content":"A"`) || !strings.Contains(string(out), `"content":"B"`) {
		t.Fatalf("expected both contents rewritten in place, got %s", out)
	}
}

func TestRewriteAtPath_WildcardOnScalarIsAnError(t *testing.T) {
	doc, _ := decodeJSONDocument([]byte(`{"messages":"not a list"}`))
	steps, _ := parseJSONPath("$.messages[*].content")
	if _, _, err := rewriteAtPath(doc, steps, func(v interface{}) (interface{}, bool) { return v, false }); err == nil ||
		err.Error() != "not an array: messages" {
		t.Fatalf("expected 'not an array: messages', got %v", err)
	}
}

func TestDecodeJSONDocument_RejectsTrailingData(t *testing.T) {
	if _, err := decodeJSONDocument([]byte(`{"a":1} {"b":2}`)); err == nil {
		t.Fatalf("expected trailing data to be rejected")
	}
	if _, err := decodeJSONDocument([]byte("{\"a\":1}\n  ")); err != nil {
		t.Fatalf("trailing whitespace must be accepted: %v", err)
	}
}
