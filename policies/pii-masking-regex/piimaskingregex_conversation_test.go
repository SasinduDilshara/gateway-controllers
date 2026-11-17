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

// These tests cover what an agent actually sends: a whole conversation on every request, content
// as a string or as parts, in more than one provider's format. Before this change the default
// JSONPath masked only the newest message, so every earlier turn reached the model unmasked, and a
// newest message sent as parts was skipped without a word.

import (
	"bytes"
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	policy "github.com/wso2/api-platform/sdk/core/policy/v1alpha2"
)

const (
	aliceEmail = "alice.chen@acmecorp.com"
	bobEmail   = "bob@example.org"
)

var emailPlaceholder = regexp.MustCompile(`\[EMAIL_[0-9a-f]{4}\]`)

// maskRequest runs the request phase and returns the body forwarded upstream (the original body
// when the policy made no change) and the request context, for its metadata.
func maskRequest(t *testing.T, params map[string]interface{}, body string) (string, *policy.RequestContext) {
	t.Helper()
	p := mustGetPIIPolicy(t, params)
	ctx := piiRequestContext(body)
	mods := mustPIIRequestMods(t, p.OnRequestBody(context.Background(), ctx, nil))
	if mods.Body == nil {
		return body, ctx
	}
	return string(mods.Body), ctx
}

func assertNoEmailLeft(t *testing.T, upstream string, emails ...string) {
	t.Helper()
	for _, email := range emails {
		if strings.Contains(upstream, email) {
			t.Fatalf("%q reached the upstream unmasked:\n%s", email, upstream)
		}
	}
}

func assertValidJSON(t *testing.T, upstream string) {
	t.Helper()
	if !json.Valid([]byte(upstream)) {
		t.Fatalf("the masked payload is not valid JSON:\n%s", upstream)
	}
}

// openAIConversation is the second turn of a chat as LangGraph/LangChain sends it to an
// OpenAI-compatible endpoint: system prompt first, then the history, then the newest message.
const openAIConversation = `{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are an IT helpdesk assistant. Escalations go to ` + bobEmail + `."},
    {"role": "user", "content": "Hi, my email is ` + aliceEmail + `. Please note it."},
    {"role": "assistant", "content": "Noted, ` + aliceEmail + `."},
    {"role": "user", "content": "Thanks. Spell my email back to me."}
  ]
}`

func TestDefaultPath_MasksEveryTurnOfTheConversation(t *testing.T) {
	upstream, ctx := maskRequest(t, map[string]interface{}{"email": true}, openAIConversation)

	assertValidJSON(t, upstream)
	assertNoEmailLeft(t, upstream, aliceEmail, bobEmail)

	mapping, ok := ctx.Metadata[MetadataKeyPIIEntities].(map[string]string)
	if !ok || len(mapping) != 2 {
		t.Fatalf("expected one placeholder per distinct email, got %#v", ctx.Metadata[MetadataKeyPIIEntities])
	}
	// The same address is the same placeholder in every message; different addresses never share one.
	if mapping[aliceEmail] == mapping[bobEmail] {
		t.Fatalf("two different emails share placeholder %q", mapping[aliceEmail])
	}
	if got := strings.Count(upstream, mapping[aliceEmail]); got != 2 {
		t.Fatalf("expected %s twice (user and assistant turn), found %d times:\n%s", mapping[aliceEmail], got, upstream)
	}
	// Structure and roles are untouched.
	var doc struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(upstream), &doc); err != nil {
		t.Fatal(err)
	}
	if doc.Model != "gpt-4o-mini" || len(doc.Messages) != 4 || doc.Messages[0].Role != "system" || doc.Messages[3].Content != "Thanks. Spell my email back to me." {
		t.Fatalf("unexpected structure after masking: %+v", doc)
	}
}

func TestDefaultPath_MasksContentParts(t *testing.T) {
	body := `{"messages":[
	  {"role":"system","content":"You are helpful."},
	  {"role":"user","content":[
	    {"type":"text","text":"My email is ` + aliceEmail + `"},
	    {"type":"image_url","image_url":{"url":"https://img.example.com/cat.png","detail":"low"}}
	  ]}]}`
	upstream, _ := maskRequest(t, map[string]interface{}{"email": true}, body)

	assertValidJSON(t, upstream)
	assertNoEmailLeft(t, upstream, aliceEmail)
	if !emailPlaceholder.MatchString(upstream) {
		t.Fatalf("expected a placeholder in the text part:\n%s", upstream)
	}
	if !strings.Contains(upstream, `"url":"https://img.example.com/cat.png"`) || !strings.Contains(upstream, `"type":"image_url"`) {
		t.Fatalf("a part without PII must pass through unchanged:\n%s", upstream)
	}
}

func TestDefaultPath_MasksOtherProviderFormats(t *testing.T) {
	tests := map[string]string{
		// Anthropic Messages: the system prompt is a top-level field and content is blocks.
		"anthropic": `{"model":"claude","system":"Escalate to ` + bobEmail + `","max_tokens":256,
		  "messages":[{"role":"user","content":[{"type":"text","text":"I am ` + aliceEmail + `"}]}]}`,
		// Gemini generateContent.
		"gemini": `{"systemInstruction":{"parts":[{"text":"Escalate to ` + bobEmail + `"}]},
		  "contents":[{"role":"user","parts":[{"text":"I am ` + aliceEmail + `"}]}]}`,
		// OpenAI Responses API.
		"responses": `{"model":"gpt-4o-mini","instructions":"Escalate to ` + bobEmail + `",
		  "input":[{"role":"user","content":[{"type":"input_text","text":"I am ` + aliceEmail + `"}]}]}`,
		// A top-level array is a JSON document too.
		"top-level array": `[{"note":"` + bobEmail + `"},{"note":"` + aliceEmail + `"}]`,
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			upstream, _ := maskRequest(t, map[string]interface{}{"email": true}, body)
			assertValidJSON(t, upstream)
			assertNoEmailLeft(t, upstream, aliceEmail, bobEmail)
		})
	}
}

// TestIssuePath_StringContentFailsClosed covers the configuration from the report:
// "$.messages[0].content[0].text". For an agent with a system prompt, messages[0] is the system
// prompt, and its content is a string, so the path does not exist in the request. The policy
// cannot know what it was meant to mask and refuses the request instead of forwarding it unmasked.
func TestIssuePath_StringContentFailsClosed(t *testing.T) {
	p := mustGetPIIPolicy(t, map[string]interface{}{"email": true, "jsonPath": "$.messages[0].content[0].text"})
	action := p.OnRequestBody(context.Background(), piiRequestContext(openAIConversation), nil)
	resp, ok := action.(policy.ImmediateResponse)
	if !ok {
		t.Fatalf("expected the request to be refused, got %T", action)
	}
	if resp.StatusCode != APIMInternalErrorCode || !resp.IsFault {
		t.Fatalf("expected a 500 mediation fault, got status=%d isFault=%v", resp.StatusCode, resp.IsFault)
	}
	if !bytes.Contains(resp.Body, []byte("error extracting value from JSONPath: not an array: content")) {
		t.Fatalf("unexpected body: %s", resp.Body)
	}
}

func TestIssuePath_ContentPartsAreMasked(t *testing.T) {
	body := `{"messages":[{"role":"user","content":[{"type":"text","text":"My email is ` + aliceEmail + `"}]}]}`
	upstream, _ := maskRequest(t, map[string]interface{}{"email": true, "jsonPath": "$.messages[0].content[0].text"}, body)
	assertValidJSON(t, upstream)
	assertNoEmailLeft(t, upstream, aliceEmail)
}

// TestExplicitPath_NonStringValueIsMaskedNotSkipped is the regression for the silent skip: a path
// that selects content parts used to forward them unmasked with no error at all.
func TestExplicitPath_NonStringValueIsMaskedNotSkipped(t *testing.T) {
	body := `{"messages":[{"role":"user","content":[{"type":"text","text":"Reach me at ` + aliceEmail + `"},{"type":"text","text":"or ` + bobEmail + `"}]}]}`
	upstream, ctx := maskRequest(t, map[string]interface{}{"email": true, "jsonPath": "$.messages[-1].content"}, body)
	assertValidJSON(t, upstream)
	assertNoEmailLeft(t, upstream, aliceEmail, bobEmail)
	if mapping, _ := ctx.Metadata[MetadataKeyPIIEntities].(map[string]string); len(mapping) != 2 {
		t.Fatalf("expected both emails recorded for restoration, got %#v", mapping)
	}
}

func TestExplicitPath_LastMessageOnlyStillMeansLastMessageOnly(t *testing.T) {
	// An explicit path keeps its meaning: only what it selects is masked.
	upstream, _ := maskRequest(t, map[string]interface{}{"email": true, "jsonPath": "$.messages[1].content"}, openAIConversation)
	if strings.Contains(upstream, "my email is "+aliceEmail) {
		t.Fatalf("the selected message must be masked:\n%s", upstream)
	}
	if !strings.Contains(upstream, bobEmail) || !strings.Contains(upstream, "Noted, "+aliceEmail) {
		t.Fatalf("messages the path does not select must be left as they are:\n%s", upstream)
	}
}

func TestWildcardPath_MasksEveryMessage(t *testing.T) {
	body := `{"messages":[
	  {"role":"user","content":"I am ` + aliceEmail + `"},
	  {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
	  {"role":"tool","tool_call_id":"c1","content":"record for ` + bobEmail + `"},
	  {"role":"user","content":[{"type":"text","text":"cc ` + bobEmail + `"}]}]}`
	upstream, _ := maskRequest(t, map[string]interface{}{"email": true, "jsonPath": "$.messages[*].content"}, body)
	assertValidJSON(t, upstream)
	assertNoEmailLeft(t, upstream, aliceEmail, bobEmail)
	if !strings.Contains(upstream, `"content":null`) {
		t.Fatalf("a null content must stay null:\n%s", upstream)
	}
}

func TestWholePayload_PreservesNumbersAndMarkup(t *testing.T) {
	// The seed does not fit a float64 exactly, and max_tokens looks like a phone number: both must
	// reach the upstream exactly as sent. Markup in a prompt must not become \u003c escapes.
	body := `{"seed":12345678901234567890,"max_tokens":2125551234,"temperature":0.25,"stream":false,"stop":null,
	  "messages":[{"role":"user","content":"<b>call 212-555-1234</b> & mail ` + aliceEmail + `"}]}`
	upstream, _ := maskRequest(t, map[string]interface{}{"email": true, "phone": true}, body)

	assertValidJSON(t, upstream)
	assertNoEmailLeft(t, upstream, aliceEmail, "212-555-1234")
	for _, want := range []string{`"seed":12345678901234567890`, `"max_tokens":2125551234`, `"temperature":0.25`, `"stream":false`, `"stop":null`, `<b>call [PHONE_`, `</b> & mail [EMAIL_`} {
		if !strings.Contains(upstream, want) {
			t.Fatalf("expected %s in the masked payload:\n%s", want, upstream)
		}
	}
}

func TestExplicitPath_NumberIsMaskedThroughItsText(t *testing.T) {
	// A path that points straight at a number masks it as a scalar, as it did before.
	upstream, _ := maskRequest(t, map[string]interface{}{"ssn": true, "jsonPath": "$.customer.ssn"}, `{"customer":{"ssn":123456789,"age":42}}`)
	assertValidJSON(t, upstream)
	if !strings.Contains(upstream, `"ssn":"[SSN_0000]"`) || !strings.Contains(upstream, `"age":42`) {
		t.Fatalf("unexpected payload: %s", upstream)
	}
}

func TestRedact_MasksEveryTurnPermanently(t *testing.T) {
	upstream, ctx := maskRequest(t, map[string]interface{}{"email": true, "redactPII": true}, openAIConversation)
	assertValidJSON(t, upstream)
	assertNoEmailLeft(t, upstream, aliceEmail, bobEmail)
	if got := strings.Count(upstream, "*****"); got != 3 {
		t.Fatalf("expected three redactions, got %d:\n%s", got, upstream)
	}
	if _, exists := ctx.Metadata[MetadataKeyPIIEntities]; exists {
		t.Fatalf("redaction must not record anything to restore")
	}
}

// TestMaskThenRestore_RoundTrip drives both phases: the model sees only placeholders, and the
// client gets every original back.
func TestMaskThenRestore_RoundTrip(t *testing.T) {
	p := mustGetPIIPolicy(t, map[string]interface{}{"email": true})
	reqCtx := piiRequestContext(openAIConversation)
	mods := mustPIIRequestMods(t, p.OnRequestBody(context.Background(), reqCtx, nil))
	assertNoEmailLeft(t, string(mods.Body), aliceEmail, bobEmail)

	mapping := reqCtx.Metadata[MetadataKeyPIIEntities].(map[string]string)
	modelReply := `{"choices":[{"index":0,"message":{"role":"assistant","content":"Your email is ` +
		mapping[aliceEmail] + `; escalations go to ` + mapping[bobEmail] + `."}}]}`
	respCtx := &policy.ResponseContext{
		SharedContext: reqCtx.SharedContext,
		ResponseBody:  &policy.Body{Content: []byte(modelReply), Present: true},
	}
	out, ok := p.OnResponseBody(context.Background(), respCtx, nil).(policy.DownstreamResponseModifications)
	if !ok {
		t.Fatalf("expected DownstreamResponseModifications")
	}
	want := "Your email is " + aliceEmail + "; escalations go to " + bobEmail + "."
	if !strings.Contains(string(out.Body), want) {
		t.Fatalf("expected %q in the restored response, got %s", want, out.Body)
	}
}

func TestExistingPlaceholderIsNeverReused(t *testing.T) {
	// The request already carries "[EMAIL_0000]" as text. A new email must get a different
	// placeholder, or the response restore would turn both into the same address.
	body := `{"messages":[{"role":"user","content":"earlier I was [EMAIL_0000]"},{"role":"user","content":"now I am ` + aliceEmail + `"}]}`
	upstream, ctx := maskRequest(t, map[string]interface{}{"email": true}, body)
	mapping := ctx.Metadata[MetadataKeyPIIEntities].(map[string]string)
	if mapping[aliceEmail] == "[EMAIL_0000]" {
		t.Fatalf("a placeholder already present in the request was reused")
	}
	if !strings.Contains(upstream, "earlier I was [EMAIL_0000]") {
		t.Fatalf("existing placeholder text must be left alone:\n%s", upstream)
	}
}

func TestMasking_IsDeterministic(t *testing.T) {
	params := map[string]interface{}{"email": true, "phone": true, "ssn": true}
	body := `{"messages":[{"role":"user","content":"` + aliceEmail + `, 212-555-1234, 123-45-6789, ` + bobEmail + `"}]}`
	first, _ := maskRequest(t, params, body)
	for i := 0; i < 20; i++ {
		if again, _ := maskRequest(t, params, body); again != first {
			t.Fatalf("masking is not deterministic:\n%s\n%s", first, again)
		}
	}
}

func TestWholePayload_NonJSONAndBinaryBodies(t *testing.T) {
	// Plain text is masked as text, as before.
	upstream, _ := maskRequest(t, map[string]interface{}{"email": true}, "contact "+aliceEmail)
	if upstream != "contact [EMAIL_0000]" {
		t.Fatalf("unexpected plain-text result %q", upstream)
	}

	// A binary body (not valid UTF-8) has no text to mask and is forwarded untouched.
	p := mustGetPIIPolicy(t, map[string]interface{}{"email": true})
	binary := append([]byte{0xff, 0xfe, 0x00}, []byte(" "+aliceEmail)...)
	ctx := &policy.RequestContext{
		SharedContext: &policy.SharedContext{RequestID: "req-id", Metadata: map[string]interface{}{}},
		Body:          &policy.Body{Content: binary, Present: true},
	}
	if mods := mustPIIRequestMods(t, p.OnRequestBody(context.Background(), ctx, nil)); mods.Body != nil {
		t.Fatalf("a binary body must not be rewritten, got %q", mods.Body)
	}
}

func TestEmptyAndWhitespaceBodiesPassThrough(t *testing.T) {
	for _, params := range []map[string]interface{}{
		{"email": true},
		{"email": true, "jsonPath": "$.messages[-1].content"},
	} {
		for _, body := range []string{"", "   \n\t"} {
			p := mustGetPIIPolicy(t, params)
			ctx := piiRequestContext(body)
			ctx.Body.Content = []byte(body)
			mods, ok := p.OnRequestBody(context.Background(), ctx, nil).(policy.UpstreamRequestModifications)
			if !ok || mods.Body != nil {
				t.Fatalf("params=%v body=%q: expected a pass-through", params, body)
			}
		}
	}
}

func TestExplicitPath_InvalidJSONBodyFailsClosed(t *testing.T) {
	p := mustGetPIIPolicy(t, map[string]interface{}{"email": true, "jsonPath": "$.messages[-1].content"})
	action := p.OnRequestBody(context.Background(), piiRequestContext(`not json `+aliceEmail), nil)
	if resp, ok := action.(policy.ImmediateResponse); !ok || resp.StatusCode != APIMInternalErrorCode {
		t.Fatalf("expected a 500 mediation fault, got %T", action)
	}
}

func TestGetPolicy_InvalidJSONPathIsAConfigurationError(t *testing.T) {
	_, err := GetPolicy(policy.PolicyMetadata{}, map[string]interface{}{"email": true, "jsonPath": "$.messages[x].content"})
	if err == nil || !strings.Contains(err.Error(), "'jsonPath' is invalid") {
		t.Fatalf("expected a configuration error, got %v", err)
	}
}
