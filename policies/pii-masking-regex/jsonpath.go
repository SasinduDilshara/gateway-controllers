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
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// The request side needs more of a JSONPath than reading one scalar out of it, which is all the
// SDK helpers do: it has to rewrite every value a path selects, wherever it sits in the
// document. A chat request carries the conversation as an array, and a path such as
// "$.messages[*].content" must mask every message, not refuse because the SDK cannot write
// through a wildcard. So the request side evaluates paths here.
//
// The grammar is the SDK's, so every path that worked before keeps working and means the same:
// dot-separated steps from an optional leading "$", where a step is an object key, "*" (every
// member of an object or array), or a key followed by array indexes. Two additions:
//   - "[*]" after a key selects every element of that array ("$.messages[*].content");
//   - indexes chain ("$.messages[-1].content[0].text" already parsed; "$.a[0][1]" now does too).
//
// Error messages for a path that does not resolve are the SDK's, word for word, because they
// reach the client in the mediation-failure body and clients may match on them.

type jsonPathStepKind int

const (
	stepKey jsonPathStepKind = iota
	stepIndex
	stepWildcard
)

// jsonPathStep is one step of a parsed JSONPath.
type jsonPathStep struct {
	kind jsonPathStepKind
	// key is the object key for stepKey. For stepIndex and an array wildcard it is the key of the
	// array being indexed, used only to word errors the way the SDK does ("not an array: content").
	key string
	// index is the array index for stepIndex; negative counts from the end.
	index int
	// raw is the index as written ("-1"), used only to word the out-of-range error.
	raw string
}

// parseJSONPath parses a JSONPath into steps. "" and "$" both parse to no steps: the whole
// document. A path that cannot be parsed is a configuration error, reported when the policy is
// created rather than on every request.
func parseJSONPath(path string) ([]jsonPathStep, error) {
	rest := strings.TrimSpace(path)
	if rest == "" || rest == "$" {
		return nil, nil
	}
	if strings.HasPrefix(rest, "$") {
		rest = rest[1:]
		// "$.a" and "$[0]" are both valid; only the "." separator after "$" is dropped.
		rest = strings.TrimPrefix(rest, ".")
	}
	if rest == "" {
		return nil, fmt.Errorf("invalid JSONPath %q", path)
	}

	var steps []jsonPathStep
	for _, part := range strings.Split(rest, ".") {
		if part == "" {
			return nil, fmt.Errorf("invalid JSONPath %q: empty step", path)
		}
		if part == "*" {
			steps = append(steps, jsonPathStep{kind: stepWildcard})
			continue
		}
		open := strings.IndexByte(part, '[')
		if open == -1 {
			if strings.ContainsAny(part, "]") {
				return nil, fmt.Errorf("invalid JSONPath %q: unexpected ']' in %q", path, part)
			}
			steps = append(steps, jsonPathStep{kind: stepKey, key: part})
			continue
		}
		name := part[:open]
		if strings.ContainsAny(name, "]*") {
			return nil, fmt.Errorf("invalid JSONPath %q: invalid key %q", path, name)
		}
		if name != "" {
			steps = append(steps, jsonPathStep{kind: stepKey, key: name})
		}
		brackets := part[open:]
		for brackets != "" {
			if brackets[0] != '[' {
				return nil, fmt.Errorf("invalid JSONPath %q: unexpected %q after an index in %q", path, brackets, part)
			}
			end := strings.IndexByte(brackets, ']')
			if end == -1 {
				return nil, fmt.Errorf("invalid JSONPath %q: unclosed '[' in %q", path, part)
			}
			inner := strings.TrimSpace(brackets[1:end])
			brackets = brackets[end+1:]
			if inner == "*" {
				steps = append(steps, jsonPathStep{kind: stepWildcard, key: name})
				continue
			}
			idx, err := strconv.Atoi(inner)
			if err != nil {
				return nil, fmt.Errorf("invalid JSONPath %q: invalid array index %q", path, inner)
			}
			steps = append(steps, jsonPathStep{kind: stepIndex, key: name, index: idx, raw: inner})
		}
	}
	return steps, nil
}

// rewriteFunc receives a value the path selected and returns its replacement and whether it
// changed. It is never handed a nil container to fill in: it only rewrites what is there.
type rewriteFunc func(value interface{}) (interface{}, bool)

// rewriteAtPath applies rewrite to every value steps select under node and returns the
// (possibly replaced) node and whether anything changed. Objects and arrays are updated in
// place, so the caller's document reflects every rewrite once this returns.
//
// A path that does not resolve is an error, worded as the SDK words it. Under a wildcard, a
// member the rest of the path does not resolve against is skipped rather than failing the whole
// path, as the SDK's wildcard does: "$.messages[*].content" must not reject a conversation
// because one assistant turn carries tool calls and no content.
func rewriteAtPath(node interface{}, steps []jsonPathStep, rewrite rewriteFunc) (interface{}, bool, error) {
	if len(steps) == 0 {
		replaced, changed := rewrite(node)
		return replaced, changed, nil
	}
	step, rest := steps[0], steps[1:]

	switch step.kind {
	case stepKey:
		obj, ok := node.(map[string]interface{})
		if !ok {
			return node, false, errors.New("invalid structure for key: " + step.key)
		}
		child, exists := obj[step.key]
		if !exists {
			return node, false, errors.New("key not found: " + step.key)
		}
		replaced, changed, err := rewriteAtPath(child, rest, rewrite)
		if err != nil {
			return node, false, err
		}
		if changed {
			obj[step.key] = replaced
		}
		return node, changed, nil

	case stepIndex:
		arr, ok := node.([]interface{})
		if !ok {
			return node, false, errors.New("not an array: " + step.key)
		}
		idx := step.index
		if idx < 0 {
			idx += len(arr)
		}
		if idx < 0 || idx >= len(arr) {
			return node, false, errors.New("array index out of range: " + step.raw)
		}
		replaced, changed, err := rewriteAtPath(arr[idx], rest, rewrite)
		if err != nil {
			return node, false, err
		}
		if changed {
			arr[idx] = replaced
		}
		return node, changed, nil

	case stepWildcard:
		anyChanged := false
		switch container := node.(type) {
		case map[string]interface{}:
			// Sorted so placeholder numbering does not depend on map iteration order.
			keys := make([]string, 0, len(container))
			for k := range container {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				replaced, changed, err := rewriteAtPath(container[k], rest, rewrite)
				if err != nil {
					continue
				}
				if changed {
					container[k] = replaced
					anyChanged = true
				}
			}
		case []interface{}:
			for i := range container {
				replaced, changed, err := rewriteAtPath(container[i], rest, rewrite)
				if err != nil {
					continue
				}
				if changed {
					container[i] = replaced
					anyChanged = true
				}
			}
		default:
			if step.key != "" {
				return node, false, errors.New("not an array: " + step.key)
			}
			return node, false, errors.New("wildcard used on non-iterable node")
		}
		return node, anyChanged, nil
	}
	return node, false, fmt.Errorf("unsupported JSONPath step")
}
