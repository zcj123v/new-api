package setting

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/QuantumNous/new-api/common"
)

// ModelMaxInputTokensOptionKey is the global option that caps how many input
// tokens the gateway accepts for a model, e.g. {"gpt-6-sol":272000}. A model
// missing from the object has no limit.
const ModelMaxInputTokensOptionKey = "ModelMaxInputTokens"

// modelMaxInputTokens holds the parsed option as an immutable map, so the relay
// read path never races with an option update.
var modelMaxInputTokens atomic.Pointer[map[string]int]

// ModelMaxInputTokens2JSONString returns the default option value: no limits.
func ModelMaxInputTokens2JSONString() string {
	return "{}"
}

// GetModelMaxInputTokens returns the configured limit for model. 0 means the
// model has no configured limit.
func GetModelMaxInputTokens(model string) int {
	limits := modelMaxInputTokens.Load()
	if limits == nil || model == "" {
		return 0
	}
	return (*limits)[model]
}

// LoadModelMaxInputTokensFromJSONString replaces the configured limits. A
// broken option must never reject traffic, so invalid input falls back to
// "no limit" (whole object) or skips the offending entry, and only logs.
func LoadModelMaxInputTokensFromJSONString(value string) {
	limits, err := parseModelMaxInputTokens(value)
	if err != nil {
		common.SysError("invalid " + ModelMaxInputTokensOptionKey + " option, all models stay unlimited: " + err.Error())
		limits = map[string]int{}
	}
	modelMaxInputTokens.Store(&limits)
}

func parseModelMaxInputTokens(value string) (map[string]int, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return map[string]int{}, nil
	}
	raw := json.RawMessage(trimmed)
	if common.GetJsonType(raw) != "object" {
		return nil, fmt.Errorf("expected a JSON object of model name to token limit")
	}
	var rawLimits map[string]json.RawMessage
	if err := common.Unmarshal(raw, &rawLimits); err != nil {
		return nil, err
	}

	limits := make(map[string]int, len(rawLimits))
	invalid := make([]string, 0)
	for model, rawLimit := range rawLimits {
		var limit int
		switch {
		case model == "":
			invalid = append(invalid, "empty model name")
		case common.GetJsonType(rawLimit) != "number":
			invalid = append(invalid, model+" (not a number)")
		case common.Unmarshal(rawLimit, &limit) != nil:
			invalid = append(invalid, model+" (not an integer)")
		case limit < 0:
			invalid = append(invalid, fmt.Sprintf("%s (%d)", model, limit))
		case limit == 0:
			// 0 means "no limit", same as omitting the model.
		default:
			limits[model] = limit
		}
	}
	if len(invalid) > 0 {
		sort.Strings(invalid)
		common.SysError(ModelMaxInputTokensOptionKey + " entries ignored (negative or malformed, treated as unlimited): " + strings.Join(invalid, ", "))
	}
	return limits, nil
}
