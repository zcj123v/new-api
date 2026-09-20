package deepseek

import (
	"encoding/json"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/stretchr/testify/require"
)

func TestApplyResponsesReasoningText(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{
			name:  "summary-only reasoning item gains reasoning_text",
			input: `[{"type":"reasoning","summary":[{"type":"summary_text","text":"draft A"}],"content":[],"encrypted_content":""}]`,
			want:  []string{"draft A"},
		},
		{
			name:  "multiple summary parts are mirrored in order",
			input: `[{"type":"reasoning","summary":[{"type":"summary_text","text":"a"},{"type":"summary_text","text":"b"}]}]`,
			want:  []string{"a", "b"},
		},
		{
			name:  "existing reasoning_text is left untouched",
			input: `[{"type":"reasoning","summary":[{"type":"summary_text","text":"ignored"}],"content":[{"type":"reasoning_text","text":"kept"}]}]`,
			want:  []string{"kept"},
		},
		{
			name:  "non-array input is left untouched",
			input: `"plain string prompt"`,
			want:  nil,
		},
		{
			name:  "reasoning item without summary is left untouched",
			input: `[{"type":"reasoning","content":[]}]`,
			want:  nil,
		},
		{
			name:  "non-reasoning items are ignored",
			input: `[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]`,
			want:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := dto.OpenAIResponsesRequest{Input: json.RawMessage(tt.input)}
			applyResponsesReasoningText(&request)

			if tt.want == nil {
				require.Equal(t, tt.input, string(request.Input))
				return
			}

			var items []map[string]any
			require.NoError(t, common.Unmarshal(request.Input, &items))
			require.Len(t, items, 1)

			parts, ok := items[0]["content"].([]any)
			require.True(t, ok, "content should be an array")

			texts := make([]string, 0, len(parts))
			for _, part := range parts {
				p, ok := part.(map[string]any)
				require.True(t, ok)
				require.Equal(t, "reasoning_text", p["type"])
				texts = append(texts, p["text"].(string))
			}
			require.Equal(t, tt.want, texts)
		})
	}
}
