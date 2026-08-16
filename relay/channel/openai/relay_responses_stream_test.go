package openai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// runCompletedEventStream feeds a single upstream response.completed event
// carrying the given usage JSON through OaiResponsesStreamHandler and returns
// the relayed usage plus the response.completed event as the client received
// it on the wire.
func runCompletedEventStream(t *testing.T, usageJSON string) (*dto.Usage, dto.ResponsesStreamResponse) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	completed := `{"type":"response.completed","response":{"status":"completed","output":[],"usage":` + usageJSON + `}}`
	body := "data: " + completed + "\n\n" + "data: [DONE]\n\n"

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set(common.RequestIdKey, "responses-stream-usage-test")
	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.1",
		DisablePing:     true,
		ChannelMeta: &relaycommon.ChannelMeta{
			UpstreamModelName: "gpt-5.1",
		},
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
	}

	usage, apiErr := OaiResponsesStreamHandler(c, info, resp)
	require.Nil(t, apiErr)
	require.NotNil(t, usage)

	var completedEvent dto.ResponsesStreamResponse
	found := false
	for _, line := range strings.Split(w.Body.String(), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var event dto.ResponsesStreamResponse
		require.NoError(t, common.UnmarshalJsonStr(data, &event))
		if event.Type == "response.completed" {
			completedEvent = event
			found = true
		}
	}
	require.True(t, found, "response.completed event not found in client stream")
	require.NotNil(t, completedEvent.Response)
	require.NotNil(t, completedEvent.Response.Usage)
	return usage, completedEvent
}

// Regression test: the response.completed event sent to the client must carry
// usage.output_tokens_details even when the upstream reports reasoning_tokens
// only at the top level of usage (e.g. Moonshot/Kimi). The synthesis must
// happen before the event is re-serialized and written to the stream.
func TestOaiResponsesStreamHandlerCompletedEventIncludesOutputTokensDetails(t *testing.T) {
	usage, completedEvent := runCompletedEventStream(t,
		`{"input_tokens":10,"output_tokens":5,"total_tokens":15,"reasoning_tokens":3}`)

	assert.Equal(t, 3, usage.ReasoningTokens)
	require.NotNil(t, completedEvent.Response.Usage.OutputTokensDetails,
		"output_tokens_details missing from response.completed event sent to client")
	assert.Equal(t, 3, completedEvent.Response.Usage.OutputTokensDetails.ReasoningTokens)
}

// Regression test: upstream OpenAI always emits output_tokens_details (with
// reasoning_tokens possibly 0) and strict clients such as Codex require the
// field to exist. When the upstream usage carries no reasoning tokens at all
// (e.g. Kimi local counting), the completed event must still include
// output_tokens_details with reasoning_tokens 0.
func TestOaiResponsesStreamHandlerCompletedEventIncludesZeroOutputTokensDetails(t *testing.T) {
	usage, completedEvent := runCompletedEventStream(t,
		`{"input_tokens":10,"output_tokens":5,"total_tokens":15}`)

	assert.Equal(t, 0, usage.ReasoningTokens)
	require.NotNil(t, completedEvent.Response.Usage.OutputTokensDetails,
		"output_tokens_details missing from response.completed event sent to client")
	assert.Equal(t, 0, completedEvent.Response.Usage.OutputTokensDetails.ReasoningTokens)
}
