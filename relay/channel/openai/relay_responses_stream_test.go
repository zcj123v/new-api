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

// Regression test: the response.completed event sent to the client must carry
// usage.output_tokens_details even when the upstream reports reasoning_tokens
// only at the top level of usage (e.g. Moonshot/Kimi). The synthesis must
// happen before the event is re-serialized and written to the stream.
func TestOaiResponsesStreamHandlerCompletedEventIncludesOutputTokensDetails(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	completed := `{"type":"response.completed","response":{"status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"reasoning_tokens":3}}}`
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
	assert.Equal(t, 3, usage.ReasoningTokens)

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
	require.NotNil(t, completedEvent.Response.Usage.OutputTokensDetails,
		"output_tokens_details missing from response.completed event sent to client")
	assert.Equal(t, 3, completedEvent.Response.Usage.OutputTokensDetails.ReasoningTokens)
}
