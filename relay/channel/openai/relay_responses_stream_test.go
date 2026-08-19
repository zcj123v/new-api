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

// Regression test: upstream events must pass through byte-identical except for
// completed/done usage patches. Typed-struct re-serialization previously
// dropped custom_tool_call input, reasoning encrypted_content and
// sequence_number, which broke strict clients (e.g. Codex lost tool calls).
func TestOaiResponsesStreamHandlerPreservesUnknownFields(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	events := []string{
		`{"type":"response.output_item.added","sequence_number":3,"output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"ENC_DATA","summary":[]}}`,
		`{"type":"response.output_item.added","sequence_number":5,"output_index":1,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"exec","input":""}}`,
		`{"type":"response.custom_tool_call_input.delta","sequence_number":6,"item_id":"ctc_1","output_index":1,"delta":"const r = 1;"}`,
		`{"type":"response.custom_tool_call_input.done","sequence_number":7,"item_id":"ctc_1","output_index":1,"input":"const r = 1;"}`,
		`{"type":"response.output_item.done","sequence_number":8,"output_index":1,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"exec","input":"const r = 1;"}}`,
	}
	body := ""
	for _, e := range events {
		body += "data: " + e + "\n\n"
	}
	body += "data: [DONE]\n\n"

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set(common.RequestIdKey, "responses-stream-passthrough-test")
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

	_, apiErr := OaiResponsesStreamHandler(c, info, resp)
	require.Nil(t, apiErr)

	out := w.Body.String()
	assert.Contains(t, out, `"encrypted_content":"ENC_DATA"`,
		"reasoning encrypted_content must pass through (store:false clients chain reasoning across turns)")
	assert.Contains(t, out, `"input":"const r = 1;"`,
		"custom_tool_call input must pass through (clients read the final tool input from done events)")
	assert.Contains(t, out, `"sequence_number":8`,
		"sequence_number must pass through")
	for _, e := range events {
		assert.Contains(t, out, e, "non-completed events must be relayed byte-identical")
	}
}

// Regression test: the response.completed patch must preserve unknown fields
// (e.g. custom_tool_call input inside output items) while injecting
// output_tokens_details.
func TestOaiResponsesStreamHandlerCompletedPatchPreservesFields(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	completed := `{"type":"response.completed","sequence_number":20,"response":{"id":"resp_1","created_at":1786885892.5,"status":"completed","output":[{"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"exec","input":"const r = 1;"}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`
	body := "data: " + completed + "\n\n" + "data: [DONE]\n\n"

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set(common.RequestIdKey, "responses-stream-completed-patch-test")
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

	_, apiErr := OaiResponsesStreamHandler(c, info, resp)
	require.Nil(t, apiErr)

	out := w.Body.String()
	assert.Contains(t, out, `"output_tokens_details":{"reasoning_tokens":0}`,
		"completed event must be patched with output_tokens_details")
	assert.Contains(t, out, `"input":"const r = 1;"`,
		"completed patch must not drop custom_tool_call input")
	assert.Contains(t, out, `"sequence_number":20`,
		"completed patch must preserve sequence_number")
	assert.Contains(t, out, `"created_at":1786885892`,
		"completed patch must round created_at to an integer")
	assert.NotContains(t, out, `1786885892.5`)
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

// Regression test: an upstream stream that ends (EOF) without any terminal
// event must not leave strict clients (e.g. Codex) waiting forever. The
// handler synthesizes response.incomplete before the trailing [DONE].
func TestOaiResponsesStreamHandlerSynthesizesIncompleteOnEOF(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	events := []string{
		`{"type":"response.created","sequence_number":0,"response":{"id":"resp_eof1","object":"response","created_at":1786885892,"status":"in_progress","model":"gpt-5.1","output":[]}}`,
		`{"type":"response.output_item.added","sequence_number":3,"output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`,
	}
	body := ""
	for _, e := range events {
		body += "data: " + e + "\n\n"
	}
	// 注意：没有 response.completed，也没有 [DONE]——上游直接 EOF。

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set(common.RequestIdKey, "responses-stream-eof-test")
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

	_, apiErr := OaiResponsesStreamHandler(c, info, resp)
	require.Nil(t, apiErr)

	out := w.Body.String()
	assert.Contains(t, out, "event: response.incomplete",
		"must synthesize response.incomplete when upstream ends without terminal event")
	assert.Contains(t, out, `"id":"resp_eof1"`,
		"synthesized event must reuse the upstream response id")
	assert.Contains(t, out, `"status":"incomplete"`)
	assert.Contains(t, out, `"incomplete_details":{"reason":"upstream_eof"}`)
	assert.Contains(t, out, `"output_tokens_details":{"reasoning_tokens":0}`,
		"synthesized usage must carry output_tokens_details for strict clients")
	assert.Contains(t, out, `"input_tokens_details":{"cached_tokens":0}`,
		"synthesized usage must carry input_tokens_details for strict clients")
	assert.True(t, strings.HasSuffix(strings.TrimSpace(out), "data: [DONE]"),
		"stream must still end with [DONE]")
}

// 正常收到终止事件的流不得再合成 response.incomplete。
func TestOaiResponsesStreamHandlerNoSynthesisOnCompleted(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	completed := `{"type":"response.completed","response":{"id":"resp_ok","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`
	body := "data: " + completed + "\n\n" + "data: [DONE]\n\n"

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set(common.RequestIdKey, "responses-stream-no-synthesis-test")
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

	_, apiErr := OaiResponsesStreamHandler(c, info, resp)
	require.Nil(t, apiErr)
	assert.NotContains(t, w.Body.String(), "response.incomplete",
		"normal completed stream must not get a synthesized incomplete event")
}
