package controller

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/middleware"
	"github.com/QuantumNous/new-api/model"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	kitdto "github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/service"
	"github.com/QuantumNous/new-api/setting"
	"github.com/QuantumNous/new-api/setting/ratio_setting"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The limit must accept every request the operator did not configure, and reject
// only what exceeds the resolved limit, before any channel work or billing.
func TestModelMaxInputTokensDecision(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Cleanup(func() { setting.LoadModelMaxInputTokensFromJSONString("{}") })

	longText := strings.Repeat("token ", 200)
	tools := fmt.Sprintf(`{"type":"function","function":{"name":"lookup","description":%q,"parameters":{"type":"object"}}}`, longText)

	decodeChat := func(body string) *kitdto.GeneralOpenAIRequest {
		request := &kitdto.GeneralOpenAIRequest{}
		require.NoError(t, common.UnmarshalJsonStr(body, request))
		return request
	}
	decodeResponses := func(body string) *kitdto.OpenAIResponsesRequest {
		request := &kitdto.OpenAIResponsesRequest{}
		require.NoError(t, common.UnmarshalJsonStr(body, request))
		return request
	}
	responsesBody := func(text string, toolJSON string) string {
		if toolJSON == "" {
			return fmt.Sprintf(`{"model":"limit-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":%q}]}]}`, text)
		}
		return fmt.Sprintf(`{"model":"limit-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":%q}]}],"tools":[%s]}`, text, toolJSON)
	}

	cases := []struct {
		name       string
		config     string
		channelMax int
		format     types.RelayFormat
		request    kitdto.Request
		wantLimit  int
	}{
		{"no config keeps a large chat request unlimited", `{}`, 0, types.RelayFormatOpenAI,
			decodeChat(fmt.Sprintf(`{"model":"limit-model","messages":[{"role":"user","content":%q}]}`, longText)), 0},
		{"chat request over the model limit is rejected", `{"limit-model":50}`, 0, types.RelayFormatOpenAI,
			decodeChat(fmt.Sprintf(`{"model":"limit-model","messages":[{"role":"user","content":%q}]}`, longText)), 50},
		{"chat request under the model limit passes", `{"limit-model":50}`, 0, types.RelayFormatOpenAI,
			decodeChat(`{"model":"limit-model","messages":[{"role":"user","content":"hi"}]}`), 0},
		{"tool definitions count towards the chat request", `{"limit-model":50}`, 0, types.RelayFormatOpenAI,
			decodeChat(fmt.Sprintf(`{"model":"limit-model","messages":[{"role":"user","content":"hi"}],"tools":[%s]}`, tools)), 50},
		{"responses request over the model limit is rejected", `{"limit-model":50}`, 0, types.RelayFormatOpenAIResponses,
			decodeResponses(responsesBody(longText, "")), 50},
		{"responses request under the model limit passes", `{"limit-model":50}`, 0, types.RelayFormatOpenAIResponses,
			decodeResponses(responsesBody("hi", "")), 0},
		{"tool definitions count towards the responses request", `{"limit-model":50}`, 0, types.RelayFormatOpenAIResponses,
			decodeResponses(responsesBody("hi", tools)), 50},
		{"channel override wins over the model option", `{"limit-model":100000}`, 50, types.RelayFormatOpenAI,
			decodeChat(fmt.Sprintf(`{"model":"limit-model","messages":[{"role":"user","content":%q}]}`, longText)), 50},
		{"channel override -1 disables the limit", `{"limit-model":50}`, -1, types.RelayFormatOpenAI,
			decodeChat(fmt.Sprintf(`{"model":"limit-model","messages":[{"role":"user","content":%q}]}`, longText)), 0},
		{"broken option value leaves every request unlimited", `{"limit-model":50`, 0, types.RelayFormatOpenAI,
			decodeChat(fmt.Sprintf(`{"model":"limit-model","messages":[{"role":"user","content":%q}]}`, longText)), 0},
		{"non object option value leaves every request unlimited", `"50"`, 0, types.RelayFormatOpenAI,
			decodeChat(fmt.Sprintf(`{"model":"limit-model","messages":[{"role":"user","content":%q}]}`, longText)), 0},
		{"negative option entry is ignored as unlimited", `{"limit-model":-5}`, 0, types.RelayFormatOpenAI,
			decodeChat(fmt.Sprintf(`{"model":"limit-model","messages":[{"role":"user","content":%q}]}`, longText)), 0},
		{"unrecognized request shape is skipped", `{"limit-model":50}`, 0, types.RelayFormatEmbedding,
			&kitdto.EmbeddingRequest{Model: "limit-model"}, 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			setting.LoadModelMaxInputTokensFromJSONString(tc.config)
			ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
			if tc.channelMax != 0 {
				common.SetContextKey(ctx, constant.ContextKeyChannelSetting, kitdto.ChannelSettings{MaxInputTokens: tc.channelMax})
			}
			info := &relaycommon.RelayInfo{Request: tc.request, RelayFormat: tc.format, OriginModelName: "limit-model"}

			apiErr := service.CheckModelMaxInputTokens(ctx, info)

			if tc.wantLimit == 0 {
				require.Nil(t, apiErr)
				return
			}
			require.NotNil(t, apiErr)
			assert.Equal(t, http.StatusBadRequest, apiErr.StatusCode)
			assert.Equal(t, types.ErrorCodeContextLengthExceeded, apiErr.GetErrorCode())
			assert.True(t, types.IsSkipRetryError(apiErr), "the limit is a model property, retrying another channel cannot help")
			assert.Equal(t, types.ErrorCodeContextLengthExceeded, apiErr.ToOpenAIError().Code)
			assert.Contains(t, apiErr.Error(), fmt.Sprintf("This model's maximum context length is %d tokens", tc.wantLimit))
			assert.Regexp(t, `however you requested \d+ tokens\. Please reduce the length of your input\.$`, apiErr.Error())
		})
	}
}

type inputLimitChannel struct {
	model   string
	setting kitdto.ChannelSettings
}

type inputLimitFixture struct {
	engine        *gin.Engine
	tokenKey      string
	user          *model.User
	token         *model.Token
	useChannels   []string
	upstreamCalls atomic.Int64
}

func newInputLimitFixture(t *testing.T, channels []inputLimitChannel) *inputLimitFixture {
	t.Helper()
	user, token := setupResponsesWSRequestTest(t)
	// InitDB skips migrations for non-master nodes, so the fixture migrates the
	// tables this flow touches: options (option writes), user_subscriptions and
	// logs (pre-consume and consume-log assertions), channels/abilities (routing).
	require.NoError(t, model.DB.AutoMigrate(&model.Channel{}, &model.Ability{}, &model.Option{}, &model.UserSubscription{}))
	require.NoError(t, model.LOG_DB.AutoMigrate(&model.Log{}))
	// The option map is process state built at startup; UpdateOption writes
	// through it, so the fixture must initialize it like the real server does.
	model.InitOptionMap()
	common.OptionMapRWMutex.RLock()
	_, optionRegistered := common.OptionMap[setting.ModelMaxInputTokensOptionKey]
	common.OptionMapRWMutex.RUnlock()
	require.True(t, optionRegistered, "ModelMaxInputTokens must be a registered global option")

	previousRatios := ratio_setting.ModelRatio2JSONString()
	require.NoError(t, ratio_setting.UpdateModelRatioByJSONString(
		`{"limit-chat":1,"limit-responses":1,"limit-override":1,"limit-unlimited":1,"limit-free":1}`))
	t.Cleanup(func() { require.NoError(t, ratio_setting.UpdateModelRatioByJSONString(previousRatios)) })

	previousBatch, previousLogs := common.BatchUpdateEnabled, common.LogConsumeEnabled
	common.BatchUpdateEnabled, common.LogConsumeEnabled = false, true
	t.Cleanup(func() { common.BatchUpdateEnabled, common.LogConsumeEnabled = previousBatch, previousLogs })

	require.NoError(t, model.DB.Model(user).Update("quota", 100000000).Error)
	require.NoError(t, model.DB.Model(token).Update("remain_quota", 100000000).Error)

	fixture := &inputLimitFixture{tokenKey: token.Key, user: user, token: token}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fixture.upstreamCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/responses") {
			_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","created_at":1786588600,"status":"completed","model":"limit-responses","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"limit-chat","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`))
	}))
	t.Cleanup(upstream.Close)

	for _, spec := range channels {
		channel := &model.Channel{
			Name:    "input-limit-upstream",
			Key:     "sk-input-limit",
			Status:  common.ChannelStatusEnabled,
			Type:    constant.ChannelTypeOpenAI,
			Group:   "default",
			Models:  spec.model,
			BaseURL: &upstream.URL,
		}
		if spec.setting.MaxInputTokens != 0 {
			channel.SetSetting(spec.setting)
		}
		require.NoError(t, model.DB.Create(channel).Error)
		require.NoError(t, model.DB.Create(&model.Ability{ChannelId: channel.Id, Model: spec.model, Group: "default", Enabled: true}).Error)
	}

	engine := gin.New()
	engine.POST("/v1/chat/completions", middleware.TokenAuth(), middleware.Distribute(), func(c *gin.Context) {
		c.Set(common.RequestIdKey, "input-limit-chat")
		Relay(c, types.RelayFormatOpenAI)
		fixture.useChannels = c.GetStringSlice("use_channel")
	})
	engine.POST("/v1/responses", middleware.TokenAuth(), middleware.Distribute(), func(c *gin.Context) {
		c.Set(common.RequestIdKey, "input-limit-responses")
		Relay(c, types.RelayFormatOpenAIResponses)
		fixture.useChannels = c.GetStringSlice("use_channel")
	})
	fixture.engine = engine
	return fixture
}

func (f *inputLimitFixture) setLimit(t *testing.T, value string) {
	t.Helper()
	require.NoError(t, model.UpdateOption(setting.ModelMaxInputTokensOptionKey, value))
	t.Cleanup(func() { require.NoError(t, model.UpdateOption(setting.ModelMaxInputTokensOptionKey, "{}")) })
}

func (f *inputLimitFixture) post(t *testing.T, path string, body string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer sk-"+f.tokenKey)
	recorder := httptest.NewRecorder()
	f.engine.ServeHTTP(recorder, request)
	return recorder
}

func (f *inputLimitFixture) errorPayload(t *testing.T, recorder *httptest.ResponseRecorder) struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
} {
	t.Helper()
	var payload struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
	}
	require.NoError(t, common.Unmarshal(recorder.Body.Bytes(), &payload))
	return payload.Error
}

func (f *inputLimitFixture) userQuota(t *testing.T) int {
	t.Helper()
	fresh := &model.User{}
	require.NoError(t, model.DB.Select("quota").First(fresh, f.user.Id).Error)
	return fresh.Quota
}

func (f *inputLimitFixture) tokenRemainQuota(t *testing.T) int {
	t.Helper()
	fresh := &model.Token{}
	require.NoError(t, model.DB.Select("remain_quota").First(fresh, f.token.Id).Error)
	return fresh.RemainQuota
}

func (f *inputLimitFixture) consumeLogCount(t *testing.T) int64 {
	t.Helper()
	var count int64
	require.NoError(t, model.LOG_DB.Model(&model.Log{}).Where("token_id = ?", f.token.Id).Count(&count).Error)
	return count
}

func TestRelayRejectsOverLimitChatRequestWithoutBilling(t *testing.T) {
	fixture := newInputLimitFixture(t, []inputLimitChannel{{model: "limit-chat"}})
	fixture.setLimit(t, `{"limit-chat":1000}`)

	quotaBefore := fixture.userQuota(t)
	remainBefore := fixture.tokenRemainQuota(t)
	logsBefore := fixture.consumeLogCount(t)

	recorder := fixture.post(t, "/v1/chat/completions", fmt.Sprintf(
		`{"model":"limit-chat","messages":[{"role":"user","content":%q}]}`, strings.Repeat("token ", 1500)))

	require.Equal(t, http.StatusBadRequest, recorder.Code)
	payload := fixture.errorPayload(t, recorder)
	assert.Equal(t, "context_length_exceeded", payload.Code)
	assert.Contains(t, payload.Message, "This model's maximum context length is 1000 tokens")
	assert.Contains(t, payload.Message, "Please reduce the length of your input.")
	assert.Zero(t, fixture.upstreamCalls.Load(), "a rejected request must not reach the upstream")
	assert.Empty(t, fixture.useChannels, "a rejected request is stopped before the attempt loop, so no channel is tried or retried")
	assert.Equal(t, quotaBefore, fixture.userQuota(t), "a rejected request must not change user quota")
	assert.Equal(t, remainBefore, fixture.tokenRemainQuota(t), "a rejected request must not pre-consume token quota")
	assert.Equal(t, logsBefore, fixture.consumeLogCount(t), "a rejected request must not write a consume log")
}

func TestRelayAllowsRequestsUnderOrWithoutConfiguredLimit(t *testing.T) {
	fixture := newInputLimitFixture(t, []inputLimitChannel{{model: "limit-chat"}, {model: "limit-free"}})
	fixture.setLimit(t, `{"limit-chat":1000}`)

	underLimit := fixture.post(t, "/v1/chat/completions", `{"model":"limit-chat","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusOK, underLimit.Code)
	assert.NotContains(t, underLimit.Body.String(), "maximum context length")

	unconfigured := fixture.post(t, "/v1/chat/completions", fmt.Sprintf(
		`{"model":"limit-free","messages":[{"role":"user","content":%q}]}`, strings.Repeat("token ", 1500)))
	assert.Equal(t, http.StatusOK, unconfigured.Code)

	assert.Equal(t, int64(2), fixture.upstreamCalls.Load())
	assert.Positive(t, fixture.consumeLogCount(t), "allowed requests keep their consume log")
}

func TestRelayHonorsChannelMaxInputTokensOverride(t *testing.T) {
	fixture := newInputLimitFixture(t, []inputLimitChannel{
		{model: "limit-unlimited", setting: kitdto.ChannelSettings{MaxInputTokens: -1}},
		{model: "limit-override", setting: kitdto.ChannelSettings{MaxInputTokens: 100}},
	})
	fixture.setLimit(t, `{"limit-unlimited":1000,"limit-override":1000000}`)

	big := strings.Repeat("token ", 1500)

	unlimited := fixture.post(t, "/v1/chat/completions", fmt.Sprintf(
		`{"model":"limit-unlimited","messages":[{"role":"user","content":%q}]}`, big))
	assert.Equal(t, http.StatusOK, unlimited.Code, "channel max_input_tokens=-1 disables the limit")

	override := fixture.post(t, "/v1/chat/completions", fmt.Sprintf(
		`{"model":"limit-override","messages":[{"role":"user","content":%q}]}`, big))
	require.Equal(t, http.StatusBadRequest, override.Code)
	assert.Contains(t, fixture.errorPayload(t, override).Message, "maximum context length is 100 tokens")
	assert.Equal(t, int64(1), fixture.upstreamCalls.Load())
}

func TestRelayRejectsOverLimitResponsesRequest(t *testing.T) {
	fixture := newInputLimitFixture(t, []inputLimitChannel{{model: "limit-responses"}})
	fixture.setLimit(t, `{"limit-responses":1000}`)

	over := fixture.post(t, "/v1/responses", fmt.Sprintf(
		`{"model":"limit-responses","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":%q}]}]}`,
		strings.Repeat("token ", 1500)))
	require.Equal(t, http.StatusBadRequest, over.Code)
	assert.Equal(t, "context_length_exceeded", fixture.errorPayload(t, over).Code)
	assert.Zero(t, fixture.upstreamCalls.Load())

	under := fixture.post(t, "/v1/responses",
		`{"model":"limit-responses","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
	assert.Equal(t, http.StatusOK, under.Code)
	assert.Equal(t, int64(1), fixture.upstreamCalls.Load())
}
