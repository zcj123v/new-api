package relayconvert

import (
	"context"
)

// Codex 的 freeform 工具（type:"custom"，如 apply_patch）在 Responses→Chat
// 请求转换时被伪装成普通 function。这里把被伪装工具的名字暂存在请求上下文
// 里，响应方向（chat→responses）据此把同名 function_call 还原成
// custom_tool_call。上下文在运行时是 gin.Context（实现 Set/Get），用最小
// 接口断言以避免 relaykit 依赖 gin。
const responsesCustomToolNamesContextKey = "responses_custom_tool_names"

type contextKVSetter interface {
	Set(string, any)
}

type contextKVGetter interface {
	Get(string) (any, bool)
}

func stashResponsesCustomToolNames(c context.Context, names []string) {
	if len(names) == 0 {
		return
	}
	if s, ok := c.(contextKVSetter); ok {
		s.Set(responsesCustomToolNamesContextKey, names)
	}
}

func responsesCustomToolNamesFromContext(c context.Context) map[string]bool {
	g, ok := c.(contextKVGetter)
	if !ok {
		return nil
	}
	v, ok := g.Get(responsesCustomToolNamesContextKey)
	if !ok {
		return nil
	}
	names, ok := v.([]string)
	if !ok || len(names) == 0 {
		return nil
	}
	set := make(map[string]bool, len(names))
	for _, name := range names {
		set[name] = true
	}
	return set
}
