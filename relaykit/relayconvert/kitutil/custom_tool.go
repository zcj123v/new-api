package kitutil

// Codex 等 Responses 客户端的 freeform 工具（type:"custom"，如 apply_patch）
// 在 Responses→Chat 转换时被伪装成普通 function：参数统一包成 {"input": "..."}。
// 响应方向再把对应 function_call 的 arguments 解包回 custom_tool_call.input。

// WrapCustomToolInput wraps a freeform tool input string into the chat-side
// function arguments JSON shape {"input": "..."}.
func WrapCustomToolInput(input string) string {
	raw, err := Marshal(map[string]string{"input": input})
	if err != nil {
		return "{}"
	}
	return string(raw)
}

// UnwrapCustomToolInput extracts the freeform input back out of chat-side
// function arguments. Falls back to the raw arguments string when the wrapper
// shape is not present (e.g. the model emitted non-JSON arguments).
func UnwrapCustomToolInput(arguments string) string {
	var m map[string]any
	if err := UnmarshalJsonStr(arguments, &m); err != nil {
		return arguments
	}
	if v, ok := m["input"].(string); ok {
		return v
	}
	return arguments
}
