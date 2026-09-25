"""接入 Hermes 前的两项关键体检：①工具调用(function calling) ②服务端计时口径"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8710"


def post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ① 工具调用体检：给一个工具，看模型会不会吐出 OpenAI 格式的 tool_calls
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的当前天气",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "城市名"}},
                       "required": ["city"]},
    },
}]
body = {"model": "r1-14b-local",
        "messages": [{"role": "user", "content": "帮我查一下杭州现在天气怎么样。"}],
        "tools": tools, "tool_choice": "auto", "temperature": 0, "max_tokens": 200}
try:
    r = post("/v1/chat/completions", body)
    msg = r["choices"][0]["message"]
    tc = msg.get("tool_calls")
    print("① 工具调用:", "✅ 支持" if tc else "❌ 未返回 tool_calls")
    if tc:
        print("   ->", json.dumps(tc, ensure_ascii=False)[:300])
    else:
        print("   content 前 200 字:", repr((msg.get("content") or "")[:200]))
        print("   原始 message keys:", list(msg.keys()))
except Exception as e:
    print("① 工具调用: 报错", type(e).__name__, str(e)[:200])

# ② 服务端计时（/completion 带 timings）
t0 = time.time()
r2 = post("/completion", {"prompt": "请用 100 字介绍混合专家模型的原理。", "n_predict": 128,
                          "temperature": 0, "cache_prompt": False}, timeout=300)
tm = r2.get("timings", {})
print(f"② /completion 墙钟 {time.time()-t0:.1f}s")
for k in ["prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
          "predicted_per_second"]:
    if k in tm:
        print(f"   {k} = {tm[k]}")
