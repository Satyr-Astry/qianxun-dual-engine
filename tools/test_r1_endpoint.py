"""用 urllib 测 OpenAI 兼容端点（不用 curl：shell 引号会把中文 payload 搞坏）"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8710/v1"


def post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")), time.time() - t0


# 1) /v1/models
m, dt = post("/models", {}) if False else (json.loads(urllib.request.urlopen(BASE + "/models", timeout=30).read()), 0)
print("[models]", [x.get("id") for x in m.get("data", [])], flush=True)

# 2) 中文短问答（首 token 延迟 + 解码速度）
body = {"model": "r1-14b-local", "messages": [{"role": "user", "content": "用一句话解释什么是 KV 缓存。"}],
        "temperature": 0, "max_tokens": 128, "stream": False}
r, dt = post("/chat/completions", body)
msg = r["choices"][0]["message"]
txt = (msg.get("content") or "").strip()
print(f"[首答] {dt:.1f}s 完成", flush=True)
print("[内容]", txt[:300].replace("\n", " "), flush=True)
print("[用量]", r.get("usage"), flush=True)

# 3) 长一点的推理题（看质量与速度）
body2 = {"model": "r1-14b-local",
         "messages": [{"role": "user", "content": "一个水池有甲乙两个进水管，甲单独注满需 6 小时，乙单独需 4 小时。两管同开需几小时注满？请给出计算过程。"}],
         "temperature": 0, "max_tokens": 400, "stream": False}
r2, dt2 = post("/chat/completions", body2, timeout=300)
m2 = r2["choices"][0]["message"]
print(f"\n[推理题] {dt2:.1f}s", flush=True)
print("[内容]", (m2.get("content") or "").strip()[:600].replace("\n", " "), flush=True)
print("[用量]", r2.get("usage"), flush=True)
