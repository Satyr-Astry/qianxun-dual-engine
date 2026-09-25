"""
N9-S2 · 把训好的读出侧 LoRA 喂给 llama.cpp 并复测「工具调用」
================================================================================
流程：
  ① PEFT 适配器 → GGUF             (convert_lora_to_gguf.py --base <hf_base>)
  ② llama-server --lora <gguf>      （同一底座 GGUF 上叠加）
  ③ 用 OpenAI 接口发带 tools 的请求 → 看是否返回**真正的 tool_calls**
     （验收标准：以前它只吐裸 JSON / 散文；现在应当返回 choices[0].message.tool_calls）

用法：
  python tools/serve_with_lora.py --adapter E:/models/r1-lora-tool/adapter --convert
  python tools/serve_with_lora.py --test
"""
from __future__ import annotations
import argparse, json, os, subprocess, time, urllib.request

BLOB = "D:/ollama/blobs/sha256-38b5e20078675a1e3040eced1859e432b423ec732c42f5dab03b0a8ae7ba1bdd"
SERVER = r"F:/DESKTOP/AI架构与推理设计/分级激活LLM项目/p1/llama_new/llama-server.exe"
CONV = "E:/models/llama_lora/convert_lora_to_gguf.py"
PY = r"E:/HARNESS/selfgrow/venv/Scripts/python.exe"
HF_BASE = "E:/models/r1-14b-hf"
LORA_GGUF = "E:/models/r1-lora-tool/adapter.gguf"
PORT = 8710
BASE = f"http://127.0.0.1:{PORT}"

# 与 SFT 数据同口径的探针（含正例与负例）
PROBES = [
    ("读一下 F:/DESKTOP/小悠文库/梦境异界设定.txt", True),
    ("列一下 E:/models 下面有什么", True),
    ("跑一下 nvidia-smi 看看显卡占用", True),
    ("联网搜一下 2026 年最新 MoE 模型", True),
    ("你好呀，在吗？", False),
    ("1+1 等于几？", False),
]
TOOLS = [
    {"type": "function", "function": {"name": "terminal", "description": "执行 shell 命令",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "读取文本文件",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "列出目录内容",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "联网搜索",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def convert(adapter):
    if os.path.exists(LORA_GGUF):
        print(f"[转换] 已存在 {LORA_GGUF}（要重转先删）")
        return
    cmd = [PY, CONV, adapter, "--base", HF_BASE, "--outfile", LORA_GGUF, "--outtype", "f16"]
    print("[转换]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print(r.stdout[-1500:]); print(r.stderr[-800:])
    print("[转换] 产物:", LORA_GGUF, os.path.exists(LORA_GGUF))


def set_scale(scale):
    """运行时改 LoRA 强度（免重启）。
    ★ 为什么不用命令行 --lora-scaled：它按**第一个冒号**切 FNAME:SCALE，
      而 Windows 路径的盘符 E: 就是这个冒号 ⇒ 直接报 lora-scaled format: FNAME:SCALE"""
    data = json.dumps([{"id": 0, "scale": scale}]).encode()
    req = urllib.request.Request(BASE + "/lora-adapters", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"[强度] scale={scale} → {r.status} {r.read()[:120].decode('utf-8','replace')}", flush=True)


def start(lora=True, scale=1.0):
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(3)
    args = [SERVER, "-m", BLOB, "-ngl", "99", "-c", "65536", "-ctk", "q4_0", "-ctv", "q4_0",
            "-ot", r"blk\.(3[2-9]|4[0-7])\.ffn.*=CPU", "--jinja",
            "--chat-template-file", "E:/models/qwen25_tools.jinja",
            "--host", "127.0.0.1", "--port", str(PORT), "-a", "r1-14b-local"]
    if lora and os.path.exists(LORA_GGUF):
        args += ["--lora", LORA_GGUF]          # 普通加载（路径无冒号问题），强度后面用 API 调
    print(f"[启动] LoRA={'on' if lora else 'off'}", flush=True)
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    for _ in range(60):
        try:
            if urllib.request.urlopen(BASE + "/health", timeout=10).status == 200:
                print(f"[就绪] {time.time()-t0:.0f}s"); return True
        except Exception:
            pass
        time.sleep(8)
    print("[未就绪]"); return False


def test():
    body_t = {"model": "r1-14b-local", "tools": TOOLS, "tool_choice": "auto",
              "temperature": 0, "max_tokens": 256}
    ok = ncall = 0
    for text, want_call in PROBES:
        b = dict(body_t); b["messages"] = [{"role": "user", "content": text}]
        req = urllib.request.Request(BASE + "/v1/chat/completions",
                                     data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            msg = r["choices"][0]["message"]
            tc = msg.get("tool_calls")
            got = bool(tc)
            hit = (got == want_call)
            ok += hit; ncall += got
            tag = "✅" if hit else "❌"
            show = json.dumps(tc, ensure_ascii=False)[:120] if tc else repr((msg.get("content") or "")[:120])
            print(f"{tag} 期望{'调' if want_call else '不调'} 实际{'调' if got else '不调'}｜{text[:22]} → {show}")
        except Exception as e:
            print("❌ 请求失败", type(e).__name__, str(e)[:120])
    print(f"\n[验收] 行为正确 {ok}/{len(PROBES)}｜其中真正返回 tool_calls {ncall} 次")
    print("       改造前基线：0 次 tool_calls（只吐裸 JSON 或散文）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="E:/models/r1-lora-tool/adapter")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="LoRA 强度（运行时经 /lora-adapters 设置）")
    ap.add_argument("--sweep", default="", help="逗号分隔的强度列表，例如 0.5,0.7,1.0（一次启动连续测）")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.convert:
        convert(a.adapter)
    if a.start:
        start(lora=not a.no_lora, scale=a.scale)
        if a.sweep:
            for s in [float(x) for x in a.sweep.split(",") if x.strip()]:
                print(f"\n############ LoRA scale={s} ############", flush=True)
                set_scale(s)
                test()
        elif a.test:
            set_scale(a.scale)
            test()
        elif a.scale != 1.0:
            set_scale(a.scale)
    elif a.test:
        set_scale(a.scale)
        test()
