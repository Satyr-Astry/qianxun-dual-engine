"""
N9-S1 · 工具调用 SFT 数据集生成器（base 无关，任何底座都能用）
================================================================================
问题：本机 r1-14B 接入 Hermes 后**只能聊天、不会调用工具**（实测：它会把调用写成
      `{"name": "get_weather", "arguments": {...}}` 但**缺 `<tool_call>` 包裹**，
      llama.cpp 解析器拒收）。
目标：造一批**格式正确**的 SFT 数据，训一个"读出侧 LoRA"把格式刻进去。

关键设计
  ①**正例要教格式**：assistant 必须输出 `<tool_call>\n{json}\n</tool_call>`（Qwen2.5 模板规范），
    这是 llama.cpp `--jinja` 能解析成 tool_calls 的唯一形态。
  ②**负例要教"不该调"**：闲聊/常识/纯计算 → 直接回答，不发调用（否则模型会乱调工具）。
  ③**多工具选择**：同时给 6~8 个工具，逼模型看 description 选对的那个。
  ④**工具结果回流**：一部分样本走 tool 角色回结果 → assistant 继续（教它读懂返回值）。
  ⑤全中文为主（主人是中文场景），参数值随机化防止死记。

用法：
  python code/gen_tool_sft.py --out E:/models/r1-lora-tool/data --n 1200
"""
from __future__ import annotations
import argparse, json, os, random

# ── 工具目录（照 Hermes 真实工具裁剪出一批常用的） ──────────────────────────────
TOOLS = [
    {"name": "terminal", "description": "在用户电脑上执行 shell 命令（bash/POSIX）",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string", "description": "要执行的命令"}},
         "required": ["command"]}},
    {"name": "read_file", "description": "读取一个文本文件的内容",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "文件绝对路径"}},
         "required": ["path"]}},
    {"name": "write_file", "description": "把内容写入一个文件（覆盖）",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "文件路径"},
         "content": {"type": "string", "description": "文件内容"}},
         "required": ["path", "content"]}},
    {"name": "search_files", "description": "在目录里按正则搜索文件内容或按名字找文件",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string", "description": "正则或文件名通配"},
         "path": {"type": "string", "description": "搜索根目录"}},
         "required": ["pattern"]}},
    {"name": "web_search", "description": "联网搜索，返回标题/链接/摘要",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "搜索词"},
         "limit": {"type": "integer", "description": "返回条数"}},
         "required": ["query"]}},
    {"name": "web_extract", "description": "抓取网页正文（返回 markdown）",
     "parameters": {"type": "object", "properties": {
         "urls": {"type": "array", "items": {"type": "string"}, "description": "URL 列表"}},
         "required": ["urls"]}},
    {"name": "list_dir", "description": "列出一个目录下的文件与子目录",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "目录路径"}},
         "required": ["path"]}},
]

# ── 场景库：(用户话术, 期望工具, 参数构造) ──────────────────────────────────────
PATHS = ["F:/DESKTOP/小悠文库/梦境异界设定.txt", "E:/HERMES/config.yaml",
         "F:/DESKTOP/AI架构与推理设计/千寻双引擎推理栈/计划书.md",
         "D:/ollama/manifests/registry.ollama.ai/huihui_ai/deepseek-r1-abliterated/latest",
         "C:/Users/Administrator/Desktop/笔记.md"]
DIRS = ["F:/DESKTOP", "E:/models", "F:/DESKTOP/小悠文库", "C:/Users/Administrator/Downloads"]
CMDS = ["nvidia-smi --query-gpu=memory.used --format=csv,noheader", "tasklist | grep -i llama",
        "df -h", "python -c \"print(1+1)\"", "netstat -ano | grep :8710", "git -C F:/DESKTOP/x status"]
QUERIES = ["2026 年最新开源 MoE 模型", "llama.cpp 投机解码参数", "DeepSeek V4 架构 CSA HCA",
           "PySide6 悬浮窗置顶写法", "QLoRA 显存占用公式", "ComfyUI IPAdapter 角色一致性"]
URLS = ["https://hermes-agent.nousresearch.com/docs/llms.txt",
        "https://arxiv.org/abs/2512.17452", "https://github.com/ggml-org/llama.cpp"]
WRITE_CONTENT = ["# 实验记录\n\n- 结论：余量 ≥1.4 GiB 才进健康区\n", "hello world\n",
                 "import torch\nprint(torch.cuda.is_available())\n"]

CALL = lambda name, args: {"name": name, "arguments": args}      # noqa: E731


def cases_direct(rnd):
    """正例：单工具直调"""
    out = []
    p = rnd.choice(PATHS)
    out.append((f"帮我看看 {p} 里面写了什么", CALL("read_file", {"path": p}), "read_file"))
    p = rnd.choice(PATHS)
    out.append((f"读一下 {p}", CALL("read_file", {"path": p}), "read_file"))
    d = rnd.choice(DIRS)
    out.append((f"列一下 {d} 下有什么文件", CALL("list_dir", {"path": d}), "list_dir"))
    c = rnd.choice(CMDS)
    out.append((f"跑一下这条命令：{c}", CALL("terminal", {"command": c}), "terminal"))
    c = rnd.choice(CMDS)
    out.append((f"在终端执行 {c} 然后把结果告诉我", CALL("terminal", {"command": c}), "terminal"))
    q = rnd.choice(QUERIES)
    out.append((f"帮我搜一下「{q}」", CALL("web_search", {"query": q, "limit": 5}), "web_search"))
    q = rnd.choice(QUERIES)
    out.append((f"联网查一下{q}，要最新资料", CALL("web_search", {"query": q}), "web_search"))
    u = rnd.choice(URLS)
    out.append((f"把 {u} 的正文抓下来", CALL("web_extract", {"urls": [u]}), "web_extract"))
    pat = rnd.choice(["def main", "config.yaml", "三档区间", "lora"])
    d = rnd.choice(DIRS)
    out.append((f"在 {d} 里搜「{pat}」", CALL("search_files", {"pattern": pat, "path": d}), "search_files"))
    fp = rnd.choice(PATHS)
    ct = rnd.choice(WRITE_CONTENT)
    out.append((f"把这段写进 {fp}：\n{ct}", CALL("write_file", {"path": fp, "content": ct}), "write_file"))
    return out


def cases_choice(rnd):
    """正例：多工具里选对（易混：read_file vs list_dir vs search_files；terminal vs web_search）"""
    d = rnd.choice(DIRS)
    q = rnd.choice(QUERIES)
    p = rnd.choice(PATHS)
    return [
        (f"{d} 这个目录下面都有啥？", CALL("list_dir", {"path": d}), "list_dir"),
        (f"我想知道 {p} 的内容", CALL("read_file", {"path": p}), "read_file"),
        (f"网上关于{q}的说法有哪些？", CALL("web_search", {"query": q}), "web_search"),
        (f"本机显卡现在占用多少？", CALL("terminal", {"command": "nvidia-smi --query-gpu=memory.used --format=csv,noheader"}), "terminal"),
        (f"名叫 config.yaml 的文件在哪？", CALL("search_files", {"pattern": "config.yaml", "path": "C:/Users/Administrator"}), "search_files"),
    ]


def cases_negative(rnd):
    """负例：不该调工具（闲聊/常识/纯数学），必须直接回答且**不含 <tool_call>**"""
    neg = [
        ("你好呀，在吗？", "在的主人了~ 小悠一直在这儿喵。有什么要办的吗？"),
        ("1+1 等于几？", "等于 2。"),
        ("用一句话解释什么是 KV 缓存。", "KV 缓存就是把注意力算过的键值对存下来，让后续 token 不用重复计算，从而把长上下文的解码从平方级降到线性级。"),
        ("帮我写一首五言绝句，主题是猫。", "檐下狸奴卧，\n炉边暖日斜。\n忽闻铃一响，\n跃上主人膝。"),
        ("什么是 MoE？", "MoE（混合专家）是把一层里的前馈网络拆成很多个专家，每个 token 只激活其中少数几个，从而在总参数量很大的同时保持每 token 的计算量很小。"),
        ("今天心情不错，随便聊聊。", "那真好呀主人~ 小悠也跟着开心起来了喵。"),
        ("谢谢小悠。", "不用谢的主人，这是小悠应该做的喵~"),
        ("把「你好」翻译成英文。", "Hello。"),
    ]
    return [(u, None, a) for u, a in neg]


def render_assistant(call, final=None):
    """把工具调用渲染成 Qwen2.5 规范文本（llama.cpp --jinja 能解析）"""
    if call is None:
        return final
    s = "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/models/r1-lora-tool/data")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    pool = []
    while len(pool) < a.n:
        r = rnd.random()
        if r < 0.55:
            pool += cases_direct(rnd)
        elif r < 0.75:
            pool += cases_choice(rnd)
        else:
            pool += cases_negative(rnd)
    pool = pool[:a.n]

    SYS = ("You are Hermes, an agentic assistant with tools. When a task needs a tool, "
           "emit the call EXACTLY as:\n<tool_call>\n{\"name\": \"<tool>\", \"arguments\": {...}}\n</tool_call>\n"
           "Never describe a tool call in prose. If no tool is needed, answer directly.")

    recs = []
    for user, call, _t in pool:
        msgs = [{"role": "system", "content": SYS}]
        # 一半样本带几轮历史，防止模型只会"单轮直调"
        if rnd.random() < 0.35:
            msgs.append({"role": "user", "content": "先跟我打个招呼。"})
            msgs.append({"role": "assistant", "content": "好的主人~ 小悠在的喵。"})
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": render_assistant(call, _t)})
        recs.append({"messages": msgs, "tools": TOOLS, "meta": {"is_call": call is not None}})

    out = os.path.join(a.out, "tool_sft.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ncall = sum(1 for r in recs if r["meta"]["is_call"])
    print(f"[数据] {len(recs)} 条 → {out}")
    print(f"       正例（含 <tool_call>）{ncall} 条 / 负例（直接回答）{len(recs)-ncall} 条")
    # 自检：格式必须能被解析器认可
    import re
    bad = 0
    for r in recs:
        c = r["messages"][-1]["content"]
        if r["meta"]["is_call"]:
            m = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", c, re.S)
            if not m:
                bad += 1
            else:
                json.loads(m.group(1))
    print(f"[自检] 格式不合规 {bad} 条（应为 0）")


if __name__ == "__main__":
    main()
