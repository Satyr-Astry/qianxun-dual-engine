"""
N7/N8 实验工装 · 上下文容量 + 速度 + 投机解码（本机 r1-14B）
================================================================================
用 llama-server 的 /completion timings 取数（比 llama-cli 更准，且能拿到投机接受率）：
  · predicted_per_second    → decode 速度（tok/s）
  · prompt_per_second       → prefill 速度
  · draft_n / draft_n_accepted → 投机解码的接受数（llama.cpp 有投机时会报）

测什么（对应计划书 §12A）：
  A 基线 f16 KV（d=0）                → 速度天花板复核
  B q8 KV（d=0）                      → 量化开销
  C q8 KV（d=32K）                    → 长上下文「每步不变慢」检验（N7 主指标）
  D ngram 投机（若本 build 支持）      → N8 的免训练加速档
  E f16 KV（d=32K）                    → 对照：f16 在 32K 是否装得下

用法：
  python tools/n7_n8_bench.py --blob <gguf> [--only A,B] [--n-predict 128]
结果：写到 <项目>/bench/n7n8_<时间戳>.json 与 .md
"""
from __future__ import annotations
import argparse, glob, json, os, re, subprocess, sys, time, urllib.request, urllib.error
import socket

EXE_DIRS = [
    r"F:/DESKTOP/AI架构与推理设计/分级激活LLM项目/p1/llama_new",   # build 10938：CUDA + 投机
    r"F:/DESKTOP/AI架构与推理设计/分级激活LLM项目/p1",
]
PROMPT = ("请用中文详细解释下面这段代码的意图，并指出潜在的性能问题：\n"
          "def f(xs):\n    s = 0\n    for i in range(len(xs)):\n        s += xs[i]\n    return s\n"
          "然后给出三种优化方案，每种都要说明适用场景。")
COPY_PROMPT = ("下面是一段会议记录，请逐字重抄一遍，不要改动任何字符：\n"
               "第一条：本周完成接口对齐。第二条：下周一提交测试报告。第三条：预算追加百分之十。"
               "第四条：上线时间定在月底。第五条：回滚方案必须有。\n")


FILLER = ("在分布式系统的设计中，缓存一致性、故障恢复与延迟预算三者往往互相牵制。"
          "工程师需要先量化瓶颈，再选择分层策略，最后用可观测性数据验证收敛。"
          "任何跳过量化直接调参的做法，都会把系统推向不可解释的状态。" * 8)


def build_prompt(fill_tokens: int, tail: str) -> str:
    """★ 真喂长上下文：按目标 token 数填充（中文约 1.5 字符/token），再附上真正的问题"""
    if fill_tokens <= 0:
        return tail
    reps = max(1, int(fill_tokens * 1.5 / len(FILLER)))
    return "以下是一份背景资料：\n" + (FILLER * reps) + "\n\n" + tail


def find_exe(name="llama-server.exe") -> str:
    for d in EXE_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    sys.exit(f"找不到 {name}，请把目录加到 EXE_DIRS")


def find_blob() -> str:
    """自动定位本机 r1 的 GGUF blob（从 ollama manifest 读，取最大的那个模型层）"""
    best = None
    for man in glob.glob("D:/ollama/manifests/**/*", recursive=True):
        if not os.path.isfile(man) or "r1" not in man.lower():
            continue
        try:
            j = json.load(open(man, encoding="utf-8"))
        except Exception:
            continue
        for l in j.get("layers", []):
            if "model" in l.get("mediaType", ""):
                p = f"D:/ollama/blobs/{l['digest'].replace('sha256:', 'sha256-')}"
                if os.path.exists(p) and (best is None or os.path.getsize(p) > os.path.getsize(best)):
                    best = p
    if not best:
        sys.exit("未找到本机 r1 blob")
    return best


def help_text(exe: str) -> str:
    return subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=120).stdout


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def wait_health(port: int, timeout=600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if json.loads(r.read().decode()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(1.5)
    return False


def cleanup_servers():
    """★ 仪器纪律：每次测量前清干净，否则残留 server 占着显存会污染后续配置（实测踩过）"""
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True, text=True)
    time.sleep(2.0)


BUFS = re.compile(r"(CUDA0|CUDA_Host|CPU_Mapped|CPU)\s+(\S+ buffer size|KV buffer size)\s*=\s*([\d.]+) MiB")


def probe_config(exe, blob, name, ctx, extra, n_gpu=99, timeout=200):
    """★ 秒级显存预算探针：只加载、不推理，从 `-v` 日志里抠出 buffer 分配。
    比跑长上下文快 100×（用于扫 config 矩阵，先筛出"装得下且有余量"的组合）"""
    cleanup_servers()
    port = free_port()
    cmd = [exe, "-m", blob, "-ngl", str(n_gpu), "-c", str(ctx), "--port", str(port),
           "--host", "127.0.0.1", "-np", "1", "-fa", "auto", "-v"] + extra
    lp = os.path.join(OUT, f"probe_{name}.log")
    with open(lp, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        ok = wait_health(port, timeout=timeout)
        time.sleep(1.0)
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
    txt = open(lp, encoding="utf-8", errors="replace").read()
    last = {}
    for m in re.finditer(r"(CUDA0|CUDA_Host|CPU_Mapped|CPU)\s+(model buffer size|KV buffer size|compute buffer size)\s*=\s*([\d.]+) MiB", txt):
        last[(m.group(1), m.group(2))] = float(m.group(3))
    model = last.get(("CUDA0", "model buffer size"), 0.0)
    kv = last.get(("CUDA0", "KV buffer size"), 0.0)
    comp = last.get(("CUDA0", "compute buffer size"), 0.0) + last.get(("CUDA_Host", "compute buffer size"), 0.0)
    total = model + kv + comp
    return {"name": name, "ctx": ctx, "flags": " ".join(extra), "loaded": ok,
            "model_mib": round(model), "kv_mib": round(kv), "compute_mib": round(comp),
            "total_mib": round(total), "headroom_mib": round(12282 - total),
            "fits": total <= 12282}


def run_config(exe, blob, name, ctx, extra, prompt, n_predict, n_gpu=99, fill=0):
    cleanup_servers()
    port = free_port()
    cmd = [exe, "-m", blob, "-ngl", str(n_gpu), "-c", str(ctx), "--port", str(port),
           "--host", "127.0.0.1", "-np", "1", "-fa", "auto"] + extra
    log = open(os.path.join(OUT, f"{name}.log"), "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_health(port, timeout=210):
            tail = ""
            try:
                tail = open(os.path.join(OUT, f"{name}.log"), encoding="utf-8", errors="replace").read()[-400:]
            except Exception:
                pass
            return {"name": name, "ctx": ctx, "fill": fill, "error": "加载失败/超时（装不下？）",
                    "log_tail": tail.replace("\n", " | ")[-300:], "total_s": round(time.time() - t0, 1)}
        body = json.dumps({"prompt": build_prompt(fill, prompt), "n_predict": n_predict,
                           "temperature": 0, "cache_prompt": False}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read().decode())
        t = d.get("timings", {})
        return {"name": name, "ctx": ctx, "flags": " ".join(extra),
                "load_s": round(time.time() - t0 - t.get("predicted_ms", 0) / 1000, 1),
                "prompt_tok": t.get("prompt_n"), "prompt_tps": round(t.get("prompt_per_second", 0), 1),
                "pred_n": t.get("predicted_n"), "decode_tps": round(t.get("predicted_per_second", 0), 1),
                "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                "accept_rate": (round(t["draft_n_accepted"] / t["draft_n"], 3)
                                if t.get("draft_n") else None),
                "total_s": round((time.time() - t0), 1)}
    except Exception as e:
        return {"name": name, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        log.close()


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", default=None)
    ap.add_argument("--only", default=None, help="逗号分隔，如 A,C")
    ap.add_argument("--probe", action="store_true", help="★ 只加载不推理，秒级产出显存预算表")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    OUT = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench")
    OUT = os.path.abspath(OUT); os.makedirs(OUT, exist_ok=True)

    exe = find_exe(); blob = a.blob or find_blob()
    h = help_text(exe)
    has_spec_type = "--spec-type" in h
    ngram_hint = re.findall(r"ngram[\w-]*", h)[:5]
    print(f"[环境] exe={exe}\n[环境] blob={blob} ({os.path.getsize(blob)/2**30:.2f} GiB)")
    print(f"[环境] --spec-type 支持={has_spec_type}  线索={ngram_hint}")

    CONFIGS = {
        "A": dict(name="A_f16_d0", ctx=8192, extra=["-ctk", "f16", "-ctv", "f16"], prompt=PROMPT, fill=0),
        "B": dict(name="B_q8_d0", ctx=8192, extra=["-ctk", "q8_0", "-ctv", "q8_0"], prompt=PROMPT, fill=0),
        "C": dict(name="C_q8_d16k", ctx=40960, extra=["-ctk", "q8_0", "-ctv", "q8_0"], prompt=PROMPT, fill=16384),
        "D": dict(name="D_ngram_d0", ctx=8192,
                  extra=["--spec-type", "ngram-mod"] if has_spec_type else [], prompt=COPY_PROMPT, fill=0),
        "E": dict(name="E_f16_d32k", ctx=40960, extra=["-ctk", "f16", "-ctv", "f16"], prompt=PROMPT, fill=32768),
        "F": dict(name="F_q8_d32k", ctx=40960, extra=["-ctk", "q8_0", "-ctv", "q8_0"], prompt=PROMPT, fill=32768),
        # ★ 定性对照（4.73 tok/s @16K 的成因）
        "G": dict(name="G_q8_d4k", ctx=40960, extra=["-ctk", "q8_0", "-ctv", "q8_0"], prompt=PROMPT, fill=4096),
        "H": dict(name="H_q8_d16k_verbose", ctx=40960,
                  extra=["-v", "-ctk", "q8_0", "-ctv", "q8_0"], prompt=PROMPT, fill=16384),
        "I": dict(name="I_q8_d16k_ngl30", ctx=40960,
                  extra=["-ctk", "q8_0", "-ctv", "q8_0"], prompt=PROMPT, fill=16384, ngl=30),
        # ★ 因果验证：同样 16K 真上下文，只把 ctx 从 40960 收到 32768
        #   ⇒ KV 3072 MiB，合计 11460 MiB ≤ 12282 MiB（不超额）→ 若 decode 回升则超额=根因成立
        "J": dict(name="J_q8_d16k_ctx32k", ctx=32768,
                  extra=["-v", "-ctk", "q8_0", "-ctv", "q8_0"], prompt=PROMPT, fill=16384),
        # ★ 因果三连（都不带 -v；Q1 是 C 的"只收 ctx"对照）
        "Q1": dict(name="Q1_fit_16k", ctx=32768, extra=["-ctk", "q8_0", "-ctv", "q8_0"],
                   prompt=PROMPT, fill=16384),
        "Q2": dict(name="Q2_otFFN8_16k", ctx=32768,
                   extra=["-ctk", "q8_0", "-ctv", "q8_0", "-ot", r"blk\.(4[0-7])\.ffn.*=CPU"],
                   prompt=PROMPT, fill=16384),
        "Q3": dict(name="Q3_nkvo_16k", ctx=32768,
                   extra=["-ctk", "q8_0", "-ctv", "q8_0", "-nkvo"], prompt=PROMPT, fill=16384),
        # ★ 目标档：32K 真上下文 + 权重让位（探针：合计 11520 MiB，余量 ~762）
        "Q4": dict(name="Q4_otFFN8_32k", ctx=40960,
                   extra=["-ctk", "q8_0", "-ctv", "q8_0", "-ot", r"blk\.(4[0-7])\.ffn.*=CPU"],
                   prompt=PROMPT, fill=32768),
        # ★ 机制验证：不卸载权重，只把 prefill 的批量压小（看能否不牺牲 decode 就救回 prefill）
        "Q5": dict(name="Q5_smallbatch_16k", ctx=32768,
                   extra=["-ctk", "q8_0", "-ctv", "q8_0", "-b", "512", "-ub", "256"],
                   prompt=PROMPT, fill=16384),
        # ★ 折中档验证：不卸载、ctx 收到 24576（探针预计 KV 2448 → 余量 ~1434）
        "Q6": dict(name="Q6_ctx24k_16k", ctx=24576,
                   extra=["-ctk", "q8_0", "-ctv", "q8_0"], prompt=PROMPT, fill=16384),
        # ★ K4：投机解码对照（必须放在"健康区"余量下测，否则可能根本没启用）
        "S1": dict(name="S1_spec_ngram_24k", ctx=24576,
                   extra=["-ctk", "q8_0", "-ctv", "q8_0", "--spec-type", "ngram-mod"],
                   prompt=COPY_PROMPT, fill=0),
        "S2": dict(name="S2_nospec_copy_24k", ctx=24576,
                   extra=["-ctk", "q8_0", "-ctv", "q8_0"], prompt=COPY_PROMPT, fill=0),
        "S3": dict(name="S3_spec_ngram_simple_24k", ctx=24576,
                   extra=["-ctk", "q8_0", "-ctv", "q8_0", "--spec-type", "ngram-simple",
                          "--spec-draft-n-max", "8"],
                   prompt=COPY_PROMPT, fill=0),
    }
    keys = [k.strip().upper() for k in a.only.split(",")] if a.only else list(CONFIGS)

    if a.probe:
        PROBES = {
            "P32q8":  dict(name="P32k_q8", ctx=32768, extra=["-ctk", "q8_0", "-ctv", "q8_0"]),
            "P32q8b": dict(name="P32k_q8_otFFN8", ctx=32768,
                           extra=["-ctk", "q8_0", "-ctv", "q8_0", "-ot", r"blk\.(4[0-7])\.ffn.*=CPU"]),
            "P32f16": dict(name="P32k_f16", ctx=32768, extra=["-ctk", "f16", "-ctv", "f16"]),
            "P40q8":  dict(name="P40k_q8", ctx=40960, extra=["-ctk", "q8_0", "-ctv", "q8_0"]),
            "P32q8nv": dict(name="P32k_q8_nkvo", ctx=32768, extra=["-ctk", "q8_0", "-ctv", "q8_0", "-nkvo"]),
            "P32q8v4": dict(name="P32k_q8_q4v", ctx=32768, extra=["-ctk", "q8_0", "-ctv", "q4_0"]),
        }
        rows = []
        for k in (keys if a.only else list(PROBES)):
            if k not in PROBES:
                continue
            c = PROBES[k]
            r = probe_config(exe, blob, c["name"], c["ctx"], c["extra"])
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        jp = os.path.join(OUT, f"probe_{ts}.json")
        json.dump(rows, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        lines = ["# 显存预算探针（只加载不推理）", "",
                 "| 配置 | ctx | 标志 | model | KV | compute | 合计 | 余量 | 装得下 |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['name']} | {r['ctx']} | `{r['flags']}` | {r['model_mib']} | {r['kv_mib']} | "
                         f"{r['compute_mib']} | **{r['total_mib']}** | {r['headroom_mib']} | "
                         f"{'✅' if r['fits'] else '❌超额'} |")
        mp = os.path.join(OUT, f"probe_{ts}.md")
        open(mp, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"[完成] {jp}\n[完成] {mp}")
        return

    rows = []
    for k in keys:
        c = CONFIGS[k]
        print(f"\n=== {c['name']}  ({c['flags'] if 'flags' in c else ' '.join(c['extra'])}) ===", flush=True)
        r = run_config(exe, blob, c["name"], c["ctx"], c["extra"], c["prompt"], a.n_predict,
                       fill=c.get("fill", 0), n_gpu=c.get("ngl", 99))
        print("   ", json.dumps(r, ensure_ascii=False), flush=True)
        rows.append(r)

    ts = time.strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(OUT, f"n7n8_{ts}.json")
    json.dump({"blob": blob, "exe": exe, "spec_type_supported": has_spec_type, "rows": rows},
              open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    lines = ["# N7/N8 实测（本机 r1-14B）", "",
             f"- exe: `{exe}`　blob: `{os.path.basename(blob)}`　`--spec-type` 支持：{has_spec_type}", "",
             "| 配置 | ctx | 标志 | prefill tok/s | **decode tok/s** | 投机接受 | 加载 s |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | | | | ❌ {r['error']} | | |"); continue
        lines.append(f"| {r['name']} | {r['ctx']} | `{r['flags']}` | {r['prompt_tps']} | "
                     f"**{r['decode_tps']}** | {r.get('accept_rate', '—')} | {r['load_s']} |")
    mp = os.path.join(OUT, f"n7n8_{ts}.md")
    open(mp, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\n[完成] {jp}\n[完成] {mp}")


if __name__ == "__main__":
    main()
