"""
GGUF(Ollama blob) → HF safetensors 转换工装
================================================================================
**为什么需要它**：作者指定基底为「**本机的 r1**」= `huihui_ai/deepseek-r1-abliterated:14b`
（Ollama 的 GGUF blob，Q4_K_M）。但架构实验（忆态分支、注入深度扫描、自学习闭环）都要在 **PyTorch**
里做 ⇒ 必须把 GGUF 反量化成可被 transformers 加载的 safetensors。**不下载任何权重**，全部用本机已有的那份。

支持：qwen2 / qwen3 系（本项目当前基底）；其余架构按需扩展。
反量化：走 `gguf.quants.dequantize`（Q4_K / Q6_K / Q8_0 / … 全覆盖）。

用法：
  # 1) 先干跑，只校验张量名映射（秒级，不写盘）
  python tools/gguf_to_hf.py --blob <blob路径> --out E:/models/r1-14b-hf --dry-run

  # 2) 真转换（写 bf16 分片 + config/tokenizer）
  python tools/gguf_to_hf.py --blob <blob路径> --out E:/models/r1-14b-hf --dtype bf16

  # 3) 只转部分层（显存/磁盘紧张时先试）
  python tools/gguf_to_hf.py --blob ... --out ... --tensor-filter "blk.[0-3]\\.|token_embd|output"
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys
from collections import OrderedDict

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.exit("需要 gguf：pip install gguf")

try:
    from safetensors.torch import save_file
except ImportError:
    sys.exit("需要 safetensors：pip install safetensors")


# ── GGUF 名 → HF 名（qwen2 / qwen3 同构） ────────────────────────────────
def hf_name(gguf_name: str, n_layer: int) -> str:
    m = re.match(r"blk\.(\d+)\.(.+)", gguf_name)
    if m:
        i, rest = int(m.group(1)), m.group(2)
        p = f"model.layers.{i}."
        table = {
            "attn_q.weight": p + "self_attn.q_proj.weight",
            "attn_q.bias":   p + "self_attn.q_proj.bias",
            "attn_k.weight": p + "self_attn.k_proj.weight",
            "attn_k.bias":   p + "self_attn.k_proj.bias",
            "attn_v.weight": p + "self_attn.v_proj.weight",
            "attn_v.bias":   p + "self_attn.v_proj.bias",
            "attn_output.weight": p + "self_attn.o_proj.weight",
            "attn_norm.weight":   p + "input_layernorm.weight",
            "ffn_norm.weight":    p + "post_attention_layernorm.weight",
            "ffn_gate.weight":    p + "mlp.gate_proj.weight",
            "ffn_up.weight":      p + "mlp.up_proj.weight",
            "ffn_down.weight":    p + "mlp.down_proj.weight",
        }
        if rest in table:
            return table[rest]
        raise KeyError(f"未映射的层内张量：{gguf_name}")
    top = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight":      "lm_head.weight",
    }
    if gguf_name in top:
        return top[gguf_name]
    raise KeyError(f"未映射的顶层张量：{gguf_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True, help="Ollama blob（GGUF）路径")
    ap.add_argument("--out", required=True, help="输出目录（HF 格式）")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--shard-gb", type=float, default=4.5, help="单分片上限（GB）")
    ap.add_argument("--rows-chunk", type=int, default=4096,
                    help="大张量分块行数（★ 16GB RAM 必须分块，否则反量化 embed 会 OOM）")
    ap.add_argument("--dry-run", action="store_true", help="只校验映射，不写盘")
    ap.add_argument("--tensor-filter", default=None, help="正则：只转匹配的张量")
    a = ap.parse_args()

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    reader = gguf.GGUFReader(a.blob)
    n_layer = int(reader.fields["qwen2.block_count"].contents())
    print(f"[读入] {a.blob}\n[信息] 层数 {n_layer}，张量 {len(reader.tensors)}，dtype→{a.dtype}")

    pat = re.compile(a.tensor_filter) if a.tensor_filter else None
    plan, unmapped = [], []
    for t in reader.tensors:
        if pat and not pat.search(t.name):
            continue
        try:
            plan.append((t, hf_name(t.name, n_layer)))
        except KeyError as e:
            unmapped.append(str(e))
    print(f"[映射] 待转换 {len(plan)} 张量；未映射 {len(unmapped)}")
    if unmapped:
        print("  ⚠️", unmapped[:5], "…" if len(unmapped) > 5 else "")
    # 展示映射样例
    for t, h in plan[:4]:
        print(f"    {t.name:34s} → {h}")
    if a.dry_run:
        print("[dry-run] 校验完成，未写盘。")
        return

    os.makedirs(a.out, exist_ok=True)
    total_bytes, shard, shard_bytes, shards = 0, OrderedDict(), 0, []
    limit = a.shard_gb * 1024**3

    def flush():
        nonlocal shard, shard_bytes
        if not shard:
            return
        idx = len(shards)
        tmp = os.path.join(a.out, f"part-{idx+1}.safetensors")
        save_file(shard, tmp, metadata={"format": "pt"})
        shards.append((tmp, list(shard.keys())))
        print(f"  [写出] {os.path.basename(tmp)}  {shard_bytes/2**30:.2f} GiB  ({len(shard)} 张量)", flush=True)
        shard, shard_bytes = OrderedDict(), 0

    # ★ 分块流式反量化：16 GB RAM 装不下一次性 fp32 副本（原版曾 OOM）
    ROWS = a.rows_chunk
    for n_i, (t, h) in enumerate(plan, 1):
        src = t.data
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            w = torch.from_numpy(np.ascontiguousarray(src.astype(np.float32) if a.dtype == "fp32" else src))
            if a.dtype != "fp32":
                w = w.to(dt)
        elif src.ndim == 1 or src.shape[0] <= ROWS:
            arr = gguf.quants.dequantize(src, t.tensor_type)
            w = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32) if a.dtype == "fp32" else arr))
            if a.dtype != "fp32":
                w = w.to(dt)
            del arr
        else:
            rows = src.shape[0]
            # ★ 用第一块的**反量化后形状**来分配（量化行宽是按字节算的，≠ 值宽；曾因此 shape 不匹配）
            first = gguf.quants.dequantize(src[0:ROWS], t.tensor_type)
            w = torch.empty((rows,) + tuple(first.shape[1:]), dtype=dt)
            w[0:first.shape[0]] = torch.from_numpy(first).to(dt)
            del first
            for r0 in range(ROWS, rows, ROWS):
                blk = src[r0:r0 + ROWS]
                arr = gguf.quants.dequantize(blk, t.tensor_type)
                w[r0:r0 + arr.shape[0]] = torch.from_numpy(arr).to(dt)
                del arr, blk
            print(f"  [分块] {t.name} rows={rows} → {h}", flush=True)
        shard[h] = w.contiguous()
        shard_bytes += w.numel() * w.element_size()
        total_bytes += w.numel() * w.element_size()
        if n_i % 60 == 0:
            print(f"  [{n_i}/{len(plan)}] 累计 {total_bytes/2**30:.2f} GiB", flush=True)
        del w
        if shard_bytes >= limit:
            flush()
    flush()

    # 重命名为标准分片名
    n = len(shards)
    weight_map = {}
    for i, (tmp, keys) in enumerate(shards, 1):
        final = os.path.join(a.out, f"model-{i:05d}-of-{n:05d}.safetensors")
        os.replace(tmp, final)
        for k in keys:
            weight_map[k] = os.path.basename(final)
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, ensure_ascii=False, indent=2)
    print(f"[完成] 共 {total_bytes/2**30:.2f} GiB，{n} 个分片 → {a.out}")
    print("  ⚠️ 还需把 HF 的 config.json / tokenizer* 放进该目录：")
    print("     HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \\")
    print("       huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\")
    print("       --local-dir <out> --include 'config.json' 'tokenizer*' '*.txt'  # 只拉几十 KB")


if __name__ == "__main__":
    main()
