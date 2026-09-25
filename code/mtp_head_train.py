"""
N8-S1 · 给本机 r1-14B 自训一个 MTP 头（DeepSeek-V3 式，共享 embed / lm_head）
================================================================================
命题（计划书 §12A.2 的 S1）：本基底**没有 MTP 头**，而投机解码是唯一"一次读权重、多产出 token"的免费加速。
⇒ 自训 1 层 MTP 头（≈0.275 B，共享词表嵌入与输出头），目标 **首 token 接受率 α₁ ≥ 0.6**。

两个关键工程决定（都是为了让 12 GB 单机能训）：
  ①**隐状态预缓存**：先跑一遍主干把所有 `h_t`（fp16）写盘，训练时**只训头、不再跑主干** ⇒ 显存只剩头 + 优化器。
  ②**复用主干自己的层定义**（`Qwen2DecoderLayer`）而不是手写 —— 少一个写错 RoPE/attention 的机会。

流程：
  step1  cache : 读语料 → 主干前向 → 存 E:/models/r1-14b-mtp/hidden/*.npy（fp16）+ token ids
  step2  train : 冻结主干（不加载也行，本步只读缓存）→ 训 MTP 头
  step3  eval  : 在同一模型上比"头的预测"与"主干实际 greedy 下一个 token" ⇒ α₁ / α₂
用法：
  python code/mtp_head_train.py --stage cache --corpus "F:/DESKTOP/AI架构与推理设计/千寻双引擎推理栈"
  python code/mtp_head_train.py --stage train --steps 800
  python code/mtp_head_train.py --stage eval
"""
from __future__ import annotations
import argparse, glob, json, os, random, time

import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = os.environ.get("R1_HF", "E:/models/r1-14b-hf")
WORK = os.environ.get("MTP_WORK", "E:/models/r1-14b-mtp")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ── 组装 MTP 头 ────────────────────────────────────────────────────────────
class MTPHead(nn.Module):
    """1 层解码块 + 输入融合；**输出复用主干 lm_head**（共享，不新增词表参数）"""

    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        d = config.hidden_size
        self.proj = nn.Linear(2 * d, d, bias=False)      # 融合 [h_t ; emb(x_{t+1})]
        self.norm = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.block = Qwen2DecoderLayer(config, layer_idx=0)

    def forward(self, h_prev, emb_next, position_embeddings, attention_mask=None, position_ids=None):
        x = self.proj(torch.cat([h_prev, emb_next], dim=-1))
        x = self.norm(x)
        out = self.block(x, attention_mask=attention_mask, position_ids=position_ids,
                         position_embeddings=position_embeddings, past_key_value=None,
                         use_cache=False)
        return out[0] if isinstance(out, tuple) else out


# ── stage 1：隐状态缓存（主干前向一次，供后面反复训头） ──────────────────────
def stage_cache(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.makedirs(os.path.join(WORK, "hidden"), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"[cache] 载入主干 {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(DEV).eval()
    texts = []
    for p in sorted(glob.glob(os.path.join(args.corpus, "**", "*.md"), recursive=True)):
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
            if len(t) > 4000:
                texts.append(t[:200000])
        except Exception:
            pass
    print(f"[cache] 语料 {len(texts)} 篇，合计 {sum(len(t) for t in texts)/1e6:.2f} M 字符")
    CH = args.chunk
    saved = 0
    with torch.no_grad():
        for ti, t in enumerate(texts):
            ids = tok(t, return_tensors="pt").input_ids[0]
            for s in range(0, len(ids) - CH - 1, CH):
                seg = ids[s:s + CH + 1].to(DEV)                     # 多留 1 个 token 作 label
                out = model(seg.unsqueeze(0), output_hidden_states=True)
                h = out.hidden_states[args.layer][0].to(torch.float16).cpu().numpy()  # [CH+1, d]
                np.save(os.path.join(WORK, "hidden", f"c{ti:03d}_{s:07d}_h.npy"), h)
                np.save(os.path.join(WORK, "hidden", f"c{ti:03d}_{s:07d}_x.npy"), seg.cpu().numpy())
                saved += 1
                if saved % 20 == 0:
                    print(f"  [cache] {saved} chunks …", flush=True)
    print(f"[cache] 完成：{saved} 个 chunk → {WORK}/hidden")


# ── stage 2：训头 ────────────────────────────────────────────────────────
def stage_train(args):
    from transformers import AutoConfig, AutoModelForCausalLM
    files = sorted(glob.glob(os.path.join(WORK, "hidden", "*_h.npy")))
    print(f"[train] 缓存 chunk {len(files)} 个")
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    head = MTPHead(cfg).to(DEV).bfloat16()
    n_par = sum(p.numel() for p in head.parameters())
    print(f"[train] MTP 头参数 {n_par/1e6:.1f} M（主干冻结、共享 lm_head）")
    # 只为了拿 lm_head 与 embed_tokens（冻结）。主干用 bf16 加载会占 28 G ⇒ 只加载这两块
    print("[train] 仅载入 embed_tokens / lm_head（不载主干）")
    sd = {}
    from safetensors import safe_open
    idx = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json"), encoding="utf-8"))
    want = {"model.embed_tokens.weight": None, "lm_head.weight": None}
    for name in list(want):
        f = idx["weight_map"].get(name)
        if f is None:
            raise SystemExit(f"找不到 {name}")
        with safe_open(os.path.join(MODEL_DIR, f), framework="pt") as fh:
            want[name] = fh.get_tensor(name).to(DEV).to(torch.bfloat16)
    embed, lm_head = want["model.embed_tokens.weight"], want["lm_head.weight"]
    print(f"[train] embed {tuple(embed.shape)}  lm_head {tuple(lm_head.shape)}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95))
    rnd = random.Random(0)
    losses = []
    for step in range(args.steps):
        f = rnd.choice(files)
        h = torch.from_numpy(np.load(f)).to(DEV).to(torch.bfloat16)          # [T+1, d]
        x = torch.from_numpy(np.load(f.replace("_h.npy", "_x.npy"))).to(DEV)  # [T+1]
        h_prev = h[:-2].unsqueeze(0)                       # h_t
        x_next = x[1:-1].unsqueeze(0)                      # token t+1
        y = x[2:].unsqueeze(0)                             # 预测 token t+2
        emb_next = embed[x_next]
        pos_ids = torch.arange(h_prev.shape[1], device=DEV).unsqueeze(0)
        # RoPE：复用主干同款（qwen2 全维 rotary，theta 1e6）
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        rope = Qwen2RotaryEmbedding(cfg).to(DEV)
        cos, sin = rope(h_prev, pos_ids)
        out = head(h_prev, emb_next, (cos, sin), attention_mask=None, position_ids=pos_ids)
        logits = out @ lm_head.t()
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  [train] step {step:5d} loss {loss.item():.4f}", flush=True)
    os.makedirs(WORK, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(WORK, "mtp_head.pt"))
    json.dump({"steps": args.steps, "lr": args.lr, "params": n_par, "losses": losses},
              open(os.path.join(WORK, "train_log.json"), "w"), indent=2)
    print(f"[train] 完成 loss {losses[0]:.3f} → {np.mean(losses[-20:]):.3f}；已存 {WORK}/mtp_head.pt")


# ── stage 3：评估接受率（不需要 llama.cpp 集成，直接在 PyTorch 里数） ────────
def stage_eval(args):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(DEV).eval()
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    head = MTPHead(cfg).to(DEV).bfloat16()
    head.load_state_dict(torch.load(os.path.join(WORK, "mtp_head.pt"), map_location=DEV))
    head.eval()
    prompts = ["请解释什么是混合专家模型。", "写一段 Python 代码实现快速排序。",
               "把下面这段话翻译成英文：显存带宽决定了推理速度的上限。"]
    a1 = a2 = tot = 0
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(DEV)
            gen = model.generate(ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            seq = gen[0]
            out = model(seq.unsqueeze(0), output_hidden_states=True)
            h = out.hidden_states[args.layer][0]
            for t in range(len(ids[0]) - 1, len(seq) - 2):
                hp = h[t:t + 1].unsqueeze(0)
                pos = torch.arange(t, t + 1, device=DEV).unsqueeze(0)
                from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
                cos, sin = Qwen2RotaryEmbedding(cfg).to(DEV)(hp, pos)
                en = model.get_input_embeddings()(seq[t + 1:t + 2].unsqueeze(0))
                lo = head(hp, en, (cos, sin), position_ids=pos) @ model.lm_head.weight.t()
                pred = lo[0, -1].argmax().item()
                truth = seq[t + 2].item() if t + 2 < len(seq) else None
                if truth is None:
                    continue
                tot += 1
                if pred == truth:
                    a1 += 1
    print(f"[eval] 首 token 接受率 α₁ = {a1/max(1,tot):.3f}  (n={tot})  目标 ≥0.60")
    json.dump({"alpha1": a1 / max(1, tot), "n": tot}, open(os.path.join(WORK, "eval.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cache", "train", "eval"])
    ap.add_argument("--corpus", default=r"F:/DESKTOP/AI架构与推理设计")
    ap.add_argument("--layer", type=int, default=-1, help="取第几层隐状态（-1=最后层）")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    {"cache": stage_cache, "train": stage_train, "eval": stage_eval}[a.stage](a)
