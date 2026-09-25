"""
N8-S1 · 本机 r1-14B 自训 MTP 头（12 GB 单机可行版）
================================================================================
背景：`mtp_head_train.py` 的 cache/eval 阶段要整模型加载（bf16 28 GB），本机 16 GB 内存 + 12 GB 显存
**跑不动**；bitsandbytes 4/8bit 量化在本机对 14B **必 segfault**（1.5B 正常，已实测）。
⇒ 本文件用**逐层流式加载**替代：按 safetensors 分片把每层权重搬进显存、算完即卸，峰值 <6 GB。

关键观察（省掉了最贵的一步）：
  · 训 MTP 头只需要 **隐状态 h_t + token id x_t**，不需要主干常驻；
  · 评估 α₁ 也**不需要生成**：主干在位置 t 的贪心 token = argmax(lm_head · norm(h_t))，
    而 draft 的猜测来自 (h_t, emb(x_{t+1})) ⇒ 两边都能用缓存好的 h 直接算。

数据布局：E:/models/r1-14b-mtp/hidden/c{chunk:03d}.npz  {h: fp16 [L,d], x: int32 [L]}
用法：
  python code/mtp_n8s1.py --stage cache --chunks 160 --chunk-len 512
  python code/mtp_n8s1.py --stage train --steps 600
  python code/mtp_n8s1.py --stage eval
"""
from __future__ import annotations
import argparse, glob, json, math, os, random, time

import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = os.environ.get("R1_HF", "E:/models/r1-14b-hf")
WORK = os.environ.get("MTP_WORK", "E:/models/r1-14b-mtp")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16


# ══ 逐层流式主干（只求最后层隐状态，无需整模型） ═══════════════════════════════
class StreamQwen2:
    def __init__(self, md=MODEL_DIR, device=DEV, dtype=DT):
        from transformers import AutoConfig
        self.cfg = AutoConfig.from_pretrained(md)
        self.cfg._attn_implementation = "sdpa"      # ★ 必须显式设，否则注意力模块分发拿到 None（本机实测会 segfault）
        self.md, self.dev, self.dt = md, device, dtype
        idx = json.load(open(os.path.join(md, "model.safetensors.index.json"), encoding="utf-8"))
        self.wmap = idx["weight_map"]
        from safetensors import safe_open
        with safe_open(os.path.join(md, self.wmap["model.embed_tokens.weight"]), framework="pt") as fh:
            self.embed = fh.get_tensor("model.embed_tokens.weight").to(device, dtype)
        with safe_open(os.path.join(md, self.wmap["model.norm.weight"]), framework="pt") as fh:
            self.norm_w = fh.get_tensor("model.norm.weight").to(device, dtype)
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        self.rope = Qwen2RotaryEmbedding(self.cfg).to(device)
        # ★ 分片会从中间切断某一层 ⇒ 必须建"层→(分片,键)"索引，而不是按分片顺序取
        self.layer_keys = {}
        for k, shard in self.wmap.items():
            if k.startswith("model.layers."):
                self.layer_keys.setdefault(int(k.split(".")[2]), []).append((shard, k))
        self.n_layers = self.cfg.num_hidden_layers
        print(f"[stream] 分片 {len(self.wmap.values()) and len(set(self.wmap.values()))}，层 {self.n_layers}，d {self.cfg.hidden_size}", flush=True)

    def forward_hidden(self, ids):
        """ids:[B,L] → 最后层隐状态（pre-norm）[B,L,d]"""
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from safetensors import safe_open
        B, L = ids.shape
        pos = torch.arange(L, device=self.dev).unsqueeze(0)
        cos, sin = self.rope(self.embed[:1, :1].new_zeros(1, 1, 1), pos)
        h = self.embed[ids.to(self.dev)]
        t0 = time.time()
        # ★ 只开"当前层需要的分片"，用完即关（6 个分片全映射 = 27GB 虚拟内存，本机实测会 segfault）
        layer = Qwen2DecoderLayer(self.cfg, layer_idx=0).to(self.dev, self.dt)
        import gc
        try:
            for li in range(self.n_layers):
                need = {}
                for s, k in self.layer_keys[li]:
                    need.setdefault(s, []).append(k)
                sd = {}
                for s, ks in need.items():
                    fh = safe_open(os.path.join(self.md, s), framework="pt")
                    try:
                        for k in ks:
                            sd[k[len(f"model.layers.{li}."):]] = fh.get_tensor(k).to(self.dev, self.dt)
                    finally:
                        fh.__exit__(None, None, None)
                layer.load_state_dict(sd, strict=True)
                with torch.no_grad():
                    out = layer(h, attention_mask=None, position_ids=pos, past_key_value=None,
                                use_cache=False, position_embeddings=(cos, sin))
                h = out[0] if isinstance(out, tuple) else out
                del sd, out
                gc.collect()
                if li % 8 == 0 or li == self.n_layers - 1:
                    print(f"  [stream] L{li:02d} ok {time.time()-t0:5.1f}s 显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
        finally:
            del layer
        torch.cuda.empty_cache()
        print(f"[stream] 前向完成 {time.time()-t0:.1f}s（{self.n_layers} 层逐层装卸）", flush=True)
        return h

    @torch.no_grad()
    def normed(self, h):
        w = self.norm_w.float()
        v = h.float()
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)) * w


# ══ MTP 头（DeepSeek-V3 式：融合 h_t 与 emb(x_{t+1})，输出复用主干 lm_head） ══════
class MTPHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        cfg._attn_implementation = "sdpa"     # ★ 独立子模块必须显式设
        d = cfg.hidden_size
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        # ★ layer_idx 必须落在 [0, num_hidden_layers) 内：传 48 会 IndexError（config.layer_types[48]）
        self.block = Qwen2DecoderLayer(cfg, layer_idx=0)

    def forward(self, h_prev, emb_next, pos_emb, position_ids=None):
        x = self.norm(self.proj(torch.cat([h_prev, emb_next], dim=-1)))
        out = self.block(x, attention_mask=None, position_ids=position_ids,
                         past_key_value=None, use_cache=False, position_embeddings=pos_emb)
        return out[0] if isinstance(out, tuple) else out


# ══ 小工具：只取 embed / lm_head / norm 三个大张量（≈2.9 GB） ═══════════════════
def load_small(md=MODEL_DIR, dev=DEV, dt=DT):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(md, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
    out = {}
    for name in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]:
        with safe_open(os.path.join(md, idx[name]), framework="pt") as fh:
            out[name] = fh.get_tensor(name).to(dev, dt)
    return out


def corpus_texts(root, max_files=40):
    pats = [os.path.join(root, "**", "*.md"), os.path.join(root, "**", "*.py")]
    files = sorted({f for p in pats for f in glob.glob(p, recursive=True)})
    files = [f for f in files if os.path.getsize(f) > 2000][:max_files]
    return files


# ══ stage 1 cache ════════════════════════════════════════════════════════════
def stage_cache(a):
    from transformers import AutoTokenizer
    os.makedirs(os.path.join(WORK, "hidden"), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    files = corpus_texts(a.corpus)
    print(f"[cache] 语料 {len(files)} 个文件", flush=True)
    trunk = StreamQwen2()
    CH, n, t0 = a.chunk_len, 0, time.time()
    # ★ 批量喂：一遍读盘 27 GB 摊到 BC 个 chunk 上（否则每 chunk 一遍盘）
    BC = max(1, a.batch_chunks)
    buf_h, buf_x, buf_ids = [], [], []
    def flush_buf():
        nonlocal n, buf_h, buf_x, buf_ids
        if not buf_ids:
            return
        batch = torch.stack(buf_ids)                       # [B, L]
        h = trunk.forward_hidden(batch).to(torch.float16).cpu()
        for i, ids_seg in enumerate(buf_x):
            np.savez_compressed(os.path.join(WORK, "hidden", f"c{n:04d}.npz"),
                                h=h[i].numpy(), x=np.array(ids_seg, dtype=np.int32))
            n += 1
        buf_h, buf_x, buf_ids = [], [], []
        print(f"  [cache] {n} chunk  用时 {time.time()-t0:.0f}s", flush=True)

    stop = False
    for fi, f in enumerate(files):
        txt = open(f, encoding="utf-8", errors="replace").read()[:120000]
        ids = tok(txt).input_ids
        for s in range(0, max(0, len(ids) - CH - 2), CH):
            seg = ids[s:s + CH + 1]
            buf_x.append(seg)
            buf_ids.append(torch.tensor(seg))
            if len(buf_ids) >= BC:
                flush_buf()
            if n + len(buf_ids) >= a.chunks:
                stop = True
                break
        if stop:
            break
    flush_buf()
    print(f"[cache] 完成 {n} chunk → {WORK}/hidden（{time.time()-t0:.0f}s）", flush=True)


# ══ stage 2 train ════════════════════════════════════════════════════════════
def stage_train(a):
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    files = sorted(glob.glob(os.path.join(WORK, "hidden", "*.npz")))
    rnd = random.Random(0)
    held = set(rnd.sample(range(len(files)), max(4, len(files) // 8)))
    train_files = [f for i, f in enumerate(files) if i not in held]
    print(f"[train] chunk 总 {len(files)}｜训练 {len(train_files)}｜留出 {len(held)}", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    small = load_small()
    embed, lm_head, norm_w = (small["model.embed_tokens.weight"], small["lm_head.weight"],
                              small["model.norm.weight"])
    head = MTPHead(cfg).to(DEV, DT).train()
    npar = sum(p.numel() for p in head.parameters())
    print(f"[train] 头参数 {npar/1e6:.1f} M（主干冻结）｜显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    rope = Qwen2RotaryEmbedding(cfg).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=(0.9, 0.95))
    # ★ 余弦退火 + 线性 warmup（首轮 lr 3e-4 恒定导致 loss 剧烈震荡 7.7↔2.8）
    warm = max(10, int(0.05 * a.steps))
    def lr_at(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, a.steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    losses = []
    for step in range(a.steps):
        f = rnd.choice(train_files)
        z = np.load(f)
        x = torch.from_numpy(z["x"].astype(np.int64)).to(DEV)
        h = torch.from_numpy(z["h"]).to(DEV, DT)
        h_prev, x_next, y = h[:-2].unsqueeze(0), x[1:-1].unsqueeze(0), x[2:].unsqueeze(0)
        pos = torch.arange(h_prev.shape[1], device=DEV).unsqueeze(0)
        cos, sin = rope(h_prev, pos)
        out = head(h_prev, embed[x_next], (cos, sin), position_ids=pos)
        logits = (out.float() @ lm_head.float().t())
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  [train] step {step:4d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} 显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    torch.save(head.state_dict(), os.path.join(WORK, "mtp_head.pt"))
    json.dump({"steps": a.steps, "lr": a.lr, "params": npar, "losses": losses,
               "held_out": sorted(held)}, open(os.path.join(WORK, "train_log.json"), "w"), indent=2)
    print(f"[train] loss {np.mean(losses[:20]):.3f} → {np.mean(losses[-20:]):.3f}｜已存 {WORK}/mtp_head.pt", flush=True)


# ══ stage 3 eval（纯离线算 α₁，无需生成） ════════════════════════════════════
def stage_eval(a):
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    small = load_small()
    embed, lm_head, norm_w = (small["model.embed_tokens.weight"], small["lm_head.weight"],
                              small["model.norm.weight"])
    head = MTPHead(cfg).to(DEV, DT)
    head.load_state_dict(torch.load(os.path.join(WORK, "mtp_head.pt"), map_location=DEV))
    head.eval()
    rope = Qwen2RotaryEmbedding(cfg).to(DEV)
    log = json.load(open(os.path.join(WORK, "train_log.json")))
    files = sorted(glob.glob(os.path.join(WORK, "hidden", "*.npz")))
    held = [f for i, f in enumerate(files) if i in set(log.get("held_out", []))]
    use = held or files[-8:]
    print(f"[eval] 留出 chunk {len(use)}（共 {len(files)}）", flush=True)

    def normed(h):
        v = h.float(); return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)) * norm_w.float()

    a1 = a2 = tot = 0
    acc_same_prefix = 0
    ok_numel = 0
    base_rep = base_copy = 0
    with torch.no_grad():
        for f in use:
            z = np.load(f)
            x = torch.from_numpy(z["x"].astype(np.int64)).to(DEV)
            h = torch.from_numpy(z["h"]).to(DEV, DT)
            # 主干贪心：位置 t 的下一 token = argmax(lm_head·norm(h_t))
            trunk_next = (normed(h) @ lm_head.float().t()).argmax(-1)          # [L]
            # 若主干贪心 == 真实下一 token（teacher forcing 自洽），则该位置可用于接受率统计
            ok = trunk_next[:-1] == x[1:]
            acc_same_prefix += int(ok.sum()); ok_numel += int(ok.numel())
            h_prev, x_next = h[:-2].unsqueeze(0), x[1:-1].unsqueeze(0)
            pos = torch.arange(h_prev.shape[1], device=DEV).unsqueeze(0)
            cos, sin = rope(h_prev, pos)
            out = head(h_prev, embed[x_next], (cos, sin), position_ids=pos)
            draft = (out.float() @ lm_head.float().t()).argmax(-1)[0]          # 预测 t+2
            tgt = trunk_next[1:-1]                                             # 主干在 t+1 的贪心 = t+2
            m = ok[:-1]
            tot += int(m.sum())
            a1 += int(((draft == tgt) & m).sum())
            # 基线（同样只在自洽位置上比）
            base_rep += int(((trunk_next[:-2] == tgt) & m).sum())   # 直接复读主干上一步的预测
            base_copy += int(((x[1:-1] == tgt) & m).sum())          # 复读当前 token
    print(f"[eval] ★ draft vs 主干 greedy 的 α₁ = {a1/max(1,tot):.4f}  (n={tot}，留出集)")
    print(f"[eval] 基线A·复读主干上一步 α₁ = {base_rep/max(1,tot):.4f}｜基线B·复读当前 token α₁ = {base_copy/max(1,tot):.4f}")
    print(f"[eval] 主干贪心与原文一致率 = {acc_same_prefix/max(1,ok_numel):.3f}（teacher-forcing 自洽性）")
    json.dump({"alpha1": a1 / max(1, tot), "n": tot,
               "base_repeat_prev": base_rep / max(1, tot), "base_copy_cur": base_copy / max(1, tot),
               "trunk_selfconsistency": acc_same_prefix / max(1, ok_numel)},
              open(os.path.join(WORK, "eval.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cache", "train", "eval"])
    ap.add_argument("--corpus", default=r"F:/DESKTOP/AI架构与推理设计/千寻双引擎推理栈")
    ap.add_argument("--chunks", type=int, default=160)
    ap.add_argument("--chunk-len", type=int, default=512, dest="chunk_len")
    ap.add_argument("--batch-chunks", type=int, default=8, dest="batch_chunks",
                    help="一遍读盘同时算几个 chunk（越大越省盘读，显存相应上升）")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    {"cache": stage_cache, "train": stage_train, "eval": stage_eval}[a.stage](a)
