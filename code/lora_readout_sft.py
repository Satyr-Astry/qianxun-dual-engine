"""
N9-S1 · 读出侧 LoRA 微调（把「工具调用格式」刻进本机 r1-14B）
================================================================================
为什么走这条路（12 GB 单机的唯一可行解，且踩在我们自己的实验结论上）：
  · 14B 全层 QLoRA 在本机**不可能**：bnb 4bit/8bit 加载 14B 必 segfault（28GB bf16 过 16GB 内存）。
  · N2c/N2d 已实测：**读出侧才是瓶颈**（Arm1 记忆分支+读出侧有效、只训记忆分支 0%）。
  ⇒ 冻结主干，只训「末 K 层 + lm_head」的 LoRA，**用缓存好的隐状态**驱动（不需要主干常驻）。
  ⇒ 显存 = 末 K 层权重(bf16) + lm_head + LoRA 适配器；K=4 时约 3.7 GiB，稳。

三阶段：
  cache : 流式跑主干 0..K-1 层 → 存每条的「进入第 K 层的隐状态」+ token ids
  train : 载入 K..47 层 + lm_head（一次性搬进显存）+ LoRA → 在缓存 h 上算 SFT loss
          （只在 assistant 片段上回传，其余 mask 掉；正例/负例都学）
  eval  : 留出集上报两个数：①assistant 片段 token 命中率 ②`<tool_call>` 开场命中率

用法：
  python code/lora_readout_sft.py --stage cache --layers-from 44
  python code/lora_readout_sft.py --stage train --layers-from 44 --steps 1200 --lr 2e-4
  python code/lora_readout_sft.py --stage eval  --layers-from 44
"""
from __future__ import annotations
import argparse, glob, json, math, os, random, time

import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = os.environ.get("R1_HF", "E:/models/r1-14b-hf")
WORK = os.environ.get("LORA_WORK", "E:/models/r1-lora-tool")
DATA = os.path.join(WORK, "data", "tool_sft.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16


# ══ 流式主干：只跑到第 K 层之前，返回「进入第 K 层的隐状态」 ═══════════════════════
class StemPrefix:
    def __init__(self, layers_from: int, md=MODEL_DIR, device=DEV, dtype=DT):
        from transformers import AutoConfig
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        self.cfg = AutoConfig.from_pretrained(md)
        self.cfg._attn_implementation = "sdpa"     # ★ 独立子模块必须显式设
        self.K, self.md, self.dev, self.dt = layers_from, md, device, dtype
        idx = json.load(open(os.path.join(md, "model.safetensors.index.json"), encoding="utf-8"))
        self.wmap = idx["weight_map"]
        from safetensors import safe_open
        for name in ["model.embed_tokens.weight"]:
            with safe_open(os.path.join(md, self.wmap[name]), framework="pt") as fh:
                self.embed = fh.get_tensor(name).to(device, dtype)
        self.rope = Qwen2RotaryEmbedding(self.cfg).to(device)
        self.layer_keys = {}
        for k, shard in self.wmap.items():
            if k.startswith("model.layers."):
                self.layer_keys.setdefault(int(k.split(".")[2]), []).append((shard, k))
        print(f"[stem] 只跑到第 {self.K} 层（前 {self.K} 层冻结、只做前向）", flush=True)

    def hidden_into_K(self, ids):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from safetensors import safe_open
        pos = torch.arange(ids.shape[1], device=self.dev).unsqueeze(0)
        cos, sin = self.rope(self.embed[:1, :1].new_zeros(1, 1, 1), pos)
        h = self.embed[ids.to(self.dev)]
        t0 = time.time()
        # ★ 只开当前层需要的分片（同时映射 27GB 会 segfault）
        layer = Qwen2DecoderLayer(self.cfg, layer_idx=0).to(self.dev, self.dt)
        import gc
        try:
            for li in range(self.K):
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
                del sd, out; gc.collect()
        finally:
            del layer
        torch.cuda.empty_cache()
        print(f"  [stem] 前 {self.K} 层前向 {time.time()-t0:.1f}s 显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
        return h


# ══ 读出侧：末 K 层 + lm_head（常驻显存，加 LoRA） ═══════════════════════════════
class ReadoutStack(nn.Module):
    def __init__(self, cfg, layers_from: int, md=MODEL_DIR, device=DEV, dtype=DT):
        super().__init__()
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from safetensors import safe_open
        self.cfg, self.K, self.dev = cfg, layers_from, device
        # ★ 模块树必须与 transformers 同构：model.layers.{全局层号}.* + lm_head
        #   否则 peft 存出的键是 blocks.0.mlp.down_proj.weight，
        #   官方 convert_lora_to_gguf.py 会 ValueError: Can not map tensor
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict()
        idx = json.load(open(os.path.join(md, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
        for li in range(layers_from, cfg.num_hidden_layers):
            blk = Qwen2DecoderLayer(cfg, layer_idx=0)          # ★ layer_idx 必须在 [0, L) 内
            keys = [k for k in idx if k.startswith(f"model.layers.{li}.")]
            sd = {}
            by_shard = {}
            for k in keys:
                by_shard.setdefault(idx[k], []).append(k)
            for s, ks in by_shard.items():
                with safe_open(os.path.join(md, s), framework="pt") as fh:
                    for k in ks:
                        sd[k[len(f"model.layers.{li}."):]] = fh.get_tensor(k).to(device, dtype)
            blk.load_state_dict(sd, strict=True)
            self.model.layers[str(li)] = blk
            del sd
        self.model.norm = nn.Module()
        with safe_open(os.path.join(md, idx["model.norm.weight"]), framework="pt") as fh:
            self.model.norm.weight = nn.Parameter(fh.get_tensor("model.norm.weight").to(device, dtype))
        with safe_open(os.path.join(md, idx["lm_head.weight"]), framework="pt") as fh:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(fh.get_tensor("lm_head.weight").to(device, dtype))
        self.rope = None

    def forward(self, h, pos_ids):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        if self.rope is None:
            self.rope = Qwen2RotaryEmbedding(self.cfg).to(self.dev)
        cos, sin = self.rope(h, pos_ids)
        for key in sorted(self.model.layers.keys(), key=int):
            blk = self.model.layers[key]
            out = blk(h, attention_mask=None, position_ids=pos_ids, past_key_value=None,
                      use_cache=False, position_embeddings=(cos, sin))
            h = out[0] if isinstance(out, tuple) else out
        v = h.float()
        hn = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)) * self.model.norm.weight.float()
        # ★ 输出头是 bf16：不要把 fp32 直接喂进去（会 RuntimeError: float != BFloat16）
        return self.lm_head(hn.to(h.dtype))


def lora_attach(stack: nn.Module, r=16, alpha=32, dropout=0.05):
    """只给末 K 层的 q/k/v/o + gate/up/down 与 lm_head 挂 LoRA；主干其余冻结"""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj",
                                     "lm_head"],   # ★ 输出头必须挂：格式塑造最吃它
                     modules_to_save=None)
    model = get_peft_model(stack, cfg)
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)
    model.print_trainable_parameters()
    return model


# ══ stage cache ═══════════════════════════════════════════════════════════════
def build_sequences(tok, limit=None):
    """把 jsonl 渲染成 (ids, mask)：mask=1 表示该位置属于 assistant 需要学的内容
    ★ 用 fast tokenizer 的 offset_mapping 做**字符级→token 级**精确对齐（分段单独编码在
       BPE 边界上会错位，mask 一错 loss 就学到不该学的地方）"""
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    if limit:
        recs = recs[:limit]
    out = []
    for r in recs:
        text, spans = "", []
        for m in r["messages"]:
            text += f"<|im_start|>{m['role']}\n"
            a0 = len(text)
            text += m["content"] + "<|im_end|>\n"
            spans.append((m["role"], a0, a0 + len(m["content"])))
        enc = tok(text, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        mask = [0] * len(ids)
        for role, a0, a1 in spans:
            if role != "assistant":
                continue
            for i, (s, e) in enumerate(offs):
                if s >= a0 and e <= a1 and e > s:
                    mask[i] = 1
        out.append({"ids": ids, "mask": mask, "is_call": r["meta"]["is_call"]})
    return out


def stage_cache(a):
    from transformers import AutoTokenizer
    os.makedirs(os.path.join(WORK, "h"), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    seqs = build_sequences(tok)
    # ★ 按长度分桶：同一批里长度接近，padding 浪费最小（否则长样本拖累整批）
    seqs.sort(key=lambda s: len(s["ids"]))
    print(f"[cache] 样本 {len(seqs)}，平均 token {np.mean([len(s['ids']) for s in seqs]):.0f}"
          f"，最长 {max(len(s['ids']) for s in seqs)}", flush=True)
    stem = StemPrefix(a.layers_from)
    CH = a.chunk_len
    # 每条按 CH 切块缓存（长序列分块，块间断层——对 SFT 可接受，优先覆盖开头/结尾）
    idx = []
    t0, n = time.time(), 0
    batch, meta = [], []
    def flush():
        nonlocal n, idx, batch, meta
        if not batch:
            return
        L = max(x.shape[0] for x in batch)
        pad = torch.zeros(len(batch), L, dtype=torch.long)
        for i, x in enumerate(batch):
            pad[i, :x.shape[0]] = x
        h = stem.hidden_into_K(pad).to(torch.float16).cpu().numpy()
        for i, m in enumerate(meta):
            Lx = batch[i].shape[0]
            np.savez_compressed(os.path.join(WORK, "h", f"s{n:05d}.npz"),
                                h=h[i, :Lx], ids=batch[i].numpy(), mask=np.array(m["mask"][:Lx], dtype=np.int8))
            n += 1
        batch, meta = [], []
        print(f"  [cache] {n} 条 用时 {time.time()-t0:.0f}s", flush=True)
    for s in seqs:
        ids = s["ids"][:CH]
        batch.append(torch.tensor(ids))
        meta.append({"mask": s["mask"][:len(ids)]})
        if len(batch) >= a.batch_items:
            flush()
    flush()
    json.dump({"n": n, "layers_from": a.layers_from, "chunk": CH},
              open(os.path.join(WORK, "cache_meta.json"), "w"), indent=2)
    print(f"[cache] 完成 {n} 条 → {WORK}/h（{time.time()-t0:.0f}s）", flush=True)


# ══ stage train ═══════════════════════════════════════════════════════════════
def stage_train(a):
    from transformers import AutoConfig
    files = sorted(glob.glob(os.path.join(WORK, "h", "*.npz")))
    rnd = random.Random(0)
    held = set(rnd.sample(range(len(files)), max(20, len(files) // 10)))
    train_f = [f for i, f in enumerate(files) if i not in held]
    print(f"[train] 总 {len(files)}｜训练 {len(train_f)}｜留出 {len(held)}", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg._attn_implementation = "sdpa"
    stack = ReadoutStack(cfg, a.layers_from).to(DEV, DT)
    model = lora_attach(stack, r=a.lora_r, alpha=2 * a.lora_r)
    print(f"[train] 常驻显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95))
    warm = max(10, int(0.05 * a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    losses = []
    for step in range(a.steps):
        f = rnd.choice(train_f)
        z = np.load(f)
        ids = torch.from_numpy(z["ids"].astype(np.int64)).to(DEV).unsqueeze(0)
        m = torch.from_numpy(z["mask"].astype(np.int64)).to(DEV).unsqueeze(0)
        h = torch.from_numpy(z["h"]).to(DEV, DT).unsqueeze(0)
        pos = torch.arange(h.shape[1], device=DEV).unsqueeze(0)
        logits = model(h, pos)                                  # [1, L, V]
        # 预测下一 token：只在 mask 位置（assistant 片段）算 loss
        lg = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
        tgt = ids[:, 1:].reshape(-1)
        w = m[:, 1:].reshape(-1).float()
        loss_all = nn.functional.cross_entropy(lg, tgt, reduction="none")
        loss = (loss_all * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  [train] step {step:4d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    os.makedirs(os.path.join(WORK, "adapter"), exist_ok=True)
    model.save_pretrained(os.path.join(WORK, "adapter"))
    json.dump({"steps": a.steps, "lr": a.lr, "lora_r": a.lora_r, "layers_from": a.layers_from,
               "losses": losses, "held_out": sorted(held)},
              open(os.path.join(WORK, "train_log.json"), "w"), indent=2)
    print(f"[train] loss {np.mean(losses[:20]):.3f} → {np.mean(losses[-20:]):.3f}｜适配器 → {WORK}/adapter", flush=True)


# ══ stage eval：留出集上量两个硬指标 ═══════════════════════════════════════════
def stage_eval(a):
    from transformers import AutoConfig
    from peft import PeftModel
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg._attn_implementation = "sdpa"
    stack = ReadoutStack(cfg, a.layers_from).to(DEV, DT)
    model = PeftModel.from_pretrained(stack, os.path.join(WORK, "adapter")).to(DEV, DT).eval()
    log = json.load(open(os.path.join(WORK, "train_log.json")))
    files = sorted(glob.glob(os.path.join(WORK, "h", "*.npz")))
    held = [f for i, f in enumerate(files) if i in set(log["held_out"])]
    tok_hit = tok_n = 0
    call_hit = call_n = 0
    with torch.no_grad():
        for f in held:
            z = np.load(f)
            ids = torch.from_numpy(z["ids"].astype(np.int64)).to(DEV).unsqueeze(0)
            m = torch.from_numpy(z["mask"].astype(np.int64)).to(DEV).unsqueeze(0)
            h = torch.from_numpy(z["h"]).to(DEV, DT).unsqueeze(0)
            pos = torch.arange(h.shape[1], device=DEV).unsqueeze(0)
            pred = model(h, pos).argmax(-1)
            w = m[:, 1:].reshape(-1).bool()
            hit = (pred[:, :-1].reshape(-1)[w] == ids[:, 1:].reshape(-1)[w])
            tok_hit += int(hit.sum()); tok_n += int(w.sum())
            # 教它开场：assistant 片段**第一个** token 是否命中（正例里通常是 '<'）
            first = (m[0] == 1).nonzero()
            if len(first):
                t = int(first[0])
                call_n += 1
                if pred[0, max(0, t - 1)].item() == ids[0, t].item():
                    call_hit += 1
    print(f"[eval] assistant 片段 token 命中率 = {tok_hit/max(1,tok_n):.4f} (n={tok_n})")
    print(f"[eval] assistant 开场 token 命中率 = {call_hit/max(1,call_n):.4f} (n={call_n})")
    json.dump({"assist_token_acc": tok_hit / max(1, tok_n), "assist_open_acc": call_hit / max(1, call_n),
               "n_tok": tok_n, "n_seq": call_n},
              open(os.path.join(WORK, "eval.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cache", "train", "eval"])
    ap.add_argument("--layers-from", type=int, default=44, dest="layers_from",
                    help="从第几层开始挂 LoRA（前 K 层冻结）")
    ap.add_argument("--chunk-len", type=int, default=768, dest="chunk_len")
    ap.add_argument("--batch-items", type=int, default=4, dest="batch_items")
    ap.add_argument("--lora-r", type=int, default=16, dest="lora_r")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    {"cache": stage_cache, "train": stage_train, "eval": stage_eval}[a.stage](a)
