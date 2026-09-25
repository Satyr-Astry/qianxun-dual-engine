"""验证 r1-14b-hf：用 accelerate 流式加载 + 逐分片 4bit 量化（避免 28GB 挤爆 16GB 内存）"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

P = "E:/models/r1-14b-hf"
tok = AutoTokenizer.from_pretrained(P)
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_enable_fp32_cpu_offload=True)   # ★ 允许少量模块回退 CPU
t0 = time.time()
try:
    m = AutoModelForCausalLM.from_pretrained(
        P, quantization_config=qc,
        device_map="auto", max_memory={0: "11GiB", "cpu": "11GiB"},
        low_cpu_mem_usage=True)
    m.eval()
    n = sum(x.numel() for x in m.parameters()) / 1e9
    print(f"[加载OK] {time.time()-t0:.1f}s  参数 {n:.2f}B  显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    ids = tok("用一句话解释什么是 KV 缓存：", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    print("[输出]", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:160]), flush=True)
except Exception as e:
    print(f"[加载失败] {type(e).__name__}: {str(e)[:300]}", flush=True)
