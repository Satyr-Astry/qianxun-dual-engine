import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
P="E:/models/r1-14b-hf"
tok=AutoTokenizer.from_pretrained(P)
qc=BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
t0=time.time()
try:
    m=AutoModelForCausalLM.from_pretrained(P, quantization_config=qc, device_map="auto",
          max_memory={0:"10GiB","cpu":"12GiB"}, low_cpu_mem_usage=True)
    m.eval()
    print(f"[8bit OK] {time.time()-t0:.1f}s 显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    ids=tok("用一句话解释什么是 KV 缓存：", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out=m.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    print("[输出]", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:150]), flush=True)
except Exception as e:
    print(f"[8bit 失败] {type(e).__name__}: {str(e)[:250]}", flush=True)
