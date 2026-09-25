@echo off
chcp 65001 >nul
REM ============================================================
REM  一键把本机 r1-14B 接入 Hermes（已实测跑通的唯一配置）
REM  用法：双击本文件；然后在 Hermes 里 /model 选 r1-local，或
REM        hermes chat -q "..." --model r1-14b-local --provider r1-local
REM ============================================================
set BLOB=D:/ollama/blobs/sha256-38b5e20078675a1e3040eced1859e432b423ec732c42f5dab03b0a8ae7ba1bdd
set SERVER=F:/DESKTOP/AI架构与推理设计/分级激活LLM项目/p1/llama_new/llama-server.exe
set TEMPLATE=E:/models/qwen25_tools.jinja

echo [1/2] 清掉旧的 llama-server（Windows 允许多实例占同端口，必须先清）
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 3 /nobreak >nul

echo [2/2] 启动：ctx 65536（Hermes 硬要求 >=64K）+ q4 KV + 卸 16 层 FFN 到 CPU
"%SERVER%" -m "%BLOB%" ^
  -ngl 99 -c 65536 -ctk q4_0 -ctv q4_0 ^
  -ot "blk\.(3[2-9]|4[0-7])\.ffn.*=CPU" ^
  --jinja --chat-template-file "%TEMPLATE%" ^
  --host 127.0.0.1 --port 8710 -a r1-14b-local

REM 冷加载约 60-90 秒（要等 /health 返回 200 才算就绪）
REM 停止：taskkill /F /IM llama-server.exe
