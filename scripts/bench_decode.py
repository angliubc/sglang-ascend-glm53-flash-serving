#!/usr/bin/env python3
"""decode 对照尺：SSE 事件流法，任意引擎/并行度 A/B 通用（910B/H800/vLLM/SGLang）。

原理：MTP/EAGLE 下每个 SSE 事件 ≈ 1 个 verify step 的产出 token。
  - median gap = step 时间（跨任务可比的引擎指标，抗 stall）
  - window tok/s = (末事件 tokens-首事件 tokens)/window
  - tokens/event ≈ accept length（任务相关！A/B 必须同 prompt 同 temperature）

用法: python3 bench_decode.py [base_url] [n_concurrent] [max_tokens]
示例: python3 bench_decode.py http://<node-ip>:8100 1 320
基线（GLM-5.2-w4a8c8 SGLang TP16, 0830 夜干净流量）:
  1路 step 52.1ms / 108.9 tok/s | 2路 57.4ms / 聚合 141.6 | 4路 63.6ms / 聚合 198.4
注意: 测试机有 http_proxy 时 urllib 会走代理——已用 ProxyHandler({}) 绕开。
"""
import sys, time, json, statistics, urllib.request, threading

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8100'
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MT = int(sys.argv[3]) if len(sys.argv) > 3 else 320
MODEL = sys.argv[4] if len(sys.argv) > 4 else 'glm5.2'
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROMPT = "写一篇关于秋天的短文，400字左右，写得优美一些。"

def one_stream(i, results):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': PROMPT}],
                       'max_tokens': MT, 'temperature': 0.7, 'stream': True}).encode()
    req = urllib.request.Request(BASE + '/v1/chat/completions', body,
                                 {'Content-Type': 'application/json'})
    events, ntok = [], 0
    with op.open(req, timeout=600) as r:
        buf = b''
        for chunk in r:
            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                line = line.strip()
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    continue
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                d = j.get('choices', [{}])[0].get('delta', {})
                n = len(d.get('content') or '') + len(d.get('reasoning_content') or '')
                if n:
                    ntok += n
                    events.append((time.time(), ntok))
    results[i] = (events, ntok)

results = {}
threads = [threading.Thread(target=one_stream, args=(i, results)) for i in range(NC)]
t_start = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.time() - t_start

for i in sorted(results):
    events, ntok = results[i]
    if len(events) >= 3:
        gaps = [events[k + 1][0] - events[k][0] for k in range(len(events) - 1)]
        med = statistics.median(gaps) * 1000
        win = events[-1][0] - events[0][0]
        rate = (ntok - events[0][1]) / win if win > 0 else 0
        acc = (ntok - events[0][1]) / (len(events) - 1) if len(events) > 1 else 0
        print(f"  stream{i}: tokens={ntok} events={len(events)} win={win:.2f}s | "
              f"step(med)={med:.1f}ms | decode={rate:.1f} tok/s | accept~{acc:.2f}")
tot = sum(r[1] for r in results.values())
print(f"TOTAL: {tot} tokens / {wall:.1f}s wall | aggregate {tot / wall:.1f} tok/s | {BASE} NC={NC}")
