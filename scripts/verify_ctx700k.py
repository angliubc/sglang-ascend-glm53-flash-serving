#!/usr/bin/env python3
"""700K 上下文验证: 短算术 + 长文暗号检索 (sparse-layer 池补丁正确性)."""
import json, random, sys, time
import urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://190.168.24.33:8078"

def chat(messages, max_tokens=512, temperature=0.0):
    req = urllib.request.Request(
        API + "/v1/chat/completions",
        data=json.dumps({
            "model": "glm53-flash",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d, time.time() - t0

# --- 1. 短 prompt 算术 ---
d, el = chat([{"role": "user", "content": "17 * 23 = ? 只回答数字"}], max_tokens=64)
ans1 = d["choices"][0]["message"].get("content", "")
ok1 = "391" in ans1
print(f"[{'PASS' if ok1 else 'FAIL'}] 短算术 17*23: {ans1.strip()[:40]!r} ({el:.1f}s)")

# --- 2. 长文暗号检索 ---
CANARY = "ZQX-7719-斑马线"
TARGET_TOKENS = 600_000
para = []
words = ["系统", "数据", "分析", "模型", "结构", "过程", "参数", "优化", "指标", "方法",
         "测试", "结果", "性能", "配置", "环境", "计算", "资源", "调度", "传输", "缓存"]
random.seed(42)
# ~1.6 token/词, 填充到 ~60 万 token
n_words = int(TARGET_TOKENS / 1.6)
for i in range(0, n_words, 10):
    para.append("".join(random.choice(words) for _ in range(10)))
filler = "，".join(para)
# 暗号埋在文本中段
mid = len(filler) // 2
doc = filler[:mid] + f"\n\n重要备注：本段的核心验证码是 {CANARY}，请务必记住。\n\n" + filler[mid:]

d, el = chat([
    {"role": "user", "content": doc + "\n\n请回答：上文“重要备注”中的核心验证码是什么？只回答验证码本身。"}
], max_tokens=64)
ans2 = d["choices"][0]["message"].get("content", "")
u = d.get("usage", {})
ok2 = CANARY in ans2
print(f"[{'PASS' if ok2 else 'FAIL'}] 暗号检索: {ans2.strip()[:50]!r} ({el:.0f}s, prompt_tokens={u.get('prompt_tokens')})")

# --- 3. 二次检索(同一前缀 radix 命中路径) ---
d, el = chat([
    {"role": "user", "content": doc + "\n\n请回答：验证码中的四位数字是什么？只回答数字。"}
], max_tokens=64)
ans3 = d["choices"][0]["message"].get("content", "")
ok3 = "7719" in ans3
print(f"[{'PASS' if ok3 else 'FAIL'}] 二次检索(radix): {ans3.strip()[:30]!r} ({el:.0f}s)")

print("ALL PASS" if (ok1 and ok2 and ok3) else "SOME FAILED")
sys.exit(0 if (ok1 and ok2 and ok3) else 1)
