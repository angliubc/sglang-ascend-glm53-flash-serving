#!/usr/bin/env python3
"""Long-context serving E2E battery (any sglang-compatible endpoint).

Reusable validation suite for long-context deployments (written for the
GLM-5.3-Flash MTP 32768-context port on 910B, 2026-09-07; see
references/glm53-longctx-kpool-npu-port-20260907.md). Run INSIDE the serving
container or any host that can reach the endpoint:

    python3 e2e_longctx_battery.py            # BASE default 127.0.0.1:8100
    BASE=http://10.0.0.1:9000 python3 ...     # override endpoint

Checks:
  [1] server_info context_length / pool size
  [2] short-prompt arithmetic battery (greedy) — compare failure MODES against
      the pre-change baseline before blaming the change (a same-question
      failure that also failed before = pre-existing, not a regression)
  [3] needle-in-haystack: ~10K-token prompt, needle near the START (max
      distance from the query) — proves prefill>2048 + sparse/retrieval chain
  [4] end-of-prompt recall combo (short-range within the same long ctx)
  [5] long-form generation coherence

Companion one-liners (on the serving node, after the run):
  accept length:  docker logs <ctr> 2>&1 | grep -aoE 'accept len: [0-9.]+' \
                    | awk -F': ' '{s+=$2;n++} END{print s/n, n}'
  graph replay:   docker logs <ctr> 2>&1 | grep -a 'npu graph: True' | head
                  (look for large #full token values = long-ctx decode in graph)
  decode tput:    grep -aoE 'gen throughput \(token/s\): [0-9.]+' (scheduler
                  self-report — NOT wall-clock aggregate, which is
                  prefill-dominated at long prompts)
"""
import os
import time
import requests

BASE = os.environ.get("BASE", "http://127.0.0.1:8100")


def gen(prompt, max_new=128, temp=0.0):
    r = requests.post(
        f"{BASE}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": temp,
                "max_new_tokens": max_new,
                "ignore_eos": False,
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


QUESTIONS = [
    ("17 * 23 = ? Reply with the number only.", 16, "391"),
    ("12 * 14 = ? Reply with the number only.", 16, "168"),
    ("45 + 38 = ? Reply with the number only.", 16, "83"),
    ("The capital of France is", 12, "Paris"),
    ("If a train travels 60 km in 1.5 hours, what is its speed in km/h? Reply with the number only.", 16, "40"),
]


def main():
    info = requests.get(f"{BASE}/get_server_info", timeout=30).json()
    print("[1] context_length =", info.get("context_length"),
          "pool =", info.get("max_total_num_tokens"))

    print("\n[2] short-prompt battery (greedy):")
    for q, mx, expect in QUESTIONS:
        t0 = time.time()
        out = gen(q, max_new=mx)
        txt = out["text"].replace("\n", "\\n")
        print(f"    Q: {q[:50]!r:52s} -> {txt[:60]!r}  (expect {expect}, {time.time()-t0:.1f}s)")

    docs = []
    marker = "ZQX742"  # change per run if you suspect cache effects
    for i in range(300):
        if i == 3:
            docs.append(
                f"Document {i}: The secret access code for the vault is {marker}. "
                f"Keep it confidential. The vault is located in the east wing."
            )
        else:
            docs.append(
                f"Document {i}: Routine inventory log for sector {i % 9}. "
                f"Items checked: {i * 7 % 13} boxes, {i * 3 % 11} crates. "
                f"All seals verified. Nothing unusual to report today."
            )
    prompt = (
        "Below is a series of inventory documents. Read them carefully.\n\n"
        + "\n\n".join(docs)
        + "\n\nQuestion: What is the secret access code for the vault mentioned "
        "in the documents? Reply with the code only."
    )
    print(f"\n[3] long needle: ~{len(prompt)} chars (~{len(prompt)//4} tokens), needle at doc 3/300")
    t0 = time.time()
    out = gen(prompt, max_new=32)
    meta = out.get("meta_info", {})
    print(f"    -> {out['text']!r}  HIT={marker in out['text']}  ({time.time()-t0:.1f}s)")
    print(f"    prompt_tokens={meta.get('prompt_tokens')} completion_tokens={meta.get('completion_tokens')}")

    out = gen(prompt.replace(
        "Reply with the code only.",
        "Also, what is 13 * 12? Reply with the code and the product, comma separated.",
    ), max_new=32)
    print(f"[4] end-recall combo -> {out['text']!r}  (expect {marker}, 156)")

    prompt_sum = ("Below is a series of inventory documents.\n\n" + "\n\n".join(docs)
                  + "\n\nSummarize the overall state of the inventory in 2 sentences.")
    t0 = time.time()
    out = gen(prompt_sum, max_new=160)
    print(f"\n[5] long-gen ({time.time()-t0:.1f}s) -> {out['text'][:400]!r}")

    print("\nDONE")


if __name__ == "__main__":
    main()
