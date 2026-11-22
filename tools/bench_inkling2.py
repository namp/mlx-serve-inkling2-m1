#!/usr/bin/env python3
"""
6-workload streaming benchmark (and optional bracketed A/B comparison) for Inkling-2-Mini on mlx-serve.

Workloads:
  1K_raw  : ~1.2K-token raw completion (temperature=0)
  code    : Python LRU cache implementation (reasoning_effort=low, temp=0)
  eng     : English technical prose (reasoning_effort=low, temp=0)
  greek   : Greek technical explanation (reasoning_effort=low, temp=0)
  xhigh   : Math proof with reasoning_effort=xhigh, temp=1.0, top_p=1.0
  8K_raw  : ~8K-token raw completion (temperature=0)

Example:
  python3 tools/bench_inkling2.py \
      --bin ./zig-out/bin/mlx-serve \
      --model /path/to/inkling2-q4attn \
      --baseline-bin /path/to/stock-mlx-serve \
      --baseline-model /path/to/inkling2-canonical
"""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request

PROMPTS = {
    "1K_raw": {
        "kind": "completions",
        "prompt": ("The quick brown fox jumps over the lazy dog. " * 110)[:4500]
        + "\nSummarize the key pattern in three short bullet points:\n-",
        "max_tokens": 128,
        "temperature": 0.0,
    },
    "code": {
        "kind": "chat",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a clean Python implementation of an LRU cache using a doubly linked list "
                    "and a dict, with O(1) get and put and brief docstrings. Keep reasoning brief."
                ),
            }
        ],
        "max_tokens": 256,
        "temperature": 0.0,
        "reasoning_effort": "low",
    },
    "eng": {
        "kind": "chat",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain in clear, concise prose how speculative decoding guarantees exact "
                    "output distribution equivalence with autoregressive decoding. Keep reasoning brief."
                ),
            }
        ],
        "max_tokens": 220,
        "temperature": 0.0,
        "reasoning_effort": "low",
    },
    "greek": {
        "kind": "chat",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Εξήγησε με απλά και καθαρά ελληνικά τι είναι η στοχαστική κάθοδος κλίσης (SGD) "
                    "και πώς διαφέρει από τη μέθοδο Newton στη βελτιστοποίηση. Σύντομη σκέψη."
                ),
            }
        ],
        "max_tokens": 220,
        "temperature": 0.0,
        "reasoning_effort": "low",
    },
    "xhigh": {
        "kind": "chat",
        "messages": [
            {
                "role": "user",
                "content": "Prove that there are infinitely many primes congruent to 3 mod 4, concisely.",
            }
        ],
        "max_tokens": 400,
        "temperature": 1.0,
        "top_p": 1.0,
        "reasoning_effort": "xhigh",
    },
    "8K_raw": {
        "kind": "completions",
        "prompt": (
            "In distributed systems, consensus protocols like Raft separate leader election from "
            "log replication to maintain linearizability across partitions. " * 330
        )
        + "\nNow list 5 crisp engineering takeaways:\n1.",
        "max_tokens": 128,
        "temperature": 0.0,
    },
}


def wait_ready(proc: subprocess.Popen, base_url: str, timeout: float = 90.0) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=2) as r:
                if r.status == 200:
                    data = json.loads(r.read().decode("utf-8"))
                    return data["data"][0]["id"]
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Server did not become ready before timeout")


def run_stream(base_url: str, model_id: str, spec: dict) -> dict:
    url = f"{base_url}/completions" if spec["kind"] == "completions" else f"{base_url}/chat/completions"
    body = {k: v for k, v in spec.items() if k != "kind"}
    body["model"] = model_id
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    t_first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=240) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if spec["kind"] == "completions":
                if ch.get("text") and t_first is None:
                    t_first = time.perf_counter()
            else:
                delta = ch.get("delta") or {}
                if (delta.get("reasoning_content") or delta.get("content")) and t_first is None:
                    t_first = time.perf_counter()
    t_end = time.perf_counter()
    ctok = usage.get("completion_tokens", 0)
    ptok = usage.get("prompt_tokens", 0)
    ttft = (t_first - t0) if t_first else (t_end - t0)
    gen_s = (t_end - t_first) if (t_first and t_end > t_first) else (t_end - t0)
    tps = (ctok - 1) / gen_s if ctok > 1 and gen_s > 0 else (ctok / gen_s if gen_s > 0 else 0.0)
    return {"ptok": ptok, "ctok": ctok, "ttft": ttft, "tps": tps}


def bench_server(
    label: str, bin_path: str, model_path: str, port: int, kv_quant: str, keys: list[str], extra_env: dict
) -> dict:
    base_url = f"http://127.0.0.1:{port}/v1"
    env = os.environ.copy()
    env.update(extra_env)
    cmd = [
        bin_path,
        "--model",
        model_path,
        "--serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "1048576",
        "--max-tokens",
        "65536",
        "--os-reserve-gib",
        "16",
        "--max-concurrent",
        "1",
        "--kv-quant",
        kv_quant,
        "--no-vision",
        "--prefix-cache-entries",
        "0",
        "--no-prefix-cache-ram",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
    try:
        model_id = wait_ready(proc, base_url)
        run_stream(base_url, model_id, {"kind": "completions", "prompt": "Hello", "max_tokens": 8, "temperature": 0.0})
        out = {}
        for k in keys:
            r = run_stream(base_url, model_id, PROMPTS[k])
            out[k] = r
            print(
                f"[{label:12s}] {k:7s}: ptok={r['ptok']:5d} ctok={r['ctok']:3d} "
                f"ttft={r['ttft']:5.2f}s tps={r['tps']:6.2f}",
                flush=True,
            )
        return out
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        time.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Inkling-2-Mini on mlx-serve")
    ap.add_argument("--bin", required=True, help="Path to optimized mlx-serve binary")
    ap.add_argument("--model", required=True, help="Path to optimized Inkling-2-Mini model directory")
    ap.add_argument("--baseline-bin", default=None, help="Optional baseline mlx-serve binary for bracketed A/B")
    ap.add_argument("--baseline-model", default=None, help="Optional baseline model directory")
    ap.add_argument("--port", type=int, default=18087)
    ap.add_argument("--kv-quant", default="4")
    ap.add_argument("--workloads", default="1K_raw,code,eng,greek,xhigh,8K_raw")
    ap.add_argument("--force-draft", action="store_true", help="Set INKLING_FORCE_DRAFT=1 on optimized arm")
    args = ap.parse_args()

    keys = [k.strip() for k in args.workloads.split(",") if k.strip()]
    opt_env = {"INKLING_FORCE_DRAFT": "1"} if args.force_draft else {}

    r1 = bench_server("OPT_1", args.bin, args.model, args.port, args.kv_quant, keys, opt_env)
    if args.baseline_bin and args.baseline_model:
        rb = bench_server("BASELINE", args.baseline_bin, args.baseline_model, args.port, args.kv_quant, keys, {})
        r2 = bench_server("OPT_2", args.bin, args.model, args.port, args.kv_quant, keys, opt_env)
        print("\n=== BRACKETED A/B SUMMARY (OPT mean vs BASELINE) ===")
        for k in keys:
            opt_tps = 0.5 * (r1[k]["tps"] + r2[k]["tps"])
            base_tps = rb[k]["tps"]
            opt_ttft = 0.5 * (r1[k]["ttft"] + r2[k]["ttft"])
            base_ttft = rb[k]["ttft"]
            gain = (opt_tps / base_tps - 1.0) * 100.0 if base_tps > 0 else 0.0
            print(
                f"  {k:7s}: BASE {base_tps:5.2f} tok/s (ttft {base_ttft:5.2f}s) -> "
                f"OPT {opt_tps:5.2f} tok/s (ttft {opt_ttft:5.2f}s)  [{gain:+5.1f}%]"
            )


if __name__ == "__main__":
    main()
