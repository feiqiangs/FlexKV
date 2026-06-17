# -*- coding: utf-8 -*-
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Bounded FlexKV benchmark")
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1/")
    parser.add_argument("--model", default="glm-5")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.data, errors="ignore") as f:
        rows = json.load(f)[: args.limit]

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)

    def run_one(i, row):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=row["messages"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=False,
            )
            dt = time.perf_counter() - t0
            usage = resp.usage
            cached = 0
            details = getattr(usage, "prompt_tokens_details", None) if usage else None
            if details:
                cached = getattr(details, "cached_tokens", 0) or 0
            return {
                "idx": i,
                "ok": True,
                "latency": dt,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "cached_tokens": cached,
                "finish_reason": resp.choices[0].finish_reason if resp.choices else None,
                "error": "",
            }
        except Exception as exc:
            return {
                "idx": i,
                "ok": False,
                "latency": time.perf_counter() - t0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "finish_reason": None,
                "error": repr(exc),
            }

    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_one, i, row) for i, row in enumerate(rows)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result["ok"] else "ERR"
            print(
                "[{n}] {status} idx={idx} latency={latency:.4f}s "
                "prompt={prompt} completion={completion} cached={cached} "
                "finish={finish} err={error}".format(
                    n=len(results),
                    status=status,
                    idx=result["idx"],
                    latency=result["latency"],
                    prompt=result["prompt_tokens"],
                    completion=result["completion_tokens"],
                    cached=result["cached_tokens"],
                    finish=result["finish_reason"],
                    error=result["error"],
                ),
                flush=True,
            )

    wall = time.perf_counter() - start
    oks = [r for r in results if r["ok"]]
    lats = [r["latency"] for r in oks]
    summary = {
        "submitted": len(rows),
        "completed": len(results),
        "success": len(oks),
        "failed": len(results) - len(oks),
        "wall_s": round(wall, 4),
        "avg_latency_s": round(statistics.mean(lats), 4) if lats else None,
        "max_latency_s": round(max(lats), 4) if lats else None,
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in oks),
        "total_completion_tokens": sum(r["completion_tokens"] for r in oks),
        "total_cached_tokens": sum(r["cached_tokens"] for r in oks),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
