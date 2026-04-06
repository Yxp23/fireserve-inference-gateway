"""
FireServe Load Tester — generates benchmark data for README.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --requests 200 --concurrency 20
"""

import asyncio
import argparse
import time
import statistics
import httpx
import json
import sys


PROMPTS = [
    "a futuristic cityscape at sunset, digital art",
    "portrait of a cyberpunk samurai, neon lighting",
    "underwater coral reef with bioluminescent creatures",
    "ancient temple ruins overgrown with vines, concept art",
    "steampunk airship flying through storm clouds",
    "minimalist geometric abstract art, vibrant colors",
    "cozy cabin in snowy mountains, warm lighting",
    "astronaut exploring alien marketplace, sci-fi illustration",
]


async def send_request(client: httpx.AsyncClient, url: str, prompt: str) -> dict:
    """Send a single generate request and return timing data."""
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{url}/v1/generate",
            json={"prompt": prompt, "width": 512, "height": 512, "num_inference_steps": 4},
            timeout=30.0,
        )
        latency = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "total_latency_ms": latency,
                "inference_ms": data.get("inference_time_ms", 0),
                "queue_wait_ms": data.get("queue_wait_ms", 0),
            }
        return {"success": False, "total_latency_ms": latency, "error": resp.status_code}
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return {"success": False, "total_latency_ms": latency, "error": str(e)}


async def run_benchmark(url: str, total_requests: int, concurrency: int):
    """Run the full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"  FireServe Load Test")
    print(f"  Target: {url}")
    print(f"  Requests: {total_requests} | Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    # Health check
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{url}/health", timeout=5)
            print(f"  Health: {health.json()['status']}")
        except Exception as e:
            print(f"  ⚠ Health check failed: {e}")
            return

    results = []
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def bounded_request(client, url, prompt):
        nonlocal completed
        async with semaphore:
            result = await send_request(client, url, prompt)
            completed += 1
            if completed % 20 == 0:
                print(f"  Progress: {completed}/{total_requests}")
            return result

    start_time = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [
            bounded_request(client, url, PROMPTS[i % len(PROMPTS)])
            for i in range(total_requests)
        ]
        results = await asyncio.gather(*tasks)

    total_time = time.monotonic() - start_time

    # Calculate stats
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["total_latency_ms"] for r in successes]

    if not latencies:
        print("\n  All requests failed!")
        return

    latencies.sort()

    report = {
        "summary": {
            "total_requests": total_requests,
            "successful": len(successes),
            "failed": len(failures),
            "error_rate": f"{len(failures)/total_requests*100:.1f}%",
            "total_time_seconds": round(total_time, 2),
            "throughput_rps": round(total_requests / total_time, 2),
        },
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p95": round(latencies[int(len(latencies) * 0.95)], 2),
            "p99": round(latencies[int(len(latencies) * 0.99)], 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
            "stddev": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
        },
    }

    print(f"\n{'─'*60}")
    print(f"  RESULTS")
    print(f"{'─'*60}")
    print(f"  Throughput:    {report['summary']['throughput_rps']} req/s")
    print(f"  Success Rate:  {len(successes)}/{total_requests} ({100-float(report['summary']['error_rate'][:-1]):.1f}%)")
    print(f"  Avg Latency:   {report['latency_ms']['avg']}ms")
    print(f"  p50 Latency:   {report['latency_ms']['median']}ms")
    print(f"  p95 Latency:   {report['latency_ms']['p95']}ms")
    print(f"  p99 Latency:   {report['latency_ms']['p99']}ms")
    print(f"{'─'*60}\n")

    # Save report
    with open("benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved to benchmark_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FireServe Load Tester")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway URL")
    parser.add_argument("--requests", type=int, default=200, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent requests")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.requests, args.concurrency))
