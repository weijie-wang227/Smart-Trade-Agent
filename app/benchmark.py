from time import perf_counter
from typing import Callable, Optional
from app.data import TEST_CASES


def run_benchmark(
    agent,
    use_hybrid: bool = False,
) -> list[dict]:
    results = []

    for tc in TEST_CASES:
        t0 = perf_counter()
        s = agent.suggest(tc["text"], use_hybrid=use_hybrid)
        t1 = perf_counter()

        results.append({
            "case": tc["id"],
            "input": tc["text"],
            "suggested_code": s.suggested_hs_code or "MANUAL_REVIEW",
            "confidence": round(s.confidence, 3),
            "manual_review": s.manual_review,
            "latency_ms": int((t1 - t0) * 1000),
            "cost_usd": s.cost,
            "reason": s.reason,
        })

    return results
