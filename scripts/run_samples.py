"""Run the public sample cases against the service and grade them.

    python -m scripts.run_samples                          # in-process app
    python -m scripts.run_samples --url http://localhost:8000
    python -m scripts.run_samples --url https://<deployed-host>

For every case it checks: HTTP 200, scenario_id echo, directive
interpretation against the reference (type, applies, hours, numeric value
within 0.01), an independent replay of the plan against the reference
directives, and cost ratio vs the organizer optimum.
Exit code is non-zero if any case fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.replay import replay  # noqa: E402

NUMERIC_KEYS = ("factor", "minimum_energy_kwh", "max_grid_kwh")


def compare_interpretation(got, want):
    problems = []
    if len(got) != len(want):
        return [f"expected {len(want)} entries, got {len(got)}"]
    for g, w in zip(got, want):
        i = w["note_index"]
        if g.get("note_index") != i:
            problems.append(f"note {i}: wrong note_index {g.get('note_index')}")
        if g.get("directive_type") != w["directive_type"] or g.get("applies") != w["applies"]:
            problems.append(f"note {i}: got {g.get('directive_type')} want {w['directive_type']}")
            continue
        ga, wa = g.get("structured_adjustment"), w["structured_adjustment"]
        if wa is None:
            if ga is not None:
                problems.append(f"note {i}: no_op must have null adjustment")
            continue
        if ga.get("hours") != wa["hours"]:
            problems.append(f"note {i}: hours {ga.get('hours')} want {wa['hours']}")
        for k in NUMERIC_KEYS:
            if k in wa and abs(float(ga.get(k, -1e9)) - wa[k]) > 0.01:
                problems.append(f"note {i}: {k} {ga.get(k)} want {wa[k]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="base URL of a running service (default: in-process)")
    ap.add_argument("--cases", default=str(ROOT / "data" / "public_samples.json"))
    ap.add_argument("--only", help="comma-separated case ids")
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    if args.url:
        import httpx

        client = httpx.Client(base_url=args.url.rstrip("/"), timeout=35)
    else:
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app).__enter__()

    health = client.get("/health")
    print(f"/health -> {health.status_code} {health.text}")

    failures, latencies = 0, []
    for case in cases:
        inp, exp = case["input"], case["expected_output"]
        t0 = time.monotonic()
        resp = client.post("/optimize-energy", json=inp)
        latencies.append(time.monotonic() - t0)
        if resp.status_code != 200:
            print(f"FAIL {case['id']}: HTTP {resp.status_code} {resp.text[:200]}")
            failures += 1
            continue
        out = resp.json()
        problems = []
        if out.get("scenario_id") != inp["scenario_id"]:
            problems.append("scenario_id not echoed")
        problems += compare_interpretation(out.get("directive_interpretation", []), exp["directive_interpretation"])
        truth = [d for d in exp["directive_interpretation"] if d["applies"]]
        problems += [f"replay: {v}" for v in replay(inp["hours"], inp["battery"], truth, out)]
        ratio = min(1.0, exp["total_cost_bdt"] / out["total_cost_bdt"]) if out.get("total_cost_bdt") else 0.0
        if ratio < 0.9999:
            problems.append(f"cost {out['total_cost_bdt']} vs optimal {exp['total_cost_bdt']}")
        status = "PASS" if not problems else "FAIL"
        failures += bool(problems)
        print(f"{status} {case['id']:<10} cost={out['total_cost_bdt']:<9g} ratio={ratio:.4f} {latencies[-1]:.2f}s")
        for p in problems:
            print(f"     - {p}")

    latencies.sort()
    p95 = latencies[max(0, int(round(0.95 * len(latencies))) - 1)] if latencies else 0
    print(f"\n{len(cases) - failures}/{len(cases)} passed; p95 latency {p95:.2f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
