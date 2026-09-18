"""GridWise API: LLM -> guardrails -> LP optimizer -> replay."""

from __future__ import annotations

import itertools
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

try:  # local development convenience; containers get env vars directly
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from .interpret import interpret
from .llm import NoteInterpreter
from .optimizer import InfeasibleError, fallback_plan, optimize
from .replay import replay
from .schemas import ScenarioRequest, semantic_errors

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("gridwise")

llm = NoteInterpreter()


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("GridWise starting; LLM configured=%s (%s)", llm.configured, llm.provider_label)
    yield
    await llm.close()


app = FastAPI(title="GridWise", version="1.0.0", lifespan=lifespan)


# ---- error handling: controlled JSON errors, never stack traces ------------
@app.exception_handler(RequestValidationError)
async def _bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"loc": [str(x) for x in e.get("loc", ())], "msg": str(e.get("msg", ""))}
        for e in exc.errors()[:20]
    ]
    return JSONResponse(status_code=400, content={"error": "invalid request", "details": details})


@app.exception_handler(Exception)
async def _internal(_: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"error": "internal error"})


# ---- endpoints --------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize-energy")
async def optimize_energy(req: ScenarioRequest) -> JSONResponse:
    t0 = time.monotonic()
    problems = semantic_errors(req)
    if problems:
        return JSONResponse(status_code=422, content={"error": "invalid scenario", "details": problems})

    hours = [h.model_dump() for h in req.hours]
    battery = req.battery.model_dump()

    interpretations, source = await interpret(llm, req.operator_notes, battery)
    directives = [e for e in interpretations if e["applies"]]

    result, used = await run_in_threadpool(_schedule, hours, battery, directives)
    result.pop("effective_solar", None)

    log.info(
        "scenario=%s notes=%d source=%s directives=%s cost=%.2f in %.2fs",
        req.scenario_id,
        len(req.operator_notes),
        source,
        [d["directive_type"] for d in directives],
        result["total_cost_bdt"],
        time.monotonic() - t0,
    )
    return JSONResponse(
        {
            "scenario_id": req.scenario_id,
            "directive_interpretation": interpretations,
            "hourly_plan": result["hourly_plan"],
            "total_grid_kwh": result["total_grid_kwh"],
            "total_cost_bdt": result["total_cost_bdt"],
            "peak_grid_kwh": result["peak_grid_kwh"],
            "plan_summary": _summary(interpretations, used, result),
        }
    )


def _schedule(hours, battery, directives):
    """Optimal plan that passes replay; degrade safely if it cannot.

    Judge scenarios are feasible under the true directives, so infeasibility
    means one of our readings is wrong. Keep as many directives as possible
    (largest feasible subset, then cheapest) rather than failing the request.
    """
    for keep in range(len(directives), -1, -1):
        best = None
        for subset in itertools.combinations(directives, keep):
            subset = list(subset)
            try:
                res = optimize(hours, battery, subset)
            except InfeasibleError:
                continue
            violations = replay(hours, battery, subset, res)
            if violations:
                log.error("replay rejected optimizer plan: %s", violations[:5])
                continue
            if best is None or res["total_cost_bdt"] < best[0]["total_cost_bdt"]:
                best = (res, subset)
        if best is not None:
            if keep < len(directives):
                log.warning("dropped %d infeasible directive(s)", len(directives) - keep)
            return best
    log.error("no feasible optimized plan; returning idle-battery plan")
    return fallback_plan(hours, battery, directives), directives


def _summary(interpretations: List[Dict[str, Any]], used: List[Dict[str, Any]], result) -> str:
    plan = result["hourly_plan"]
    charged = [p["hour"] for p in plan if p["battery_action"] == "charge"]
    discharged = [p["hour"] for p in plan if p["battery_action"] == "discharge"]
    ignored = sum(1 for e in interpretations if not e["applies"])
    parts = []
    if used:
        names = ", ".join(sorted({d["directive_type"].replace("_", " ") for d in used}))
        parts.append(f"Applies {len(used)} operator directive(s) as hard constraints ({names})")
    else:
        parts.append("No operator directive changes the schedule")
    if ignored:
        parts.append(f"ignores {ignored} unrelated note(s)")
    text = "; ".join(parts) + ". "
    if charged or discharged:
        text += (
            f"Charges the battery in {len(charged)} lower-cost hour(s) and discharges in "
            f"{len(discharged)} higher-tariff hour(s), returning to the initial battery level. "
        )
    else:
        text += "Keeps the battery idle because shifting energy does not lower cost. "
    text += (
        f"Grid import {result['total_grid_kwh']:g} kWh, cost {result['total_cost_bdt']:g} BDT, "
        f"peak {result['peak_grid_kwh']:g} kWh."
    )
    return text
