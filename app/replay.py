"""Independent replay of a finished plan -- the same checks the judge runs.

Deliberately does not reuse the optimizer's internal state: it rebuilds
effective solar and per-hour limits from the scenario and the directives and
walks the plan hour by hour.
"""

from __future__ import annotations

from typing import Any, Dict, List

TOL = 0.005  # judge tolerance is 0.01; stay well inside it


def replay(
    hours: List[Dict[str, float]],
    battery: Dict[str, float],
    directives: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> List[str]:
    """Return a list of violations; empty means the plan is valid."""
    v: List[str] = []
    plan = result["hourly_plan"]
    if [p["hour"] for p in plan] != list(range(24)):
        return ["hourly_plan must list hours 0..23 in order"]

    eff_solar = [float(h["solar_kwh"]) for h in hours]
    floor = [float(battery["minimum_energy_kwh"])] * 24
    no_charge, no_discharge = set(), set()
    grid_cap: Dict[int, float] = {}
    for d in directives:
        adj = d["structured_adjustment"]
        t = d["directive_type"]
        for h in adj["hours"]:
            if t == "solar_reduction":
                eff_solar[h] = float(hours[h]["solar_kwh"]) * adj["factor"]
            elif t == "minimum_battery_reserve":
                floor[h] = max(floor[h], adj["minimum_energy_kwh"])
            elif t == "no_charge_window":
                no_charge.add(h)
            elif t == "no_discharge_window":
                no_discharge.add(h)
            elif t == "max_grid_window":
                grid_cap[h] = min(grid_cap.get(h, float("inf")), adj["max_grid_kwh"])

    cap = float(battery["capacity_kwh"])
    energy = float(battery["initial_energy_kwh"])
    for p, h in zip(plan, hours):
        i = p["hour"]
        g, s, kwh, act = p["grid_kwh"], p["solar_used_kwh"], p["battery_kwh"], p["battery_action"]
        if min(g, s, kwh, p["battery_energy_after_kwh"]) < -TOL:
            v.append(f"h{i}: negative value")
        if act not in ("charge", "discharge", "idle"):
            v.append(f"h{i}: bad battery_action")
            continue
        if act == "idle" and abs(kwh) > TOL:
            v.append(f"h{i}: idle with non-zero battery_kwh")
        ch = kwh if act == "charge" else 0.0
        dis = kwh if act == "discharge" else 0.0
        if s > eff_solar[i] + TOL:
            v.append(f"h{i}: solar_used exceeds effective solar")
        if abs(g + s + dis - float(h["demand_kwh"]) - ch) > TOL:
            v.append(f"h{i}: energy balance broken")
        if ch > float(battery["max_charge_kwh_per_hour"]) + TOL:
            v.append(f"h{i}: charge rate exceeded")
        if dis > float(battery["max_discharge_kwh_per_hour"]) + TOL:
            v.append(f"h{i}: discharge rate exceeded")
        if i in no_charge and ch > TOL:
            v.append(f"h{i}: charging inside no_charge_window")
        if i in no_discharge and dis > TOL:
            v.append(f"h{i}: discharging inside no_discharge_window")
        if i in grid_cap and g > grid_cap[i] + TOL:
            v.append(f"h{i}: grid import above max_grid_window cap")
        energy += ch - dis
        if abs(energy - p["battery_energy_after_kwh"]) > TOL:
            v.append(f"h{i}: battery_energy_after inconsistent with action")
        if energy < floor[i] - TOL or energy > cap + TOL:
            v.append(f"h{i}: battery energy outside [{floor[i]}, {cap}]")

    if abs(plan[-1]["battery_energy_after_kwh"] - float(battery["initial_energy_kwh"])) > TOL:
        v.append("end-of-day battery energy differs from initial")

    tariffs = [float(h["tariff_bdt_per_kwh"]) for h in hours]
    if abs(result["total_grid_kwh"] - sum(p["grid_kwh"] for p in plan)) > TOL:
        v.append("total_grid_kwh mismatch")
    if abs(result["total_cost_bdt"] - sum(p["grid_kwh"] * tariffs[p["hour"]] for p in plan)) > TOL:
        v.append("total_cost_bdt mismatch")
    if abs(result["peak_grid_kwh"] - max(p["grid_kwh"] for p in plan)) > TOL:
        v.append("peak_grid_kwh mismatch")
    return v
