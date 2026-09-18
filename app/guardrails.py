"""Deterministic guardrails between the LLM and the optimizer.

The LLM reports only what it read in a note: the directive type, the time
window(s) as clock hours, and the raw quantity together with what kind of
quantity it is ("80% reduction", "50% of capacity", "155 kWh"). This module
does all the arithmetic -- expanding windows into hour lists and converting
quantities into the final factor / kWh -- and rejects anything that does not
fit the Problem Statement contract. Nothing unvalidated reaches the optimizer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

DIRECTIVE_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)

QUANTITY_KINDS = (
    "percent_remaining",
    "fraction_remaining",
    "percent_reduction",
    "fraction_reduction",
    "kwh",
    "percent_of_capacity",
    "fraction_of_capacity",
    "none",
)

DEFAULT_EXPLANATIONS = {
    "solar_reduction": "Usable solar is reduced to the stated fraction during the stated hours.",
    "minimum_battery_reserve": "Battery energy must stay at or above the stated reserve during the stated hours.",
    "no_charge_window": "Battery charging is unavailable during the stated hours.",
    "no_discharge_window": "Battery discharging is unavailable during the stated hours.",
    "max_grid_window": "Grid import is capped at the stated amount in each of the stated hours.",
    "no_op": "This note does not affect today's 24-hour energy schedule.",
}


class GuardrailError(ValueError):
    """The model output for one note cannot be turned into a valid directive."""


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool):
        raise GuardrailError(f"{what} is not a number")
    if isinstance(value, str):
        value = value.strip().rstrip("%").strip()
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise GuardrailError(f"{what} is not a number") from None
    if not math.isfinite(x):
        raise GuardrailError(f"{what} is not finite")
    return x


def _clock_hour(value: Any, what: str, allow_24: bool) -> int:
    x = _number(value, what)
    if x != int(x):
        raise GuardrailError(f"{what} is not a whole hour")
    h = int(x)
    if not (0 <= h <= (24 if allow_24 else 23)):
        raise GuardrailError(f"{what} out of range")
    return h


def expand_windows(windows: Any) -> List[int]:
    """[{start_hour, end_hour}, ...] -> sorted unique hour list.

    Start is inclusive, end is exclusive. end_hour 24 (or 0 after a later
    start) means midnight at the end of the day. A window whose end precedes
    its start wraps past midnight.
    """
    if not isinstance(windows, list) or not windows:
        raise GuardrailError("at least one time window is required")
    hours = set()
    for w in windows:
        if not isinstance(w, dict):
            raise GuardrailError("time window must be an object")
        start = _clock_hour(w.get("start_hour"), "start_hour", allow_24=False)
        end = _clock_hour(w.get("end_hour"), "end_hour", allow_24=True)
        if end == start:
            raise GuardrailError("empty time window")
        if end > start:
            hours.update(range(start, end))
        else:  # wraps past midnight, e.g. 22 -> 2
            hours.update(range(start, 24))
            hours.update(range(0, end))
    return sorted(hours)


def _factor(quantity: Any, kind: str) -> float:
    q = _number(quantity, "quantity")
    if kind == "percent_remaining":
        f = q / 100.0
    elif kind == "fraction_remaining":
        f = q
    elif kind == "percent_reduction":
        f = 1.0 - q / 100.0
    elif kind == "fraction_reduction":
        f = 1.0 - q
    else:
        raise GuardrailError(f"quantity kind {kind!r} is not valid for solar_reduction")
    f = round(f, 6)
    if not (0.0 <= f <= 1.0):
        raise GuardrailError("solar factor must be between 0 and 1")
    return f


def _reserve(quantity: Any, kind: str, capacity: float) -> float:
    q = _number(quantity, "quantity")
    if kind == "kwh":
        kwh = q
    elif kind == "percent_of_capacity":
        kwh = capacity * q / 100.0
    elif kind == "fraction_of_capacity":
        kwh = capacity * q
    else:
        raise GuardrailError(f"quantity kind {kind!r} is not valid for minimum_battery_reserve")
    kwh = round(kwh, 6)
    if not (0.0 <= kwh <= capacity + 1e-9):
        raise GuardrailError("reserve must be non-negative and not exceed battery capacity")
    return kwh


def _grid_cap(quantity: Any, kind: str) -> float:
    if kind != "kwh":
        raise GuardrailError(f"quantity kind {kind!r} is not valid for max_grid_window")
    q = round(_number(quantity, "quantity"), 6)
    if q < 0:
        raise GuardrailError("max_grid_kwh must be non-negative")
    return q


def _tidy(x: float) -> float | int:
    """Print 100.0 as 100 so responses read like the organizer samples."""
    return int(x) if float(x).is_integer() else x


def no_op(note_index: int, explanation: Optional[str] = None) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": explanation or DEFAULT_EXPLANATIONS["no_op"],
    }


def normalize(raw: Any, note_index: int, battery: Dict[str, float]) -> Dict[str, Any]:
    """Turn one raw model reading into a contract-exact interpretation entry.

    Raises GuardrailError when the reading is unusable; the caller decides the
    safe fallback. Never invents a directive type.
    """
    if not isinstance(raw, dict):
        raise GuardrailError("interpretation must be an object")

    dtype = raw.get("directive_type")
    if dtype not in DIRECTIVE_TYPES:
        raise GuardrailError(f"unsupported directive_type {dtype!r}")

    explanation = raw.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = DEFAULT_EXPLANATIONS[dtype]
    explanation = " ".join(explanation.split())[:300]

    if dtype == "no_op":
        return no_op(note_index, explanation)

    hours = expand_windows(raw.get("windows"))
    kind = str(raw.get("quantity_kind") or "none").strip().lower()
    quantity = raw.get("quantity")

    if dtype == "solar_reduction":
        adjustment = {"hours": hours, "factor": _tidy(_factor(quantity, kind))}
    elif dtype == "minimum_battery_reserve":
        capacity = float(battery["capacity_kwh"])
        adjustment = {"hours": hours, "minimum_energy_kwh": _tidy(_reserve(quantity, kind, capacity))}
    elif dtype == "max_grid_window":
        adjustment = {"hours": hours, "max_grid_kwh": _tidy(_grid_cap(quantity, kind))}
    else:  # no_charge_window / no_discharge_window
        adjustment = {"hours": hours}

    return {
        "note_index": note_index,
        "applies": True,
        "directive_type": dtype,
        "structured_adjustment": adjustment,
        "explanation": explanation,
    }


def validate_final(entries: List[Dict[str, Any]], n_notes: int, capacity: float) -> None:
    """Last contract check on the assembled directive_interpretation list."""
    if [e["note_index"] for e in entries] != list(range(n_notes)):
        raise GuardrailError("interpretations must cover every note once, in order")
    for e in entries:
        t, adj = e["directive_type"], e["structured_adjustment"]
        if t not in DIRECTIVE_TYPES:
            raise GuardrailError("unsupported directive_type")
        if t == "no_op":
            if e["applies"] is not False or adj is not None:
                raise GuardrailError("no_op must have applies=false and null adjustment")
            continue
        if e["applies"] is not True or not isinstance(adj, dict):
            raise GuardrailError("directives must have applies=true and an adjustment")
        hrs = adj.get("hours")
        if (
            not isinstance(hrs, list)
            or not hrs
            or any(not isinstance(h, int) or not 0 <= h <= 23 for h in hrs)
            or hrs != sorted(set(hrs))
        ):
            raise GuardrailError("hours must be unique ascending integers 0..23")
        if t == "solar_reduction" and not 0 <= adj["factor"] <= 1:
            raise GuardrailError("factor out of range")
        if t == "minimum_battery_reserve" and not 0 <= adj["minimum_energy_kwh"] <= capacity:
            raise GuardrailError("reserve out of range")
        if t == "max_grid_window" and not adj["max_grid_kwh"] >= 0:
            raise GuardrailError("grid cap out of range")
