"""Rule-based note reader -- OUTAGE FALLBACK ONLY.

The LLM is the interpreter. This module runs only when the model provider is
unreachable, returns unusable output, or a single note's reading fails the
guardrails. It emits the same raw reading shape as the LLM so it passes
through exactly the same guardrails in app/guardrails.py.

It is deliberately conservative: when it cannot find both a directive and a
time window with confidence, it says no_op rather than inventing a constraint.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

FRACTION_WORDS = [
    (r"\bthree[- ]quarters?\b", 0.75),
    (r"\bone[- ]quarter\b|\ba quarter\b|\bquarter of\b", 0.25),
    (r"\btwo[- ]thirds?\b", 2 / 3),
    (r"\bone[- ]third\b|\ba third\b", 1 / 3),
    (r"\bone[- ]fifth\b|\ba fifth\b", 0.2),
    (r"\bone[- ]tenth\b|\ba tenth\b", 0.1),
    (r"\bhalf\b", 0.5),
]

_MERIDIEM = r"(a\.?m\.?|p\.?m\.?)"
_NUM = r"(\d{1,2}(?::\d{2})?|" + "|".join(NUMBER_WORDS) + r")"
TIME_TOKEN = re.compile(
    r"\b(noon|midday|midnight|" + _NUM + r"(?:\s*o'?clock)?\s*" + _MERIDIEM + r"?)(?![\w%])",
    re.IGNORECASE,
)
RANGE_SEP = re.compile(r"^\s*(?:-|–|—|to|until|till|til|through|thru|and)\s*$", re.IGNORECASE)

NEGATION = r"(not|no|never|disabled?|unavailable|isolated|offline|off-line|blocked|prohibited|forbidden|suspended|locked out|out of service|cannot|can't|must not|mustn't|avoid|halt|pause|stop)"


def _parse_time(tok: str) -> Tuple[Optional[int], Optional[str], bool]:
    """-> (hour value, meridiem 'am'/'pm'/None, is_24h_explicit)."""
    t = tok.lower().strip()
    if t in ("noon", "midday"):
        return 12, "pm", True
    if t == "midnight":
        return 0, "am", True
    m = re.match(_NUM + r"(?:\s*o'?clock)?\s*" + _MERIDIEM + "?$", t)
    if not m:
        return None, None, False
    num, mer = m.group(1), m.group(2)
    explicit24 = False
    if ":" in num:
        num = num.split(":")[0]
        explicit24 = mer is None
    value = int(num) if num.isdigit() else NUMBER_WORDS[num]
    if mer:
        mer = "pm" if mer.startswith("p") else "am"
    if value > 12:
        explicit24 = True
    return value, mer, explicit24


def _to_24(value: int, mer: Optional[str]) -> int:
    if mer == "am":
        return 0 if value == 12 else value
    if mer == "pm":
        return 12 if value == 12 else value + 12
    return value


def find_window(note: str, solar: bool) -> Optional[Tuple[int, int]]:
    tokens = [m for m in TIME_TOKEN.finditer(note)]
    for a, b in zip(tokens, tokens[1:]):
        between = note[a.end(): b.start()]
        if not RANGE_SEP.match(between):
            continue
        # "from 1 PM to 3 PM", "13:00-15:00", "noon until 2 PM"
        v1, m1, e1 = _parse_time(a.group(1))
        v2, m2, e2 = _parse_time(b.group(1))
        if v1 is None or v2 is None:
            continue
        is_midnight_end = b.group(1).lower() == "midnight"
        if m1 is None and not e1 and m2:
            # shared meridiem: "1-3 PM"; "11 to 2 PM" means 11 AM.
            cand = _to_24(v1, m2)
            m1 = m2 if cand < _to_24(v2, m2) else ("am" if m2 == "pm" else "pm")
        if m2 is None and not e2 and m1:
            m2 = m1 if _to_24(v2, m1) > _to_24(v1, m1) else ("pm" if m1 == "am" else "am")
        if m1 is None and m2 is None and not (e1 or e2):
            # No clock hint at all: solar work is daytime; otherwise small
            # numbers before 7 are assumed afternoon/evening.
            m1 = m2 = "pm" if (v1 < 7 or (solar and v1 < 12 and v2 <= 6)) else "am"
            if solar and v1 >= 7 and v2 < v1:
                m1, m2 = "am", "pm"
        start = v1 if (e1 and m1 is None) else _to_24(v1, m1)
        end = v2 if (e2 and m2 is None) else _to_24(v2, m2)
        if is_midnight_end or (end == 0 and start > 0):
            end = 24
        if start == end or not (0 <= start <= 23 and 0 <= end <= 24):
            continue
        return start, end
    # "all day", "for the whole day"
    if re.search(r"\b(all day|entire day|whole day|full day|all 24 hours|around the clock)\b", note, re.I):
        return 0, 24
    return None


def _percent(note: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|per cent)", note, re.I)
    return float(m.group(1)) if m else None


def _kwh(note: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kwh|kilowatt[- ]hours?)", note, re.I)
    return float(m.group(1)) if m else None


def _solar_quantity(note: str) -> Optional[Tuple[float, str]]:
    low = note.lower()
    pct = _percent(note)
    if pct is not None:
        reduction = re.search(
            r"(\d+(?:\.\d+)?)\s*(%|percent|per cent)\s*(reduction|less|lower|decrease|drop|cut|loss|curtail)"
            r"|(reduc\w*|cut|lower\w*|decreas\w*|drop\w*|curtail\w*)\s+(by|of)\s+(about\s+|roughly\s+|around\s+)?\d",
            low,
        )
        return (pct, "percent_reduction") if reduction else (pct, "percent_remaining")
    for pattern, frac in FRACTION_WORDS:
        if re.search(pattern, low):
            lost = re.search(r"(lose|losing|lost|reduc\w*|cut)\s+(by\s+)?(about\s+|roughly\s+)?" + pattern.split("|")[0].strip(r"\b"), low)
            return (frac, "fraction_reduction") if lost else (frac, "fraction_remaining")
    if re.search(r"\b(no solar|zero|unavailable|offline|shut ?down|disconnected|nothing)\b", low):
        return 0.0, "fraction_remaining"
    return None


def interpret_note(note: str) -> Dict[str, Any]:
    low = note.lower()
    solar = bool(re.search(r"\b(solar|pv|photovoltaic|panels?|rooftop array)\b", low))
    window = find_window(note, solar)
    reading: Dict[str, Any] = {"directive_type": "no_op", "explanation": "Fallback reader: no supported directive recognised."}
    if window is None:
        return reading
    windows = [{"start_hour": window[0], "end_hour": window[1]}]

    def make(dtype: str, quantity=None, kind="none", why="") -> Dict[str, Any]:
        return {
            "directive_type": dtype,
            "windows": windows,
            "quantity": quantity,
            "quantity_kind": kind,
            "explanation": f"Fallback reader: {why}",
        }

    kwh = _kwh(note)
    pct = _percent(note)

    if re.search(r"\b(grid|import|feeder|transformer|intake|substation|utility supply|draw)\b", low) and kwh is not None:
        return make("max_grid_window", kwh, "kwh", "grid import is capped.")
    if re.search(r"\bdischarg", low) and re.search(NEGATION, low):
        return make("no_discharge_window", why="battery discharge is unavailable.")
    if re.search(r"\b(charg(e|er|ing)|recharg\w*)\b", low) and re.search(NEGATION, low):
        return make("no_charge_window", why="battery charging is unavailable.")
    if re.search(r"\b(battery|storage|reserve|stored|state of charge|soc)\b", low) and re.search(
        r"\b(at least|minimum|reserve|keep|maintain|remain|retain|hold|no less than|not below|not fall below)\b", low
    ):
        if kwh is not None:
            return make("minimum_battery_reserve", kwh, "kwh", "battery reserve is raised.")
        if pct is not None:
            return make("minimum_battery_reserve", pct, "percent_of_capacity", "battery reserve is raised.")
        if re.search(r"\bhalf\b", low):
            return make("minimum_battery_reserve", 0.5, "fraction_of_capacity", "battery reserve is raised.")
    if solar:
        q = _solar_quantity(note)
        if q is not None:
            return make("solar_reduction", q[0], q[1], "usable solar is reduced.")
    return reading


def interpret_notes(notes: List[str]) -> List[Dict[str, Any]]:
    return [dict(interpret_note(n), note_index=i) for i, n in enumerate(notes)]
