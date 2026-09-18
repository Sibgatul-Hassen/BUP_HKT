"""Notes -> validated directive_interpretation list.

Order of trust, per note:
  1. LLM reading (cached by note text) -> guardrails.
  2. If the provider is down, or that note's reading fails the guardrails:
     the outage fallback reader -> the same guardrails.
  3. If that also fails: no_op. Never an invented constraint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from . import fallback_parser
from .guardrails import GuardrailError, no_op, normalize, validate_final
from .llm import LLMUnavailable, NoteInterpreter

log = logging.getLogger("gridwise.interpret")


async def interpret(
    llm: NoteInterpreter, notes: List[str], battery: Dict[str, float]
) -> Tuple[List[Dict[str, Any]], str]:
    """Returns (interpretations, source) where source is llm|cache|fallback|mixed."""
    source = "cache"
    readings = llm.cache_get(notes)
    if readings is None:
        try:
            readings = await llm.read_notes(notes)
            source = "llm"
        except LLMUnavailable as e:
            log.warning("LLM unavailable (%s); using outage fallback reader", e)
            readings = [None] * len(notes)
            source = "fallback"

    entries: List[Dict[str, Any]] = []
    all_llm_valid = source != "fallback"
    for i, note in enumerate(notes):
        entry = None
        raw = readings[i]
        if raw is not None:
            try:
                entry = normalize(raw, i, battery)
            except GuardrailError as e:
                log.warning("note %d: model reading rejected by guardrails (%s)", i, e)
        if entry is None:
            all_llm_valid = False
            try:
                entry = normalize(fallback_parser.interpret_note(note), i, battery)
            except GuardrailError as e:
                log.warning("note %d: fallback reading rejected (%s); no_op", i, e)
                entry = no_op(i)
            if source != "fallback":
                source = "mixed"
        entries.append(entry)

    validate_final(entries, len(notes), float(battery["capacity_kwh"]))
    if all_llm_valid and source == "llm":
        llm.cache_put(notes, readings)
    return entries, source
