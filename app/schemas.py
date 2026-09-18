"""Request models for POST /optimize-energy.

Structural problems (wrong types, missing fields, not exactly 24 hours) are
reported as HTTP 400. Well-formed but physically inconsistent batteries are
reported as HTTP 422 by `semantic_errors`.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonNegative = Field(ge=0, allow_inf_nan=False)


class HourEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hour: int = Field(ge=0, le=23)
    demand_kwh: float = NonNegative
    solar_kwh: float = NonNegative
    tariff_bdt_per_kwh: float = NonNegative


class Battery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: float = NonNegative
    initial_energy_kwh: float = NonNegative
    minimum_energy_kwh: float = NonNegative
    max_charge_kwh_per_hour: float = NonNegative
    max_discharge_kwh_per_hour: float = NonNegative


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourEntry] = Field(min_length=24, max_length=24)
    battery: Battery

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, notes: List[str]) -> List[str]:
        if any(not n.strip() for n in notes):
            raise ValueError("operator_notes must contain non-empty strings")
        return notes

    @model_validator(mode="after")
    def _hours_cover_day(self) -> "ScenarioRequest":
        if sorted(h.hour for h in self.hours) != list(range(24)):
            raise ValueError("hours must contain each hour 0..23 exactly once")
        self.hours = sorted(self.hours, key=lambda h: h.hour)
        return self


def semantic_errors(req: ScenarioRequest) -> List[str]:
    b = req.battery
    errors = []
    if b.minimum_energy_kwh > b.capacity_kwh:
        errors.append("battery.minimum_energy_kwh exceeds capacity_kwh")
    if not (b.minimum_energy_kwh <= b.initial_energy_kwh <= b.capacity_kwh):
        errors.append("battery.initial_energy_kwh must lie between minimum_energy_kwh and capacity_kwh")
    return errors
