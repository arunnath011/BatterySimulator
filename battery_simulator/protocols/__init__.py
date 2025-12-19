"""Test protocol definitions for battery cycling."""

from __future__ import annotations

from typing import Optional, List

from battery_simulator.protocols.base_protocol import (
    BaseProtocol,
    ProtocolStep,
    ChargeStep,
    DischargeStep,
    RestStep,
)
from battery_simulator.protocols.formation import FormationProtocol
from battery_simulator.protocols.cycle_life import CycleLifeProtocol
from battery_simulator.protocols.rate_capability import RateCapabilityProtocol
from battery_simulator.protocols.calendar_aging import CalendarAgingProtocol
from battery_simulator.protocols.rpt import RPTProtocol


class Protocol:
    """Factory for creating test protocols."""

    @staticmethod
    def formation(
        cycles: int = 3,
        initial_rate: float = 0.1,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
        rest_time: float = 1800.0,
    ) -> FormationProtocol:
        """Create formation cycling protocol."""
        return FormationProtocol(
            cycles=cycles,
            initial_rate=initial_rate,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
            rest_time=rest_time,
        )

    @staticmethod
    def cycle_life(
        charge_rate: float = 1.0,
        discharge_rate: float = 1.0,
        cycles: int = 1000,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
        rest_time: float = 300.0,
        end_capacity_retention: float = 0.80,
    ) -> CycleLifeProtocol:
        """Create cycle life test protocol."""
        return CycleLifeProtocol(
            charge_rate=charge_rate,
            discharge_rate=discharge_rate,
            cycles=cycles,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
            rest_time=rest_time,
            end_capacity_retention=end_capacity_retention,
        )

    @staticmethod
    def rate_capability(
        rates: Optional[List[float]] = None,
        cycles_per_rate: int = 3,
        charge_rate: float = 0.5,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
    ) -> RateCapabilityProtocol:
        """Create rate capability test protocol."""
        return RateCapabilityProtocol(
            rates=rates or [0.2, 0.5, 1.0, 2.0, 3.0, 5.0],
            cycles_per_rate=cycles_per_rate,
            charge_rate=charge_rate,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
        )

    @staticmethod
    def calendar_aging(
        target_soc: float = 0.5,
        storage_days: int = 90,
        temperature: float = 25.0,
        checkup_interval_days: int = 30,
    ) -> CalendarAgingProtocol:
        """Create calendar aging test protocol."""
        return CalendarAgingProtocol(
            target_soc=target_soc,
            storage_days=storage_days,
            temperature=temperature,
            checkup_interval_days=checkup_interval_days,
        )

    @staticmethod
    def rpt(
        charge_rate: float = 0.33,
        discharge_rate: float = 0.33,
        pulse_soc_points: Optional[List[float]] = None,
        pulse_current: float = 1.0,
        pulse_duration: float = 10.0,
    ) -> RPTProtocol:
        """Create Reference Performance Test protocol."""
        return RPTProtocol(
            charge_rate=charge_rate,
            discharge_rate=discharge_rate,
            pulse_soc_points=pulse_soc_points or [0.2, 0.5, 0.8],
            pulse_current=pulse_current,
            pulse_duration=pulse_duration,
        )


__all__ = [
    "Protocol",
    "BaseProtocol",
    "ProtocolStep",
    "ChargeStep",
    "DischargeStep",
    "RestStep",
    "FormationProtocol",
    "CycleLifeProtocol",
    "RateCapabilityProtocol",
    "CalendarAgingProtocol",
    "RPTProtocol",
]

