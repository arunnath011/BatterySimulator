"""Calendar aging test protocol."""

from dataclasses import dataclass, field

from battery_simulator.protocols.base_protocol import (
    BaseProtocol,
    ProtocolStep,
    ChargeStep,
    DischargeStep,
    RestStep,
)


@dataclass
class CalendarAgingProtocol(BaseProtocol):
    """
    Calendar aging test protocol.
    
    Measures capacity loss during storage (non-cycling conditions).
    Cell is charged to target SOC, stored for a period, then
    capacity is measured periodically.
    """

    name: str = field(default="Calendar Aging", init=False)
    target_soc: float = 0.5  # Storage SOC
    storage_days: int = 90
    temperature: float = 25.0  # °C
    checkup_interval_days: int = 30
    voltage_max: float = 4.2
    voltage_min: float = 3.0

    def __post_init__(self):
        # Calculate number of checkup cycles
        self.cycles = self.storage_days // self.checkup_interval_days
        self.name = f"Calendar Aging ({self.storage_days}d @ {self.target_soc*100:.0f}% SOC)"
        self.steps = self._build_steps()

    def _build_steps(self) -> list[ProtocolStep]:
        """Build calendar aging test steps."""
        steps = []

        # Initial full charge
        steps.append(
            ChargeStep(
                current_rate=0.5,
                voltage_cutoff=self.voltage_max,
                current_cutoff=0.05,
                mode="CC-CV",
            )
        )

        # Rest
        steps.append(RestStep(duration=300.0))

        # Discharge to target SOC
        # Calculate discharge to reach target SOC
        # We'll discharge at low rate and stop at target SOC
        steps.append(
            DischargeStep(
                current_rate=0.33,
                voltage_cutoff=self._soc_to_voltage(self.target_soc),
            )
        )

        # Long storage rest (simulated with shorter actual rest)
        # The degradation model handles calendar aging
        storage_seconds = self.checkup_interval_days * 24 * 3600
        # For simulation, we use shorter rest but apply calendar degradation
        steps.append(RestStep(duration=min(storage_seconds, 3600.0)))

        # Checkup: Full charge to measure capacity
        steps.append(
            ChargeStep(
                current_rate=0.33,
                voltage_cutoff=self.voltage_max,
                current_cutoff=0.05,
                mode="CC-CV",
            )
        )

        # Rest
        steps.append(RestStep(duration=300.0))

        # Capacity test discharge
        steps.append(
            DischargeStep(
                current_rate=0.33,
                voltage_cutoff=self.voltage_min,
            )
        )

        # Rest before next storage period
        steps.append(RestStep(duration=300.0))

        return steps

    def _soc_to_voltage(self, soc: float) -> float:
        """Estimate voltage for target SOC (rough approximation)."""
        # Linear approximation between min and max voltage
        return self.voltage_min + soc * (self.voltage_max - self.voltage_min) * 0.9

    @classmethod
    def standard_50_soc(
        cls,
        storage_days: int = 90,
        temperature: float = 25.0,
    ) -> "CalendarAgingProtocol":
        """Create standard 50% SOC calendar aging test."""
        return cls(
            target_soc=0.5,
            storage_days=storage_days,
            temperature=temperature,
            checkup_interval_days=30,
        )

    @classmethod
    def high_soc_stress(
        cls,
        storage_days: int = 90,
        temperature: float = 45.0,
    ) -> "CalendarAgingProtocol":
        """Create high-stress calendar aging test (high SOC, high temp)."""
        return cls(
            target_soc=0.8,
            storage_days=storage_days,
            temperature=temperature,
            checkup_interval_days=14,
        )

