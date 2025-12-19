"""Cycle life testing protocol."""

from dataclasses import dataclass, field

from battery_simulator.protocols.base_protocol import (
    BaseProtocol,
    ProtocolStep,
    ChargeStep,
    DischargeStep,
    RestStep,
)


@dataclass
class CycleLifeProtocol(BaseProtocol):
    """
    Cycle life testing protocol.
    
    Standard cycling to evaluate battery degradation over repeated
    charge/discharge cycles. Typically continues until capacity
    drops to 80% of initial (End of Life).
    """

    name: str = field(default="Cycle Life", init=False)
    cycles: int = 1000
    charge_rate: float = 1.0  # C-rate
    discharge_rate: float = 1.0  # C-rate
    voltage_max: float = 4.2
    voltage_min: float = 3.0
    rest_time: float = 300.0  # 5 minutes
    end_capacity_retention: float = 0.80  # Stop at 80% retention
    cv_cutoff_rate: float = 0.05  # CV termination at C/20

    def _build_steps(self) -> list[ProtocolStep]:
        """Build cycle life test steps."""
        steps = []

        # CC-CV charge
        steps.append(
            ChargeStep(
                current_rate=self.charge_rate,
                voltage_cutoff=self.voltage_max,
                current_cutoff=self.cv_cutoff_rate,
                mode="CC-CV",
            )
        )

        # Rest after charge
        steps.append(RestStep(duration=self.rest_time))

        # CC discharge
        steps.append(
            DischargeStep(
                current_rate=self.discharge_rate,
                voltage_cutoff=self.voltage_min,
            )
        )

        # Rest after discharge
        steps.append(RestStep(duration=self.rest_time))

        return steps

    @classmethod
    def standard_1c(
        cls,
        cycles: int = 1000,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
    ) -> "CycleLifeProtocol":
        """Create standard 1C/1C cycle life protocol."""
        return cls(
            cycles=cycles,
            charge_rate=1.0,
            discharge_rate=1.0,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
        )

    @classmethod
    def accelerated(
        cls,
        cycles: int = 500,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
    ) -> "CycleLifeProtocol":
        """Create accelerated aging protocol (higher rates)."""
        return cls(
            cycles=cycles,
            charge_rate=2.0,
            discharge_rate=2.0,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
            rest_time=60.0,  # Shorter rest
        )

    @classmethod
    def conservative(
        cls,
        cycles: int = 2000,
        voltage_max: float = 4.1,  # Lower max voltage
        voltage_min: float = 3.0,
    ) -> "CycleLifeProtocol":
        """Create conservative protocol (lower stress)."""
        return cls(
            cycles=cycles,
            charge_rate=0.5,
            discharge_rate=0.5,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
            rest_time=600.0,  # Longer rest
        )

