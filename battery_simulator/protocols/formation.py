"""Formation cycling protocol."""

from dataclasses import dataclass, field

from battery_simulator.protocols.base_protocol import (
    BaseProtocol,
    ProtocolStep,
    ChargeStep,
    DischargeStep,
    RestStep,
)


@dataclass
class FormationProtocol(BaseProtocol):
    """
    Formation cycling protocol for new cells.
    
    Formation cycles are low-rate cycles performed on new cells to:
    - Form the SEI (Solid Electrolyte Interface) layer
    - Stabilize initial capacity
    - Identify defective cells
    
    Typically uses low C-rates (0.1C - 0.2C) with gradual increases.
    """

    name: str = field(default="Formation", init=False)
    cycles: int = 3
    initial_rate: float = 0.1  # Starting C-rate
    voltage_max: float = 4.2
    voltage_min: float = 3.0
    rest_time: float = 1800.0  # 30 minutes

    def _build_steps(self) -> list[ProtocolStep]:
        """Build formation cycle steps."""
        steps = []

        # Low rate CC-CV charge
        steps.append(
            ChargeStep(
                current_rate=self.initial_rate,
                voltage_cutoff=self.voltage_max,
                current_cutoff=self.initial_rate / 2,
                mode="CC-CV",
            )
        )

        # Rest period
        steps.append(RestStep(duration=self.rest_time))

        # Low rate discharge
        steps.append(
            DischargeStep(
                current_rate=self.initial_rate,
                voltage_cutoff=self.voltage_min,
            )
        )

        # Rest period
        steps.append(RestStep(duration=self.rest_time))

        return steps

    @classmethod
    def standard_3_cycle(
        cls,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
    ) -> "FormationProtocol":
        """Create standard 3-cycle formation protocol."""
        return cls(
            cycles=3,
            initial_rate=0.1,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
            rest_time=1800.0,
        )

    @classmethod
    def fast_formation(
        cls,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
    ) -> "FormationProtocol":
        """Create faster formation protocol (higher rate)."""
        return cls(
            cycles=2,
            initial_rate=0.2,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
            rest_time=600.0,  # 10 minutes
        )

