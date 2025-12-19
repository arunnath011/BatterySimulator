"""Rate capability testing protocol."""

from dataclasses import dataclass, field

from battery_simulator.protocols.base_protocol import (
    BaseProtocol,
    ProtocolStep,
    ChargeStep,
    DischargeStep,
    RestStep,
)


@dataclass
class RateCapabilityProtocol(BaseProtocol):
    """
    Rate capability testing protocol.
    
    Evaluates battery performance at different discharge rates.
    Typically charges at a fixed rate and discharges at varying rates
    to measure capacity retention at high power.
    """

    name: str = field(default="Rate Capability", init=False)
    rates: list[float] = field(default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 3.0, 5.0])
    cycles_per_rate: int = 3
    charge_rate: float = 0.5  # Fixed charge rate
    voltage_max: float = 4.2
    voltage_min: float = 3.0
    rest_time: float = 300.0

    def __post_init__(self):
        # Calculate total cycles
        self.cycles = len(self.rates) * self.cycles_per_rate
        self.name = f"Rate Capability ({len(self.rates)} rates)"
        # Build steps for first rate (will be updated during simulation)
        self._current_rate_index = 0
        self._cycle_in_rate = 0
        self.steps = self._build_steps()

    def _build_steps(self) -> list[ProtocolStep]:
        """Build steps for current discharge rate."""
        # Get current discharge rate
        if self._current_rate_index < len(self.rates):
            discharge_rate = self.rates[self._current_rate_index]
        else:
            discharge_rate = self.rates[-1]

        steps = []

        # CC-CV charge at fixed rate
        steps.append(
            ChargeStep(
                current_rate=self.charge_rate,
                voltage_cutoff=self.voltage_max,
                current_cutoff=0.05,
                mode="CC-CV",
            )
        )

        # Rest
        steps.append(RestStep(duration=self.rest_time))

        # Discharge at test rate
        steps.append(
            DischargeStep(
                current_rate=discharge_rate,
                voltage_cutoff=self.voltage_min,
            )
        )

        # Rest
        steps.append(RestStep(duration=self.rest_time))

        return steps

    def advance_rate(self) -> bool:
        """
        Advance to next cycle/rate.
        
        Returns:
            True if advanced to new rate, False if completed
        """
        self._cycle_in_rate += 1

        if self._cycle_in_rate >= self.cycles_per_rate:
            self._cycle_in_rate = 0
            self._current_rate_index += 1

            if self._current_rate_index >= len(self.rates):
                return False

            # Rebuild steps for new rate
            self.steps = self._build_steps()
            return True

        return True

    def get_current_rate(self) -> float:
        """Get the current discharge rate being tested."""
        if self._current_rate_index < len(self.rates):
            return self.rates[self._current_rate_index]
        return self.rates[-1]

    @classmethod
    def standard(
        cls,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
    ) -> "RateCapabilityProtocol":
        """Create standard rate capability test."""
        return cls(
            rates=[0.2, 0.5, 1.0, 2.0, 3.0, 5.0],
            cycles_per_rate=3,
            charge_rate=0.5,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
        )

    @classmethod
    def high_power(
        cls,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
    ) -> "RateCapabilityProtocol":
        """Create high-power rate capability test."""
        return cls(
            rates=[1.0, 2.0, 5.0, 10.0, 15.0, 20.0],
            cycles_per_rate=3,
            charge_rate=1.0,
            voltage_max=voltage_max,
            voltage_min=voltage_min,
        )

