"""Reference Performance Test (RPT) protocol."""

from dataclasses import dataclass, field

from battery_simulator.protocols.base_protocol import (
    BaseProtocol,
    ProtocolStep,
    ChargeStep,
    DischargeStep,
    RestStep,
    PulseStep,
)


@dataclass
class RPTProtocol(BaseProtocol):
    """
    Reference Performance Test (RPT) protocol.
    
    Comprehensive characterization test including:
    - Capacity test at low rate
    - DC resistance measurement via current pulses
    - Rate capability at multiple rates
    
    Typically performed periodically during cycle life testing
    to track degradation.
    """

    name: str = field(default="Reference Performance Test", init=False)
    cycles: int = 1  # RPT is typically a single comprehensive test
    charge_rate: float = 0.33  # C/3 for capacity test
    discharge_rate: float = 0.33
    pulse_soc_points: list[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    pulse_current: float = 1.0  # C-rate for resistance pulses
    pulse_duration: float = 10.0  # seconds
    rate_test_rates: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    voltage_max: float = 4.2
    voltage_min: float = 3.0

    def __post_init__(self):
        self.name = "Reference Performance Test (RPT)"
        self.steps = self._build_steps()

    def _build_steps(self) -> list[ProtocolStep]:
        """Build comprehensive RPT steps."""
        steps = []

        # ==========================================
        # Part 1: Capacity Test (C/3)
        # ==========================================

        # Full charge
        steps.append(
            ChargeStep(
                current_rate=self.charge_rate,
                voltage_cutoff=self.voltage_max,
                current_cutoff=0.02,  # Lower cutoff for accurate capacity
                mode="CC-CV",
            )
        )
        steps.append(RestStep(duration=3600.0))  # 1 hour rest

        # Full discharge for capacity
        steps.append(
            DischargeStep(
                current_rate=self.discharge_rate,
                voltage_cutoff=self.voltage_min,
            )
        )
        steps.append(RestStep(duration=3600.0))

        # ==========================================
        # Part 2: DC Resistance Test (Pulse)
        # ==========================================

        # Charge back to 100%
        steps.append(
            ChargeStep(
                current_rate=0.5,
                voltage_cutoff=self.voltage_max,
                current_cutoff=0.05,
                mode="CC-CV",
            )
        )
        steps.append(RestStep(duration=1800.0))

        # Pulse tests at different SOC points (discharge direction)
        for soc in sorted(self.pulse_soc_points, reverse=True):
            # Discharge to target SOC
            target_voltage = self._soc_to_voltage(soc)
            steps.append(
                DischargeStep(
                    current_rate=0.33,
                    voltage_cutoff=target_voltage,
                )
            )
            steps.append(RestStep(duration=600.0))  # 10 min rest

            # Discharge pulse
            steps.append(
                PulseStep(
                    current_rate=self.pulse_current,
                    duration=self.pulse_duration,
                    is_discharge=True,
                )
            )
            steps.append(RestStep(duration=60.0))

            # Charge pulse
            steps.append(
                PulseStep(
                    current_rate=self.pulse_current,
                    duration=self.pulse_duration,
                    is_discharge=False,
                )
            )
            steps.append(RestStep(duration=300.0))

        # ==========================================
        # Part 3: Rate Capability (optional)
        # ==========================================

        for rate in self.rate_test_rates:
            # Full charge
            steps.append(
                ChargeStep(
                    current_rate=0.5,
                    voltage_cutoff=self.voltage_max,
                    current_cutoff=0.05,
                    mode="CC-CV",
                )
            )
            steps.append(RestStep(duration=300.0))

            # Discharge at test rate
            steps.append(
                DischargeStep(
                    current_rate=rate,
                    voltage_cutoff=self.voltage_min,
                )
            )
            steps.append(RestStep(duration=300.0))

        return steps

    def _soc_to_voltage(self, soc: float) -> float:
        """Estimate voltage for target SOC."""
        # Rough linear approximation
        return self.voltage_min + soc * (self.voltage_max - self.voltage_min) * 0.85

    @classmethod
    def standard(
        cls,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
    ) -> "RPTProtocol":
        """Create standard RPT protocol."""
        return cls(
            charge_rate=0.33,
            discharge_rate=0.33,
            pulse_soc_points=[0.2, 0.5, 0.8],
            pulse_current=1.0,
            pulse_duration=10.0,
            rate_test_rates=[0.5, 1.0, 2.0],
            voltage_max=voltage_max,
            voltage_min=voltage_min,
        )

    @classmethod
    def quick(
        cls,
        voltage_max: float = 4.2,
        voltage_min: float = 3.0,
    ) -> "RPTProtocol":
        """Create quick RPT (capacity test only)."""
        return cls(
            charge_rate=0.5,
            discharge_rate=0.5,
            pulse_soc_points=[0.5],  # Single SOC point
            pulse_current=1.0,
            pulse_duration=10.0,
            rate_test_rates=[],  # Skip rate test
            voltage_max=voltage_max,
            voltage_min=voltage_min,
        )

