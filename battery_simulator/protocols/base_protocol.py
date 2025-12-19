"""Base protocol and step classes for battery testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from battery_simulator.core.battery_model import BatteryModel


@dataclass
class ProtocolStep(ABC):
    """Abstract base class for protocol steps."""

    step_type: str = field(default="unknown", init=False)

    @abstractmethod
    def get_current(self, battery: BatteryModel) -> float:
        """
        Get the current to apply for this step.
        
        Args:
            battery: Battery model instance
            
        Returns:
            Current in A (positive = charge, negative = discharge)
        """
        pass

    @abstractmethod
    def is_complete(self, battery: BatteryModel, step_time: float) -> bool:
        """
        Check if the step is complete.
        
        Args:
            battery: Battery model instance
            step_time: Time elapsed in this step (seconds)
            
        Returns:
            True if step is complete
        """
        pass

    @property
    def is_charge(self) -> bool:
        """Check if this is a charge step."""
        step_lower = self.step_type.lower()
        return "charge" in step_lower and "discharge" not in step_lower

    @property
    def is_discharge(self) -> bool:
        """Check if this is a discharge step."""
        return "discharge" in self.step_type.lower()

    @property
    def is_rest(self) -> bool:
        """Check if this is a rest step."""
        return "rest" in self.step_type.lower()


@dataclass
class ChargeStep(ProtocolStep):
    """
    Constant current (CC) or CC-CV charge step.
    
    Modes:
    - CC: Charge at constant current until voltage cutoff
    - CC-CV: CC until voltage, then CV until current cutoff
    """

    current_rate: float = 1.0  # C-rate
    voltage_cutoff: float = 4.2  # V
    current_cutoff: float = 0.05  # C-rate for CV termination
    mode: str = "CC-CV"  # "CC" or "CC-CV"

    def __post_init__(self):
        self.step_type = "charge_cc_cv" if self.mode == "CC-CV" else "charge_cc"
        self._in_cv_mode = False
        self._cv_current = None

    def get_current(self, battery: BatteryModel) -> float:
        """Get charge current based on mode and state."""
        # Calculate C-rate current
        cc_current = self.current_rate * battery.capacity_nominal

        if self.mode == "CC":
            return cc_current

        # CC-CV mode
        if battery.state.voltage >= self.voltage_cutoff:
            self._in_cv_mode = True
            # CV mode: reduce current to maintain voltage
            # Simple proportional control
            v_error = self.voltage_cutoff - battery.state.voltage
            cv_adjustment = max(0.1, 1.0 + v_error * 10)  # Reduce current
            self._cv_current = cc_current * cv_adjustment
            return max(self._cv_current, self.current_cutoff * battery.capacity_nominal)

        return cc_current

    def is_complete(self, battery: BatteryModel, step_time: float) -> bool:
        """Check if charge is complete."""
        if self.mode == "CC":
            return battery.state.voltage >= self.voltage_cutoff

        # CC-CV mode: complete when current drops below cutoff
        if self._in_cv_mode:
            cv_cutoff_current = self.current_cutoff * battery.capacity_nominal
            if self._cv_current is not None and self._cv_current <= cv_cutoff_current:
                return True

        # Also complete if SOC reaches 1.0
        return battery.state.soc >= 0.9999


@dataclass
class DischargeStep(ProtocolStep):
    """Constant current discharge step."""

    current_rate: float = 1.0  # C-rate
    voltage_cutoff: float = 3.0  # V
    power_mode: bool = False  # If True, use constant power instead
    power_watts: float = 0.0  # For constant power mode

    def __post_init__(self):
        self.step_type = "discharge_cc" if not self.power_mode else "discharge_cp"

    def get_current(self, battery: BatteryModel) -> float:
        """Get discharge current (negative)."""
        if self.power_mode and self.power_watts > 0:
            # Constant power: I = P / V
            return -self.power_watts / battery.state.voltage

        # Constant current (negative for discharge)
        return -self.current_rate * battery.capacity_nominal

    def is_complete(self, battery: BatteryModel, step_time: float) -> bool:
        """Check if discharge is complete (voltage cutoff reached)."""
        return battery.state.voltage <= self.voltage_cutoff or battery.state.soc <= 0.0001


@dataclass
class RestStep(ProtocolStep):
    """Rest/relaxation step with zero current."""

    duration: float = 300.0  # seconds

    def __post_init__(self):
        self.step_type = "rest"

    def get_current(self, battery: BatteryModel) -> float:
        """Rest step has zero current."""
        return 0.0

    def is_complete(self, battery: BatteryModel, step_time: float) -> bool:
        """Check if rest duration is complete."""
        return step_time >= self.duration


@dataclass
class PulseStep(ProtocolStep):
    """Current pulse for resistance measurement."""

    current_rate: float = 1.0  # C-rate
    duration: float = 10.0  # seconds
    is_discharge: bool = True  # Discharge pulse by default

    def __post_init__(self):
        self.step_type = "pulse_discharge" if self.is_discharge else "pulse_charge"

    def get_current(self, battery: BatteryModel) -> float:
        """Get pulse current."""
        current = self.current_rate * battery.capacity_nominal
        return -current if self.is_discharge else current

    def is_complete(self, battery: BatteryModel, step_time: float) -> bool:
        """Check if pulse duration is complete."""
        return step_time >= self.duration


@dataclass
class BaseProtocol(ABC):
    """Abstract base class for test protocols."""

    name: str = "BaseProtocol"
    cycles: int = 1
    steps: list[ProtocolStep] = field(default_factory=list)

    @abstractmethod
    def _build_steps(self) -> list[ProtocolStep]:
        """Build the list of steps for one cycle. Must be implemented by subclasses."""
        pass

    def __post_init__(self):
        """Initialize protocol steps."""
        self.steps = self._build_steps()

    def get_total_steps(self) -> int:
        """Get total number of steps per cycle."""
        return len(self.steps)

    def to_dict(self) -> dict:
        """Convert protocol to dictionary."""
        return {
            "name": self.name,
            "cycles": self.cycles,
            "steps": [
                {
                    "type": step.step_type,
                    "is_charge": step.is_charge,
                    "is_discharge": step.is_discharge,
                }
                for step in self.steps
            ],
        }

