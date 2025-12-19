"""Base chemistry class for battery configurations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DegradationParameters:
    """
    Semi-empirical degradation parameters for Arrhenius-based aging models.
    
    Cycle aging model:
        Q_loss,cyc = k_cyc * N^z * exp(-E_a,cyc/R * (1/T - 1/T_ref)) * (C_ch/C_ref)^α
    
    Calendar aging model:
        Q_loss,cal = k_cal * t^b * exp(-E_a,cal/R * (1/T - 1/T_ref)) * exp(β_SOC * (SOC - SOC_ref))
    
    Attributes:
        k_cyc: Cycle degradation pre-factor (calibrated for target life)
        e_a_cyc: Activation energy for cycle aging (J/mol)
        z: Cycle number exponent (typically 0.5-0.6)
        alpha: C-rate exponent (chemistry-dependent)
        
        k_cal: Calendar degradation pre-factor
        e_a_cal: Activation energy for calendar aging (J/mol)
        b: Time exponent (typically 0.4-0.5)
        beta_soc: SOC sensitivity coefficient
        
        k_resistance: Resistance growth pre-factor
        e_a_resistance: Activation energy for resistance growth (J/mol)
    """
    # Cycle aging parameters
    k_cyc: float = 0.003  # Pre-factor (fit to match target cycle life)
    e_a_cyc: float = 45000.0  # Activation energy (J/mol)
    z: float = 0.6  # Cycle exponent
    alpha: float = 0.7  # C-rate exponent
    
    # Calendar aging parameters
    k_cal: float = 0.001  # Pre-factor
    e_a_cal: float = 40000.0  # Activation energy (J/mol)
    b: float = 0.5  # Time exponent
    beta_soc: float = 1.5  # SOC sensitivity (positive = high SOC hurts)
    soc_ref: float = 0.5  # Reference SOC
    
    # Resistance growth parameters
    k_resistance: float = 0.002  # Resistance growth pre-factor
    e_a_resistance: float = 35000.0  # Activation energy (J/mol)
    
    # Reference conditions
    t_ref: float = 298.0  # Reference temperature (K) = 25°C
    c_ref: float = 1.0  # Reference C-rate


@dataclass
class BaseChemistry(ABC):
    """
    Abstract base class for battery chemistry configurations.
    
    Defines all parameters needed for battery simulation:
    - Voltage limits and nominal values
    - Capacity and resistance
    - OCV vs SOC curve
    - Performance limits
    - Degradation parameters (semi-empirical Arrhenius-based)
    """

    # Chemistry identification
    name: str = field(default="BaseChemistry", init=False)
    cathode: str = field(default="Unknown", init=False)
    anode: str = field(default="Unknown", init=False)

    # Voltage specifications (V)
    voltage_nominal: float = field(default=3.7, init=False)
    voltage_max: float = field(default=4.2, init=False)
    voltage_min: float = field(default=3.0, init=False)

    # Capacity and resistance
    capacity: float = field(default=3.0, init=False)  # Ah
    resistance_initial: float = field(default=0.050, init=False)  # Ohm

    # Performance limits
    max_charge_rate: float = field(default=1.0, init=False)  # C-rate
    max_discharge_rate: float = field(default=2.0, init=False)  # C-rate
    coulombic_efficiency: float = field(default=0.9995, init=False)
    energy_density: float = field(default=250.0, init=False)  # Wh/kg

    # Legacy degradation parameters (kept for backward compatibility)
    capacity_fade_rate: float = field(default=0.05, init=False)  # % per 100 cycles
    resistance_growth_rate: float = field(default=0.02, init=False)  # % per 100 cycles
    calendar_fade_rate: float = field(default=0.03, init=False)  # % per year at 25°C

    # Semi-empirical degradation parameters
    degradation_params: DegradationParameters = field(
        default_factory=DegradationParameters, init=False
    )

    # OCV table: list of [SOC, OCV] pairs
    ocv_table: list[list[float]] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Initialize chemistry-specific parameters."""
        self._init_parameters()

    @abstractmethod
    def _init_parameters(self) -> None:
        """Initialize chemistry-specific parameters. Must be implemented by subclasses."""
        pass

    def get_ocv(self, soc: float) -> float:
        """
        Get open circuit voltage for given SOC using linear interpolation.
        
        Args:
            soc: State of charge (0-1)
            
        Returns:
            Open circuit voltage (V)
        """
        import numpy as np
        from scipy.interpolate import interp1d

        soc_points = np.array([p[0] for p in self.ocv_table])
        ocv_points = np.array([p[1] for p in self.ocv_table])
        interp = interp1d(soc_points, ocv_points, kind="cubic", fill_value="extrapolate")
        return float(interp(np.clip(soc, 0.0, 1.0)))

    def validate(self) -> list[str]:
        """
        Validate chemistry parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.voltage_min >= self.voltage_max:
            errors.append(f"voltage_min ({self.voltage_min}) must be < voltage_max ({self.voltage_max})")

        if not (self.voltage_min <= self.voltage_nominal <= self.voltage_max):
            errors.append(
                f"voltage_nominal ({self.voltage_nominal}) must be between "
                f"voltage_min ({self.voltage_min}) and voltage_max ({self.voltage_max})"
            )

        if self.capacity <= 0:
            errors.append(f"capacity ({self.capacity}) must be > 0")

        if self.resistance_initial <= 0:
            errors.append(f"resistance_initial ({self.resistance_initial}) must be > 0")

        if not (0 < self.coulombic_efficiency <= 1):
            errors.append(f"coulombic_efficiency ({self.coulombic_efficiency}) must be in (0, 1]")

        if len(self.ocv_table) < 2:
            errors.append("ocv_table must have at least 2 points")

        return errors

    def to_dict(self) -> dict:
        """Convert chemistry to dictionary."""
        return {
            "name": self.name,
            "cathode": self.cathode,
            "anode": self.anode,
            "voltage_nominal": self.voltage_nominal,
            "voltage_max": self.voltage_max,
            "voltage_min": self.voltage_min,
            "capacity": self.capacity,
            "resistance_initial": self.resistance_initial,
            "max_charge_rate": self.max_charge_rate,
            "max_discharge_rate": self.max_discharge_rate,
            "coulombic_efficiency": self.coulombic_efficiency,
            "energy_density": self.energy_density,
            "capacity_fade_rate": self.capacity_fade_rate,
            "resistance_growth_rate": self.resistance_growth_rate,
            "calendar_fade_rate": self.calendar_fade_rate,
            "degradation_params": {
                "k_cyc": self.degradation_params.k_cyc,
                "e_a_cyc": self.degradation_params.e_a_cyc,
                "z": self.degradation_params.z,
                "alpha": self.degradation_params.alpha,
                "k_cal": self.degradation_params.k_cal,
                "e_a_cal": self.degradation_params.e_a_cal,
                "b": self.degradation_params.b,
                "beta_soc": self.degradation_params.beta_soc,
            },
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"capacity={self.capacity}Ah, "
            f"voltage={self.voltage_min}-{self.voltage_max}V)"
        )
