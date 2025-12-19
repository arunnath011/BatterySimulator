"""Physics-based battery model for lithium-ion cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from battery_simulator.chemistry.base_chemistry import BaseChemistry


@dataclass
class BatteryState:
    """Current state of the battery."""

    soc: float = 0.5  # State of charge (0-1)
    voltage: float = 3.7  # Terminal voltage (V)
    current: float = 0.0  # Current (A)
    temperature: float = 25.0  # Temperature (°C)
    capacity_current: float = 3.0  # Current capacity (Ah)
    resistance_current: float = 0.05  # Current internal resistance (Ohm)
    cycle_count: int = 0
    step_capacity: float = 0.0  # Capacity accumulated in current step (Ah)
    step_energy: float = 0.0  # Energy accumulated in current step (Wh)
    total_capacity: float = 0.0  # Total capacity throughput (Ah)
    total_energy: float = 0.0  # Total energy throughput (Wh)


class BatteryModel:
    """
    Physics-based battery model for lithium-ion cells.
    
    Implements:
    - Voltage vs. State-of-Charge (SOC) curves based on OCV lookup tables
    - Internal resistance model (DC resistance + polarization)
    - Coulombic efficiency modeling
    - Temperature dependence of voltage and resistance
    - Self-discharge modeling
    """

    def __init__(
        self,
        chemistry: BaseChemistry,
        capacity: float | None = None,
        temperature: float = 25.0,
        initial_soc: float = 0.5,
    ):
        """
        Initialize the battery model.
        
        Args:
            chemistry: Chemistry configuration object
            capacity: Nominal capacity in Ah (uses chemistry default if None)
            temperature: Initial temperature in °C
            initial_soc: Initial state of charge (0-1)
        """
        self.chemistry = chemistry
        self.capacity_nominal = capacity or chemistry.capacity
        self.temperature_ref = 298.0  # Reference temperature (K)

        # Initialize state
        self.state = BatteryState(
            soc=initial_soc,
            temperature=temperature,
            capacity_current=self.capacity_nominal,
            resistance_current=chemistry.resistance_initial,
        )

        # Create OCV interpolator
        self._ocv_interp = self._create_ocv_interpolator()

        # Polarization time constant (seconds)
        self._tau_polarization = 30.0
        self._polarization_current = 0.0

    def _create_ocv_interpolator(self) -> interp1d:
        """Create interpolation function for OCV vs SOC."""
        soc_points = np.array([p[0] for p in self.chemistry.ocv_table])
        ocv_points = np.array([p[1] for p in self.chemistry.ocv_table])
        return interp1d(
            soc_points, ocv_points, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )

    def get_ocv(self, soc: float | None = None) -> float:
        """
        Get open circuit voltage for given SOC.
        
        Args:
            soc: State of charge (0-1), uses current state if None
            
        Returns:
            Open circuit voltage in V
        """
        if soc is None:
            soc = self.state.soc
        soc = np.clip(soc, 0.0, 1.0)
        return float(self._ocv_interp(soc))

    def calculate_voltage(self, current: float, dt: float = 1.0) -> float:
        """
        Calculate terminal voltage given current.
        
        V(t) = OCV(SOC) - I(t) * R_internal(SOC, T, age) - η_polarization(I, SOC)
        
        Args:
            current: Applied current in A (positive = charge, negative = discharge)
            dt: Time step in seconds
            
        Returns:
            Terminal voltage in V
        """
        # Get OCV at current SOC
        ocv = self.get_ocv(self.state.soc)

        # Get temperature-adjusted resistance
        r_internal = self._get_temperature_adjusted_resistance()

        # Calculate polarization overpotential
        v_polarization = self._calculate_polarization(current, dt)

        # Calculate voltage drop
        v_drop = current * r_internal

        # Terminal voltage (for discharge, current is negative, so v_drop adds to voltage)
        voltage = ocv - v_drop - v_polarization

        # Apply voltage limits
        voltage = np.clip(voltage, self.chemistry.voltage_min, self.chemistry.voltage_max)

        return float(voltage)

    def _get_temperature_adjusted_resistance(self) -> float:
        """
        Get temperature-adjusted internal resistance.
        
        R(T) = R_ref * exp(Ea_R / R * (1/T - 1/T_ref))
        """
        # Activation energy for resistance (J/mol)
        ea_r = 20000.0  # Typical value
        r_gas = 8.314  # Gas constant (J/mol·K)

        t_current = self.state.temperature + 273.15  # Convert to Kelvin
        t_ref = self.temperature_ref

        temp_factor = np.exp(ea_r / r_gas * (1 / t_current - 1 / t_ref))
        return self.state.resistance_current * temp_factor

    def _calculate_polarization(self, current: float, dt: float) -> float:
        """
        Calculate polarization overpotential using first-order RC model.
        
        Args:
            current: Applied current in A
            dt: Time step in seconds
            
        Returns:
            Polarization voltage in V
        """
        # First-order exponential response
        alpha = dt / (self._tau_polarization + dt)
        self._polarization_current = (1 - alpha) * self._polarization_current + alpha * current

        # Polarization resistance (increases at low/high SOC)
        r_pol = self.chemistry.resistance_initial * 0.3
        soc_factor = 1.0 + 0.5 * (abs(self.state.soc - 0.5) * 2) ** 2

        return self._polarization_current * r_pol * soc_factor

    def update_state(self, current: float, dt: float) -> None:
        """
        Update battery state after applying current for time dt.
        
        SOC(t) = SOC(t-1) + (I(t) * Δt * η_coulombic) / Q_capacity
        
        Args:
            current: Applied current in A (positive = charge, negative = discharge)
            dt: Time step in seconds
        """
        # Get coulombic efficiency
        if current > 0:  # Charging
            eta = self.chemistry.coulombic_efficiency
        else:  # Discharging
            eta = 1.0  # Full efficiency on discharge

        # Update SOC
        dq = current * dt / 3600.0 * eta  # Convert to Ah
        new_soc = self.state.soc + dq / self.state.capacity_current
        self.state.soc = np.clip(new_soc, 0.0, 1.0)

        # Update current and voltage
        self.state.current = current
        self.state.voltage = self.calculate_voltage(current, dt)

        # Update step accumulators
        dq_abs = abs(current * dt / 3600.0)
        de = abs(self.state.voltage * current * dt / 3600.0)
        self.state.step_capacity += dq_abs
        self.state.step_energy += de
        self.state.total_capacity += dq_abs
        self.state.total_energy += de

        # Update temperature (simple model)
        self._update_temperature(current, dt)

    def _update_temperature(self, current: float, dt: float) -> None:
        """
        Simple thermal model for temperature update.
        
        ΔT = Q_heat / (m * Cp) - h * A * (T - T_amb) / (m * Cp)
        """
        # Heat generation from I²R losses
        q_heat = current**2 * self.state.resistance_current

        # Simplified thermal model
        mass = 0.050  # kg (typical 18650 mass)
        cp = 1000.0  # J/kg·K
        h_conv = 10.0  # W/m²·K (convection coefficient)
        area = 0.004  # m² (surface area)
        t_amb = 25.0  # Ambient temperature

        # Temperature change
        dt_heat = q_heat * dt / (mass * cp)
        dt_cool = h_conv * area * (self.state.temperature - t_amb) * dt / (mass * cp)

        self.state.temperature += dt_heat - dt_cool

    def reset_step_accumulators(self) -> None:
        """Reset step-level capacity and energy accumulators."""
        self.state.step_capacity = 0.0
        self.state.step_energy = 0.0

    def apply_degradation(self, capacity_fade: float, resistance_growth: float) -> None:
        """
        Apply degradation to the battery.
        
        Args:
            capacity_fade: Fractional capacity loss (e.g., 0.001 for 0.1%)
            resistance_growth: Fractional resistance increase (e.g., 0.001 for 0.1%)
        """
        self.state.capacity_current *= (1 - capacity_fade)
        self.state.resistance_current *= (1 + resistance_growth)

    def get_state_dict(self) -> dict:
        """Get current state as dictionary."""
        return {
            "soc": self.state.soc,
            "voltage": self.state.voltage,
            "current": self.state.current,
            "temperature": self.state.temperature,
            "capacity_current": self.state.capacity_current,
            "resistance_current": self.state.resistance_current,
            "cycle_count": self.state.cycle_count,
            "step_capacity": self.state.step_capacity,
            "step_energy": self.state.step_energy,
        }

    @property
    def capacity_retention(self) -> float:
        """Get current capacity retention as fraction of nominal."""
        return self.state.capacity_current / self.capacity_nominal

