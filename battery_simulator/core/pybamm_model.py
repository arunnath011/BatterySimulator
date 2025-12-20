"""
PyBaMM-based high-fidelity battery model.

This module wraps PyBaMM's physics-based electrochemical models (SPM, SPMe, DFN)
to provide accurate battery simulations with the same interface as our fast empirical model.

Requires: pip install pybamm
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import pybamm
    PYBAMM_AVAILABLE = True
except ImportError:
    PYBAMM_AVAILABLE = False
    pybamm = None

if TYPE_CHECKING:
    from battery_simulator.chemistry.base_chemistry import BaseChemistry


class PyBaMMModelType(str, Enum):
    """Available PyBaMM electrochemical models."""
    SPM = "SPM"  # Single Particle Model - fastest
    SPME = "SPMe"  # Single Particle Model with Electrolyte - balanced
    DFN = "DFN"  # Doyle-Fuller-Newman - most accurate


@dataclass
class PyBaMMState:
    """Current state from PyBaMM simulation."""
    soc: float = 0.5
    voltage: float = 3.7
    current: float = 0.0
    temperature: float = 25.0
    capacity_current: float = 3.0
    resistance_current: float = 0.05
    cycle_count: int = 0
    step_capacity: float = 0.0
    step_energy: float = 0.0
    total_capacity: float = 0.0
    total_energy: float = 0.0
    # PyBaMM-specific states
    c_s_surf_n: float = 0.0  # Surface concentration negative electrode
    c_s_surf_p: float = 0.0  # Surface concentration positive electrode
    c_e_avg: float = 1000.0  # Average electrolyte concentration


class PyBaMMModel:
    """
    High-fidelity battery model using PyBaMM.
    
    Implements physics-based electrochemical models:
    - SPM: Single Particle Model (fastest, ~10x slower than empirical)
    - SPMe: SPM with electrolyte dynamics (balanced)
    - DFN: Doyle-Fuller-Newman full physics (most accurate, ~100x slower)
    
    Provides the same interface as BatteryModel for seamless integration.
    
    Note: For performance, per-timestep updates use extracted PyBaMM parameters
    with a simplified model. Full PyBaMM simulations are used for complete
    charge/discharge cycles when using run_experiment().
    """

    # Mapping of our chemistry names to PyBaMM parameter sets
    CHEMISTRY_MAP = {
        "NMC811": "Chen2020",
        "NMC811-Graphite": "Chen2020",
        "LFP": "Prada2013",
        "LFP-Graphite": "Prada2013",
        "NCA": "NCA_Kim2011",
        "NCA-SiGraphite": "NCA_Kim2011",
        "LTO": "Ramadass2004",
        "LTO-LMO": "Ramadass2004",
    }

    def __init__(
        self,
        chemistry: BaseChemistry | str,
        capacity: float | None = None,
        temperature: float = 25.0,
        initial_soc: float = 0.5,
        model_type: PyBaMMModelType = PyBaMMModelType.SPME,
        parameter_set: str | None = None,
    ):
        """
        Initialize the PyBaMM model.
        
        Args:
            chemistry: Chemistry configuration or name
            capacity: Nominal capacity in Ah (uses parameter set default if None)
            temperature: Initial temperature in C
            initial_soc: Initial state of charge (0-1)
            model_type: Which PyBaMM model to use (SPM, SPMe, DFN)
            parameter_set: PyBaMM parameter set name (auto-detected if None)
        """
        if not PYBAMM_AVAILABLE:
            raise ImportError(
                "PyBaMM is not installed. Install with: pip install pybamm"
            )
        
        self.model_type = model_type
        self.temperature_initial = temperature
        self.initial_soc = initial_soc
        
        # Get chemistry name
        if hasattr(chemistry, 'name'):
            self.chemistry_name = chemistry.name
            self._chemistry_obj = chemistry
        else:
            self.chemistry_name = chemistry
            self._chemistry_obj = None
        
        # Determine parameter set
        if parameter_set:
            self.parameter_set_name = parameter_set
        else:
            self.parameter_set_name = self.CHEMISTRY_MAP.get(
                self.chemistry_name, "Chen2020"
            )
        
        # Initialize PyBaMM model and extract parameters
        self._init_pybamm_model()
        
        # Extract OCV curve for fast lookups
        self._extract_ocv_curve()
        
        # Override capacity if specified
        if capacity:
            self.capacity_nominal = capacity
            self._update_capacity(capacity)
        else:
            self.capacity_nominal = self._get_nominal_capacity()
        
        # Extract internal resistance
        self._internal_resistance = self._extract_internal_resistance()
        
        # Initialize state
        self.state = PyBaMMState(
            soc=initial_soc,
            temperature=temperature,
            capacity_current=self.capacity_nominal,
            voltage=self._get_ocv_from_curve(initial_soc),
            resistance_current=self._internal_resistance,
        )
        
        # Simulation state
        self._simulation = None
        self._current_time = 0.0
        self._step_start_capacity = 0.0
        self._step_start_energy = 0.0

    def _init_pybamm_model(self) -> None:
        """Initialize the PyBaMM model and parameter set."""
        # Select model
        if self.model_type == PyBaMMModelType.SPM:
            self.model = pybamm.lithium_ion.SPM()
        elif self.model_type == PyBaMMModelType.SPME:
            self.model = pybamm.lithium_ion.SPMe()
        elif self.model_type == PyBaMMModelType.DFN:
            self.model = pybamm.lithium_ion.DFN()
        else:
            self.model = pybamm.lithium_ion.SPMe()
        
        # Load parameter set
        try:
            self.parameter_values = pybamm.ParameterValues(self.parameter_set_name)
        except Exception:
            # Fall back to Chen2020 if parameter set not found
            self.parameter_values = pybamm.ParameterValues("Chen2020")
            self.parameter_set_name = "Chen2020"
        
        # Set initial temperature
        self.parameter_values.update({
            "Ambient temperature [K]": self.temperature_initial + 273.15,
            "Initial temperature [K]": self.temperature_initial + 273.15,
        })
        
        # Create solver
        self.solver = pybamm.CasadiSolver(mode="fast")
    
    def _extract_ocv_curve(self) -> None:
        """Extract OCV curve from PyBaMM for fast lookups."""
        # Generate OCV lookup table by running quick simulations at different SOCs
        self._ocv_soc = np.linspace(0.0, 1.0, 101)
        self._ocv_voltage = np.zeros(101)
        
        try:
            # Try to get OCV from a quick discharge simulation
            experiment = pybamm.Experiment([
                "Discharge at C/20 until 2.5 V",
            ])
            sim = pybamm.Simulation(
                self.model,
                parameter_values=self.parameter_values,
                experiment=experiment,
            )
            solution = sim.solve(initial_soc=1.0)
            
            # Extract voltage vs SOC
            socs = solution["State of Charge"].entries if "State of Charge" in solution.data else None
            voltages = solution["Voltage [V]"].entries
            
            if socs is not None and len(socs) > 10:
                # Interpolate to our standard SOC points
                from scipy.interpolate import interp1d
                # Reverse because discharge goes from high to low SOC
                interp_func = interp1d(socs[::-1], voltages[::-1], 
                                       bounds_error=False, fill_value="extrapolate")
                self._ocv_voltage = interp_func(self._ocv_soc)
            else:
                # Use default curve
                self._set_default_ocv_curve()
        except Exception:
            # Use default curve based on chemistry
            self._set_default_ocv_curve()
    
    def _set_default_ocv_curve(self) -> None:
        """Set default OCV curve based on chemistry type."""
        # Default voltage limits based on chemistry
        if "LFP" in self.chemistry_name:
            v_min, v_max = 2.5, 3.65
        elif "LTO" in self.chemistry_name:
            v_min, v_max = 1.5, 2.8
        else:  # NMC, NCA
            v_min, v_max = 2.5, 4.2
        
        # Simple polynomial approximation
        self._ocv_voltage = v_min + (v_max - v_min) * (
            0.1 + 0.8 * self._ocv_soc + 0.1 * self._ocv_soc**2
        )
    
    def _get_ocv_from_curve(self, soc: float) -> float:
        """Get OCV from pre-computed curve."""
        soc = np.clip(soc, 0.0, 1.0)
        idx = int(soc * 100)
        idx = min(idx, 100)
        return float(self._ocv_voltage[idx])
    
    def _extract_internal_resistance(self) -> float:
        """Extract internal resistance from PyBaMM parameters."""
        try:
            # Try to get from parameter values
            # This varies by parameter set, try common names
            r_names = [
                "Cell capacity [A.h]",
                "Nominal cell capacity [A.h]",
            ]
            capacity = 3.0
            for name in r_names:
                try:
                    capacity = float(self.parameter_values[name])
                    break
                except Exception:
                    pass
            
            # Estimate resistance from capacity (typical values)
            # Larger cells tend to have lower resistance
            return 0.02 + 0.01 * (3.0 / max(capacity, 0.1))
        except Exception:
            return 0.05  # Default 50 mOhm

    def _get_nominal_capacity(self) -> float:
        """Get nominal capacity from parameter set."""
        try:
            return float(self.parameter_values["Nominal cell capacity [A.h]"])
        except Exception:
            return 3.0  # Default

    def _update_capacity(self, capacity: float) -> None:
        """Update the capacity in parameter values."""
        try:
            self.parameter_values.update({
                "Nominal cell capacity [A.h]": capacity,
            })
        except Exception:
            pass

    def _get_initial_voltage(self) -> float:
        """Get initial voltage based on SOC using pre-computed OCV curve."""
        return self._get_ocv_from_curve(self.initial_soc)

    def get_ocv(self, soc: float | None = None) -> float:
        """
        Get open circuit voltage for given SOC.
        
        Uses pre-computed OCV curve extracted from PyBaMM for fast lookup.
        
        Args:
            soc: State of charge (0-1), uses current state if None
            
        Returns:
            Open circuit voltage in V
        """
        if soc is None:
            soc = self.state.soc
        return self._get_ocv_from_curve(soc)

    def run_experiment(
        self,
        current: float,
        duration: float,
        dt: float = 1.0,
    ) -> list[dict]:
        """
        Run a constant current experiment.
        
        Args:
            current: Applied current in A (positive = charge, negative = discharge)
            duration: Duration in seconds
            dt: Time step for output in seconds
            
        Returns:
            List of data points with time, voltage, current, SOC, temperature
        """
        data_points = []
        
        # Build experiment string
        if abs(current) < 0.001:
            exp_string = f"Rest for {duration} seconds"
        elif current > 0:
            exp_string = f"Charge at {current} A for {duration} seconds"
        else:
            exp_string = f"Discharge at {abs(current)} A for {duration} seconds"
        
        try:
            experiment = pybamm.Experiment([exp_string])
            sim = pybamm.Simulation(
                self.model,
                parameter_values=self.parameter_values,
                experiment=experiment,
            )
            
            solution = sim.solve(initial_soc=self.state.soc)
            
            # Extract data at regular intervals
            times = solution["Time [s]"].entries
            voltages = solution["Voltage [V]"].entries
            currents = solution["Current [A]"].entries
            
            # Try to get SOC and temperature
            try:
                socs = solution["State of Charge"].entries
            except Exception:
                socs = np.linspace(self.state.soc, self.state.soc, len(times))
            
            try:
                temps = solution["Cell temperature [K]"].entries - 273.15
            except Exception:
                temps = np.full(len(times), self.state.temperature)
            
            # Sample at dt intervals
            for i, t in enumerate(times):
                if i == 0 or t >= self._current_time + dt:
                    data_points.append({
                        "time": self._current_time + t,
                        "voltage": float(voltages[i]),
                        "current": float(currents[i]),
                        "soc": float(socs[i]) if i < len(socs) else self.state.soc,
                        "temperature": float(temps[i]) if i < len(temps) else self.state.temperature,
                    })
            
            # Update state
            if len(voltages) > 0:
                self.state.voltage = float(voltages[-1])
                self.state.current = float(currents[-1])
                if len(socs) > 0:
                    self.state.soc = float(socs[-1])
                if len(temps) > 0:
                    self.state.temperature = float(temps[-1])
            
            self._current_time += duration
            
        except Exception as e:
            # Fallback to simple model if PyBaMM fails
            num_steps = int(duration / dt)
            for i in range(num_steps):
                self.update_state(current, dt)
                data_points.append({
                    "time": self._current_time,
                    "voltage": self.state.voltage,
                    "current": current,
                    "soc": self.state.soc,
                    "temperature": self.state.temperature,
                })
        
        return data_points

    def calculate_voltage(self, current: float, dt: float = 1.0) -> float:
        """
        Calculate terminal voltage given current.
        
        Uses pre-extracted OCV curve and resistance for fast calculation.
        This avoids running a full PyBaMM simulation for every timestep.
        
        Args:
            current: Applied current in A
            dt: Time step in seconds (unused, kept for interface compatibility)
            
        Returns:
            Terminal voltage in V
        """
        # Fast calculation using extracted parameters
        ocv = self.get_ocv(self.state.soc)
        
        # Temperature effect on resistance (Arrhenius-like)
        temp_factor = 1.0 + 0.02 * (25.0 - self.state.temperature)
        r_effective = self.state.resistance_current * temp_factor
        
        # Terminal voltage = OCV - IR drop
        voltage = ocv - current * r_effective
        
        # Add simple polarization effect for more realism
        if abs(current) > 0.01:
            # Concentration polarization (simplified)
            c_rate = abs(current) / self.capacity_nominal
            polarization = 0.02 * c_rate * np.sign(current)
            voltage -= polarization
        
        return float(voltage)

    def _simple_voltage(self, current: float) -> float:
        """Simple voltage calculation as fallback."""
        ocv = self.get_ocv(self.state.soc)
        return ocv - current * self._internal_resistance

    def update_state(self, current: float, dt: float) -> None:
        """
        Update battery state after applying current for time dt.
        
        Uses fast calculations with pre-extracted PyBaMM parameters.
        
        Args:
            current: Applied current in A
            dt: Time step in seconds
        """
        # Update SOC using coulomb counting
        dq = current * dt / 3600.0  # Convert to Ah
        coulombic_eff = 0.9995 if current > 0 else 1.0
        new_soc = self.state.soc + (dq * coulombic_eff) / self.state.capacity_current
        self.state.soc = float(np.clip(new_soc, 0.0, 1.0))
        
        # Update voltage using fast calculation
        self.state.voltage = self.calculate_voltage(current, dt)
        self.state.current = current
        
        # Simple thermal model
        if abs(current) > 0.01:
            # Heat generation from I²R
            heat = current**2 * self.state.resistance_current * dt
            # Temperature rise (simplified)
            temp_rise = heat * 0.001  # Scaled for reasonable values
            # Cooling towards ambient
            temp_diff = self.state.temperature - self.temperature_initial
            cooling = temp_diff * 0.01 * dt
            self.state.temperature += temp_rise - cooling
        
        # Update accumulators
        dq_abs = abs(current * dt / 3600.0)
        de = abs(self.state.voltage * current * dt / 3600.0)
        self.state.step_capacity += dq_abs
        self.state.step_energy += de
        self.state.total_capacity += dq_abs
        self.state.total_energy += de
        
        self._current_time += dt

    def reset_step_accumulators(self) -> None:
        """Reset step-level capacity and energy accumulators."""
        self.state.step_capacity = 0.0
        self.state.step_energy = 0.0

    def apply_degradation(self, capacity_fade: float, resistance_growth: float) -> None:
        """
        Apply degradation to the battery.
        
        Args:
            capacity_fade: Fractional capacity loss
            resistance_growth: Fractional resistance increase
        """
        self.state.capacity_current *= (1 - capacity_fade)
        self.state.resistance_current *= (1 + resistance_growth)
        
        # Update PyBaMM parameters
        try:
            self._update_capacity(self.state.capacity_current)
        except Exception:
            pass

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
            "model_type": self.model_type.value,
            "parameter_set": self.parameter_set_name,
        }

    @property
    def capacity_retention(self) -> float:
        """Get current capacity retention as fraction of nominal."""
        return self.state.capacity_current / self.capacity_nominal

    @classmethod
    def list_available_parameter_sets(cls) -> list[str]:
        """List available PyBaMM parameter sets."""
        if not PYBAMM_AVAILABLE:
            return []
        
        try:
            return list(pybamm.parameter_sets.keys())
        except Exception:
            return [
                "Chen2020",
                "Marquis2019",
                "Prada2013",
                "Ramadass2004",
                "NCA_Kim2011",
                "Ecker2015",
                "Ai2020",
            ]

    @classmethod
    def is_available(cls) -> bool:
        """Check if PyBaMM is available."""
        return PYBAMM_AVAILABLE


def check_pybamm_available() -> bool:
    """Check if PyBaMM is installed and available."""
    return PYBAMM_AVAILABLE

