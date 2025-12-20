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
        
        # Initialize PyBaMM model
        self._init_pybamm_model()
        
        # Override capacity if specified
        if capacity:
            self.capacity_nominal = capacity
            self._update_capacity(capacity)
        else:
            self.capacity_nominal = self._get_nominal_capacity()
        
        # Initialize state
        self.state = PyBaMMState(
            soc=initial_soc,
            temperature=temperature,
            capacity_current=self.capacity_nominal,
            voltage=self._get_initial_voltage(),
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
        """Get initial voltage based on SOC."""
        try:
            # Run a quick simulation at rest to get OCV
            experiment = pybamm.Experiment([
                f"Rest for 1 seconds"
            ])
            sim = pybamm.Simulation(
                self.model,
                parameter_values=self.parameter_values,
                experiment=experiment,
            )
            
            # Set initial SOC
            sim.solve(initial_soc=self.initial_soc)
            voltage = sim.solution["Voltage [V]"].entries[-1]
            return float(voltage)
        except Exception:
            # Fallback based on chemistry
            return 3.7

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
        
        try:
            # Get OCV from PyBaMM parameter functions
            ocv_p = self.parameter_values.evaluate(
                self.parameter_values["Positive electrode OCP [V]"]
            )
            ocv_n = self.parameter_values.evaluate(
                self.parameter_values["Negative electrode OCP [V]"]
            )
            # Simplified OCV calculation
            return float(ocv_p - ocv_n) if isinstance(ocv_p, (int, float)) else 3.7
        except Exception:
            # Fallback to linear approximation
            return 3.0 + soc * 1.2

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
        
        For PyBaMM, this runs a short simulation step.
        
        Args:
            current: Applied current in A
            dt: Time step in seconds
            
        Returns:
            Terminal voltage in V
        """
        try:
            # Run a short experiment
            if abs(current) < 0.001:
                exp_string = f"Rest for {dt} seconds"
            elif current > 0:
                exp_string = f"Charge at {current} A for {dt} seconds"
            else:
                exp_string = f"Discharge at {abs(current)} A for {dt} seconds"
            
            experiment = pybamm.Experiment([exp_string])
            sim = pybamm.Simulation(
                self.model,
                parameter_values=self.parameter_values,
                experiment=experiment,
            )
            
            solution = sim.solve(initial_soc=self.state.soc)
            voltage = solution["Voltage [V]"].entries[-1]
            return float(voltage)
            
        except Exception:
            # Fallback to simple model
            return self._simple_voltage(current)

    def _simple_voltage(self, current: float) -> float:
        """Simple voltage calculation as fallback."""
        ocv = self.get_ocv(self.state.soc)
        r_internal = 0.05  # Approximate internal resistance
        return ocv - current * r_internal

    def update_state(self, current: float, dt: float) -> None:
        """
        Update battery state after applying current for time dt.
        
        Args:
            current: Applied current in A
            dt: Time step in seconds
        """
        # Update SOC
        dq = current * dt / 3600.0  # Convert to Ah
        coulombic_eff = 0.9995 if current > 0 else 1.0
        new_soc = self.state.soc + (dq * coulombic_eff) / self.state.capacity_current
        self.state.soc = np.clip(new_soc, 0.0, 1.0)
        
        # Update voltage
        self.state.voltage = self.calculate_voltage(current, dt)
        self.state.current = current
        
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

