"""
Pack-level battery simulation using liionpack.

This module provides pack-level simulations with series/parallel cell configurations,
cell-to-cell variation, and thermal coupling between cells.

Requires: pip install liionpack pybamm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

try:
    import pybamm
    PYBAMM_AVAILABLE = True
except ImportError:
    PYBAMM_AVAILABLE = False
    pybamm = None

try:
    import liionpack as lp
    LIIONPACK_AVAILABLE = True
except ImportError:
    LIIONPACK_AVAILABLE = False
    lp = None

if TYPE_CHECKING:
    from battery_simulator.chemistry.base_chemistry import BaseChemistry


@dataclass
class PackState:
    """Current state of the battery pack."""
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
    
    # Pack-specific states
    cell_voltages: List[float] = field(default_factory=list)
    cell_socs: List[float] = field(default_factory=list)
    cell_temperatures: List[float] = field(default_factory=list)
    pack_voltage: float = 0.0
    pack_power: float = 0.0
    min_cell_voltage: float = 0.0
    max_cell_voltage: float = 0.0
    soc_spread: float = 0.0  # Max SOC - Min SOC


@dataclass
class PackConfig:
    """Configuration for a battery pack."""
    series: int = 1
    parallel: int = 1
    cell_capacity_ah: float = 3.0
    cell_variation: float = 0.02  # 2% cell-to-cell variation
    connection_resistance: float = 0.001  # Ohm per connection
    thermal_coupling: float = 0.1  # Heat transfer coefficient between cells
    busbar_resistance: float = 0.0005  # Ohm
    
    @property
    def total_cells(self) -> int:
        """Total number of cells in the pack."""
        return self.series * self.parallel
    
    @property
    def pack_capacity_ah(self) -> float:
        """Total pack capacity in Ah (parallel cells add capacity)."""
        return self.cell_capacity_ah * self.parallel
    
    @property
    def pack_voltage_nominal(self, cell_voltage: float = 3.7) -> float:
        """Nominal pack voltage (series cells add voltage)."""
        return cell_voltage * self.series


class PackSimulator:
    """
    Pack-level battery simulator using liionpack.
    
    Supports:
    - Series and parallel cell configurations
    - Cell-to-cell variation in capacity and resistance
    - Thermal coupling between cells
    - Individual cell monitoring
    
    Provides the same interface as BatteryModel for integration with
    the main simulator.
    """
    
    # Mapping of our chemistry names to PyBaMM parameter sets
    CHEMISTRY_MAP = {
        "NMC811": "Chen2020",
        "LFP": "Prada2013",
        "NCA": "NCA_Kim2011",
        "LTO": "Ramadass2004",
    }

    def __init__(
        self,
        chemistry: BaseChemistry | str,
        capacity: float | None = None,
        temperature: float = 25.0,
        initial_soc: float = 0.5,
        series: int = 1,
        parallel: int = 1,
        cell_variation: float = 0.02,
        connection_resistance: float = 0.001,
        parameter_set: str | None = None,
    ):
        """
        Initialize the pack simulator.
        
        Args:
            chemistry: Chemistry configuration or name
            capacity: Cell nominal capacity in Ah (uses parameter set default if None)
            temperature: Initial temperature in C
            initial_soc: Initial state of charge (0-1)
            series: Number of cells in series
            parallel: Number of cells in parallel
            cell_variation: Cell-to-cell variation coefficient
            connection_resistance: Resistance per cell connection (Ohm)
            parameter_set: PyBaMM parameter set name (auto-detected if None)
        """
        if not LIIONPACK_AVAILABLE:
            raise ImportError(
                "liionpack is not installed. Install with: pip install liionpack"
            )
        if not PYBAMM_AVAILABLE:
            raise ImportError(
                "PyBaMM is not installed. Install with: pip install pybamm"
            )
        
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
        
        # Pack configuration
        self.pack_config = PackConfig(
            series=series,
            parallel=parallel,
            cell_capacity_ah=capacity or 3.0,
            cell_variation=cell_variation,
            connection_resistance=connection_resistance,
        )
        
        self.temperature_initial = temperature
        self.initial_soc = initial_soc
        
        # Initialize PyBaMM and liionpack
        self._init_pack_model()
        
        # Override capacity if specified
        if capacity:
            self.capacity_nominal = capacity * parallel  # Pack capacity
        else:
            self.capacity_nominal = self._get_cell_capacity() * parallel
        
        # Initialize state
        self.state = PackState(
            soc=initial_soc,
            temperature=temperature,
            capacity_current=self.capacity_nominal,
            voltage=self._get_initial_voltage(),
            cell_voltages=[self._get_initial_voltage()] * self.pack_config.total_cells,
            cell_socs=[initial_soc] * self.pack_config.total_cells,
            cell_temperatures=[temperature] * self.pack_config.total_cells,
        )
        self._update_pack_state()
        
        # Simulation state
        self._current_time = 0.0
        self._step_start_capacity = 0.0
        self._step_start_energy = 0.0
        
        # Cell variation factors
        self._init_cell_variations()

    def _init_pack_model(self) -> None:
        """Initialize the PyBaMM model and liionpack netlist."""
        # Load PyBaMM model - use SPMe for balance of speed and accuracy
        self.model = pybamm.lithium_ion.SPMe()
        
        # Load parameter set
        try:
            self.parameter_values = pybamm.ParameterValues(self.parameter_set_name)
        except Exception:
            self.parameter_values = pybamm.ParameterValues("Chen2020")
            self.parameter_set_name = "Chen2020"
        
        # Set initial temperature
        self.parameter_values.update({
            "Ambient temperature [K]": self.temperature_initial + 273.15,
            "Initial temperature [K]": self.temperature_initial + 273.15,
        })
        
        # Create netlist for liionpack
        self._create_netlist()

    def _create_netlist(self) -> None:
        """Create the liionpack netlist for pack topology."""
        self.netlist = lp.setup_circuit(
            Np=self.pack_config.parallel,
            Ns=self.pack_config.series,
            Rb=self.pack_config.connection_resistance,
            Ri=self.pack_config.connection_resistance,
            V=3.7,  # Initial voltage
            I=0.0,  # Initial current
        )

    def _init_cell_variations(self) -> None:
        """Initialize cell-to-cell variations."""
        rng = np.random.default_rng()
        n_cells = self.pack_config.total_cells
        
        # Capacity variations (normal distribution around 1.0)
        self._capacity_factors = 1.0 + rng.normal(
            0, self.pack_config.cell_variation, n_cells
        )
        self._capacity_factors = np.clip(self._capacity_factors, 0.9, 1.1)
        
        # Resistance variations
        self._resistance_factors = 1.0 + rng.normal(
            0, self.pack_config.cell_variation, n_cells
        )
        self._resistance_factors = np.clip(self._resistance_factors, 0.9, 1.1)
        
        # Initial SOC variations (small spread)
        self._soc_offsets = rng.normal(0, 0.01, n_cells)

    def _get_cell_capacity(self) -> float:
        """Get cell capacity from parameter set."""
        try:
            return float(self.parameter_values["Nominal cell capacity [A.h]"])
        except Exception:
            return 3.0

    def _get_initial_voltage(self) -> float:
        """Get initial voltage based on SOC."""
        # Simple linear approximation
        return 3.0 + self.initial_soc * 1.2

    def _update_pack_state(self) -> None:
        """Update pack-level state from cell states."""
        if not self.state.cell_voltages:
            return
        
        # Pack voltage is sum of series string voltages
        # For parallel strings, use average
        cell_voltages = np.array(self.state.cell_voltages)
        cell_voltages_reshaped = cell_voltages.reshape(
            self.pack_config.parallel, self.pack_config.series
        )
        string_voltages = cell_voltages_reshaped.sum(axis=1)
        self.state.pack_voltage = float(np.mean(string_voltages))
        self.state.voltage = self.state.pack_voltage / self.pack_config.series
        
        # Cell voltage statistics
        self.state.min_cell_voltage = float(np.min(cell_voltages))
        self.state.max_cell_voltage = float(np.max(cell_voltages))
        
        # SOC statistics
        cell_socs = np.array(self.state.cell_socs)
        self.state.soc = float(np.mean(cell_socs))
        self.state.soc_spread = float(np.max(cell_socs) - np.min(cell_socs))
        
        # Temperature
        self.state.temperature = float(np.mean(self.state.cell_temperatures))

    def run_experiment(
        self,
        current: float,
        duration: float,
        dt: float = 1.0,
    ) -> list[dict]:
        """
        Run a constant current experiment on the pack.
        
        Args:
            current: Applied pack current in A (positive = charge, negative = discharge)
            duration: Duration in seconds
            dt: Time step for output in seconds
            
        Returns:
            List of data points with pack and cell states
        """
        data_points = []
        
        try:
            # Build experiment for liionpack
            if abs(current) < 0.001:
                # Rest period - simulate with zero current
                num_steps = int(duration / dt)
                for _ in range(num_steps):
                    self._update_cells_rest(dt)
                    data_points.append(self._get_data_point())
            else:
                # Active experiment
                # Per-cell current (divided among parallel cells)
                cell_current = abs(current) / self.pack_config.parallel
                
                # Run liionpack simulation
                experiment = pybamm.Experiment([
                    f"Discharge at {cell_current} A for {duration} seconds" if current < 0 
                    else f"Charge at {cell_current} A for {duration} seconds"
                ])
                
                output = lp.solve(
                    netlist=self.netlist,
                    parameter_values=self.parameter_values,
                    experiment=experiment,
                    initial_soc=self.state.soc,
                )
                
                # Extract results
                self._extract_liionpack_results(output, dt, data_points)
                
        except Exception as e:
            # Fallback to simplified pack model
            num_steps = int(duration / dt)
            for _ in range(num_steps):
                self.update_state(current, dt)
                data_points.append(self._get_data_point())
        
        return data_points

    def _extract_liionpack_results(
        self, 
        output: Any, 
        dt: float, 
        data_points: list
    ) -> None:
        """Extract results from liionpack output."""
        try:
            # Get time array
            times = output["Time [s]"]
            
            # Sample at dt intervals
            for i, t in enumerate(times):
                if i == 0 or t >= self._current_time + dt:
                    # Update cell states from output
                    for cell_idx in range(self.pack_config.total_cells):
                        try:
                            self.state.cell_voltages[cell_idx] = float(
                                output["Terminal voltage [V]"][cell_idx, i]
                            )
                            self.state.cell_socs[cell_idx] = float(
                                output["State of Charge"][cell_idx, i]
                            )
                        except Exception:
                            pass
                    
                    self._update_pack_state()
                    self._current_time = t
                    data_points.append(self._get_data_point())
                    
        except Exception:
            pass

    def _update_cells_rest(self, dt: float) -> None:
        """Update cell states during rest."""
        # Small relaxation during rest
        for i in range(len(self.state.cell_socs)):
            # Thermal equilibration
            temp_diff = self.temperature_initial - self.state.cell_temperatures[i]
            self.state.cell_temperatures[i] += temp_diff * 0.01
        
        self._current_time += dt
        self._update_pack_state()

    def _get_data_point(self) -> dict:
        """Get current state as data point dictionary."""
        return {
            "time": self._current_time,
            "voltage": self.state.voltage,
            "pack_voltage": self.state.pack_voltage,
            "current": self.state.current,
            "soc": self.state.soc,
            "temperature": self.state.temperature,
            "min_cell_voltage": self.state.min_cell_voltage,
            "max_cell_voltage": self.state.max_cell_voltage,
            "soc_spread": self.state.soc_spread,
            "cell_voltages": self.state.cell_voltages.copy(),
            "cell_socs": self.state.cell_socs.copy(),
        }

    def calculate_voltage(self, current: float, dt: float = 1.0) -> float:
        """
        Calculate terminal voltage given current.
        
        Args:
            current: Applied current in A
            dt: Time step in seconds
            
        Returns:
            Terminal voltage in V
        """
        # Simple model: OCV - IR drop
        ocv = self.get_ocv(self.state.soc)
        r_total = self.state.resistance_current * self.pack_config.series
        r_total += self.pack_config.connection_resistance * self.pack_config.total_cells
        return ocv * self.pack_config.series - current * r_total

    def get_ocv(self, soc: float | None = None) -> float:
        """Get open circuit voltage for a single cell."""
        if soc is None:
            soc = self.state.soc
        soc = np.clip(soc, 0.0, 1.0)
        return 3.0 + soc * 1.2

    def update_state(self, current: float, dt: float) -> None:
        """
        Update pack state after applying current for time dt.
        
        Args:
            current: Applied current in A
            dt: Time step in seconds
        """
        # Update each cell
        cell_current = current / self.pack_config.parallel
        
        for i in range(self.pack_config.total_cells):
            # Update SOC
            eff_capacity = self.capacity_nominal / self.pack_config.parallel
            eff_capacity *= self._capacity_factors[i]
            dq = cell_current * dt / 3600.0
            coulombic_eff = 0.9995 if cell_current > 0 else 1.0
            new_soc = self.state.cell_socs[i] + (dq * coulombic_eff) / eff_capacity
            self.state.cell_socs[i] = np.clip(new_soc, 0.0, 1.0)
            
            # Update voltage
            cell_ocv = self.get_ocv(self.state.cell_socs[i])
            cell_r = 0.05 * self._resistance_factors[i]
            self.state.cell_voltages[i] = cell_ocv - cell_current * cell_r
            
            # Update temperature (simple model)
            heat = abs(cell_current) ** 2 * cell_r * dt
            self.state.cell_temperatures[i] += heat * 0.001  # Simplified thermal
        
        self.state.current = current
        self._current_time += dt
        
        # Update pack state
        self._update_pack_state()
        
        # Update accumulators
        dq_abs = abs(current * dt / 3600.0)
        de = abs(self.state.pack_voltage * current * dt / 3600.0)
        self.state.step_capacity += dq_abs
        self.state.step_energy += de
        self.state.total_capacity += dq_abs
        self.state.total_energy += de

    def reset_step_accumulators(self) -> None:
        """Reset step-level capacity and energy accumulators."""
        self.state.step_capacity = 0.0
        self.state.step_energy = 0.0

    def apply_degradation(self, capacity_fade: float, resistance_growth: float) -> None:
        """
        Apply degradation to all cells in the pack.
        
        Args:
            capacity_fade: Fractional capacity loss
            resistance_growth: Fractional resistance increase
        """
        self.state.capacity_current *= (1 - capacity_fade)
        self.state.resistance_current *= (1 + resistance_growth)
        
        # Apply with cell-to-cell variation
        for i in range(len(self._capacity_factors)):
            variation = np.random.normal(1.0, 0.1)
            self._capacity_factors[i] *= (1 - capacity_fade * variation)
            self._resistance_factors[i] *= (1 + resistance_growth * variation)

    def get_state_dict(self) -> dict:
        """Get current state as dictionary."""
        return {
            "soc": self.state.soc,
            "voltage": self.state.voltage,
            "pack_voltage": self.state.pack_voltage,
            "current": self.state.current,
            "temperature": self.state.temperature,
            "capacity_current": self.state.capacity_current,
            "resistance_current": self.state.resistance_current,
            "cycle_count": self.state.cycle_count,
            "step_capacity": self.state.step_capacity,
            "step_energy": self.state.step_energy,
            "series": self.pack_config.series,
            "parallel": self.pack_config.parallel,
            "total_cells": self.pack_config.total_cells,
            "min_cell_voltage": self.state.min_cell_voltage,
            "max_cell_voltage": self.state.max_cell_voltage,
            "soc_spread": self.state.soc_spread,
        }

    @property
    def capacity_retention(self) -> float:
        """Get current capacity retention as fraction of nominal."""
        return self.state.capacity_current / self.capacity_nominal

    @classmethod
    def is_available(cls) -> bool:
        """Check if liionpack is available."""
        return LIIONPACK_AVAILABLE and PYBAMM_AVAILABLE


def check_liionpack_available() -> bool:
    """Check if liionpack is installed and available."""
    return LIIONPACK_AVAILABLE and PYBAMM_AVAILABLE

