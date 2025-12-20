"""Main battery simulator orchestrator."""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from battery_simulator.core.battery_model import BatteryModel
from battery_simulator.core.degradation import DegradationModel, FailureModel
from battery_simulator.core.thermal_model import ThermalModel, TemperatureProfile

if TYPE_CHECKING:
    from battery_simulator.chemistry.base_chemistry import BaseChemistry
    from battery_simulator.protocols.base_protocol import BaseProtocol


class TimingMode(str, Enum):
    """Simulation timing modes."""

    REAL_TIME = "real_time"  # Generate at actual time intervals
    ACCELERATED = "accelerated"  # Generate faster than real time
    INSTANT = "instant"  # Generate entire test immediately


class SimulationMode(str, Enum):
    """Simulation fidelity modes."""

    FAST = "fast"  # Fast empirical model (default, existing behavior)
    HIGH_FIDELITY = "high_fidelity"  # PyBaMM-based physics model
    PACK = "pack"  # Pack simulation with liionpack


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""

    timing_mode: TimingMode = TimingMode.INSTANT
    speed_factor: float = 1.0  # For accelerated mode
    data_rate: float = 1.0  # Hz (points per second)
    enable_degradation: bool = True
    enable_thermal: bool = True
    enable_failures: bool = False
    failure_probability: float = 0.01
    noise_voltage: float = 0.001  # V std dev
    noise_current: float = 0.005  # A std dev
    noise_temperature: float = 0.5  # °C std dev
    random_seed: Optional[int] = None
    
    # Simulation mode settings
    simulation_mode: SimulationMode = SimulationMode.FAST
    pybamm_model: str = "SPMe"  # SPM, SPMe, or DFN for high-fidelity mode
    pybamm_parameter_set: Optional[str] = None  # PyBaMM parameter set name
    
    # Pack simulation settings
    pack_config: Optional[Dict[str, Any]] = None  # {"series": 14, "parallel": 4}


@dataclass
class SimulationResults:
    """Results from a simulation run."""

    test_id: str
    chemistry: str
    protocol_name: str
    cycles_completed: int
    start_time: datetime
    end_time: datetime
    capacity_initial: float
    capacity_final: float
    capacity_retention: float
    resistance_initial: float
    resistance_final: float
    energy_throughput: float
    equivalent_full_cycles: float
    failure_mode: Optional[str] = None
    failure_cycle: Optional[int] = None
    cycle_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_file: Optional[str] = None
    simulation_mode: str = "fast"
    backend_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert results to dictionary."""
        return {
            "test_metadata": {
                "test_id": self.test_id,
                "chemistry": self.chemistry,
                "protocol_name": self.protocol_name,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_cycles": self.cycles_completed,
                "data_file": self.data_file,
                "simulation_mode": self.simulation_mode,
                "backend_info": self.backend_info,
            },
            "results_summary": {
                "cycles_completed": self.cycles_completed,
                "capacity_retention": self.capacity_retention,
                "capacity_fade_rate": (1 - self.capacity_retention) / max(self.cycles_completed, 1),
                "resistance_growth": (self.resistance_final - self.resistance_initial)
                / self.resistance_initial,
                "energy_throughput": self.energy_throughput,
                "equivalent_full_cycles": self.equivalent_full_cycles,
                "failure_mode": self.failure_mode,
            },
        }


class BatterySimulator:
    """
    Main simulator orchestrator for battery cycling tests.
    
    Coordinates:
    - Battery model (voltage, SOC, etc.)
    - Degradation model (capacity fade, resistance growth)
    - Thermal model (temperature evolution)
    - Protocol execution (charge, discharge, rest steps)
    - Data output (CSV, specific formats)
    
    Supports multiple simulation modes:
    - FAST: Empirical model (default, fastest)
    - HIGH_FIDELITY: PyBaMM physics-based models (SPM, SPMe, DFN)
    - PACK: Pack-level simulation with liionpack
    """

    def __init__(
        self,
        chemistry: Union[BaseChemistry, str],
        capacity: Optional[float] = None,
        temperature: float = 25.0,
        config: Optional[SimulationConfig] = None,
        mode: Optional[SimulationMode] = None,
        pybamm_model: Optional[str] = None,
        pack_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the battery simulator.
        
        Args:
            chemistry: Chemistry configuration or name string
            capacity: Nominal capacity in Ah (uses chemistry default if None)
            temperature: Initial temperature (°C)
            config: Simulation configuration
            mode: Simulation mode (FAST, HIGH_FIDELITY, PACK) - overrides config
            pybamm_model: PyBaMM model type (SPM, SPMe, DFN) - overrides config
            pack_config: Pack configuration dict - overrides config
        """
        self.config = config or SimulationConfig()
        
        # Handle mode overrides
        if mode is not None:
            self.config.simulation_mode = mode
        if pybamm_model is not None:
            self.config.pybamm_model = pybamm_model
        if pack_config is not None:
            self.config.pack_config = pack_config
        
        # Handle chemistry by name
        if isinstance(chemistry, str):
            self._chemistry_name = chemistry
            from battery_simulator.chemistry import Chemistry
            chemistry = Chemistry.from_name(chemistry)
        else:
            self._chemistry_name = getattr(chemistry, 'name', 'Unknown')

        self.chemistry = chemistry
        
        # Initialize random number generator
        self.rng = np.random.default_rng(self.config.random_seed)
        
        # Initialize appropriate backend based on mode
        self._init_backend(capacity, temperature)

        # Initialize degradation model
        self.degradation = DegradationModel(
            chemistry=chemistry,
            nominal_capacity=self.battery.capacity_nominal,
        )

        # Initialize failure model
        self.failure = FailureModel(
            enable=self.config.enable_failures,
            failure_probability=self.config.failure_probability,
            seed=self.config.random_seed,
        )

        # Initialize thermal model
        self.thermal = ThermalModel()
        self.temperature_profile = TemperatureProfile.constant(temperature)

        # Data buffers
        self._data_buffer: list[dict] = []
        self._cycle_summary: list[dict] = []

        # Callbacks for streaming
        self._callbacks: list[Callable[[dict], None]] = []

        # State
        self._test_time = 0.0
        self._timestamp = datetime.now()
        self._current_cycle = 0
        self._current_step = 0
        self._last_cycle_metrics: dict = {}
    
    def _init_backend(self, capacity: Optional[float], temperature: float) -> None:
        """Initialize the appropriate simulation backend based on mode."""
        mode = self.config.simulation_mode
        
        if mode == SimulationMode.FAST:
            # Standard empirical model
            self.battery = BatteryModel(
                chemistry=self.chemistry,
                capacity=capacity,
                temperature=temperature,
            )
            self._backend_type = "fast"
            
        elif mode == SimulationMode.HIGH_FIDELITY:
            # PyBaMM-based model
            try:
                from battery_simulator.core.pybamm_model import (
                    PyBaMMModel, 
                    PyBaMMModelType,
                    PYBAMM_AVAILABLE
                )
                
                if not PYBAMM_AVAILABLE:
                    raise ImportError("PyBaMM not available")
                
                # Map model string to enum
                model_map = {
                    "SPM": PyBaMMModelType.SPM,
                    "SPMe": PyBaMMModelType.SPME,
                    "SPME": PyBaMMModelType.SPME,
                    "DFN": PyBaMMModelType.DFN,
                }
                model_type = model_map.get(
                    self.config.pybamm_model, PyBaMMModelType.SPME
                )
                
                self.battery = PyBaMMModel(
                    chemistry=self.chemistry,
                    capacity=capacity,
                    temperature=temperature,
                    model_type=model_type,
                    parameter_set=self.config.pybamm_parameter_set,
                )
                self._backend_type = "pybamm"
                
            except ImportError:
                # Fallback to fast mode
                import warnings
                warnings.warn(
                    "PyBaMM not available, falling back to fast empirical model. "
                    "Install with: pip install pybamm"
                )
                self.battery = BatteryModel(
                    chemistry=self.chemistry,
                    capacity=capacity,
                    temperature=temperature,
                )
                self._backend_type = "fast"
                self.config.simulation_mode = SimulationMode.FAST
                
        elif mode == SimulationMode.PACK:
            # Pack simulation with liionpack
            try:
                from battery_simulator.core.pack_simulator import (
                    PackSimulator,
                    LIIONPACK_AVAILABLE
                )
                
                if not LIIONPACK_AVAILABLE:
                    raise ImportError("liionpack not available")
                
                pack_cfg = self.config.pack_config or {"series": 1, "parallel": 1}
                
                self.battery = PackSimulator(
                    chemistry=self.chemistry,
                    capacity=capacity,
                    temperature=temperature,
                    series=pack_cfg.get("series", 1),
                    parallel=pack_cfg.get("parallel", 1),
                    cell_variation=pack_cfg.get("cell_variation", 0.02),
                )
                self._backend_type = "pack"
                
            except ImportError:
                # Fallback to fast mode
                import warnings
                warnings.warn(
                    "liionpack not available, falling back to fast empirical model. "
                    "Install with: pip install liionpack"
                )
                self.battery = BatteryModel(
                    chemistry=self.chemistry,
                    capacity=capacity,
                    temperature=temperature,
                )
                self._backend_type = "fast"
                self.config.simulation_mode = SimulationMode.FAST
        else:
            # Default to fast model
            self.battery = BatteryModel(
                chemistry=self.chemistry,
                capacity=capacity,
                temperature=temperature,
            )
            self._backend_type = "fast"
    
    @property
    def backend_type(self) -> str:
        """Get the currently active backend type."""
        return self._backend_type
    
    @classmethod
    def check_backend_available(cls, mode: SimulationMode) -> bool:
        """
        Check if a specific backend is available.
        
        Args:
            mode: Simulation mode to check
            
        Returns:
            True if the backend is available
        """
        if mode == SimulationMode.FAST:
            return True
        elif mode == SimulationMode.HIGH_FIDELITY:
            try:
                from battery_simulator.core.pybamm_model import PYBAMM_AVAILABLE
                return PYBAMM_AVAILABLE
            except ImportError:
                return False
        elif mode == SimulationMode.PACK:
            try:
                from battery_simulator.core.pack_simulator import LIIONPACK_AVAILABLE
                return LIIONPACK_AVAILABLE
            except ImportError:
                return False
        return False

    def set_temperature_profile(self, profile: TemperatureProfile) -> None:
        """Set ambient temperature profile."""
        self.temperature_profile = profile

    def on_data(self, callback: Callable[[dict], None]) -> None:
        """Register callback for live data updates."""
        self._callbacks.append(callback)

    def run(
        self,
        protocol: BaseProtocol,
        output_path: str | Path | None = None,
        output_format: str = "generic",
        show_progress: bool = True,
    ) -> SimulationResults:
        """
        Run complete simulation.
        
        Args:
            protocol: Test protocol to execute
            output_path: Where to write results (None = no file output)
            output_format: Output format ('generic', 'arbin', 'neware', 'biologic')
            show_progress: Show progress bar
            
        Returns:
            SimulationResults object with summary statistics
        """
        start_time = datetime.now()
        self._timestamp = start_time
        self._test_time = 0.0
        self._current_cycle = 0

        # Store initial state
        initial_capacity = self.battery.capacity_nominal
        initial_resistance = self.chemistry.resistance_initial

        # Initialize output writer
        writer = None
        if output_path:
            from battery_simulator.outputs import get_writer

            writer = get_writer(output_format, output_path)

        # Run cycles
        cycles_iter = range(protocol.cycles)
        if show_progress:
            cycles_iter = tqdm(cycles_iter, desc="Simulating cycles", unit="cycle")

        for cycle in cycles_iter:
            self._current_cycle = cycle + 1

            # Check for failure
            if self.config.enable_failures:
                failed, failure_type = self.failure.check_failure(cycle)
                if failed:
                    break

            # Update ambient temperature
            ambient_temp = self.temperature_profile.get_temperature(
                self._test_time, self._current_cycle
            )
            self.thermal.set_ambient_temperature(ambient_temp)

            # Execute cycle
            cycle_data = self._simulate_cycle(protocol, cycle)

            # Write cycle data
            if writer:
                writer.write_data(cycle_data)

            # Apply degradation after cycle using actual cycle conditions
            if self.config.enable_degradation:
                metrics = getattr(self, "_last_cycle_metrics", {})
                self._apply_cycle_degradation(
                    cycle,
                    c_rate_charge=metrics.get("c_rate_charge", 1.0),
                    c_rate_discharge=metrics.get("c_rate_discharge", 1.0),
                    dod=metrics.get("dod", 1.0),
                    cycle_time_hours=metrics.get("cycle_time_hours", 2.0),
                    avg_soc=metrics.get("avg_soc", 0.5),
                )

            # Record cycle summary
            self._record_cycle_summary(cycle, cycle_data)

            # Check end conditions
            if self._check_end_conditions(protocol):
                break

            # Handle timing
            self._handle_timing()

        # Finalize
        end_time = datetime.now()
        if writer:
            writer.close()

        # Build backend info
        backend_info = {
            "type": self._backend_type,
        }
        if self._backend_type == "pybamm":
            backend_info["model"] = self.config.pybamm_model
            backend_info["parameter_set"] = self.config.pybamm_parameter_set
        elif self._backend_type == "pack":
            backend_info["pack_config"] = self.config.pack_config
        
        # Build results
        results = SimulationResults(
            test_id=f"SIM-{start_time.strftime('%Y%m%d-%H%M%S')}",
            chemistry=self.chemistry.name,
            protocol_name=protocol.name,
            cycles_completed=self._current_cycle,
            start_time=start_time,
            end_time=end_time,
            capacity_initial=initial_capacity,
            capacity_final=self.battery.state.capacity_current,
            capacity_retention=self.battery.capacity_retention,
            resistance_initial=initial_resistance,
            resistance_final=self.battery.state.resistance_current,
            energy_throughput=self.battery.state.total_energy,
            equivalent_full_cycles=self.degradation.state.equivalent_full_cycles,
            failure_mode=self.failure.failure_type,
            failure_cycle=self.failure.failure_cycle,
            cycle_summary=pd.DataFrame(self._cycle_summary),
            data_file=str(output_path) if output_path else None,
            simulation_mode=self.config.simulation_mode.value,
            backend_info=backend_info,
        )

        return results

    def _simulate_cycle(self, protocol: BaseProtocol, cycle_index: int) -> pd.DataFrame:
        """Simulate a single cycle."""
        cycle_data = []
        self._current_step = 0

        # Track cycle-level metrics
        charge_capacity = 0.0
        discharge_capacity = 0.0
        charge_energy = 0.0
        discharge_energy = 0.0
        
        # Track C-rates for degradation model
        charge_c_rates = []
        discharge_c_rates = []
        cycle_start_time = self._test_time
        soc_values = []

        for step in protocol.steps:
            self._current_step += 1
            self.battery.reset_step_accumulators()

            step_data = self._simulate_step(step, cycle_index)
            cycle_data.append(step_data)
            
            # Track SOC for average calculation
            if not step_data.empty:
                soc_values.extend(step_data["state_of_charge"].tolist())

            # Accumulate metrics and track C-rates
            if step.is_charge:
                charge_capacity += self.battery.state.step_capacity
                charge_energy += self.battery.state.step_energy
                # Extract C-rate from step if available
                if hasattr(step, "c_rate"):
                    charge_c_rates.append(step.c_rate)
            elif step.is_discharge:
                discharge_capacity += self.battery.state.step_capacity
                discharge_energy += self.battery.state.step_energy
                if hasattr(step, "c_rate"):
                    discharge_c_rates.append(step.c_rate)

        # Store cycle metrics for degradation calculation
        cycle_time_hours = (self._test_time - cycle_start_time) / 3600.0
        self._last_cycle_metrics = {
            "c_rate_charge": np.mean(charge_c_rates) if charge_c_rates else 1.0,
            "c_rate_discharge": np.mean(discharge_c_rates) if discharge_c_rates else 1.0,
            "dod": (charge_capacity + discharge_capacity) / (2 * self.battery.capacity_nominal) if self.battery.capacity_nominal > 0 else 1.0,
            "cycle_time_hours": max(cycle_time_hours, 0.1),  # Minimum 6 minutes
            "avg_soc": np.mean(soc_values) if soc_values else 0.5,
        }

        # Combine step data
        if cycle_data:
            return pd.concat(cycle_data, ignore_index=True)
        return pd.DataFrame()

    def _simulate_step(self, step: Any, cycle_index: int) -> pd.DataFrame:
        """
        Simulate a single protocol step.
        
        Args:
            step: Protocol step (charge, discharge, rest)
            cycle_index: Current cycle number
            
        Returns:
            DataFrame with time-series data
        """
        data_points = []
        dt = 1.0 / self.config.data_rate
        step_time = 0.0

        while not step.is_complete(self.battery, step_time):
            # Get current from step
            current = step.get_current(self.battery)

            # Update battery state
            self.battery.update_state(current, dt)

            # Update thermal model
            if self.config.enable_thermal:
                self.thermal.update(current, self.battery.state.resistance_current, dt)
                self.battery.state.temperature = self.thermal.temperature

            # Add measurement noise
            voltage = self.battery.state.voltage + self.rng.normal(0, self.config.noise_voltage)
            measured_current = current + self.rng.normal(0, self.config.noise_current)
            temperature = self.battery.state.temperature + self.rng.normal(
                0, self.config.noise_temperature
            )

            # Record data point
            data_point = {
                "timestamp": self._timestamp.isoformat(),
                "test_time": self._test_time,
                "cycle_index": cycle_index + 1,
                "step_index": self._current_step,
                "step_type": step.step_type,
                "current": measured_current,
                "voltage": voltage,
                "capacity": self.battery.state.step_capacity,
                "energy": self.battery.state.step_energy,
                "temperature": temperature,
                "state_of_charge": self.battery.state.soc,
                "power": voltage * measured_current,
                "internal_resistance": self.battery.state.resistance_current,
            }
            data_points.append(data_point)

            # Emit to callbacks
            for callback in self._callbacks:
                callback(data_point)

            # Update time
            step_time += dt
            self._test_time += dt
            self._timestamp += timedelta(seconds=dt)

        return pd.DataFrame(data_points)

    def _apply_cycle_degradation(
        self,
        cycle_index: int,
        c_rate_charge: float = 1.0,
        c_rate_discharge: float = 1.0,
        dod: float = 1.0,
        cycle_time_hours: float = 2.0,
        avg_soc: float = 0.5,
    ) -> None:
        """
        Apply degradation after a cycle using the semi-empirical model.
        
        Args:
            cycle_index: Current cycle index (0-based)
            c_rate_charge: Charge C-rate used
            c_rate_discharge: Discharge C-rate used
            dod: Depth of discharge achieved
            cycle_time_hours: Duration of the cycle in hours
            avg_soc: Average SOC during the cycle
        """
        # Use the new update_from_cycle method that handles both cyclic and calendar aging
        deg_info = self.degradation.update_from_cycle(
            cycle_number=cycle_index + 1,
            c_rate_charge=c_rate_charge,
            c_rate_discharge=c_rate_discharge,
            temperature=self.battery.state.temperature,
            dod=dod,
            cycle_time_hours=cycle_time_hours,
            avg_soc=avg_soc,
            capacity_throughput=self.battery.state.step_capacity * 2,  # Charge + discharge
        )

        # Apply degradation to battery
        total_cap_fade = deg_info["cap_loss_cyc"] + deg_info["cap_loss_cal"]
        self.battery.apply_degradation(
            total_cap_fade,
            self.degradation.state.total_resistance_growth / max(cycle_index + 1, 1)
        )

    def _record_cycle_summary(self, cycle_index: int, cycle_data: pd.DataFrame) -> None:
        """Record summary statistics for a cycle."""
        if cycle_data.empty:
            return

        # Calculate summary metrics
        charge_data = cycle_data[cycle_data["current"] > 0.01]
        discharge_data = cycle_data[cycle_data["current"] < -0.01]

        charge_capacity = charge_data["capacity"].max() if not charge_data.empty else 0
        discharge_capacity = discharge_data["capacity"].max() if not discharge_data.empty else 0
        charge_energy = charge_data["energy"].max() if not charge_data.empty else 0
        discharge_energy = discharge_data["energy"].max() if not discharge_data.empty else 0

        coulombic_eff = discharge_capacity / charge_capacity if charge_capacity > 0 else 0
        energy_eff = discharge_energy / charge_energy if charge_energy > 0 else 0

        summary = {
            "cycle_index": cycle_index + 1,
            "charge_capacity_ah": charge_capacity,
            "discharge_capacity_ah": discharge_capacity,
            "charge_energy_wh": charge_energy,
            "discharge_energy_wh": discharge_energy,
            "coulombic_efficiency": coulombic_eff,
            "energy_efficiency": energy_eff,
            "max_voltage": cycle_data["voltage"].max(),
            "min_voltage": cycle_data["voltage"].min(),
            "avg_temperature": cycle_data["temperature"].mean(),
            "capacity_retention": self.battery.capacity_retention,
            "internal_resistance": self.battery.state.resistance_current,
        }
        self._cycle_summary.append(summary)

    def _check_end_conditions(self, protocol: BaseProtocol) -> bool:
        """Check if any end conditions are met."""
        # Check capacity retention
        if hasattr(protocol, "end_capacity_retention"):
            if self.battery.capacity_retention < protocol.end_capacity_retention:
                return True

        # Check for failure
        if self.failure.has_failed:
            return True

        return False

    def _handle_timing(self) -> None:
        """Handle timing between data points based on mode."""
        if self.config.timing_mode == TimingMode.REAL_TIME:
            time_module.sleep(1.0 / self.config.data_rate)
        elif self.config.timing_mode == TimingMode.ACCELERATED:
            time_module.sleep(1.0 / (self.config.data_rate * self.config.speed_factor))
        # INSTANT mode: no delay

    def get_state(self) -> dict:
        """Get current simulator state."""
        return {
            "battery": self.battery.get_state_dict(),
            "degradation": self.degradation.get_state_dict(),
            "cycle": self._current_cycle,
            "step": self._current_step,
            "test_time": self._test_time,
        }

