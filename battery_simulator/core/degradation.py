"""
Semi-empirical degradation model for battery aging simulation.

Implements Arrhenius-based temperature dependence and C-rate effects
for both cycle and calendar aging according to:

Cycle aging:
    Q_loss,cyc = k_cyc * N^z * exp(-E_a,cyc/R * (1/T - 1/T_ref)) * (C_ch/C_ref)^alpha

Calendar aging:
    Q_loss,cal = k_cal * t^b * exp(-E_a,cal/R * (1/T - 1/T_ref)) * exp(beta_SOC * (SOC - SOC_ref))

Total loss:
    Q_loss,total = Q_loss,cyc + Q_loss,cal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from battery_simulator.chemistry.base_chemistry import BaseChemistry


# Gas constant (J/mol·K)
R_GAS = 8.314


@dataclass
class DegradationState:
    """Track degradation state over time."""

    # Capacity loss fractions
    total_capacity_loss: float = 0.0
    cyclic_capacity_loss: float = 0.0
    calendar_capacity_loss: float = 0.0
    
    # Resistance growth fraction
    total_resistance_growth: float = 0.0
    
    # Cycle counting
    equivalent_full_cycles: float = 0.0
    total_ah_throughput: float = 0.0
    
    # Time tracking (for calendar aging)
    total_time_hours: float = 0.0
    
    # Cumulative stress factors (for analysis)
    cumulative_temperature_stress: float = 0.0
    cumulative_rate_stress: float = 0.0
    
    # History for detailed analysis
    cycle_history: list = field(default_factory=list)


class DegradationModel:
    """
    Semi-empirical degradation model with Arrhenius temperature dependence.
    
    This model accurately captures:
    - Temperature effects via Arrhenius equation
    - C-rate effects via power law
    - SOC effects on calendar aging
    - Chemistry-specific degradation characteristics
    
    The model uses incremental calculations to track degradation over
    each time step, allowing for varying conditions throughout the test.
    """

    def __init__(self, chemistry: BaseChemistry, nominal_capacity: float):
        """
        Initialize degradation model.
        
        Args:
            chemistry: Chemistry configuration with degradation parameters
            nominal_capacity: Nominal cell capacity in Ah
        """
        self.chemistry = chemistry
        self.nominal_capacity = nominal_capacity
        self.state = DegradationState()
        
        # Get degradation parameters from chemistry
        self.params = chemistry.degradation_params
        
        # Pre-compute constants for efficiency
        self._inv_r = 1.0 / R_GAS
        self._inv_t_ref = 1.0 / self.params.t_ref

    def calculate_arrhenius_factor(self, temperature_c: float, activation_energy: float) -> float:
        """
        Calculate Arrhenius temperature acceleration factor.
        
        Factor = exp(-E_a/R * (1/T - 1/T_ref))
        
        At T = T_ref (25°C), factor = 1.0
        At T > T_ref, factor > 1.0 (accelerated aging)
        At T < T_ref, factor < 1.0 (slower aging)
        
        Args:
            temperature_c: Temperature in Celsius
            activation_energy: Activation energy in J/mol
            
        Returns:
            Arrhenius acceleration factor (dimensionless)
        """
        t_kelvin = temperature_c + 273.15
        exponent = -activation_energy * self._inv_r * (1.0 / t_kelvin - self._inv_t_ref)
        return np.exp(exponent)

    def calculate_c_rate_factor(self, c_rate: float) -> float:
        """
        Calculate C-rate stress factor.
        
        Factor = (C_rate / C_ref)^alpha
        
        Args:
            c_rate: Charge or discharge C-rate
            
        Returns:
            C-rate stress factor (dimensionless)
        """
        # Ensure minimum C-rate to avoid division issues
        c_rate = max(c_rate, 0.01)
        return (c_rate / self.params.c_ref) ** self.params.alpha

    def calculate_soc_factor(self, soc: float) -> float:
        """
        Calculate SOC stress factor for calendar aging.
        
        Factor = exp(beta_SOC * (SOC - SOC_ref))
        
        For beta_SOC > 0: high SOC increases aging
        For beta_SOC < 0: low SOC increases aging
        
        Args:
            soc: State of charge (0-1)
            
        Returns:
            SOC stress factor (dimensionless)
        """
        delta_soc = soc - self.params.soc_ref
        return np.exp(self.params.beta_soc * delta_soc)

    def calculate_cycle_degradation(
        self,
        cycle_number: int,
        c_rate_charge: float = 1.0,
        c_rate_discharge: float = 1.0,
        temperature: float = 25.0,
        dod: float = 1.0,
    ) -> tuple[float, float]:
        """
        Calculate incremental degradation for a single cycle.
        
        Uses the semi-empirical model:
        Q_loss,cyc = k_cyc * N^z * f_temp * f_crate * f_dod
        
        For incremental update, we calculate:
        dQ/dN = k_cyc * z * N^(z-1) * f_temp * f_crate * f_dod
        
        Args:
            cycle_number: Current cycle number (1-based)
            c_rate_charge: Charge C-rate
            c_rate_discharge: Discharge C-rate  
            temperature: Average cycle temperature (°C)
            dod: Depth of discharge (0-1)
            
        Returns:
            Tuple of (capacity_fade_increment, resistance_growth_increment)
        """
        if cycle_number < 1:
            return 0.0, 0.0
            
        # Temperature factor (Arrhenius)
        f_temp = self.calculate_arrhenius_factor(temperature, self.params.e_a_cyc)
        
        # C-rate factor (use average of charge and discharge rates)
        avg_c_rate = (c_rate_charge + c_rate_discharge) / 2.0
        f_crate = self.calculate_c_rate_factor(avg_c_rate)
        
        # DOD factor (deeper cycles cause more damage)
        # Using quadratic relationship: f_dod = 0.2 + 0.8 * DOD^2
        f_dod = 0.2 + 0.8 * dod ** 2
        
        # Calculate incremental capacity loss
        # dQ/dN = k * z * N^(z-1) for Q = k * N^z
        z = self.params.z
        if cycle_number == 1:
            # For first cycle, use total loss at N=1
            capacity_loss = self.params.k_cyc * f_temp * f_crate * f_dod
        else:
            # Incremental: derivative of N^z is z * N^(z-1)
            capacity_loss = self.params.k_cyc * z * (cycle_number ** (z - 1)) * f_temp * f_crate * f_dod
        
        # Resistance growth (typically faster than capacity fade)
        f_temp_r = self.calculate_arrhenius_factor(temperature, self.params.e_a_resistance)
        resistance_growth = self.params.k_resistance * z * (cycle_number ** (z - 1)) * f_temp_r * f_crate * f_dod
        
        # Store stress factors for analysis
        self.state.cumulative_temperature_stress += f_temp
        self.state.cumulative_rate_stress += f_crate
        
        return float(capacity_loss), float(resistance_growth)

    def calculate_calendar_degradation(
        self,
        time_hours: float,
        temperature: float = 25.0,
        soc: float = 0.5,
    ) -> tuple[float, float]:
        """
        Calculate incremental calendar (storage) degradation.
        
        Uses the semi-empirical model:
        Q_loss,cal = k_cal * t^b * f_temp * f_soc
        
        For incremental update with time step dt:
        dQ = k_cal * b * t^(b-1) * f_temp * f_soc * dt
        
        Args:
            time_hours: Time step in hours
            temperature: Storage temperature (°C)
            soc: State of charge during storage (0-1)
            
        Returns:
            Tuple of (capacity_fade_increment, resistance_growth_increment)
        """
        if time_hours <= 0:
            return 0.0, 0.0
            
        # Convert hours to days for the model (parameters calibrated for days)
        time_days = time_hours / 24.0
        total_time_days = self.state.total_time_hours / 24.0 + time_days
        
        # Temperature factor (Arrhenius)
        f_temp = self.calculate_arrhenius_factor(temperature, self.params.e_a_cal)
        
        # SOC factor
        f_soc = self.calculate_soc_factor(soc)
        
        # Calculate incremental capacity loss
        b = self.params.b
        if total_time_days <= time_days:
            # First time step
            capacity_loss = self.params.k_cal * (time_days ** b) * f_temp * f_soc
        else:
            # Incremental: use derivative
            # dQ/dt = k * b * t^(b-1) for Q = k * t^b
            prev_time_days = self.state.total_time_hours / 24.0
            prev_loss = self.params.k_cal * (prev_time_days ** b) * f_temp * f_soc
            new_loss = self.params.k_cal * (total_time_days ** b) * f_temp * f_soc
            capacity_loss = new_loss - prev_loss
        
        # Calendar resistance growth (typically slower than cyclic)
        f_temp_r = self.calculate_arrhenius_factor(temperature, self.params.e_a_resistance)
        resistance_growth = capacity_loss * 0.5 * f_temp_r / f_temp
        
        return float(max(0, capacity_loss)), float(max(0, resistance_growth))

    def update_from_cycle(
        self,
        cycle_number: int,
        c_rate_charge: float = 1.0,
        c_rate_discharge: float = 1.0,
        temperature: float = 25.0,
        dod: float = 1.0,
        cycle_time_hours: float = 2.0,
        avg_soc: float = 0.5,
        capacity_throughput: float = 0.0,
    ) -> dict:
        """
        Update degradation state after completing a cycle.
        
        Combines both cyclic and calendar aging contributions.
        
        Args:
            cycle_number: Current cycle number
            c_rate_charge: Charge C-rate used
            c_rate_discharge: Discharge C-rate used
            temperature: Average temperature during cycle (°C)
            dod: Depth of discharge achieved
            cycle_time_hours: Duration of the cycle in hours
            avg_soc: Average SOC during the cycle (for calendar aging)
            capacity_throughput: Ah throughput this cycle
            
        Returns:
            Dictionary with degradation update details
        """
        # Calculate cyclic degradation
        cap_loss_cyc, res_growth_cyc = self.calculate_cycle_degradation(
            cycle_number=cycle_number,
            c_rate_charge=c_rate_charge,
            c_rate_discharge=c_rate_discharge,
            temperature=temperature,
            dod=dod,
        )
        
        # Calculate calendar degradation for the cycle duration
        cap_loss_cal, res_growth_cal = self.calculate_calendar_degradation(
            time_hours=cycle_time_hours,
            temperature=temperature,
            soc=avg_soc,
        )
        
        # Update state
        self.state.cyclic_capacity_loss += cap_loss_cyc
        self.state.calendar_capacity_loss += cap_loss_cal
        self.state.total_capacity_loss = self.state.cyclic_capacity_loss + self.state.calendar_capacity_loss
        
        self.state.total_resistance_growth += res_growth_cyc + res_growth_cal
        
        self.state.total_time_hours += cycle_time_hours
        self.state.total_ah_throughput += capacity_throughput
        
        if capacity_throughput > 0:
            self.state.equivalent_full_cycles += capacity_throughput / (2 * self.nominal_capacity)
        
        # Store cycle info
        cycle_info = {
            "cycle": cycle_number,
            "temperature": temperature,
            "c_rate_charge": c_rate_charge,
            "c_rate_discharge": c_rate_discharge,
            "dod": dod,
            "cap_loss_cyc": cap_loss_cyc,
            "cap_loss_cal": cap_loss_cal,
            "total_cap_loss": self.state.total_capacity_loss,
            "capacity_retention": self.get_capacity_retention(),
        }
        self.state.cycle_history.append(cycle_info)
        
        return cycle_info

    def get_current_capacity(self) -> float:
        """Get current capacity after degradation."""
        return self.nominal_capacity * (1 - min(self.state.total_capacity_loss, 0.5))

    def get_current_resistance_factor(self) -> float:
        """Get resistance multiplication factor."""
        return 1 + self.state.total_resistance_growth

    def get_capacity_retention(self) -> float:
        """Get capacity retention as fraction (0-1)."""
        return max(0.5, 1 - self.state.total_capacity_loss)

    def get_state_dict(self) -> dict:
        """Get degradation state as dictionary."""
        return {
            "total_capacity_loss": self.state.total_capacity_loss,
            "cyclic_capacity_loss": self.state.cyclic_capacity_loss,
            "calendar_capacity_loss": self.state.calendar_capacity_loss,
            "total_resistance_growth": self.state.total_resistance_growth,
            "equivalent_full_cycles": self.state.equivalent_full_cycles,
            "total_ah_throughput": self.state.total_ah_throughput,
            "total_time_hours": self.state.total_time_hours,
            "capacity_retention": self.get_capacity_retention(),
            "resistance_factor": self.get_current_resistance_factor(),
        }

    def predict_cycle_life(
        self,
        target_retention: float = 0.8,
        temperature: float = 25.0,
        c_rate: float = 1.0,
    ) -> int:
        """
        Predict number of cycles to reach target capacity retention.
        
        Solves: target_retention = 1 - k_cyc * N^z * f_temp * f_crate
        For N: N = ((1 - target_retention) / (k_cyc * f_temp * f_crate))^(1/z)
        
        Args:
            target_retention: Target capacity retention (e.g., 0.8 for 80%)
            temperature: Operating temperature (°C)
            c_rate: Operating C-rate
            
        Returns:
            Predicted cycle count to reach target retention
        """
        f_temp = self.calculate_arrhenius_factor(temperature, self.params.e_a_cyc)
        f_crate = self.calculate_c_rate_factor(c_rate)
        
        loss_target = 1 - target_retention
        denominator = self.params.k_cyc * f_temp * f_crate
        
        if denominator <= 0:
            return 100000  # Very long life
            
        cycles = (loss_target / denominator) ** (1 / self.params.z)
        return int(max(1, cycles))

    def get_temperature_acceleration_factor(self, temperature: float, reference_temp: float = 25.0) -> float:
        """
        Get the acceleration factor for a given temperature relative to reference.
        
        Args:
            temperature: Test temperature (°C)
            reference_temp: Reference temperature (°C)
            
        Returns:
            Acceleration factor (>1 means faster aging)
        """
        f_temp_test = self.calculate_arrhenius_factor(temperature, self.params.e_a_cyc)
        f_temp_ref = self.calculate_arrhenius_factor(reference_temp, self.params.e_a_cyc)
        return f_temp_test / f_temp_ref

    def reset(self) -> None:
        """Reset degradation state to initial conditions."""
        self.state = DegradationState()


class FailureModel:
    """
    Models sudden failure events in batteries.
    
    Failure modes:
    - Internal short circuit
    - Lithium plating
    - Thermal runaway (catastrophic)
    - Connection failure
    """

    def __init__(
        self,
        enable: bool = False,
        failure_probability: float = 0.01,
        seed: int | None = None,
    ):
        """
        Initialize failure model.
        
        Args:
            enable: Whether failure events are enabled
            failure_probability: Probability per 100 cycles
            seed: Random seed for reproducibility
        """
        self.enable = enable
        self.failure_probability = failure_probability
        self.rng = np.random.default_rng(seed)
        self.has_failed = False
        self.failure_type: str | None = None
        self.failure_cycle: int | None = None

    def check_failure(self, cycle_number: int) -> tuple[bool, str | None]:
        """
        Check if a failure event occurs this cycle.
        
        Args:
            cycle_number: Current cycle number
            
        Returns:
            Tuple of (failure_occurred, failure_type)
        """
        if not self.enable or self.has_failed:
            return False, None

        # Check probability (scaled per cycle)
        prob_per_cycle = self.failure_probability / 100.0
        if self.rng.random() < prob_per_cycle:
            self.has_failed = True
            self.failure_cycle = cycle_number

            # Select failure type
            failure_types = ["lithium_plating", "internal_short", "contact_loss"]
            weights = [0.5, 0.3, 0.2]
            self.failure_type = self.rng.choice(failure_types, p=weights)

            return True, self.failure_type

        return False, None

    def get_failure_effect(self) -> dict:
        """Get the effect of the current failure mode."""
        effects = {
            "lithium_plating": {
                "capacity_drop": 0.10,  # 10% sudden capacity loss
                "resistance_increase": 0.20,
            },
            "internal_short": {
                "voltage_drop": 0.3,  # 0.3V drop
                "self_discharge_rate": 0.1,  # 10%/day self-discharge
            },
            "contact_loss": {
                "intermittent": True,
                "resistance_spike": 5.0,  # 5x resistance increase
            },
        }
        return effects.get(self.failure_type, {})
