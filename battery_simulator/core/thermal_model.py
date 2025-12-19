"""Thermal model for battery temperature simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalParameters:
    """Thermal model parameters."""

    mass: float = 0.050  # Cell mass (kg)
    specific_heat: float = 1000.0  # Specific heat capacity (J/kg·K)
    surface_area: float = 0.004  # Surface area (m²)
    convection_coefficient: float = 10.0  # Convection coefficient (W/m²·K)
    ambient_temperature: float = 25.0  # Ambient temperature (°C)


class ThermalModel:
    """
    Thermal model for battery temperature prediction.
    
    Implements lumped thermal model:
    m * Cp * dT/dt = Q_gen - Q_loss
    
    Where:
    - Q_gen = I²R (Joule heating) + I*η (entropic heating)
    - Q_loss = h*A*(T - T_amb) (convection)
    """

    def __init__(self, params: ThermalParameters | None = None):
        """
        Initialize thermal model.
        
        Args:
            params: Thermal parameters (uses defaults if None)
        """
        self.params = params or ThermalParameters()
        self.temperature = self.params.ambient_temperature

    def update(
        self,
        current: float,
        resistance: float,
        dt: float,
        entropic_coefficient: float = 0.0,
    ) -> float:
        """
        Update temperature for one time step.
        
        Args:
            current: Applied current (A)
            resistance: Internal resistance (Ohm)
            dt: Time step (seconds)
            entropic_coefficient: dOCV/dT coefficient (V/K)
            
        Returns:
            New temperature (°C)
        """
        p = self.params

        # Heat generation
        q_joule = current**2 * resistance  # Joule heating (W)
        q_entropic = abs(current) * entropic_coefficient * (self.temperature + 273.15)  # Entropic
        q_gen = q_joule + q_entropic

        # Heat dissipation (convection)
        q_loss = p.convection_coefficient * p.surface_area * (self.temperature - p.ambient_temperature)

        # Temperature change
        dt_temp = (q_gen - q_loss) * dt / (p.mass * p.specific_heat)
        self.temperature += dt_temp

        # Limit to reasonable range
        self.temperature = np.clip(self.temperature, -40.0, 100.0)

        return self.temperature

    def set_ambient_temperature(self, temperature: float) -> None:
        """Set ambient temperature."""
        self.params.ambient_temperature = temperature

    def reset(self, temperature: float | None = None) -> None:
        """Reset to ambient or specified temperature."""
        self.temperature = temperature or self.params.ambient_temperature


class TemperatureProfile:
    """
    Generate temperature profiles for testing.
    
    Supports:
    - Constant temperature
    - Sinusoidal (daily cycling)
    - Stepped profiles
    """

    def __init__(
        self,
        profile_type: str = "constant",
        base_temperature: float = 25.0,
        **kwargs,
    ):
        """
        Initialize temperature profile.
        
        Args:
            profile_type: 'constant', 'sinusoidal', or 'stepped'
            base_temperature: Base/mean temperature (°C)
            **kwargs: Profile-specific parameters
        """
        self.profile_type = profile_type
        self.base_temperature = base_temperature
        self.params = kwargs

    def get_temperature(self, time: float, cycle: int = 0) -> float:
        """
        Get ambient temperature at given time.
        
        Args:
            time: Time in seconds from start
            cycle: Current cycle number
            
        Returns:
            Ambient temperature (°C)
        """
        if self.profile_type == "constant":
            return self.base_temperature

        elif self.profile_type == "sinusoidal":
            amplitude = self.params.get("amplitude", 10.0)
            period = self.params.get("period", 86400.0)  # Default: daily cycle
            return self.base_temperature + amplitude * np.sin(2 * np.pi * time / period)

        elif self.profile_type == "stepped":
            schedule = self.params.get("schedule", [])
            for entry in schedule:
                cycle_range = entry.get("cycles", [0, float("inf")])
                if cycle_range[0] <= cycle <= cycle_range[1]:
                    return entry.get("temperature", self.base_temperature)
            return self.base_temperature

        else:
            return self.base_temperature

    @classmethod
    def constant(cls, temperature: float = 25.0) -> "TemperatureProfile":
        """Create constant temperature profile."""
        return cls(profile_type="constant", base_temperature=temperature)

    @classmethod
    def daily_cycle(
        cls, mean: float = 25.0, amplitude: float = 10.0
    ) -> "TemperatureProfile":
        """Create daily sinusoidal temperature cycle."""
        return cls(
            profile_type="sinusoidal",
            base_temperature=mean,
            amplitude=amplitude,
            period=86400.0,
        )

    @classmethod
    def stepped(cls, schedule: list[dict]) -> "TemperatureProfile":
        """
        Create stepped temperature profile.
        
        Args:
            schedule: List of dicts with 'cycles' (range) and 'temperature' keys
        """
        return cls(
            profile_type="stepped",
            base_temperature=25.0,
            schedule=schedule,
        )

