"""Noise generation utilities for realistic measurement simulation."""

from __future__ import annotations

import numpy as np


class NoiseGenerator:
    """
    Generate realistic measurement noise for battery test data.
    
    Supports:
    - Gaussian noise (most common)
    - Drift noise (slow variations)
    - Quantization noise (ADC effects)
    - Occasional spikes (outliers)
    """

    def __init__(
        self,
        voltage_std: float = 0.001,
        current_std: float = 0.005,
        temperature_std: float = 0.5,
        seed: int | None = None,
    ):
        """
        Initialize noise generator.
        
        Args:
            voltage_std: Standard deviation for voltage noise (V)
            current_std: Standard deviation for current noise (A)
            temperature_std: Standard deviation for temperature noise (°C)
            seed: Random seed for reproducibility
        """
        self.voltage_std = voltage_std
        self.current_std = current_std
        self.temperature_std = temperature_std
        self.rng = np.random.default_rng(seed)

        # Drift state
        self._voltage_drift = 0.0
        self._current_drift = 0.0
        self._temperature_drift = 0.0

    def add_voltage_noise(
        self,
        voltage: float,
        include_drift: bool = True,
        include_spikes: bool = False,
        spike_probability: float = 0.001,
    ) -> float:
        """
        Add noise to voltage measurement.
        
        Args:
            voltage: Clean voltage value (V)
            include_drift: Add slow drift component
            include_spikes: Add occasional spikes
            spike_probability: Probability of spike per sample
            
        Returns:
            Noisy voltage measurement
        """
        # Gaussian noise
        noise = self.rng.normal(0, self.voltage_std)

        # Drift (random walk)
        if include_drift:
            self._voltage_drift += self.rng.normal(0, self.voltage_std * 0.01)
            self._voltage_drift *= 0.999  # Decay to prevent runaway
            noise += self._voltage_drift

        # Occasional spike
        if include_spikes and self.rng.random() < spike_probability:
            noise += self.rng.choice([-1, 1]) * self.voltage_std * 10

        return voltage + noise

    def add_current_noise(
        self,
        current: float,
        include_drift: bool = True,
        include_spikes: bool = False,
        spike_probability: float = 0.001,
    ) -> float:
        """
        Add noise to current measurement.
        
        Args:
            current: Clean current value (A)
            include_drift: Add slow drift component
            include_spikes: Add occasional spikes
            spike_probability: Probability of spike per sample
            
        Returns:
            Noisy current measurement
        """
        # Gaussian noise
        noise = self.rng.normal(0, self.current_std)

        # Drift
        if include_drift:
            self._current_drift += self.rng.normal(0, self.current_std * 0.01)
            self._current_drift *= 0.999
            noise += self._current_drift

        # Spike
        if include_spikes and self.rng.random() < spike_probability:
            noise += self.rng.choice([-1, 1]) * self.current_std * 10

        return current + noise

    def add_temperature_noise(
        self,
        temperature: float,
        include_drift: bool = True,
    ) -> float:
        """
        Add noise to temperature measurement.
        
        Args:
            temperature: Clean temperature value (°C)
            include_drift: Add slow drift component
            
        Returns:
            Noisy temperature measurement
        """
        # Gaussian noise
        noise = self.rng.normal(0, self.temperature_std)

        # Drift (temperature sensors drift slowly)
        if include_drift:
            self._temperature_drift += self.rng.normal(0, self.temperature_std * 0.001)
            self._temperature_drift *= 0.9999
            noise += self._temperature_drift

        return temperature + noise

    def add_quantization_noise(
        self,
        value: float,
        resolution: float,
    ) -> float:
        """
        Add quantization noise (ADC effect).
        
        Args:
            value: Input value
            resolution: ADC resolution (smallest step)
            
        Returns:
            Quantized value
        """
        return np.round(value / resolution) * resolution

    def reset_drift(self) -> None:
        """Reset drift state to zero."""
        self._voltage_drift = 0.0
        self._current_drift = 0.0
        self._temperature_drift = 0.0

    def generate_noise_vector(
        self,
        length: int,
        noise_type: str = "voltage",
    ) -> np.ndarray:
        """
        Generate a vector of noise values.
        
        Args:
            length: Number of noise samples
            noise_type: Type of noise ('voltage', 'current', 'temperature')
            
        Returns:
            Array of noise values
        """
        std_map = {
            "voltage": self.voltage_std,
            "current": self.current_std,
            "temperature": self.temperature_std,
        }
        std = std_map.get(noise_type, self.voltage_std)
        return self.rng.normal(0, std, length)

