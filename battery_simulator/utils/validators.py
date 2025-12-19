"""Validation utilities for battery simulation data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    message: str
    details: dict | None = None


def validate_config(config: dict) -> list[ValidationResult]:
    """
    Validate simulation configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation results
    """
    results = []

    # Check required fields
    required_fields = ["cell", "protocol"]
    for field in required_fields:
        if field not in config:
            results.append(
                ValidationResult(
                    passed=False,
                    message=f"Missing required field: {field}",
                )
            )

    # Validate cell configuration
    if "cell" in config:
        cell = config["cell"]

        # Validate capacity
        capacity = cell.get("capacity", 0)
        if capacity <= 0:
            results.append(
                ValidationResult(
                    passed=False,
                    message=f"Invalid capacity: {capacity}. Must be > 0",
                )
            )
        else:
            results.append(
                ValidationResult(passed=True, message="Cell capacity valid")
            )

    # Validate protocol
    if "protocol" in config:
        protocol = config["protocol"]

        # Validate cycles
        cycles = protocol.get("cycles", 0)
        if cycles <= 0:
            results.append(
                ValidationResult(
                    passed=False,
                    message=f"Invalid cycles: {cycles}. Must be > 0",
                )
            )

        # Validate voltage limits
        v_max = protocol.get("voltage_max", 4.2)
        v_min = protocol.get("voltage_min", 3.0)
        if v_min >= v_max:
            results.append(
                ValidationResult(
                    passed=False,
                    message=f"Invalid voltage range: {v_min}-{v_max}V. Min must be < Max",
                )
            )

    return results


def validate_simulation_data(data: pd.DataFrame) -> list[ValidationResult]:
    """
    Validate simulation output data.
    
    Checks:
    - Energy conservation
    - Voltage bounds
    - Monotonic time
    - No missing values
    
    Args:
        data: Simulation output DataFrame
        
    Returns:
        List of validation results
    """
    results = []

    # Check for required columns
    required_columns = ["test_time", "voltage", "current", "state_of_charge"]
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        results.append(
            ValidationResult(
                passed=False,
                message=f"Missing required columns: {missing_cols}",
            )
        )
        return results

    # Check for NaN values
    nan_counts = data[required_columns].isna().sum()
    if nan_counts.any():
        results.append(
            ValidationResult(
                passed=False,
                message=f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}",
            )
        )
    else:
        results.append(
            ValidationResult(passed=True, message="No missing values")
        )

    # Check monotonic time
    if not data["test_time"].is_monotonic_increasing:
        results.append(
            ValidationResult(
                passed=False,
                message="Test time is not monotonically increasing",
            )
        )
    else:
        results.append(
            ValidationResult(passed=True, message="Time series monotonic")
        )

    # Check voltage bounds (typical li-ion range)
    v_min, v_max = data["voltage"].min(), data["voltage"].max()
    if v_min < 2.0 or v_max > 4.5:
        results.append(
            ValidationResult(
                passed=False,
                message=f"Voltage out of typical range: {v_min:.3f}-{v_max:.3f}V",
                details={"v_min": v_min, "v_max": v_max},
            )
        )
    else:
        results.append(
            ValidationResult(
                passed=True,
                message=f"Voltage within typical range: {v_min:.3f}-{v_max:.3f}V",
            )
        )

    # Check SOC bounds
    soc_min, soc_max = data["state_of_charge"].min(), data["state_of_charge"].max()
    if soc_min < 0 or soc_max > 1:
        results.append(
            ValidationResult(
                passed=False,
                message=f"SOC out of bounds: {soc_min:.3f}-{soc_max:.3f}",
            )
        )
    else:
        results.append(
            ValidationResult(passed=True, message="SOC within valid range")
        )

    return results


def validate_energy_conservation(
    charge_energy: float,
    discharge_energy: float,
    min_efficiency: float = 0.85,
    max_efficiency: float = 0.99,
) -> ValidationResult:
    """
    Validate energy conservation (round-trip efficiency).
    
    Args:
        charge_energy: Total charge energy (Wh)
        discharge_energy: Total discharge energy (Wh)
        min_efficiency: Minimum expected efficiency
        max_efficiency: Maximum expected efficiency
        
    Returns:
        Validation result
    """
    if charge_energy <= 0:
        return ValidationResult(
            passed=False,
            message="Charge energy must be > 0",
        )

    efficiency = discharge_energy / charge_energy

    if min_efficiency <= efficiency <= max_efficiency:
        return ValidationResult(
            passed=True,
            message=f"Energy efficiency valid: {efficiency:.2%}",
            details={"efficiency": efficiency},
        )
    else:
        return ValidationResult(
            passed=False,
            message=f"Energy efficiency out of range: {efficiency:.2%} (expected {min_efficiency:.0%}-{max_efficiency:.0%})",
            details={"efficiency": efficiency},
        )


def validate_degradation_trend(
    capacity_values: list[float] | np.ndarray,
) -> ValidationResult:
    """
    Validate that capacity degradation is monotonically decreasing.
    
    Args:
        capacity_values: List of capacity values over cycles
        
    Returns:
        Validation result
    """
    capacity_array = np.array(capacity_values)

    # Allow small increases due to noise, but overall trend should be decreasing
    differences = np.diff(capacity_array)
    increasing_count = np.sum(differences > 0.01)  # Allow 1% increases
    total_transitions = len(differences)

    if increasing_count > total_transitions * 0.1:  # >10% increases
        return ValidationResult(
            passed=False,
            message=f"Capacity not monotonically decreasing: {increasing_count}/{total_transitions} increases",
            details={"increasing_count": increasing_count, "total": total_transitions},
        )

    return ValidationResult(
        passed=True,
        message="Capacity degradation trend valid",
    )

