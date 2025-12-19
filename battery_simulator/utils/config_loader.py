"""Configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class CellConfigModel(BaseModel):
    """Cell configuration model."""

    chemistry: str = "NMC811"
    capacity: float = Field(default=3.0, gt=0)
    form_factor: str = "cylindrical"

    # Optional overrides
    resistance_initial: float | None = None
    capacity_fade_rate: float | None = None
    resistance_growth_rate: float | None = None


class ProtocolConfigModel(BaseModel):
    """Protocol configuration model."""

    type: str = "cycle_life"
    cycles: int = Field(default=1000, gt=0)
    temperature: float = Field(default=25.0, ge=-40, le=100)

    # Cycling parameters
    charge_rate: float = Field(default=1.0, gt=0)
    discharge_rate: float = Field(default=1.0, gt=0)
    voltage_max: float = Field(default=4.2, gt=0)
    voltage_min: float = Field(default=3.0, gt=0)
    rest_time: float = Field(default=300.0, ge=0)

    # RPT interval
    rpt_interval: int | None = None

    # End conditions
    end_capacity_retention: float = Field(default=0.80, gt=0, le=1)
    max_time_days: int | None = None


class OutputConfigModel(BaseModel):
    """Output configuration model."""

    format: str = "generic"
    directory: str = "./output"
    cycle_data: bool = True
    summary_data: bool = True
    metadata: bool = True

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        valid_formats = ["generic", "csv", "arbin", "neware", "biologic"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid format '{v}'. Must be one of: {valid_formats}")
        return v.lower()


class NoiseConfigModel(BaseModel):
    """Noise configuration model."""

    voltage_noise: float = Field(default=0.001, ge=0)
    current_noise: float = Field(default=0.005, ge=0)
    temperature_noise: float = Field(default=0.5, ge=0)


class DegradationConfigModel(BaseModel):
    """Degradation configuration model."""

    enable_capacity_fade: bool = True
    enable_resistance_growth: bool = True
    enable_sudden_failure: bool = False
    failure_probability: float = Field(default=0.01, ge=0, le=1)


class SimulationConfigModel(BaseModel):
    """Complete simulation configuration model."""

    name: str = "Battery Simulation"
    output_dir: str = "./output"
    timing_mode: str = "instant"
    speed_factor: float = Field(default=1.0, gt=0)
    data_rate: float = Field(default=1.0, gt=0)
    random_seed: int | None = None

    cell: CellConfigModel = Field(default_factory=CellConfigModel)
    protocol: ProtocolConfigModel = Field(default_factory=ProtocolConfigModel)
    output: OutputConfigModel = Field(default_factory=OutputConfigModel)
    noise: NoiseConfigModel = Field(default_factory=NoiseConfigModel)
    degradation: DegradationConfigModel = Field(default_factory=DegradationConfigModel)

    @field_validator("timing_mode")
    @classmethod
    def validate_timing_mode(cls, v: str) -> str:
        valid_modes = ["instant", "real_time", "accelerated"]
        if v.lower() not in valid_modes:
            raise ValueError(f"Invalid timing mode '{v}'. Must be one of: {valid_modes}")
        return v.lower()


def load_config(config_path: str | Path) -> SimulationConfigModel:
    """
    Load simulation configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated configuration model
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Map YAML structure to model
    config_dict = _map_yaml_to_model(raw_config)
    return SimulationConfigModel(**config_dict)


def _map_yaml_to_model(raw: dict[str, Any]) -> dict[str, Any]:
    """Map YAML configuration to model structure."""
    result = {}

    # Top-level simulation settings
    if "simulation" in raw:
        sim = raw["simulation"]
        result["name"] = sim.get("name", "Battery Simulation")
        result["output_dir"] = sim.get("output_dir", "./output")
        result["timing_mode"] = sim.get("timing_mode", "instant")
        result["speed_factor"] = sim.get("speed_factor", 1.0)
        result["data_rate"] = sim.get("data_rate", 1.0)
        result["random_seed"] = sim.get("random_seed")

    # Cell configuration
    if "cell" in raw:
        result["cell"] = raw["cell"]

    # Protocol configuration
    if "test_protocol" in raw:
        protocol = raw["test_protocol"]
        result["protocol"] = {
            "type": protocol.get("type", "cycle_life"),
            "cycles": protocol.get("cycles", 1000),
            "temperature": protocol.get("temperature", 25),
        }
        if "cycling" in protocol:
            cycling = protocol["cycling"]
            result["protocol"]["charge_rate"] = cycling.get("charge_rate", 1.0)
            result["protocol"]["discharge_rate"] = cycling.get("discharge_rate", 1.0)
            result["protocol"]["voltage_max"] = cycling.get("voltage_max", 4.2)
            result["protocol"]["voltage_min"] = cycling.get("voltage_min", 3.0)
            result["protocol"]["rest_time"] = cycling.get("rest_time", 300)
        if "end_conditions" in protocol:
            ec = protocol["end_conditions"]
            result["protocol"]["end_capacity_retention"] = ec.get("capacity_retention", 0.80)
            result["protocol"]["max_time_days"] = ec.get("max_time_days")
        result["protocol"]["rpt_interval"] = protocol.get("rpt_interval")

    # Output configuration
    if "output" in raw:
        result["output"] = raw["output"]

    # Noise configuration
    if "noise" in raw:
        result["noise"] = raw["noise"]

    # Degradation configuration
    if "degradation" in raw:
        result["degradation"] = raw["degradation"]

    return result


def save_config(config: SimulationConfigModel, output_path: str | Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration model
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump()

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

