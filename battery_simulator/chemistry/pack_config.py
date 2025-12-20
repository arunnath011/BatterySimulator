"""
Battery pack configuration module.

Defines pack topology, cell connections, and thermal interconnections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PackTopology(str, Enum):
    """Common pack topologies."""
    SERIES = "series"  # All cells in series (Ns, Np=1)
    PARALLEL = "parallel"  # All cells in parallel (Ns=1, Np)
    SERIES_PARALLEL = "series_parallel"  # Ns*Np grid
    MODULE_BASED = "module_based"  # Multiple modules in series


class ThermalConfiguration(str, Enum):
    """Thermal management configurations."""
    AIR_COOLED = "air_cooled"
    LIQUID_COOLED = "liquid_cooled"
    BOTTOM_COOLED = "bottom_cooled"
    SIDE_COOLED = "side_cooled"
    NONE = "none"


@dataclass
class CellConnection:
    """Defines connection between cells."""
    cell_1_index: int
    cell_2_index: int
    connection_resistance: float  # Ohms
    thermal_coupling: float  # W/(m^2*K)
    connection_type: str = "busbar"  # busbar, wire, tab


@dataclass
class ModuleConfig:
    """Configuration for a single module within a pack."""
    series_cells: int = 1
    parallel_cells: int = 1
    cell_spacing_mm: float = 2.0
    module_resistance: float = 0.001  # Additional module-level resistance
    cooling_surface_area: float = 0.01  # m^2
    
    @property
    def total_cells(self) -> int:
        """Total cells in module."""
        return self.series_cells * self.parallel_cells


@dataclass
class PackConfiguration:
    """
    Complete pack configuration.
    
    Defines:
    - Pack topology (series/parallel arrangement)
    - Cell connections and resistances
    - Thermal coupling between cells
    - Cooling configuration
    """
    
    # Basic topology
    series: int = 1
    parallel: int = 1
    topology: PackTopology = PackTopology.SERIES_PARALLEL
    
    # Cell specifications
    cell_capacity_ah: float = 3.0
    cell_nominal_voltage: float = 3.7
    cell_min_voltage: float = 2.5
    cell_max_voltage: float = 4.2
    
    # Cell-to-cell variation
    capacity_variation: float = 0.02  # 2% std dev
    resistance_variation: float = 0.02  # 2% std dev
    initial_soc_variation: float = 0.01  # 1% std dev
    
    # Electrical connections
    busbar_resistance: float = 0.0005  # Ohm per connection
    tab_resistance: float = 0.0002  # Ohm per tab
    module_interconnect_resistance: float = 0.001  # Ohm per module connection
    
    # Thermal configuration
    thermal_config: ThermalConfiguration = ThermalConfiguration.AIR_COOLED
    cell_thermal_mass: float = 100.0  # J/K per cell
    cell_surface_area: float = 0.004  # m^2 per cell
    heat_transfer_coefficient: float = 10.0  # W/(m^2*K) to ambient
    cell_to_cell_thermal_coupling: float = 5.0  # W/(m^2*K) between cells
    coolant_temperature: float = 25.0  # °C
    
    # Module configuration (for module-based topology)
    modules: List[ModuleConfig] = field(default_factory=list)
    modules_in_series: int = 1
    
    @property
    def total_cells(self) -> int:
        """Total number of cells in the pack."""
        if self.topology == PackTopology.MODULE_BASED and self.modules:
            return sum(m.total_cells for m in self.modules) * self.modules_in_series
        return self.series * self.parallel
    
    @property
    def pack_capacity_ah(self) -> float:
        """Total pack capacity in Ah."""
        return self.cell_capacity_ah * self.parallel
    
    @property
    def pack_nominal_voltage(self) -> float:
        """Nominal pack voltage."""
        return self.cell_nominal_voltage * self.series
    
    @property
    def pack_min_voltage(self) -> float:
        """Minimum pack voltage."""
        return self.cell_min_voltage * self.series
    
    @property
    def pack_max_voltage(self) -> float:
        """Maximum pack voltage."""
        return self.cell_max_voltage * self.series
    
    @property
    def pack_energy_kwh(self) -> float:
        """Pack energy capacity in kWh."""
        return (self.pack_capacity_ah * self.pack_nominal_voltage) / 1000
    
    @property
    def total_resistance(self) -> float:
        """Estimated total pack resistance."""
        # Cell resistance (series adds, parallel divides)
        cell_r = 0.05  # Approximate cell resistance
        cell_contribution = (cell_r * self.series) / self.parallel
        
        # Connection resistance
        connections = self.total_cells - 1  # Simplified
        connection_contribution = connections * self.busbar_resistance
        
        return cell_contribution + connection_contribution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "topology": {
                "type": self.topology.value,
                "series": self.series,
                "parallel": self.parallel,
                "total_cells": self.total_cells,
            },
            "electrical": {
                "capacity_ah": self.pack_capacity_ah,
                "nominal_voltage": self.pack_nominal_voltage,
                "energy_kwh": self.pack_energy_kwh,
                "total_resistance": self.total_resistance,
            },
            "thermal": {
                "configuration": self.thermal_config.value,
                "coolant_temperature": self.coolant_temperature,
                "heat_transfer_coefficient": self.heat_transfer_coefficient,
            },
            "variation": {
                "capacity_pct": self.capacity_variation * 100,
                "resistance_pct": self.resistance_variation * 100,
                "soc_pct": self.initial_soc_variation * 100,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PackConfiguration":
        """Create configuration from dictionary."""
        return cls(
            series=data.get("series", 1),
            parallel=data.get("parallel", 1),
            topology=PackTopology(data.get("topology", "series_parallel")),
            cell_capacity_ah=data.get("cell_capacity_ah", 3.0),
            cell_nominal_voltage=data.get("cell_nominal_voltage", 3.7),
            capacity_variation=data.get("capacity_variation", 0.02),
            resistance_variation=data.get("resistance_variation", 0.02),
            busbar_resistance=data.get("busbar_resistance", 0.0005),
            thermal_config=ThermalConfiguration(data.get("thermal_config", "air_cooled")),
            coolant_temperature=data.get("coolant_temperature", 25.0),
        )


# Predefined pack configurations for common applications
STANDARD_PACKS = {
    "ev_small": PackConfiguration(
        series=96,  # ~350V nominal
        parallel=2,
        cell_capacity_ah=5.0,
        thermal_config=ThermalConfiguration.LIQUID_COOLED,
    ),
    "ev_medium": PackConfiguration(
        series=108,  # ~400V nominal
        parallel=4,
        cell_capacity_ah=5.0,
        thermal_config=ThermalConfiguration.LIQUID_COOLED,
    ),
    "ev_large": PackConfiguration(
        series=120,  # ~450V nominal
        parallel=6,
        cell_capacity_ah=5.0,
        thermal_config=ThermalConfiguration.LIQUID_COOLED,
    ),
    "ess_module": PackConfiguration(
        series=14,  # ~50V module
        parallel=4,
        cell_capacity_ah=100.0,  # Large format cells
        thermal_config=ThermalConfiguration.AIR_COOLED,
    ),
    "ebike": PackConfiguration(
        series=13,  # ~48V
        parallel=4,
        cell_capacity_ah=3.5,
        thermal_config=ThermalConfiguration.NONE,
    ),
    "power_tool": PackConfiguration(
        series=5,  # ~20V
        parallel=2,
        cell_capacity_ah=2.5,
        thermal_config=ThermalConfiguration.NONE,
    ),
    "laptop": PackConfiguration(
        series=3,  # ~11V
        parallel=2,
        cell_capacity_ah=3.5,
        thermal_config=ThermalConfiguration.NONE,
    ),
}


def get_standard_pack(name: str) -> PackConfiguration:
    """
    Get a standard pack configuration by name.
    
    Args:
        name: Pack name (ev_small, ev_medium, ev_large, ess_module, ebike, power_tool, laptop)
        
    Returns:
        Pack configuration
        
    Raises:
        ValueError: If pack name is not recognized
    """
    if name not in STANDARD_PACKS:
        available = list(STANDARD_PACKS.keys())
        raise ValueError(f"Unknown pack '{name}'. Available: {available}")
    return STANDARD_PACKS[name]


def list_standard_packs() -> List[Dict[str, Any]]:
    """
    List all standard pack configurations.
    
    Returns:
        List of pack information dictionaries
    """
    result = []
    for name, config in STANDARD_PACKS.items():
        info = config.to_dict()
        info["name"] = name
        result.append(info)
    return result

