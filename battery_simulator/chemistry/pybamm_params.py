"""
PyBaMM parameter library bridge.

Maps PyBaMM's built-in parameter sets to our chemistry format and provides
access to PyBaMM's extensive validated parameter library.

Requires: pip install pybamm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import pybamm
    PYBAMM_AVAILABLE = True
except ImportError:
    PYBAMM_AVAILABLE = False
    pybamm = None

if TYPE_CHECKING:
    from battery_simulator.chemistry.base_chemistry import BaseChemistry


# Mapping of our chemistry names to PyBaMM parameter sets
CHEMISTRY_TO_PYBAMM = {
    # NMC variants
    "NMC811": "Chen2020",
    "NMC811-Graphite": "Chen2020",
    "NMC622": "Marquis2019",
    "NMC532": "Ecker2015",
    
    # LFP variants
    "LFP": "Prada2013",
    "LFP-Graphite": "Prada2013",
    
    # NCA variants
    "NCA": "NCA_Kim2011",
    "NCA-SiGraphite": "NCA_Kim2011",
    
    # LTO variants
    "LTO": "Ai2020",
    "LTO-LMO": "Ramadass2004",
    
    # Other
    "LCO": "Ai2020",
}

# PyBaMM parameter sets with their chemistry types and descriptions
PYBAMM_PARAMETER_SETS = {
    "Chen2020": {
        "chemistry": "NMC811-Graphite",
        "description": "LG M50 21700 NMC811/Graphite cell",
        "capacity_ah": 5.0,
        "nominal_voltage": 3.63,
        "source": "Chen et al. (2020)",
    },
    "Chen2020_composite": {
        "chemistry": "NMC811-SiGraphite",
        "description": "LG M50 with silicon-graphite composite",
        "capacity_ah": 5.0,
        "nominal_voltage": 3.63,
        "source": "Chen et al. (2020)",
    },
    "Marquis2019": {
        "chemistry": "NMC622-Graphite",
        "description": "Kokam SLPB75106100 pouch cell",
        "capacity_ah": 16.0,
        "nominal_voltage": 3.7,
        "source": "Marquis et al. (2019)",
    },
    "Prada2013": {
        "chemistry": "LFP-Graphite",
        "description": "A123 26650 LFP cell",
        "capacity_ah": 2.3,
        "nominal_voltage": 3.3,
        "source": "Prada et al. (2013)",
    },
    "Ramadass2004": {
        "chemistry": "LCO-Graphite",
        "description": "Sony 18650 LCO cell",
        "capacity_ah": 1.8,
        "nominal_voltage": 3.7,
        "source": "Ramadass et al. (2004)",
    },
    "NCA_Kim2011": {
        "chemistry": "NCA-Graphite",
        "description": "Generic NCA cell",
        "capacity_ah": 2.5,
        "nominal_voltage": 3.6,
        "source": "Kim et al. (2011)",
    },
    "Ecker2015": {
        "chemistry": "NMC532-Graphite",
        "description": "Kokam NMC cell with aging",
        "capacity_ah": 7.5,
        "nominal_voltage": 3.65,
        "source": "Ecker et al. (2015)",
    },
    "Ai2020": {
        "chemistry": "LCO-Graphite",
        "description": "Generic LCO cell with SEI growth",
        "capacity_ah": 1.0,
        "nominal_voltage": 3.7,
        "source": "Ai et al. (2020)",
    },
    "Mohtat2020": {
        "chemistry": "NMC-Graphite",
        "description": "Samsung INR21700-50E",
        "capacity_ah": 5.0,
        "nominal_voltage": 3.6,
        "source": "Mohtat et al. (2020)",
    },
    "OKane2022": {
        "chemistry": "NMC811-SiGraphite",
        "description": "Cell with mechanical degradation",
        "capacity_ah": 3.5,
        "nominal_voltage": 3.65,
        "source": "O'Kane et al. (2022)",
    },
    "ORegan2022": {
        "chemistry": "NMC-Graphite",
        "description": "Thermal characterization cell",
        "capacity_ah": 5.0,
        "nominal_voltage": 3.6,
        "source": "O'Regan et al. (2022)",
    },
}


@dataclass
class PyBaMMParameterInfo:
    """Information about a PyBaMM parameter set."""
    name: str
    chemistry: str
    description: str
    capacity_ah: float
    nominal_voltage: float
    source: str
    available: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "chemistry": self.chemistry,
            "description": self.description,
            "capacity_ah": self.capacity_ah,
            "nominal_voltage": self.nominal_voltage,
            "source": self.source,
            "available": self.available,
        }


class PyBaMMParameterBridge:
    """
    Bridge between PyBaMM parameter sets and our chemistry format.
    
    Provides:
    - Listing available parameter sets
    - Loading PyBaMM parameters
    - Converting PyBaMM parameters to our chemistry format
    - Extracting key parameters for simulation
    """
    
    def __init__(self):
        """Initialize the parameter bridge."""
        self._cached_params: Dict[str, Any] = {}
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if PyBaMM is available."""
        return PYBAMM_AVAILABLE
    
    @classmethod
    def list_parameter_sets(cls) -> List[PyBaMMParameterInfo]:
        """
        List all available PyBaMM parameter sets.
        
        Returns:
            List of parameter set information
        """
        result = []
        
        for name, info in PYBAMM_PARAMETER_SETS.items():
            # Check if actually available in PyBaMM
            available = False
            if PYBAMM_AVAILABLE:
                try:
                    pybamm.ParameterValues(name)
                    available = True
                except Exception:
                    pass
            
            result.append(PyBaMMParameterInfo(
                name=name,
                chemistry=info["chemistry"],
                description=info["description"],
                capacity_ah=info["capacity_ah"],
                nominal_voltage=info["nominal_voltage"],
                source=info["source"],
                available=available,
            ))
        
        # Try to discover additional parameter sets from PyBaMM
        if PYBAMM_AVAILABLE:
            try:
                for name in pybamm.parameter_sets.keys():
                    if name not in PYBAMM_PARAMETER_SETS:
                        try:
                            params = pybamm.ParameterValues(name)
                            capacity = params.get("Nominal cell capacity [A.h]", 1.0)
                            voltage = 3.7  # Default
                            
                            result.append(PyBaMMParameterInfo(
                                name=name,
                                chemistry="Unknown",
                                description=f"PyBaMM parameter set: {name}",
                                capacity_ah=float(capacity) if isinstance(capacity, (int, float)) else 1.0,
                                nominal_voltage=voltage,
                                source="PyBaMM built-in",
                                available=True,
                            ))
                        except Exception:
                            pass
            except Exception:
                pass
        
        return result
    
    @classmethod
    def get_parameter_set_for_chemistry(cls, chemistry_name: str) -> Optional[str]:
        """
        Get the recommended PyBaMM parameter set for a chemistry.
        
        Args:
            chemistry_name: Our chemistry name
            
        Returns:
            PyBaMM parameter set name or None
        """
        return CHEMISTRY_TO_PYBAMM.get(chemistry_name)
    
    def load_parameters(self, parameter_set: str) -> Dict[str, Any]:
        """
        Load PyBaMM parameter values.
        
        Args:
            parameter_set: Name of the parameter set
            
        Returns:
            Dictionary of parameter values
        """
        if not PYBAMM_AVAILABLE:
            raise ImportError("PyBaMM is not installed")
        
        if parameter_set in self._cached_params:
            return self._cached_params[parameter_set]
        
        params = pybamm.ParameterValues(parameter_set)
        
        # Extract key parameters
        extracted = self._extract_key_parameters(params)
        self._cached_params[parameter_set] = extracted
        
        return extracted
    
    def _extract_key_parameters(self, params: Any) -> Dict[str, Any]:
        """Extract key parameters from PyBaMM ParameterValues."""
        result = {
            "raw_params": params,
        }
        
        # Cell parameters
        param_map = {
            "capacity_ah": "Nominal cell capacity [A.h]",
            "electrode_height": "Electrode height [m]",
            "electrode_width": "Electrode width [m]",
            "number_of_cells": "Number of cells connected in series to make a battery",
            
            # Negative electrode
            "neg_particle_radius": "Negative particle radius [m]",
            "neg_thickness": "Negative electrode thickness [m]",
            "neg_porosity": "Negative electrode porosity",
            "neg_max_concentration": "Maximum concentration in negative electrode [mol.m-3]",
            "neg_diffusivity": "Negative electrode diffusivity [m2.s-1]",
            
            # Positive electrode
            "pos_particle_radius": "Positive particle radius [m]",
            "pos_thickness": "Positive electrode thickness [m]",
            "pos_porosity": "Positive electrode porosity",
            "pos_max_concentration": "Maximum concentration in positive electrode [mol.m-3]",
            "pos_diffusivity": "Positive electrode diffusivity [m2.s-1]",
            
            # Electrolyte
            "electrolyte_concentration": "Initial concentration in electrolyte [mol.m-3]",
            
            # Thermal
            "ambient_temperature": "Ambient temperature [K]",
            "initial_temperature": "Initial temperature [K]",
            "cell_thermal_mass": "Cell thermal mass [J.K-1]",
            
            # Voltage limits
            "upper_voltage_limit": "Upper voltage cut-off [V]",
            "lower_voltage_limit": "Lower voltage cut-off [V]",
        }
        
        for key, pybamm_key in param_map.items():
            try:
                value = params.get(pybamm_key)
                if value is not None:
                    # Handle callable parameters
                    if callable(value):
                        result[key] = "function"
                    else:
                        result[key] = float(value) if isinstance(value, (int, float)) else value
            except Exception:
                pass
        
        return result
    
    def create_chemistry_from_pybamm(
        self, 
        parameter_set: str,
        custom_capacity: Optional[float] = None,
    ) -> "BaseChemistry":
        """
        Create a chemistry configuration from PyBaMM parameters.
        
        Args:
            parameter_set: PyBaMM parameter set name
            custom_capacity: Override capacity (Ah)
            
        Returns:
            BaseChemistry instance
        """
        from battery_simulator.chemistry.base_chemistry import (
            BaseChemistry, 
            DegradationParameters
        )
        
        if not PYBAMM_AVAILABLE:
            raise ImportError("PyBaMM is not installed")
        
        params = self.load_parameters(parameter_set)
        
        # Get info about this parameter set
        info = PYBAMM_PARAMETER_SETS.get(parameter_set, {})
        
        # Determine capacity
        capacity = custom_capacity or params.get("capacity_ah", 3.0)
        
        # Determine chemistry type for degradation parameters
        chemistry_type = info.get("chemistry", "NMC-Graphite")
        
        # Create a custom chemistry class
        @dataclass
        class PyBaMMChemistry(BaseChemistry):
            """Chemistry from PyBaMM parameter set."""
            
            def _init_parameters(self) -> None:
                self.name = f"PyBaMM-{parameter_set}"
                self.description = info.get("description", f"From {parameter_set}")
                
                # Cell specs
                self.nominal_capacity = capacity
                self.nominal_voltage = params.get("nominal_voltage", info.get("nominal_voltage", 3.7))
                self.voltage_max = params.get("upper_voltage_limit", 4.2)
                self.voltage_min = params.get("lower_voltage_limit", 2.5)
                self.voltage_nominal = self.nominal_voltage
                
                # Resistance - estimate from typical values
                self.resistance_initial = 0.05  # Ohm
                self.resistance_growth_rate = 0.0001
                
                # Degradation parameters - use defaults based on chemistry type
                if "LFP" in chemistry_type:
                    self.degradation_params = DegradationParameters(
                        k_cyc=0.00003,
                        e_a_cyc=35000.0,
                        z=0.5,
                        alpha=0.4,
                        k_cal=0.000005,
                        e_a_cal=35000.0,
                        b=0.5,
                        beta_soc=1.0,
                    )
                elif "NCA" in chemistry_type:
                    self.degradation_params = DegradationParameters(
                        k_cyc=0.00006,
                        e_a_cyc=50000.0,
                        z=0.55,
                        alpha=0.8,
                        k_cal=0.00002,
                        e_a_cal=45000.0,
                        b=0.5,
                        beta_soc=2.0,
                    )
                else:  # NMC default
                    self.degradation_params = DegradationParameters(
                        k_cyc=0.00005,
                        e_a_cyc=45000.0,
                        z=0.6,
                        alpha=0.7,
                        k_cal=0.00001,
                        e_a_cal=40000.0,
                        b=0.5,
                        beta_soc=1.5,
                    )
                
                # Temperature limits
                self.temp_min = -20.0
                self.temp_max = 60.0
                self.temp_optimal = 25.0
                
                # Charge limits
                self.max_charge_rate = 2.0
                self.max_discharge_rate = 3.0
                
                # Store PyBaMM reference
                self._pybamm_parameter_set = parameter_set
                self._pybamm_params = params
            
            def get_ocv(self, soc: float) -> float:
                """Get OCV using PyBaMM functions if available."""
                # Use linear approximation
                v_range = self.voltage_max - self.voltage_min
                return self.voltage_min + soc * v_range * 0.9
        
        return PyBaMMChemistry()


def list_available_pybamm_chemistries() -> List[Dict[str, Any]]:
    """
    List all available PyBaMM chemistry configurations.
    
    Returns:
        List of dictionaries with chemistry information
    """
    bridge = PyBaMMParameterBridge()
    sets = bridge.list_parameter_sets()
    return [s.to_dict() for s in sets if s.available]


def get_pybamm_parameter_set(chemistry_name: str) -> Optional[str]:
    """
    Get the recommended PyBaMM parameter set for a chemistry.
    
    Args:
        chemistry_name: Chemistry name
        
    Returns:
        Parameter set name or None
    """
    return PyBaMMParameterBridge.get_parameter_set_for_chemistry(chemistry_name)

