"""Battery chemistry configurations."""

from battery_simulator.chemistry.base_chemistry import BaseChemistry
from battery_simulator.chemistry.nmc811 import NMC811Chemistry
from battery_simulator.chemistry.lfp import LFPChemistry
from battery_simulator.chemistry.nca import NCAChemistry
from battery_simulator.chemistry.lto import LTOChemistry


class Chemistry:
    """Factory for creating chemistry configurations."""

    NMC811 = "NMC811"
    LFP = "LFP"
    NCA = "NCA"
    LTO = "LTO"

    _registry = {
        "NMC811": NMC811Chemistry,
        "NMC811-GRAPHITE": NMC811Chemistry,
        "LFP": LFPChemistry,
        "LFP-GRAPHITE": LFPChemistry,
        "NCA": NCAChemistry,
        "NCA-SIGRAPHITE": NCAChemistry,
        "LTO": LTOChemistry,
        "LTO-LMO": LTOChemistry,
    }

    @classmethod
    def from_name(cls, name: str) -> BaseChemistry:
        """
        Create chemistry configuration from name.
        
        Args:
            name: Chemistry name (e.g., 'NMC811', 'LFP', 'NCA', 'LTO')
            
        Returns:
            Chemistry configuration object
            
        Raises:
            ValueError: If chemistry name is not recognized
        """
        name_upper = name.upper().replace(" ", "-").replace("_", "-")
        if name_upper not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown chemistry '{name}'. Available: {available}")
        return cls._registry[name_upper]()

    @classmethod
    def list_available(cls) -> list[str]:
        """List available chemistry names."""
        return list(set(cls._registry.values()))

    @classmethod
    def register(cls, name: str, chemistry_class: type) -> None:
        """Register a custom chemistry."""
        cls._registry[name.upper()] = chemistry_class


__all__ = [
    "Chemistry",
    "BaseChemistry",
    "NMC811Chemistry",
    "LFPChemistry",
    "NCAChemistry",
    "LTOChemistry",
]

