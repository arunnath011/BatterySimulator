"""Output formatters for battery test data."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from battery_simulator.outputs.base_writer import BaseWriter
from battery_simulator.outputs.csv_writer import GenericCSVWriter
from battery_simulator.outputs.arbin_format import ArbinWriter
from battery_simulator.outputs.neware_format import NewareWriter
from battery_simulator.outputs.biologic_format import BiologicWriter


def get_writer(format_name: str, output_path: Union[str, Path]) -> BaseWriter:
    """
    Get appropriate writer for the specified format.
    
    Args:
        format_name: Format name ('generic', 'arbin', 'neware', 'biologic')
        output_path: Output file path
        
    Returns:
        Writer instance
        
    Raises:
        ValueError: If format is not recognized
    """
    writers = {
        "generic": GenericCSVWriter,
        "csv": GenericCSVWriter,
        "arbin": ArbinWriter,
        "neware": NewareWriter,
        "biologic": BiologicWriter,
    }

    format_lower = format_name.lower()
    if format_lower not in writers:
        available = list(writers.keys())
        raise ValueError(f"Unknown format '{format_name}'. Available: {available}")

    return writers[format_lower](output_path)


__all__ = [
    "get_writer",
    "BaseWriter",
    "GenericCSVWriter",
    "ArbinWriter",
    "NewareWriter",
    "BiologicWriter",
]

