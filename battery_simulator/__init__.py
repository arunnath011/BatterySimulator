"""
Battery Test Data Simulator

A Python-based battery test data simulator that generates realistic 
lithium-ion battery cycling data for development, testing, and demonstration purposes.
"""

from battery_simulator.core.simulator import (
    BatterySimulator,
    SimulationConfig,
    SimulationResults,
    SimulationMode,
    TimingMode,
    StreamingMode,
    StreamingConfig,
    CycleResult,
)
from battery_simulator.core.battery_model import BatteryModel
from battery_simulator.chemistry import Chemistry
from battery_simulator.protocols import Protocol

__version__ = "1.1.0"
__all__ = [
    "BatterySimulator",
    "SimulationConfig",
    "SimulationResults",
    "SimulationMode",
    "TimingMode",
    "StreamingMode",
    "StreamingConfig",
    "CycleResult",
    "BatteryModel",
    "Chemistry",
    "Protocol",
]

