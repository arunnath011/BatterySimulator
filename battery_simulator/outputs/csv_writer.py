"""Generic CSV writer for battery test data."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from battery_simulator.outputs.base_writer import BaseWriter


class GenericCSVWriter(BaseWriter):
    """
    Generic CSV format writer.
    
    Outputs standard CSV with columns:
    - timestamp
    - test_time
    - cycle_index
    - step_index
    - step_type
    - current
    - voltage
    - capacity
    - energy
    - temperature
    - state_of_charge
    - power
    - internal_resistance
    """

    COLUMNS = [
        "timestamp",
        "test_time",
        "cycle_index",
        "step_index",
        "step_type",
        "current",
        "voltage",
        "capacity",
        "energy",
        "temperature",
        "state_of_charge",
        "power",
        "internal_resistance",
    ]

    def __init__(self, output_path: str | Path):
        """Initialize generic CSV writer."""
        super().__init__(output_path)
        # Ensure .csv extension
        if self.output_path.suffix.lower() != ".csv":
            self.output_path = self.output_path.with_suffix(".csv")

    def get_columns(self) -> list[str]:
        """Get column names."""
        return self.COLUMNS

    def _write_header(self) -> None:
        """Write CSV header row."""
        self._write_line(",".join(self.COLUMNS))

    def _write_row(self, row: dict, data_point: int) -> None:
        """Write a single data row."""
        values = []
        for col in self.COLUMNS:
            value = row.get(col, "")
            if isinstance(value, float):
                if col in ("current", "voltage", "capacity", "energy", "power"):
                    values.append(f"{value:.6f}")
                elif col in ("temperature", "state_of_charge", "internal_resistance"):
                    values.append(f"{value:.4f}")
                elif col == "test_time":
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            elif isinstance(value, int):
                values.append(str(value))
            else:
                values.append(str(value))

        self._write_line(",".join(values))

