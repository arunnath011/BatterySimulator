"""Arbin cycler format writer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Union

from battery_simulator.outputs.base_writer import BaseWriter


class ArbinWriter(BaseWriter):
    """
    Arbin cycler format writer.
    
    Outputs CSV matching Arbin export format:
    - Data_Point
    - Test_Time(s)
    - Date_Time
    - Cycle_Index
    - Step_Index
    - Current(A)
    - Voltage(V)
    - Charge_Capacity(Ah)
    - Discharge_Capacity(Ah)
    - Charge_Energy(Wh)
    - Discharge_Energy(Wh)
    - dV/dt(V/s)
    - Internal_Resistance(Ohm)
    - Temperature(C)
    """

    COLUMNS = [
        "Data_Point",
        "Test_Time(s)",
        "Date_Time",
        "Cycle_Index",
        "Step_Index",
        "Current(A)",
        "Voltage(V)",
        "Charge_Capacity(Ah)",
        "Discharge_Capacity(Ah)",
        "Charge_Energy(Wh)",
        "Discharge_Energy(Wh)",
        "dV/dt(V/s)",
        "Internal_Resistance(Ohm)",
        "Temperature(C)",
    ]

    def __init__(self, output_path: str | Path):
        """Initialize Arbin format writer."""
        super().__init__(output_path)
        if self.output_path.suffix.lower() != ".csv":
            self.output_path = self.output_path.with_suffix(".csv")
        self._prev_voltage = None
        self._prev_time = None

    def get_columns(self) -> list[str]:
        """Get Arbin column names."""
        return self.COLUMNS

    def _write_header(self) -> None:
        """Write Arbin CSV header."""
        self._write_line(",".join(self.COLUMNS))

    def _write_row(self, row: dict, data_point: int) -> None:
        """Write a row in Arbin format."""
        # Calculate dV/dt
        voltage = row.get("voltage", 0.0)
        test_time = row.get("test_time", 0.0)

        dv_dt = 0.0
        if self._prev_voltage is not None and self._prev_time is not None:
            dt = test_time - self._prev_time
            if dt > 0:
                dv_dt = (voltage - self._prev_voltage) / dt

        self._prev_voltage = voltage
        self._prev_time = test_time

        # Parse timestamp
        timestamp_str = row.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            date_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            date_time = timestamp_str

        # Separate charge/discharge capacity and energy
        current = row.get("current", 0.0)
        capacity = row.get("capacity", 0.0)
        energy = row.get("energy", 0.0)

        charge_capacity = capacity if current > 0 else 0.0
        discharge_capacity = capacity if current < 0 else 0.0
        charge_energy = energy if current > 0 else 0.0
        discharge_energy = energy if current < 0 else 0.0

        values = [
            str(data_point),
            f"{test_time:.2f}",
            date_time,
            str(row.get("cycle_index", 1)),
            str(row.get("step_index", 1)),
            f"{current:.6f}",
            f"{voltage:.6f}",
            f"{charge_capacity:.6f}",
            f"{discharge_capacity:.6f}",
            f"{charge_energy:.6f}",
            f"{discharge_energy:.6f}",
            f"{dv_dt:.8f}",
            f"{row.get('internal_resistance', 0.0):.6f}",
            f"{row.get('temperature', 25.0):.2f}",
        ]

        self._write_line(",".join(values))

