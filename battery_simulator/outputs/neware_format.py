"""Neware cycler format writer."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from battery_simulator.outputs.base_writer import BaseWriter


class NewareWriter(BaseWriter):
    """
    Neware cycler format writer.
    
    Outputs CSV matching Neware export format:
    - Record ID
    - Step ID
    - Status
    - Jump
    - Time
    - Voltage(V)
    - Current(mA)
    - CapaCity(mAh)
    - Energy(mWh)
    - CapaCity-Chg(mAh)
    - CapaCity-DChg(mAh)
    - Engy-Chg(mWh)
    - Engy-DChg(mWh)
    - Engy-Total(mWh)
    
    Note: Neware uses mA and mAh units.
    """

    COLUMNS = [
        "Record ID",
        "Step ID",
        "Status",
        "Jump",
        "Time",
        "Voltage(V)",
        "Current(mA)",
        "CapaCity(mAh)",
        "Energy(mWh)",
        "CapaCity-Chg(mAh)",
        "CapaCity-DChg(mAh)",
        "Engy-Chg(mWh)",
        "Engy-DChg(mWh)",
        "Engy-Total(mWh)",
    ]

    STATUS_MAP = {
        "charge": "CC_Chg",
        "charge_cc": "CC_Chg",
        "charge_cc_cv": "CCCV_Chg",
        "charge_cv": "CV_Chg",
        "discharge": "CC_DChg",
        "discharge_cc": "CC_DChg",
        "discharge_cp": "CP_DChg",
        "rest": "Rest",
        "pulse_charge": "Pulse_Chg",
        "pulse_discharge": "Pulse_DChg",
    }

    def __init__(self, output_path: str | Path):
        """Initialize Neware format writer."""
        super().__init__(output_path)
        if self.output_path.suffix.lower() != ".csv":
            self.output_path = self.output_path.with_suffix(".csv")
        self._total_charge_capacity = 0.0
        self._total_discharge_capacity = 0.0
        self._total_charge_energy = 0.0
        self._total_discharge_energy = 0.0

    def get_columns(self) -> list[str]:
        """Get Neware column names."""
        return self.COLUMNS

    def _write_header(self) -> None:
        """Write Neware CSV header."""
        self._write_line(",".join(self.COLUMNS))

    def _format_time(self, seconds: float) -> str:
        """Format time as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _write_row(self, row: dict, data_point: int) -> None:
        """Write a row in Neware format."""
        step_type = row.get("step_type", "rest").lower()
        status = self.STATUS_MAP.get(step_type, "Rest")

        # Convert to Neware units (mA, mAh, mWh)
        current = row.get("current", 0.0) * 1000  # A to mA
        capacity = row.get("capacity", 0.0) * 1000  # Ah to mAh
        energy = row.get("energy", 0.0) * 1000  # Wh to mWh

        # Track charge/discharge
        if current > 0:
            charge_capacity = capacity
            discharge_capacity = 0.0
            charge_energy = energy
            discharge_energy = 0.0
        elif current < 0:
            charge_capacity = 0.0
            discharge_capacity = capacity
            charge_energy = 0.0
            discharge_energy = energy
        else:
            charge_capacity = 0.0
            discharge_capacity = 0.0
            charge_energy = 0.0
            discharge_energy = 0.0

        # Update totals
        self._total_charge_capacity += charge_capacity / 1000 if current > 0 else 0
        self._total_discharge_capacity += discharge_capacity / 1000 if current < 0 else 0
        self._total_charge_energy += charge_energy / 1000 if current > 0 else 0
        self._total_discharge_energy += discharge_energy / 1000 if current < 0 else 0

        total_energy = energy

        values = [
            str(data_point),
            str(row.get("step_index", 1)),
            status,
            "0",  # Jump (not used in simulation)
            self._format_time(row.get("test_time", 0.0)),
            f"{row.get('voltage', 0.0):.4f}",
            f"{current:.2f}",
            f"{capacity:.2f}",
            f"{energy:.2f}",
            f"{charge_capacity:.2f}",
            f"{discharge_capacity:.2f}",
            f"{charge_energy:.2f}",
            f"{discharge_energy:.2f}",
            f"{total_energy:.2f}",
        ]

        self._write_line(",".join(values))

