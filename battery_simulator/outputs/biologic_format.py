"""Biologic EC-Lab format writer."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from battery_simulator.outputs.base_writer import BaseWriter


class BiologicWriter(BaseWriter):
    """
    Biologic EC-Lab format writer.
    
    Outputs CSV matching EC-Lab export format:
    - mode
    - ox/red
    - error
    - control changes
    - Ns changes
    - counter inc.
    - time/s
    - control/V/mA
    - Ewe/V
    - I/mA
    - dq/mA.h
    - Q charge/discharge/mA.h
    - half cycle
    - Ns
    
    Note: Biologic uses specific mode codes and mA/mA.h units.
    """

    COLUMNS = [
        "mode",
        "ox/red",
        "error",
        "control changes",
        "Ns changes",
        "counter inc.",
        "time/s",
        "control/V/mA",
        "Ewe/V",
        "I/mA",
        "dq/mA.h",
        "Q charge/discharge/mA.h",
        "half cycle",
        "Ns",
    ]

    # Biologic mode codes
    MODE_CODES = {
        "rest": 3,
        "charge": 1,
        "charge_cc": 1,
        "charge_cc_cv": 1,
        "charge_cv": 2,
        "discharge": 1,
        "discharge_cc": 1,
        "discharge_cp": 4,
        "pulse_charge": 1,
        "pulse_discharge": 1,
    }

    def __init__(self, output_path: str | Path):
        """Initialize Biologic format writer."""
        super().__init__(output_path)
        if self.output_path.suffix.lower() not in (".csv", ".mpt"):
            self.output_path = self.output_path.with_suffix(".csv")
        self._cumulative_charge = 0.0
        self._cumulative_discharge = 0.0
        self._prev_capacity = 0.0
        self._half_cycle = 1

    def get_columns(self) -> list[str]:
        """Get Biologic column names."""
        return self.COLUMNS

    def _write_header(self) -> None:
        """Write Biologic CSV header."""
        # Biologic files often have metadata header
        # For simplicity, we'll just write column names
        self._write_line(",".join(self.COLUMNS))

    def _write_row(self, row: dict, data_point: int) -> None:
        """Write a row in Biologic format."""
        step_type = row.get("step_type", "rest").lower()
        mode = self.MODE_CODES.get(step_type, 3)

        current = row.get("current", 0.0)
        voltage = row.get("voltage", 0.0)
        capacity = row.get("capacity", 0.0)  # In Ah

        # Determine ox/red (1 = oxidation/charge, 0 = reduction/discharge)
        ox_red = 1 if current > 0 else 0

        # Convert to Biologic units (mA, mA.h)
        current_ma = current * 1000
        capacity_mah = capacity * 1000

        # Calculate dq (incremental capacity)
        dq = (capacity - self._prev_capacity) * 1000
        self._prev_capacity = capacity

        # Track charge/discharge
        if current > 0:
            self._cumulative_charge += abs(dq)
            q_value = self._cumulative_charge
        elif current < 0:
            self._cumulative_discharge += abs(dq)
            q_value = -self._cumulative_discharge
        else:
            q_value = 0.0

        # Control value depends on mode
        control = voltage if mode == 2 else current_ma

        values = [
            str(mode),
            str(ox_red),
            "0",  # error
            "0",  # control changes
            "0",  # Ns changes
            "0",  # counter inc.
            f"{row.get('test_time', 0.0):.3f}",
            f"{control:.4f}",
            f"{voltage:.4f}",
            f"{current_ma:.4f}",
            f"{dq:.4f}",
            f"{q_value:.4f}",
            str(self._half_cycle),
            str(row.get("step_index", 0)),
        ]

        self._write_line(",".join(values))

    def increment_half_cycle(self) -> None:
        """Increment half cycle counter (call at charge/discharge transitions)."""
        self._half_cycle += 1

