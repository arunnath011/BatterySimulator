"""Base writer class for output formatters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TextIO

import pandas as pd


class BaseWriter(ABC):
    """
    Abstract base class for output writers.
    
    Handles file management and defines interface for
    format-specific implementations.
    """

    def __init__(self, output_path: str | Path):
        """
        Initialize writer.
        
        Args:
            output_path: Path to output file
        """
        self.output_path = Path(output_path)
        self._file: TextIO | None = None
        self._data_point_count = 0
        self._header_written = False

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "BaseWriter":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """Open output file for writing."""
        self._file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._header_written = False
        self._data_point_count = 0

    def close(self) -> None:
        """Close output file."""
        if self._file:
            self._file.close()
            self._file = None

    def write_data(self, data: pd.DataFrame) -> None:
        """
        Write data to output file.
        
        Args:
            data: DataFrame with time-series data
        """
        if self._file is None:
            self.open()

        # Write header on first write
        if not self._header_written:
            self._write_header()
            self._header_written = True

        # Convert and write rows
        for _, row in data.iterrows():
            self._data_point_count += 1
            self._write_row(row.to_dict(), self._data_point_count)

    @abstractmethod
    def _write_header(self) -> None:
        """Write format-specific header. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _write_row(self, row: dict, data_point: int) -> None:
        """Write a single data row. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_columns(self) -> list[str]:
        """Get column names for this format. Must be implemented by subclasses."""
        pass

    def _write_line(self, line: str) -> None:
        """Write a line to the output file."""
        if self._file:
            self._file.write(line + "\n")

    def flush(self) -> None:
        """Flush output buffer to file."""
        if self._file:
            self._file.flush()

    @property
    def row_count(self) -> int:
        """Get number of rows written."""
        return self._data_point_count

