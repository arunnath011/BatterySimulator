"""Integration tests for battery simulator."""

import tempfile
from pathlib import Path

import pytest
import pandas as pd

from battery_simulator import BatterySimulator, Protocol


class TestSimulatorIntegration:
    """Integration tests for full simulation runs."""

    def test_basic_simulation(self):
        """Test basic simulation runs without error."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False
            )

            assert results is not None
            assert results.cycles_completed == 3
            assert output_path.exists()

    def test_output_file_created(self):
        """Test output file is created correctly."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.csv"
            sim.run(protocol=protocol, output_path=output_path, show_progress=False)

            # Check file exists and has content
            assert output_path.exists()
            df = pd.read_csv(output_path)
            assert len(df) > 0
            assert "voltage" in df.columns
            assert "current" in df.columns

    def test_capacity_retention_decreases(self):
        """Test that capacity retention decreases over cycles."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False
            )

            # Capacity should have degraded
            assert results.capacity_retention < 1.0
            assert results.capacity_final < results.capacity_initial

    def test_different_chemistries(self):
        """Test simulation with different chemistries."""
        chemistries = ["NMC811", "LFP", "NCA", "LTO"]

        for chem in chemistries:
            sim = BatterySimulator(chemistry=chem, capacity=3.0)
            protocol = Protocol.cycle_life(cycles=2)

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"{chem}.csv"
                results = sim.run(
                    protocol=protocol,
                    output_path=output_path,
                    show_progress=False
                )

                assert results.chemistry == f"{chem}-Graphite" or chem in results.chemistry

    def test_different_formats(self):
        """Test different output formats."""
        formats = ["generic", "arbin", "neware", "biologic"]

        for fmt in formats:
            sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
            protocol = Protocol.cycle_life(cycles=2)

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"test_{fmt}.csv"
                results = sim.run(
                    protocol=protocol,
                    output_path=output_path,
                    output_format=fmt,
                    show_progress=False
                )

                assert output_path.exists()
                df = pd.read_csv(output_path)
                assert len(df) > 0

    def test_formation_protocol(self):
        """Test formation protocol simulation."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.formation(cycles=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "formation.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False
            )

            assert results.cycles_completed == 2

    def test_rate_capability_protocol(self):
        """Test rate capability protocol simulation."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.rate_capability(
            rates=[0.5, 1.0],
            cycles_per_rate=1
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rate.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False
            )

            assert results.cycles_completed >= 1

    def test_results_contain_metadata(self):
        """Test that results contain expected metadata."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False
            )

            assert results.test_id is not None
            assert results.chemistry is not None
            assert results.protocol_name is not None
            assert results.start_time is not None
            assert results.end_time is not None
            assert results.energy_throughput >= 0

    def test_results_to_dict(self):
        """Test results serialization to dictionary."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False
            )

            data = results.to_dict()
            assert "test_metadata" in data
            assert "results_summary" in data
            assert "capacity_retention" in data["results_summary"]

    def test_no_output_file(self):
        """Test simulation without output file."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=2)

        # Run without output path
        results = sim.run(
            protocol=protocol,
            output_path=None,
            show_progress=False
        )

        assert results is not None
        assert results.cycles_completed == 2


class TestSimulatorState:
    """Test simulator state management."""

    def test_get_state(self):
        """Test getting simulator state."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        state = sim.get_state()

        assert "battery" in state
        assert "degradation" in state
        assert "cycle" in state

    def test_initial_state(self):
        """Test initial simulator state."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0, temperature=30.0)
        state = sim.get_state()

        assert state["battery"]["temperature"] == 30.0
        assert state["cycle"] == 0


class TestDataCallbacks:
    """Test data callback functionality."""

    def test_callback_invoked(self):
        """Test that data callbacks are invoked."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=1)

        callback_data = []
        
        def callback(data):
            callback_data.append(data)

        sim.on_data(callback)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False
            )

        assert len(callback_data) > 0
        assert "voltage" in callback_data[0]
        assert "current" in callback_data[0]

