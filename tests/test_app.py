"""Tests for Streamlit app components."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from battery_simulator import BatterySimulator, Protocol
from battery_simulator.chemistry import Chemistry
from battery_simulator.core.simulator import SimulationConfig, TimingMode


class TestAppHelperFunctions:
    """Test helper functions used in the Streamlit app."""

    def test_get_chemistry_info_nmc811(self):
        """Test getting NMC811 chemistry info."""
        chem = Chemistry.from_name("NMC811")
        
        assert chem.name == "NMC811-Graphite"
        assert chem.voltage_max == 4.2
        assert chem.voltage_min == 3.0
        assert chem.capacity == 3.0
        assert chem.energy_density == 250.0

    def test_get_chemistry_info_lfp(self):
        """Test getting LFP chemistry info."""
        chem = Chemistry.from_name("LFP")
        
        assert chem.name == "LFP-Graphite"
        assert chem.voltage_max == 3.65
        assert chem.voltage_min == 2.5

    def test_get_chemistry_info_nca(self):
        """Test getting NCA chemistry info."""
        chem = Chemistry.from_name("NCA")
        
        assert chem.name == "NCA-SiGraphite"
        assert chem.voltage_max == 4.2
        assert chem.capacity == 3.5

    def test_get_chemistry_info_lto(self):
        """Test getting LTO chemistry info."""
        chem = Chemistry.from_name("LTO")
        
        assert chem.name == "LTO-LMO"
        assert chem.voltage_max == 2.8
        assert chem.voltage_min == 1.5


class TestAppSimulationRunner:
    """Test the simulation runner functionality."""

    def test_run_cycle_life_simulation(self):
        """Test running a cycle life simulation."""
        config = SimulationConfig(
            timing_mode=TimingMode.INSTANT,
            enable_degradation=True,
            noise_voltage=0.001,
            noise_current=0.005,
            noise_temperature=0.5,
        )
        
        sim = BatterySimulator(
            chemistry="NMC811",
            capacity=3.0,
            temperature=25.0,
            config=config,
        )
        
        protocol = Protocol.cycle_life(
            charge_rate=1.0,
            discharge_rate=1.0,
            cycles=3,
            voltage_max=4.2,
            voltage_min=3.0,
            rest_time=60,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                output_format="generic",
                show_progress=False,
            )
            
            assert results.cycles_completed == 3
            assert results.capacity_retention <= 1.0
            assert output_path.exists()
            
            df = pd.read_csv(output_path)
            assert len(df) > 0
            assert "voltage" in df.columns
            assert "current" in df.columns

    def test_run_formation_simulation(self):
        """Test running a formation simulation."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        
        protocol = Protocol.formation(
            cycles=2,
            initial_rate=0.1,
            voltage_max=4.2,
            voltage_min=3.0,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "formation.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False,
            )
            
            assert results.cycles_completed == 2

    def test_run_rate_capability_simulation(self):
        """Test running a rate capability simulation."""
        sim = BatterySimulator(chemistry="LFP", capacity=3.0)
        
        protocol = Protocol.rate_capability(
            rates=[0.5, 1.0],
            cycles_per_rate=1,
            charge_rate=0.5,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rate.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False,
            )
            
            assert results.cycles_completed >= 1

    def test_run_rpt_simulation(self):
        """Test running an RPT simulation."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        
        protocol = Protocol.rpt(
            charge_rate=0.33,
            discharge_rate=0.33,
            pulse_current=1.0,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rpt.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False,
            )
            
            assert results.cycles_completed == 1


class TestAppOutputFormats:
    """Test different output formats work correctly."""

    @pytest.mark.parametrize("format_name", ["generic", "arbin", "neware", "biologic"])
    def test_output_formats(self, format_name):
        """Test all output formats produce valid CSV."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"test_{format_name}.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                output_format=format_name,
                show_progress=False,
            )
            
            assert output_path.exists()
            df = pd.read_csv(output_path)
            assert len(df) > 0


class TestAppResultsExport:
    """Test results export functionality."""

    def test_results_to_dict(self):
        """Test results can be converted to dictionary."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False,
            )
            
            data = results.to_dict()
            
            assert "test_metadata" in data
            assert "results_summary" in data
            assert data["test_metadata"]["chemistry"] == "NMC811-Graphite"
            assert "capacity_retention" in data["results_summary"]

    def test_cycle_summary_dataframe(self):
        """Test cycle summary is a valid DataFrame."""
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
        protocol = Protocol.cycle_life(cycles=5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False,
            )
            
            summary = results.cycle_summary
            
            assert isinstance(summary, pd.DataFrame)
            assert len(summary) == 5
            assert "cycle_index" in summary.columns
            assert "capacity_retention" in summary.columns


class TestAppNoiseConfiguration:
    """Test noise configuration in simulations."""

    def test_zero_noise(self):
        """Test simulation with zero noise."""
        config = SimulationConfig(
            timing_mode=TimingMode.INSTANT,
            noise_voltage=0.0,
            noise_current=0.0,
            noise_temperature=0.0,
        )
        
        sim = BatterySimulator(
            chemistry="NMC811",
            capacity=3.0,
            config=config,
        )
        
        protocol = Protocol.cycle_life(cycles=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False,
            )
            
            assert results.cycles_completed == 2

    def test_high_noise(self):
        """Test simulation with high noise values."""
        config = SimulationConfig(
            timing_mode=TimingMode.INSTANT,
            noise_voltage=0.01,
            noise_current=0.05,
            noise_temperature=2.0,
        )
        
        sim = BatterySimulator(
            chemistry="NMC811",
            capacity=3.0,
            config=config,
        )
        
        protocol = Protocol.cycle_life(cycles=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            results = sim.run(
                protocol=protocol,
                output_path=output_path,
                show_progress=False,
            )
            
            assert results.cycles_completed == 2


class TestAppDegradationToggle:
    """Test degradation enable/disable functionality."""

    def test_degradation_enabled(self):
        """Test simulation with degradation enabled."""
        config = SimulationConfig(
            timing_mode=TimingMode.INSTANT,
            enable_degradation=True,
        )
        
        sim = BatterySimulator(
            chemistry="NMC811",
            capacity=3.0,
            config=config,
        )
        
        protocol = Protocol.cycle_life(cycles=50)
        
        results = sim.run(protocol=protocol, show_progress=False)
        
        # With degradation, capacity should decrease
        assert results.capacity_retention < 1.0

    def test_degradation_disabled(self):
        """Test simulation with degradation disabled."""
        config = SimulationConfig(
            timing_mode=TimingMode.INSTANT,
            enable_degradation=False,
        )
        
        sim = BatterySimulator(
            chemistry="NMC811",
            capacity=3.0,
            config=config,
        )
        
        protocol = Protocol.cycle_life(cycles=10)
        
        results = sim.run(protocol=protocol, show_progress=False)
        
        # Without degradation, capacity retention should be ~1.0
        assert results.capacity_retention > 0.99


class TestAppChemistryParameters:
    """Test chemistry-specific parameter handling."""

    def test_chemistry_voltage_limits_respected(self):
        """Test that voltage limits from chemistry are respected."""
        for chem_name in ["NMC811", "LFP", "NCA", "LTO"]:
            chem = Chemistry.from_name(chem_name)
            sim = BatterySimulator(chemistry=chem_name, capacity=3.0)
            
            protocol = Protocol.cycle_life(
                cycles=2,
                voltage_max=chem.voltage_max,
                voltage_min=chem.voltage_min,
            )
            
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"{chem_name}.csv"
                results = sim.run(
                    protocol=protocol,
                    output_path=output_path,
                    show_progress=False,
                )
                
                df = pd.read_csv(output_path)
                
                # Check voltage stays within bounds (with small tolerance for noise)
                assert df["voltage"].min() >= chem.voltage_min - 0.1
                assert df["voltage"].max() <= chem.voltage_max + 0.1

    def test_different_capacities(self):
        """Test simulation with different capacity values."""
        for capacity in [1.0, 3.0, 5.0, 10.0]:
            sim = BatterySimulator(chemistry="NMC811", capacity=capacity)
            protocol = Protocol.cycle_life(cycles=2)
            
            results = sim.run(protocol=protocol, show_progress=False)
            
            assert results.capacity_initial == capacity

