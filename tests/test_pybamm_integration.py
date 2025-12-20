"""
Tests for PyBaMM integration.

These tests verify:
1. PyBaMM model wrapper functionality
2. Parameter library import
3. Pack simulations
4. Mode selection and fallback behavior
"""

import pytest
from unittest.mock import MagicMock, patch

# Check if PyBaMM is available for conditional tests
try:
    import pybamm
    PYBAMM_AVAILABLE = True
except ImportError:
    PYBAMM_AVAILABLE = False

try:
    import liionpack
    LIIONPACK_AVAILABLE = True
except ImportError:
    LIIONPACK_AVAILABLE = False


class TestSimulationModeEnum:
    """Test SimulationMode enum and routing."""
    
    def test_simulation_mode_values(self):
        """Test that SimulationMode enum has expected values."""
        from battery_simulator.core.simulator import SimulationMode
        
        assert SimulationMode.FAST.value == "fast"
        assert SimulationMode.HIGH_FIDELITY.value == "high_fidelity"
        assert SimulationMode.PACK.value == "pack"
    
    def test_simulation_config_mode_default(self):
        """Test that SimulationConfig defaults to FAST mode."""
        from battery_simulator.core.simulator import SimulationConfig, SimulationMode
        
        config = SimulationConfig()
        assert config.simulation_mode == SimulationMode.FAST
    
    def test_simulation_config_pybamm_settings(self):
        """Test PyBaMM-specific config settings."""
        from battery_simulator.core.simulator import SimulationConfig, SimulationMode
        
        config = SimulationConfig(
            simulation_mode=SimulationMode.HIGH_FIDELITY,
            pybamm_model="DFN",
            pybamm_parameter_set="Chen2020",
        )
        
        assert config.simulation_mode == SimulationMode.HIGH_FIDELITY
        assert config.pybamm_model == "DFN"
        assert config.pybamm_parameter_set == "Chen2020"
    
    def test_simulation_config_pack_settings(self):
        """Test pack simulation config settings."""
        from battery_simulator.core.simulator import SimulationConfig, SimulationMode
        
        pack_cfg = {"series": 14, "parallel": 4}
        config = SimulationConfig(
            simulation_mode=SimulationMode.PACK,
            pack_config=pack_cfg,
        )
        
        assert config.simulation_mode == SimulationMode.PACK
        assert config.pack_config == pack_cfg


class TestBatterySimulatorModeRouting:
    """Test BatterySimulator mode routing."""
    
    def test_fast_mode_initialization(self):
        """Test that FAST mode uses BatteryModel."""
        from battery_simulator import BatterySimulator
        from battery_simulator.core.simulator import SimulationMode
        from battery_simulator.core.battery_model import BatteryModel
        
        sim = BatterySimulator(
            chemistry="NMC811",
            mode=SimulationMode.FAST,
        )
        
        assert sim.backend_type == "fast"
        assert isinstance(sim.battery, BatteryModel)
    
    def test_check_backend_available_fast(self):
        """Test that FAST backend is always available."""
        from battery_simulator import BatterySimulator
        from battery_simulator.core.simulator import SimulationMode
        
        assert BatterySimulator.check_backend_available(SimulationMode.FAST) is True
    
    def test_check_backend_available_high_fidelity(self):
        """Test HIGH_FIDELITY backend availability check."""
        from battery_simulator import BatterySimulator
        from battery_simulator.core.simulator import SimulationMode
        
        result = BatterySimulator.check_backend_available(SimulationMode.HIGH_FIDELITY)
        assert result == PYBAMM_AVAILABLE
    
    def test_check_backend_available_pack(self):
        """Test PACK backend availability check."""
        from battery_simulator import BatterySimulator
        from battery_simulator.core.simulator import SimulationMode
        
        result = BatterySimulator.check_backend_available(SimulationMode.PACK)
        # Pack requires both PyBaMM and liionpack
        expected = PYBAMM_AVAILABLE and LIIONPACK_AVAILABLE
        assert result == expected
    
    @pytest.mark.skipif(not PYBAMM_AVAILABLE, reason="PyBaMM not installed")
    def test_high_fidelity_mode_initialization(self):
        """Test that HIGH_FIDELITY mode uses PyBaMMModel."""
        from battery_simulator import BatterySimulator
        from battery_simulator.core.simulator import SimulationMode
        from battery_simulator.core.pybamm_model import PyBaMMModel
        
        sim = BatterySimulator(
            chemistry="NMC811",
            mode=SimulationMode.HIGH_FIDELITY,
        )
        
        assert sim.backend_type == "pybamm"
        assert isinstance(sim.battery, PyBaMMModel)
    
    def test_high_fidelity_fallback_when_unavailable(self):
        """Test fallback to FAST mode when PyBaMM unavailable."""
        from battery_simulator import BatterySimulator
        from battery_simulator.core.simulator import SimulationMode, SimulationConfig
        
        if PYBAMM_AVAILABLE:
            pytest.skip("PyBaMM is available, can't test fallback")
        
        # Should fall back to FAST mode with warning
        import warnings
        with warnings.catch_warnings(record=True):
            sim = BatterySimulator(
                chemistry="NMC811",
                mode=SimulationMode.HIGH_FIDELITY,
            )
        
        assert sim.backend_type == "fast"


class TestPyBaMMModelWrapper:
    """Test PyBaMM model wrapper functionality."""
    
    @pytest.mark.skipif(not PYBAMM_AVAILABLE, reason="PyBaMM not installed")
    def test_pybamm_model_types(self):
        """Test available PyBaMM model types."""
        from battery_simulator.core.pybamm_model import PyBaMMModelType
        
        assert PyBaMMModelType.SPM.value == "SPM"
        assert PyBaMMModelType.SPME.value == "SPMe"
        assert PyBaMMModelType.DFN.value == "DFN"
    
    @pytest.mark.skipif(not PYBAMM_AVAILABLE, reason="PyBaMM not installed")
    def test_pybamm_model_initialization(self):
        """Test PyBaMMModel initialization."""
        from battery_simulator.core.pybamm_model import PyBaMMModel, PyBaMMModelType
        from battery_simulator.chemistry import Chemistry
        
        chem = Chemistry.from_name("NMC811")
        model = PyBaMMModel(
            chemistry=chem,
            capacity=3.0,
            temperature=25.0,
            model_type=PyBaMMModelType.SPME,
        )
        
        assert model.model_type == PyBaMMModelType.SPME
        assert model.capacity_nominal == 3.0
        assert model.state.temperature == 25.0
    
    @pytest.mark.skipif(not PYBAMM_AVAILABLE, reason="PyBaMM not installed")
    def test_pybamm_state_dict(self):
        """Test PyBaMMModel state dictionary."""
        from battery_simulator.core.pybamm_model import PyBaMMModel
        from battery_simulator.chemistry import Chemistry
        
        chem = Chemistry.from_name("NMC811")
        model = PyBaMMModel(chemistry=chem)
        
        state = model.get_state_dict()
        
        assert "soc" in state
        assert "voltage" in state
        assert "model_type" in state
        assert "parameter_set" in state
    
    @pytest.mark.skipif(not PYBAMM_AVAILABLE, reason="PyBaMM not installed")
    def test_list_parameter_sets(self):
        """Test listing available parameter sets."""
        from battery_simulator.core.pybamm_model import PyBaMMModel
        
        sets = PyBaMMModel.list_available_parameter_sets()
        
        assert isinstance(sets, list)
        assert len(sets) > 0
        # Check for some known parameter sets
        assert "Chen2020" in sets or len(sets) > 0


class TestPyBaMMParameterBridge:
    """Test PyBaMM parameter library bridge."""
    
    def test_chemistry_to_pybamm_mapping(self):
        """Test chemistry name to PyBaMM parameter set mapping."""
        from battery_simulator.chemistry.pybamm_params import CHEMISTRY_TO_PYBAMM
        
        assert "NMC811" in CHEMISTRY_TO_PYBAMM
        assert "LFP" in CHEMISTRY_TO_PYBAMM
        assert "NCA" in CHEMISTRY_TO_PYBAMM
    
    def test_parameter_set_info(self):
        """Test parameter set information structure."""
        from battery_simulator.chemistry.pybamm_params import PYBAMM_PARAMETER_SETS
        
        chen2020 = PYBAMM_PARAMETER_SETS.get("Chen2020")
        assert chen2020 is not None
        assert "chemistry" in chen2020
        assert "description" in chen2020
        assert "capacity_ah" in chen2020
    
    @pytest.mark.skipif(not PYBAMM_AVAILABLE, reason="PyBaMM not installed")
    def test_list_parameter_sets(self):
        """Test listing parameter sets from bridge."""
        from battery_simulator.chemistry.pybamm_params import PyBaMMParameterBridge
        
        bridge = PyBaMMParameterBridge()
        sets = bridge.list_parameter_sets()
        
        assert len(sets) > 0
        # Check structure
        info = sets[0]
        assert hasattr(info, "name")
        assert hasattr(info, "chemistry")
        assert hasattr(info, "available")
    
    @pytest.mark.skipif(not PYBAMM_AVAILABLE, reason="PyBaMM not installed")
    def test_load_parameters(self):
        """Test loading PyBaMM parameters."""
        from battery_simulator.chemistry.pybamm_params import PyBaMMParameterBridge
        
        bridge = PyBaMMParameterBridge()
        params = bridge.load_parameters("Chen2020")
        
        assert "raw_params" in params
        assert "capacity_ah" in params or len(params) > 1
    
    @pytest.mark.skipif(not PYBAMM_AVAILABLE, reason="PyBaMM not installed")
    def test_create_chemistry_from_pybamm(self):
        """Test creating chemistry from PyBaMM parameter set."""
        from battery_simulator.chemistry.pybamm_params import PyBaMMParameterBridge
        
        bridge = PyBaMMParameterBridge()
        chem = bridge.create_chemistry_from_pybamm("Chen2020")
        
        assert chem is not None
        assert hasattr(chem, "name")
        assert hasattr(chem, "nominal_capacity")
    
    def test_chemistry_from_pybamm_factory(self):
        """Test Chemistry.from_pybamm factory method."""
        from battery_simulator.chemistry import Chemistry
        
        if not PYBAMM_AVAILABLE:
            with pytest.raises(ImportError):
                Chemistry.from_pybamm("Chen2020")
        else:
            chem = Chemistry.from_pybamm("Chen2020")
            assert chem is not None


class TestPackSimulator:
    """Test pack simulator functionality."""
    
    def test_pack_config_structure(self):
        """Test PackConfiguration structure."""
        from battery_simulator.chemistry.pack_config import PackConfiguration
        
        config = PackConfiguration(series=14, parallel=4)
        
        assert config.series == 14
        assert config.parallel == 4
        assert config.total_cells == 56
        assert config.pack_capacity_ah == config.cell_capacity_ah * 4
    
    def test_pack_config_energy_calculation(self):
        """Test pack energy calculation."""
        from battery_simulator.chemistry.pack_config import PackConfiguration
        
        config = PackConfiguration(
            series=14,
            parallel=4,
            cell_capacity_ah=5.0,
            cell_nominal_voltage=3.7,
        )
        
        expected_energy = (5.0 * 4) * (3.7 * 14) / 1000  # kWh
        assert abs(config.pack_energy_kwh - expected_energy) < 0.01
    
    def test_standard_packs(self):
        """Test standard pack configurations."""
        from battery_simulator.chemistry.pack_config import (
            STANDARD_PACKS, 
            get_standard_pack,
            list_standard_packs,
        )
        
        assert "ev_small" in STANDARD_PACKS
        assert "ebike" in STANDARD_PACKS
        
        ev_pack = get_standard_pack("ev_small")
        assert ev_pack.series > 10
        assert ev_pack.parallel > 1
        
        packs = list_standard_packs()
        assert len(packs) > 0
    
    def test_pack_config_to_dict(self):
        """Test pack configuration serialization."""
        from battery_simulator.chemistry.pack_config import PackConfiguration
        
        config = PackConfiguration(series=14, parallel=4)
        data = config.to_dict()
        
        assert "topology" in data
        assert "electrical" in data
        assert "thermal" in data
        assert data["topology"]["series"] == 14
    
    @pytest.mark.skipif(
        not (PYBAMM_AVAILABLE and LIIONPACK_AVAILABLE),
        reason="PyBaMM or liionpack not installed"
    )
    def test_pack_simulator_initialization(self):
        """Test PackSimulator initialization."""
        from battery_simulator.core.pack_simulator import PackSimulator
        from battery_simulator.chemistry import Chemistry
        
        chem = Chemistry.from_name("NMC811")
        pack = PackSimulator(
            chemistry=chem,
            capacity=3.0,
            series=14,
            parallel=4,
        )
        
        assert pack.pack_config.series == 14
        assert pack.pack_config.parallel == 4
        assert pack.pack_config.total_cells == 56
    
    @pytest.mark.skipif(
        not (PYBAMM_AVAILABLE and LIIONPACK_AVAILABLE),
        reason="PyBaMM or liionpack not installed"
    )
    def test_pack_state(self):
        """Test pack state tracking."""
        from battery_simulator.core.pack_simulator import PackSimulator
        from battery_simulator.chemistry import Chemistry
        
        chem = Chemistry.from_name("NMC811")
        pack = PackSimulator(
            chemistry=chem,
            series=4,
            parallel=2,
        )
        
        state = pack.get_state_dict()
        
        assert "soc" in state
        assert "pack_voltage" in state
        assert "series" in state
        assert "parallel" in state
        assert "total_cells" in state
        assert state["total_cells"] == 8


class TestSimulationResults:
    """Test simulation results with mode information."""
    
    def test_results_include_mode(self):
        """Test that results include simulation mode info."""
        from battery_simulator import BatterySimulator, Protocol
        from battery_simulator.core.simulator import SimulationMode
        
        sim = BatterySimulator(
            chemistry="NMC811",
            mode=SimulationMode.FAST,
        )
        
        protocol = Protocol.cycle_life(cycles=1)
        results = sim.run(protocol, show_progress=False)
        
        assert results.simulation_mode == "fast"
        assert results.backend_info is not None
        assert results.backend_info["type"] == "fast"
    
    def test_results_to_dict_includes_mode(self):
        """Test that results dict includes simulation mode."""
        from battery_simulator import BatterySimulator, Protocol
        
        sim = BatterySimulator(chemistry="NMC811")
        protocol = Protocol.cycle_life(cycles=1)
        results = sim.run(protocol, show_progress=False)
        
        results_dict = results.to_dict()
        
        assert "simulation_mode" in results_dict["test_metadata"]
        assert "backend_info" in results_dict["test_metadata"]


class TestModeComparison:
    """Test comparing outputs between modes."""
    
    @pytest.mark.skipif(not PYBAMM_AVAILABLE, reason="PyBaMM not installed")
    def test_fast_vs_pybamm_output_structure(self):
        """Test that FAST and PyBaMM modes produce compatible outputs."""
        from battery_simulator import BatterySimulator, Protocol
        from battery_simulator.core.simulator import SimulationMode
        
        # Run FAST simulation
        sim_fast = BatterySimulator(
            chemistry="NMC811",
            mode=SimulationMode.FAST,
        )
        protocol = Protocol.cycle_life(cycles=1)
        results_fast = sim_fast.run(protocol, show_progress=False)
        
        # Run PyBaMM simulation
        sim_hifi = BatterySimulator(
            chemistry="NMC811",
            mode=SimulationMode.HIGH_FIDELITY,
            pybamm_model="SPM",  # Fastest PyBaMM model
        )
        results_hifi = sim_hifi.run(protocol, show_progress=False)
        
        # Check both have same structure
        assert hasattr(results_fast, 'capacity_retention')
        assert hasattr(results_hifi, 'capacity_retention')
        assert hasattr(results_fast, 'cycles_completed')
        assert hasattr(results_hifi, 'cycles_completed')
        
        # Both should complete 1 cycle
        assert results_fast.cycles_completed == 1
        assert results_hifi.cycles_completed == 1

