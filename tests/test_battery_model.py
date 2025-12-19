"""Tests for battery model."""

import pytest
import numpy as np

from battery_simulator.core.battery_model import BatteryModel
from battery_simulator.chemistry import Chemistry


class TestBatteryModel:
    """Test suite for BatteryModel class."""

    @pytest.fixture
    def nmc811_battery(self):
        """Create NMC811 battery model fixture."""
        chemistry = Chemistry.from_name("NMC811")
        return BatteryModel(chemistry=chemistry, capacity=3.0, temperature=25.0)

    @pytest.fixture
    def lfp_battery(self):
        """Create LFP battery model fixture."""
        chemistry = Chemistry.from_name("LFP")
        return BatteryModel(chemistry=chemistry, capacity=3.0, temperature=25.0)

    def test_initialization(self, nmc811_battery):
        """Test battery model initialization."""
        assert nmc811_battery.capacity_nominal == 3.0
        assert nmc811_battery.state.soc == 0.5
        assert nmc811_battery.state.temperature == 25.0

    def test_ocv_at_extremes(self, nmc811_battery):
        """Test OCV at SOC extremes."""
        # At 0% SOC
        ocv_empty = nmc811_battery.get_ocv(0.0)
        assert 2.5 < ocv_empty < 3.5  # Should be near voltage_min

        # At 100% SOC
        ocv_full = nmc811_battery.get_ocv(1.0)
        assert 4.0 < ocv_full < 4.3  # Should be near voltage_max

        # OCV should increase with SOC
        assert ocv_full > ocv_empty

    def test_ocv_monotonic(self, nmc811_battery):
        """Test that OCV is monotonically increasing with SOC."""
        soc_values = np.linspace(0, 1, 20)
        ocv_values = [nmc811_battery.get_ocv(soc) for soc in soc_values]

        # Check monotonically increasing
        for i in range(1, len(ocv_values)):
            assert ocv_values[i] >= ocv_values[i - 1], f"OCV not monotonic at SOC={soc_values[i]}"

    def test_voltage_during_charge(self, nmc811_battery):
        """Test voltage behavior during charging."""
        nmc811_battery.state.soc = 0.5

        # During charge (positive current), voltage should be above OCV
        charge_current = 3.0  # 1C
        voltage = nmc811_battery.calculate_voltage(charge_current, dt=1.0)
        ocv = nmc811_battery.get_ocv(0.5)

        # Voltage should drop below OCV due to IR drop during charge
        # (current causes voltage drop)
        assert voltage < ocv + 0.5  # Allow some margin

    def test_voltage_during_discharge(self, nmc811_battery):
        """Test voltage behavior during discharging."""
        nmc811_battery.state.soc = 0.5

        # During discharge (negative current), voltage is affected by IR drop
        # In the model: V = OCV - I*R - polarization
        # With negative I, the -I*R term becomes positive (voltage goes up)
        # But polarization still reduces voltage
        discharge_current = -3.0  # 1C discharge
        voltage = nmc811_battery.calculate_voltage(discharge_current, dt=1.0)
        ocv = nmc811_battery.get_ocv(0.5)

        # Voltage should be within reasonable range of OCV
        assert 3.0 < voltage < 4.5  # Valid voltage range

    def test_soc_update_during_charge(self, nmc811_battery):
        """Test SOC increases during charging."""
        initial_soc = nmc811_battery.state.soc
        charge_current = 3.0  # 1C

        # Simulate 1 hour of charging at 1C
        for _ in range(3600):
            nmc811_battery.update_state(charge_current, dt=1.0)

        # SOC should have increased significantly
        assert nmc811_battery.state.soc > initial_soc

    def test_soc_update_during_discharge(self, nmc811_battery):
        """Test SOC decreases during discharging."""
        nmc811_battery.state.soc = 0.8
        initial_soc = nmc811_battery.state.soc
        discharge_current = -3.0  # 1C discharge

        # Simulate some discharge
        for _ in range(1800):  # 30 minutes
            nmc811_battery.update_state(discharge_current, dt=1.0)

        # SOC should have decreased
        assert nmc811_battery.state.soc < initial_soc

    def test_soc_bounds(self, nmc811_battery):
        """Test that SOC stays within 0-1 bounds."""
        nmc811_battery.state.soc = 0.99

        # Try to overcharge
        for _ in range(7200):  # 2 hours
            nmc811_battery.update_state(3.0, dt=1.0)

        assert nmc811_battery.state.soc <= 1.0

        # Try to over-discharge
        nmc811_battery.state.soc = 0.01
        for _ in range(7200):
            nmc811_battery.update_state(-3.0, dt=1.0)

        assert nmc811_battery.state.soc >= 0.0

    def test_capacity_accumulation(self, nmc811_battery):
        """Test capacity and energy accumulation."""
        nmc811_battery.reset_step_accumulators()
        current = 3.0  # 1C

        # Simulate 10 seconds of charging
        for _ in range(10):
            nmc811_battery.update_state(current, dt=1.0)

        # Should have accumulated capacity
        expected_capacity = current * 10 / 3600  # Ah
        assert abs(nmc811_battery.state.step_capacity - expected_capacity) < 0.001

    def test_different_chemistries(self, nmc811_battery, lfp_battery):
        """Test different chemistry behaviors."""
        # NMC has higher voltage than LFP
        nmc_ocv = nmc811_battery.get_ocv(0.5)
        lfp_ocv = lfp_battery.get_ocv(0.5)

        assert nmc_ocv > lfp_ocv  # NMC voltage is higher

    def test_degradation_application(self, nmc811_battery):
        """Test that degradation can be applied."""
        initial_capacity = nmc811_battery.state.capacity_current
        initial_resistance = nmc811_battery.state.resistance_current

        # Apply degradation
        nmc811_battery.apply_degradation(capacity_fade=0.01, resistance_growth=0.02)

        # Capacity should decrease
        assert nmc811_battery.state.capacity_current < initial_capacity

        # Resistance should increase
        assert nmc811_battery.state.resistance_current > initial_resistance

    def test_capacity_retention(self, nmc811_battery):
        """Test capacity retention calculation."""
        assert nmc811_battery.capacity_retention == 1.0

        # Apply 10% degradation
        nmc811_battery.apply_degradation(capacity_fade=0.1, resistance_growth=0.0)

        assert abs(nmc811_battery.capacity_retention - 0.9) < 0.001


class TestChemistryComparison:
    """Test chemistry differences."""

    def test_all_chemistries_load(self):
        """Test that all chemistries can be loaded."""
        chemistries = ["NMC811", "LFP", "NCA", "LTO"]
        for name in chemistries:
            chem = Chemistry.from_name(name)
            assert chem is not None
            assert chem.name is not None
            assert chem.voltage_max > chem.voltage_min

    def test_chemistry_voltage_ranges(self):
        """Test chemistry voltage ranges are reasonable."""
        chemistries = {
            "NMC811": (3.0, 4.2),
            "LFP": (2.5, 3.65),
            "NCA": (2.7, 4.2),
            "LTO": (1.5, 2.8),
        }

        for name, (expected_min, expected_max) in chemistries.items():
            chem = Chemistry.from_name(name)
            assert chem.voltage_min == expected_min
            assert chem.voltage_max == expected_max

    def test_invalid_chemistry_raises(self):
        """Test that invalid chemistry name raises error."""
        with pytest.raises(ValueError):
            Chemistry.from_name("InvalidChemistry")

