"""Tests for protocol definitions."""

import pytest

from battery_simulator.protocols import Protocol
from battery_simulator.protocols.base_protocol import ChargeStep, DischargeStep, RestStep
from battery_simulator.chemistry import Chemistry
from battery_simulator.core.battery_model import BatteryModel


class TestProtocolSteps:
    """Test individual protocol steps."""

    @pytest.fixture
    def battery(self):
        """Create battery fixture."""
        chemistry = Chemistry.from_name("NMC811")
        return BatteryModel(chemistry=chemistry, capacity=3.0)

    def test_charge_step_current(self, battery):
        """Test charge step returns positive current."""
        step = ChargeStep(current_rate=1.0, voltage_cutoff=4.2)
        current = step.get_current(battery)
        
        assert current > 0
        assert abs(current - 3.0) < 0.001  # 1C = 3A for 3Ah battery

    def test_discharge_step_current(self, battery):
        """Test discharge step returns negative current."""
        step = DischargeStep(current_rate=1.0, voltage_cutoff=3.0)
        current = step.get_current(battery)
        
        assert current < 0
        assert abs(current + 3.0) < 0.001  # -1C = -3A

    def test_rest_step_zero_current(self, battery):
        """Test rest step returns zero current."""
        step = RestStep(duration=300)
        current = step.get_current(battery)
        
        assert current == 0

    def test_rest_step_completion(self, battery):
        """Test rest step completes after duration."""
        step = RestStep(duration=300)
        
        assert not step.is_complete(battery, step_time=100)
        assert not step.is_complete(battery, step_time=299)
        assert step.is_complete(battery, step_time=300)
        assert step.is_complete(battery, step_time=500)

    def test_step_type_properties(self, battery):
        """Test step type detection properties."""
        charge = ChargeStep(current_rate=1.0, voltage_cutoff=4.2)
        discharge = DischargeStep(current_rate=1.0, voltage_cutoff=3.0)
        rest = RestStep(duration=300)

        assert charge.is_charge
        assert not charge.is_discharge
        assert not charge.is_rest

        assert not discharge.is_charge
        assert discharge.is_discharge
        assert not discharge.is_rest

        assert not rest.is_charge
        assert not rest.is_discharge
        assert rest.is_rest


class TestCycleLifeProtocol:
    """Test cycle life protocol."""

    def test_creation(self):
        """Test protocol creation."""
        protocol = Protocol.cycle_life(
            charge_rate=1.0,
            discharge_rate=1.0,
            cycles=100
        )
        
        assert protocol.cycles == 100
        assert len(protocol.steps) == 4  # Charge, rest, discharge, rest

    def test_step_order(self):
        """Test steps are in correct order."""
        protocol = Protocol.cycle_life(cycles=10)
        
        assert protocol.steps[0].is_charge
        assert protocol.steps[1].is_rest
        assert protocol.steps[2].is_discharge
        assert protocol.steps[3].is_rest

    def test_custom_rates(self):
        """Test custom charge/discharge rates."""
        protocol = Protocol.cycle_life(
            charge_rate=0.5,
            discharge_rate=2.0,
            cycles=10
        )
        
        charge_step = protocol.steps[0]
        discharge_step = protocol.steps[2]
        
        assert charge_step.current_rate == 0.5
        assert discharge_step.current_rate == 2.0


class TestFormationProtocol:
    """Test formation protocol."""

    def test_creation(self):
        """Test formation protocol creation."""
        protocol = Protocol.formation(cycles=3)
        
        assert protocol.cycles == 3
        assert len(protocol.steps) > 0

    def test_low_rate(self):
        """Test formation uses low rate."""
        protocol = Protocol.formation(initial_rate=0.1)
        
        charge_step = protocol.steps[0]
        assert charge_step.current_rate == 0.1


class TestRateCapabilityProtocol:
    """Test rate capability protocol."""

    def test_creation(self):
        """Test rate capability protocol creation."""
        protocol = Protocol.rate_capability(
            rates=[0.2, 0.5, 1.0, 2.0],
            cycles_per_rate=3
        )
        
        assert protocol.cycles == 12  # 4 rates * 3 cycles

    def test_rate_list(self):
        """Test custom rate list."""
        rates = [0.5, 1.0, 2.0]
        protocol = Protocol.rate_capability(rates=rates)
        
        assert protocol.rates == rates


class TestRPTProtocol:
    """Test RPT protocol."""

    def test_creation(self):
        """Test RPT protocol creation."""
        protocol = Protocol.rpt()
        
        assert protocol.cycles == 1  # RPT is a single test
        assert len(protocol.steps) > 0

    def test_includes_pulse_steps(self):
        """Test RPT includes pulse steps for resistance."""
        protocol = Protocol.rpt(pulse_soc_points=[0.5])
        
        step_types = [step.step_type for step in protocol.steps]
        assert any("pulse" in st for st in step_types)


class TestProtocolFactory:
    """Test Protocol factory methods."""

    def test_all_protocols_create(self):
        """Test all protocol types can be created."""
        protocols = [
            Protocol.formation(),
            Protocol.cycle_life(),
            Protocol.rate_capability(),
            Protocol.rpt(),
        ]
        
        for p in protocols:
            assert p is not None
            assert p.cycles >= 1
            assert len(p.steps) >= 1

    def test_protocol_to_dict(self):
        """Test protocol serialization."""
        protocol = Protocol.cycle_life(cycles=10)
        data = protocol.to_dict()
        
        assert "name" in data
        assert "cycles" in data
        assert "steps" in data
        assert data["cycles"] == 10

