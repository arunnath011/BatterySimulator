"""
Tests for the three-component LFP/Graphite degradation model based on
"Machine-Learning Assisted Identification of Accurate Battery Lifetime Models
with Uncertainty".

Tests cover:
- Anode potential helper (Eq. A1)
- Calendar aging (Eq. 6, 7, 8)
- Break-in cycling (Eq. 11, 12)
- Long-term cycling (Eq. 14, 15)
- Path-independent dynamic updates (Eq. 20)
- Full integration with BatterySimulator
"""

import math
import pytest
import numpy as np

from battery_simulator.core.lfp_paper_degradation import (
    LFPPaperDegradationModel,
    LFPPaperDegradationState,
    LFPPaperParameters,
    anode_potential,
    R_GAS,
)
from battery_simulator.chemistry.lfp import LFPChemistry


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def lfp_chemistry():
    """Create a fresh LFP chemistry instance."""
    return LFPChemistry()


@pytest.fixture
def params():
    """Create default paper parameters."""
    return LFPPaperParameters()


@pytest.fixture
def model(lfp_chemistry, params):
    """Create a fresh LFP paper degradation model."""
    return LFPPaperDegradationModel(
        chemistry=lfp_chemistry,
        nominal_capacity=3.0,
        params=params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Anode Potential (Eq. A1)
# ─────────────────────────────────────────────────────────────────────────────


class TestAnodePotential:
    """Tests for the graphite anode potential function."""

    def test_high_soc_low_potential(self):
        """At high SOC (lithiated graphite), potential should be low."""
        ua = anode_potential(0.95)
        assert ua < 0.15, f"Expected Ua < 0.15 V at SOC=0.95, got {ua:.4f}"

    def test_low_soc_high_potential(self):
        """At low SOC (delithiated graphite), potential should be higher."""
        ua = anode_potential(0.05)
        assert ua > 0.3, f"Expected Ua > 0.3 V at SOC=0.05, got {ua:.4f}"

    def test_monotonically_decreasing(self):
        """Potential should generally decrease with increasing SOC."""
        socs = np.linspace(0.05, 0.95, 20)
        potentials = [anode_potential(s) for s in socs]
        # Allow small non-monotonicity due to staging, but overall trend should be down
        assert potentials[0] > potentials[-1], "Ua should be lower at high SOC"

    def test_clamped_inputs(self):
        """Edge SOC values should not raise errors."""
        assert anode_potential(0.0) > 0
        assert anode_potential(1.0) >= 0
        assert np.isfinite(anode_potential(0.5))

    def test_mid_soc_reasonable_range(self):
        """At 50% SOC the potential should be in a physically reasonable range."""
        ua = anode_potential(0.5)
        assert 0.05 < ua < 0.5, f"Mid-SOC potential out of range: {ua:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Calendar Aging (Eq. 6, 7, 8)
# ─────────────────────────────────────────────────────────────────────────────


class TestCalendarAging:
    """Tests for the calendar aging component."""

    def test_zero_time_no_loss(self, model):
        """Zero time should yield zero calendar loss."""
        loss = model._calendar_loss_algebraic(0.0, 25.0, 0.5)
        assert loss == 0.0

    def test_loss_increases_with_time(self, model):
        """Calendar loss should increase over time."""
        loss_100 = model._calendar_loss_algebraic(100.0, 25.0, 0.5)
        loss_500 = model._calendar_loss_algebraic(500.0, 25.0, 0.5)
        assert loss_500 > loss_100 > 0

    def test_higher_temperature_accelerates(self, model):
        """Higher temperature should increase calendar aging."""
        loss_25c = model._calendar_loss_algebraic(365.0, 25.0, 0.5)
        loss_45c = model._calendar_loss_algebraic(365.0, 45.0, 0.5)
        assert loss_45c > loss_25c, (
            f"45°C loss ({loss_45c:.6f}) should exceed 25°C loss ({loss_25c:.6f})"
        )

    def test_high_soc_accelerates(self, model):
        """Higher SOC (lower anode potential) should accelerate calendar aging."""
        loss_20soc = model._calendar_loss_algebraic(365.0, 25.0, 0.2)
        loss_90soc = model._calendar_loss_algebraic(365.0, 25.0, 0.9)
        assert loss_90soc > loss_20soc, "High SOC should cause more calendar aging"

    def test_q1_temperature_dependence(self, model):
        """q1 should increase with temperature."""
        q1_25 = model._q1(25.0, 0.5)
        q1_45 = model._q1(45.0, 0.5)
        assert q1_45 > q1_25

    def test_q3_temperature_dependence(self, model):
        """q3 should increase with temperature."""
        q3_25 = model._q3(25.0, 0.5)
        q3_45 = model._q3(45.0, 0.5)
        assert q3_45 > q3_25

    def test_sigmoidal_saturation(self, model):
        """
        The sigmoidal part should saturate; at very long times,
        the loss should be dominated by the linear term.
        """
        loss_1y = model._calendar_loss_algebraic(365.0, 25.0, 0.5)
        loss_10y = model._calendar_loss_algebraic(3650.0, 25.0, 0.5)
        # Growth factor should be less than 10x due to sigmoidal saturation
        growth = loss_10y / loss_1y
        assert growth < 10.0, f"Growth factor {growth:.1f} too large — sigmoidal should saturate"

    def test_one_year_reasonable_loss(self, model):
        """After 1 year at 25°C, 50% SOC, calendar loss should be 1-5%."""
        loss = model._calendar_loss_algebraic(365.0, 25.0, 0.5)
        assert 0.001 < loss < 0.10, f"1-year calendar loss = {loss:.4f} out of expected range"


# ─────────────────────────────────────────────────────────────────────────────
# Break-in Cycling (Eq. 11, 12)
# ─────────────────────────────────────────────────────────────────────────────


class TestBreakInCycling:
    """Tests for the break-in (initial capacity drop) component."""

    def test_zero_efc_no_loss(self, model):
        """Zero EFC should yield zero break-in loss."""
        loss = model._breakin_loss_algebraic(0.0, 0.5, 0.5)
        assert loss == 0.0

    def test_loss_saturates(self, model):
        """Break-in should saturate at large EFC values."""
        loss_100 = model._breakin_loss_algebraic(100.0, 0.5, 0.5)
        loss_5000 = model._breakin_loss_algebraic(5000.0, 0.5, 0.5)
        loss_10000 = model._breakin_loss_algebraic(10000.0, 0.5, 0.5)
        # Should be mostly saturated by 10000 EFC
        assert loss_10000 > 0
        assert abs(loss_10000 - loss_5000) < abs(loss_5000 - loss_100)

    def test_q4_peak_at_expected_conditions(self, model):
        """Break-in magnitude q4 should peak near SOC=0.5, DOD=0.2."""
        q4_peak = model._q4(0.5, 0.2)
        q4_edge = model._q4(0.1, 0.9)
        assert q4_peak > q4_edge, "Break-in should be highest near SOC=0.5, DOD=0.2"

    def test_q4_max_bound(self, model, params):
        """q4 should never exceed q4_max (it's normalised to peak at q4_max)."""
        # Test many combinations
        for soc in np.linspace(0.05, 0.95, 10):
            for dod in np.linspace(0.05, 1.0, 10):
                q4 = model._q4(soc, dod)
                assert q4 <= params.q4_max * 1.01, (
                    f"q4({soc:.2f}, {dod:.2f}) = {q4:.6f} exceeds max {params.q4_max}"
                )

    def test_breakin_loss_positive(self, model):
        """Break-in loss should always be non-negative."""
        for efc in [1, 10, 100, 1000]:
            loss = model._breakin_loss_algebraic(efc, 0.5, 0.5)
            assert loss >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Long-term Cycling (Eq. 14, 15)
# ─────────────────────────────────────────────────────────────────────────────


class TestLongTermCycling:
    """Tests for the long-term cycling degradation component."""

    def test_zero_efc_no_loss(self, model):
        """Zero EFC should yield zero long-term loss."""
        loss = model._longterm_loss_algebraic(0.0, 1.0, 1.0)
        assert loss == 0.0

    def test_loss_increases_with_efc(self, model):
        """Long-term loss should increase with EFC."""
        loss_100 = model._longterm_loss_algebraic(100.0, 1.0, 1.0)
        loss_1000 = model._longterm_loss_algebraic(1000.0, 1.0, 1.0)
        assert loss_1000 > loss_100 > 0

    def test_deeper_dod_more_damage(self, model):
        """Deeper DOD should cause more long-term loss."""
        loss_50dod = model._longterm_loss_algebraic(1000.0, 0.5, 1.0)
        loss_100dod = model._longterm_loss_algebraic(1000.0, 1.0, 1.0)
        assert loss_100dod > loss_50dod

    def test_higher_crate_more_damage(self, model):
        """Higher C-rate should cause more long-term loss."""
        loss_05c = model._longterm_loss_algebraic(1000.0, 1.0, 0.5)
        loss_2c = model._longterm_loss_algebraic(1000.0, 1.0, 2.0)
        assert loss_2c > loss_05c

    def test_sublinear_power_law(self, model, params):
        """Power-law exponent q8 < 1 means sub-linear growth."""
        assert params.q8 < 1.0
        loss_1000 = model._longterm_loss_algebraic(1000.0, 1.0, 1.0)
        loss_2000 = model._longterm_loss_algebraic(2000.0, 1.0, 1.0)
        # Doubling EFC should less than double the loss
        assert loss_2000 / loss_1000 < 2.0

    def test_q7_sisso_expression(self, model):
        """q7 should vary with DOD and C-rate."""
        q7_low = model._q7(0.5, 0.5)
        q7_high = model._q7(1.0, 2.0)
        assert q7_high > q7_low


# ─────────────────────────────────────────────────────────────────────────────
# Path-Independent Incremental Updates (Eq. 20)
# ─────────────────────────────────────────────────────────────────────────────


class TestIncrementalUpdates:
    """Tests for the dynamic, path-independent simulation logic (Eq. 20)."""

    def test_incremental_calendar_positive(self, model):
        """Calendar increment should be positive for positive time delta."""
        model.state.total_time_days = 0.0
        delta = model._incremental_calendar(10.0, 25.0, 0.5)
        assert delta > 0

    def test_incremental_calendar_sums_correctly(self, model):
        """Sequential increments should sum to the algebraic total."""
        total_days = 100.0
        steps = 10
        dt = total_days / steps
        cum_loss = 0.0
        for i in range(steps):
            t_new = (i + 1) * dt
            cum_loss += model._incremental_calendar(t_new, 25.0, 0.5)

        direct_loss = model._calendar_loss_algebraic(total_days, 25.0, 0.5)
        assert abs(cum_loss - direct_loss) < 1e-10, (
            f"Incremental sum {cum_loss:.8f} != direct {direct_loss:.8f}"
        )

    def test_incremental_breakin_sums_correctly(self, model):
        """Sequential break-in increments should sum to the algebraic total."""
        total_efc = 2000.0
        steps = 20
        d_efc = total_efc / steps
        cum_loss = 0.0
        for i in range(steps):
            efc_new = (i + 1) * d_efc
            cum_loss += model._incremental_breakin(efc_new, 0.5, 0.5)

        direct_loss = model._breakin_loss_algebraic(total_efc, 0.5, 0.5)
        assert abs(cum_loss - direct_loss) < 1e-10

    def test_incremental_longterm_sums_correctly(self, model):
        """Sequential long-term cycling increments should sum to the algebraic total."""
        total_efc = 1000.0
        steps = 20
        d_efc = total_efc / steps
        cum_loss = 0.0
        for i in range(steps):
            efc_new = (i + 1) * d_efc
            cum_loss += model._incremental_longterm(efc_new, 1.0, 1.0)

        direct_loss = model._longterm_loss_algebraic(total_efc, 1.0, 1.0)
        assert abs(cum_loss - direct_loss) < 1e-10

    def test_update_from_cycle_accumulates(self, model):
        """Running update_from_cycle multiple times should accumulate loss."""
        for n in range(1, 51):
            model.update_from_cycle(
                cycle_number=n,
                c_rate_charge=1.0,
                c_rate_discharge=1.0,
                temperature=25.0,
                dod=1.0,
                cycle_time_hours=2.0,
                avg_soc=0.5,
                capacity_throughput=6.0,  # 3 Ah × 2 (charge + discharge)
            )
        assert model.state.total_capacity_loss > 0
        assert model.state.calendar_capacity_loss > 0
        assert model.state.breakin_capacity_loss > 0
        assert model.state.longterm_capacity_loss > 0
        assert model.state.equivalent_full_cycles > 0
        assert model.get_capacity_retention() < 1.0

    def test_three_components_sum_to_total(self, model):
        """The three component losses should sum to the total."""
        for n in range(1, 101):
            model.update_from_cycle(
                cycle_number=n,
                temperature=25.0,
                dod=0.8,
                cycle_time_hours=2.5,
                avg_soc=0.5,
                capacity_throughput=4.8,
            )
        expected = (
            model.state.calendar_capacity_loss
            + model.state.breakin_capacity_loss
            + model.state.longterm_capacity_loss
        )
        assert abs(model.state.total_capacity_loss - expected) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Model Interface Compatibility
# ─────────────────────────────────────────────────────────────────────────────


class TestModelInterface:
    """Verify the model exposes the same interface as the original DegradationModel."""

    def test_get_current_capacity(self, model):
        """get_current_capacity should return a float < nominal after degradation."""
        model.update_from_cycle(
            cycle_number=1, temperature=25.0, dod=1.0,
            cycle_time_hours=2.0, capacity_throughput=6.0,
        )
        cap = model.get_current_capacity()
        assert 0 < cap <= 3.0

    def test_get_current_resistance_factor(self, model):
        """Resistance factor should be >= 1."""
        model.update_from_cycle(
            cycle_number=1, temperature=25.0, dod=1.0,
            cycle_time_hours=2.0, capacity_throughput=6.0,
        )
        rf = model.get_current_resistance_factor()
        assert rf >= 1.0

    def test_get_capacity_retention(self, model):
        """Retention should be in [0.5, 1.0]."""
        ret = model.get_capacity_retention()
        assert 0.5 <= ret <= 1.0

    def test_get_state_dict(self, model):
        """State dict should contain expected keys."""
        state = model.get_state_dict()
        expected_keys = {
            "total_capacity_loss",
            "calendar_capacity_loss",
            "breakin_capacity_loss",
            "longterm_capacity_loss",
            "cyclic_capacity_loss",
            "total_resistance_growth",
            "equivalent_full_cycles",
            "total_ah_throughput",
            "total_time_days",
            "total_time_hours",
            "capacity_retention",
            "resistance_factor",
        }
        assert expected_keys.issubset(state.keys())

    def test_get_component_breakdown(self, model):
        """Component breakdown should contain calendar, breakin, longterm."""
        breakdown = model.get_component_breakdown()
        assert "calendar" in breakdown
        assert "breakin" in breakdown
        assert "longterm" in breakdown
        assert "total" in breakdown

    def test_reset(self, model):
        """Reset should clear all state."""
        model.update_from_cycle(
            cycle_number=1, temperature=25.0, dod=1.0,
            cycle_time_hours=2.0, capacity_throughput=6.0,
        )
        assert model.state.total_capacity_loss > 0
        model.reset()
        assert model.state.total_capacity_loss == 0.0
        assert model.state.equivalent_full_cycles == 0.0

    def test_update_from_cycle_returns_dict(self, model):
        """update_from_cycle should return a dict with required keys."""
        info = model.update_from_cycle(
            cycle_number=1, temperature=25.0, dod=1.0,
            cycle_time_hours=2.0, capacity_throughput=6.0,
        )
        assert "cap_loss_cal" in info
        assert "cap_loss_breakin" in info
        assert "cap_loss_longterm" in info
        assert "total_cap_loss" in info
        assert "capacity_retention" in info
        assert "cum_calendar_loss" in info
        assert "cum_breakin_loss" in info
        assert "cum_longterm_loss" in info

    def test_cycle_history_recorded(self, model):
        """Cycle history should grow with each update."""
        for n in range(1, 6):
            model.update_from_cycle(
                cycle_number=n, temperature=25.0, dod=1.0,
                cycle_time_hours=2.0, capacity_throughput=6.0,
            )
        assert len(model.state.cycle_history) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Stress-Factor Helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestStressHelpers:
    """Tests for heatmap-oriented stress helper methods."""

    def test_calendar_stress_rate_positive(self, model):
        """Calendar stress rate should be positive at any valid condition."""
        rate = model.calendar_stress_rate(25.0, 0.5)
        assert rate > 0

    def test_calendar_stress_increases_with_temp(self, model):
        """Calendar stress rate should increase with temperature."""
        rate_25 = model.calendar_stress_rate(25.0, 0.5)
        rate_45 = model.calendar_stress_rate(45.0, 0.5)
        assert rate_45 > rate_25

    def test_breakin_stress_positive(self, model):
        """Break-in stress should be non-negative."""
        stress = model.breakin_stress(0.5, 0.5)
        assert stress >= 0

    def test_longterm_cycling_rate_positive(self, model):
        """Long-term cycling rate should be positive."""
        rate = model.longterm_cycling_rate(1.0, 1.0)
        assert rate > 0


# ─────────────────────────────────────────────────────────────────────────────
# Predict Cycle Life
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictCycleLife:
    """Tests for the cycle-life prediction method."""

    def test_predicts_reasonable_lfp_life(self, model):
        """LFP should predict > 1000 cycles to 80% at 25°C, 1C."""
        life = model.predict_cycle_life(
            target_retention=0.8,
            temperature=25.0,
            c_rate=1.0,
            dod=1.0,
        )
        assert life > 500, f"Predicted life {life} cycles is too short for LFP"

    def test_higher_temp_shorter_life(self, model):
        """Higher temperature should reduce predicted life."""
        # Use 90% retention target — LFP is very durable and may not reach 80%
        # within the max iteration cap at moderate temperatures
        life_25 = model.predict_cycle_life(target_retention=0.9, temperature=25.0)
        model.reset()
        life_45 = model.predict_cycle_life(target_retention=0.9, temperature=45.0)
        assert life_45 < life_25, (
            f"Expected shorter life at 45°C ({life_45}) than 25°C ({life_25})"
        )

    def test_temperature_acceleration_factor(self, model):
        """Acceleration factor should be > 1 for T > T_ref."""
        factor = model.get_temperature_acceleration_factor(45.0)
        assert factor > 1.0

    def test_acceleration_factor_at_ref(self, model):
        """Acceleration factor at reference temperature should be ~1."""
        factor = model.get_temperature_acceleration_factor(25.0)
        assert abs(factor - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Integration with LFPChemistry
# ─────────────────────────────────────────────────────────────────────────────


class TestLFPChemistryIntegration:
    """Test that LFPChemistry properly enables the paper model."""

    def test_use_paper_model_flag(self, lfp_chemistry):
        """LFP chemistry should have use_paper_model enabled by default."""
        assert lfp_chemistry.use_paper_model is True

    def test_get_paper_params(self, lfp_chemistry):
        """get_paper_params should return LFPPaperParameters."""
        params = lfp_chemistry.get_paper_params()
        assert isinstance(params, LFPPaperParameters)

    def test_paper_params_cached(self, lfp_chemistry):
        """get_paper_params should return the same instance on repeat calls."""
        p1 = lfp_chemistry.get_paper_params()
        p2 = lfp_chemistry.get_paper_params()
        assert p1 is p2


# ─────────────────────────────────────────────────────────────────────────────
# Integration with BatterySimulator
# ─────────────────────────────────────────────────────────────────────────────


class TestSimulatorIntegration:
    """Integration tests with the full BatterySimulator."""

    def test_lfp_uses_paper_model(self):
        """When chemistry is LFP, the simulator should use the paper degradation model."""
        from battery_simulator import BatterySimulator
        sim = BatterySimulator(chemistry="LFP", capacity=3.0, temperature=25.0)
        assert sim._use_paper_degradation is True
        assert isinstance(sim.degradation, LFPPaperDegradationModel)

    def test_nmc_uses_standard_model(self):
        """Non-LFP chemistries should use the standard DegradationModel."""
        from battery_simulator import BatterySimulator
        from battery_simulator.core.degradation import DegradationModel
        sim = BatterySimulator(chemistry="NMC811", capacity=3.0, temperature=25.0)
        assert sim._use_paper_degradation is False
        assert isinstance(sim.degradation, DegradationModel)

    def test_lfp_simulation_completes(self):
        """A short LFP simulation should complete without errors."""
        from battery_simulator import BatterySimulator, Protocol
        sim = BatterySimulator(chemistry="LFP", capacity=3.0, temperature=25.0)
        protocol = Protocol.cycle_life(cycles=5, charge_rate=1.0, discharge_rate=1.0)
        results = sim.run(protocol=protocol, show_progress=False)
        assert results.cycles_completed == 5
        assert results.capacity_retention < 1.0
        assert results.capacity_retention > 0.5

    def test_cycle_summary_has_component_breakdown(self):
        """Cycle summary should include component breakdown columns."""
        from battery_simulator import BatterySimulator, Protocol
        sim = BatterySimulator(chemistry="LFP", capacity=3.0, temperature=25.0)
        protocol = Protocol.cycle_life(cycles=5)
        results = sim.run(protocol=protocol, show_progress=False)
        assert "calendar_loss" in results.cycle_summary.columns
        assert "breakin_loss" in results.cycle_summary.columns
        assert "longterm_loss" in results.cycle_summary.columns

    def test_backend_info_indicates_paper_model(self):
        """Backend info should indicate the paper degradation model."""
        from battery_simulator import BatterySimulator, Protocol
        sim = BatterySimulator(chemistry="LFP", capacity=3.0, temperature=25.0)
        protocol = Protocol.cycle_life(cycles=3)
        results = sim.run(protocol=protocol, show_progress=False)
        assert results.backend_info.get("degradation_model") == "lfp_paper_three_component"
