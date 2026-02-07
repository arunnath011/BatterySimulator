"""
Three-component LFP/Graphite degradation model based on:
"Machine-Learning Assisted Identification of Accurate Battery Lifetime Models
with Uncertainty"

This model separates capacity loss into three physically motivated components:

1. Calendar Aging (Eq. 6):
    Q_loss_cal = q1 * (1 - exp(-t / q2)) + q3 * t
    where q1 = f(T, Ua) via Eq. 7 and q3 = f(T, Ua) via Eq. 8

2. Cycling Break-in (Eq. 11):
    Q_loss_breakin = q4 * (1 - exp(-EFC / q5))
    where q4 = f(SOC_avg, DOD) via Eq. 12 (skewed normal distribution)

3. Long-term Cycling (Eq. 14):
    Q_loss_lt = q7 * EFC^q8
    where q7 = f(DOD, C_rate) via Eq. 15 (SISSO-identified expression)

Dynamic Simulation (Eq. 20):
    Incremental path-independent updates using virtual time/EFC:
    delta_Q = Q(state_n) - Q(state_{n-1})

Cell: Sony/Murata US26650FTC1 (LFP/Graphite)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.stats import skewnorm

if TYPE_CHECKING:
    from battery_simulator.chemistry.base_chemistry import BaseChemistry

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

R_GAS = 8.314          # Universal gas constant (J mol⁻¹ K⁻¹)
FARADAY = 96485.0      # Faraday constant (C mol⁻¹)


# ─────────────────────────────────────────────────────────────────────────────
# Anode Potential Helper  (Eq. A1)
# ─────────────────────────────────────────────────────────────────────────────

def anode_potential(soc: float) -> float:
    """
    Graphite anode equilibrium potential vs Li/Li+ as a function of SOC.

    Implements Eq. A1 from the paper appendix.  The expression captures the
    staging plateaus that are characteristic of graphite intercalation.

    At high cell SOC  → anode is lithiated → low Ua  → faster SEI growth
    At low  cell SOC  → anode is delithiated → high Ua → slower SEI growth

    Args:
        soc: Cell state of charge (0–1).  Mapped to graphite lithiation x.

    Returns:
        Anode potential Ua in V vs Li/Li+.
    """
    x = np.clip(soc, 0.01, 0.99)

    ua = (
        0.6379
        + 0.5416 * np.exp(-305.5309 * x)
        + 0.044  * np.tanh(-(x - 0.1958) / 0.1088)
        - 0.1978 * np.tanh((x - 1.0571) / 0.0854)
        - 0.6875 * np.tanh((x + 0.0117) / 0.0529)
        - 0.0175 * np.tanh((x - 0.5692) / 0.0875)
    )
    return float(ua)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Dataclass  (Table IV)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LFPPaperParameters:
    """
    Parameters for the three-component LFP degradation model (Table IV).

    All parameters have been calibrated / identified from accelerated aging
    tests on Sony/Murata US26650FTC1 cells.

    The parameter names follow the paper convention (q1 … q8 plus auxiliary).
    """

    # ── Calendar Aging (Eq. 6, 7, 8) ────────────────────────────────────
    # Sigmoidal component: Q_cal_sig = q1 * (1 - exp(-t / q2))
    # Linear component:    Q_cal_lin = q3 * t
    # q1 = q1_a * exp(-q1_Ea / (R·T)) * exp(-q1_c · Ua)
    q1_a: float = 8.98e6           # Pre-exponential factor for q1
    q1_Ea: float = 54_000.0        # Activation energy for q1 (J mol⁻¹)
    q1_c: float = 4.56             # Anode-potential sensitivity for q1

    q2: float = 556.0              # Time constant for sigmoidal term (days)

    # q3 = q3_a * exp(-q3_Ea / (R·T)) * exp(-q3_c · Ua)
    q3_a: float = 3.21e4           # Pre-exponential factor for q3
    q3_Ea: float = 58_000.0        # Activation energy for q3 (J mol⁻¹)
    q3_c: float = 4.02             # Anode-potential sensitivity for q3

    # ── Cycling Break-in (Eq. 11, 12) ───────────────────────────────────
    # Q_breakin = q4 * (1 - exp(-EFC / q5))
    # q4 = f(SOC_avg, DOD) via skewed-normal (Eq. 12)
    q4_max: float = 0.022          # Maximum break-in capacity loss (~2.2 %)
    q5: float = 650.0              # Saturation time constant (EFC)

    # Skewed-normal parameters for q4(SOC_avg, DOD) — Eq. 12
    # q4 = q4_max * SN_soc(SOC_avg) * SN_dod(DOD)
    # SOC component
    sn_soc_loc: float = 0.50       # Location (μ) for SOC skewed normal
    sn_soc_scale: float = 0.28     # Scale (σ) for SOC skewed normal
    sn_soc_alpha: float = 0.0      # Skewness (α) for SOC component (symmetric)
    # DOD component
    sn_dod_loc: float = 0.20       # Location (μ) for DOD skewed normal
    sn_dod_scale: float = 0.18     # Scale (σ) for DOD skewed normal
    sn_dod_alpha: float = 2.5      # Skewness (α) for DOD component

    # ── Long-term Cycling (Eq. 14, 15) ──────────────────────────────────
    # Q_lt = q7 * EFC^q8
    # q7 = f(DOD, C_rate) — SISSO-identified (Eq. 15)
    q7_a: float = 1.20e-5          # Base rate coefficient
    q7_b: float = 1.48             # DOD exponent in SISSO expression
    q7_c: float = 0.362            # C-rate coefficient in SISSO expression
    q7_d: float = 0.0              # Interaction term (DOD × C-rate)
    q8: float = 0.786              # Power-law exponent (sub-linear)

    # ── Resistance growth (coupled to capacity loss) ────────────────────
    r_cal_factor: float = 0.60     # Fraction of calendar cap loss → R growth
    r_cyc_factor: float = 0.45     # Fraction of cycling cap loss → R growth

    # ── Reference conditions ────────────────────────────────────────────
    t_ref: float = 298.15          # Reference temperature (K) — 25 °C


# ─────────────────────────────────────────────────────────────────────────────
# Degradation State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LFPPaperDegradationState:
    """Track the three-component degradation state for the paper model."""

    # ── Component losses (fractional, 0–1) ──────────────────────────────
    calendar_capacity_loss: float = 0.0
    breakin_capacity_loss: float = 0.0
    longterm_capacity_loss: float = 0.0
    total_capacity_loss: float = 0.0

    # ── Resistance growth (fractional) ──────────────────────────────────
    total_resistance_growth: float = 0.0

    # ── Accumulators ────────────────────────────────────────────────────
    total_time_days: float = 0.0           # Total elapsed time in days
    equivalent_full_cycles: float = 0.0    # Total EFC
    total_ah_throughput: float = 0.0       # Total Ah throughput

    # ── Virtual accumulators for path-independent simulation (Eq. 20) ──
    # These store the *previous* value of each component's algebraic
    # expression so we can compute the incremental delta.
    _prev_calendar_loss: float = 0.0
    _prev_breakin_loss: float = 0.0
    _prev_longterm_loss: float = 0.0

    # ── Cumulative stress tracking ──────────────────────────────────────
    cumulative_temperature_stress: float = 0.0
    cumulative_rate_stress: float = 0.0

    # ── History ─────────────────────────────────────────────────────────
    cycle_history: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class LFPPaperDegradationModel:
    """
    Three-component LFP/Graphite degradation model.

    Separates capacity loss into calendar, break-in, and long-term cycling
    components with path-independent incremental updates suitable for
    dynamic (time-varying) operating profiles.

    This model can be used as a drop-in replacement for the generic
    ``DegradationModel`` when simulating LFP cells.
    """

    def __init__(
        self,
        chemistry: BaseChemistry,
        nominal_capacity: float,
        params: Optional[LFPPaperParameters] = None,
    ):
        """
        Initialise the three-component degradation model.

        Args:
            chemistry: Chemistry object (used for interface compatibility).
            nominal_capacity: Nominal cell capacity in Ah.
            params: Paper model parameters.  If *None*, uses default
                    calibrated values from Table IV.
        """
        self.chemistry = chemistry
        self.nominal_capacity = nominal_capacity
        self.params = params or LFPPaperParameters()
        self.state = LFPPaperDegradationState()

        # Pre-compute skewed-normal normalization so the peak value = 1.
        # For skewed normals, the mode (peak) is shifted from the location
        # parameter, so we find the true peak numerically.
        self._sn_soc_peak = self._find_skewnorm_peak(
            self.params.sn_soc_loc,
            self.params.sn_soc_scale,
            self.params.sn_soc_alpha,
        )
        self._sn_dod_peak = self._find_skewnorm_peak(
            self.params.sn_dod_loc,
            self.params.sn_dod_scale,
            self.params.sn_dod_alpha,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helper: Skewed Normal PDF  (used in Eq. 12)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _skewed_normal_pdf(
        x: float, loc: float, scale: float, alpha: float
    ) -> float:
        """Evaluate the skewed-normal PDF at *x*."""
        return float(skewnorm.pdf(x, a=alpha, loc=loc, scale=scale))

    @staticmethod
    def _find_skewnorm_peak(
        loc: float, scale: float, alpha: float
    ) -> float:
        """Find the peak (mode) value of a skewed-normal PDF numerically."""
        xs = np.linspace(loc - 4 * scale, loc + 4 * scale, 1000)
        vals = skewnorm.pdf(xs, a=alpha, loc=loc, scale=scale)
        return float(np.max(vals))

    # ─────────────────────────────────────────────────────────────────────
    # Eq. 7 — Calendar parameter q1(T, Ua)
    # ─────────────────────────────────────────────────────────────────────

    def _q1(self, temperature_c: float, soc: float) -> float:
        """
        Calculate calendar sigmoidal magnitude q1.

        Eq. 7:  q1 = q1_a · exp(-q1_Ea / (R·T)) · exp(-q1_c · Ua)

        Lower anode potential (high SOC) → larger q1 → more calendar loss.
        Higher temperature → larger q1 → more calendar loss.
        """
        p = self.params
        t_k = temperature_c + 273.15
        ua = anode_potential(soc)
        return p.q1_a * math.exp(-p.q1_Ea / (R_GAS * t_k)) * math.exp(-p.q1_c * ua)

    # ─────────────────────────────────────────────────────────────────────
    # Eq. 8 — Calendar parameter q3(T, Ua)
    # ─────────────────────────────────────────────────────────────────────

    def _q3(self, temperature_c: float, soc: float) -> float:
        """
        Calculate calendar linear-aging rate q3.

        Eq. 8:  q3 = q3_a · exp(-q3_Ea / (R·T)) · exp(-q3_c · Ua)
        """
        p = self.params
        t_k = temperature_c + 273.15
        ua = anode_potential(soc)
        return p.q3_a * math.exp(-p.q3_Ea / (R_GAS * t_k)) * math.exp(-p.q3_c * ua)

    # ─────────────────────────────────────────────────────────────────────
    # Eq. 6 — Calendar aging algebraic expression
    # ─────────────────────────────────────────────────────────────────────

    def _calendar_loss_algebraic(
        self, time_days: float, temperature_c: float, soc: float
    ) -> float:
        """
        Calendar aging (Eq. 6):
            Q_loss_cal = q1 · (1 - exp(-t / q2)) + q3 · t

        Returns fractional capacity loss.
        """
        if time_days <= 0:
            return 0.0
        q1 = self._q1(temperature_c, soc)
        q2 = self.params.q2
        q3 = self._q3(temperature_c, soc)

        return q1 * (1.0 - math.exp(-time_days / q2)) + q3 * time_days

    # ─────────────────────────────────────────────────────────────────────
    # Eq. 12 — Break-in parameter q4(SOC_avg, DOD)
    # ─────────────────────────────────────────────────────────────────────

    def _q4(self, soc_avg: float, dod: float) -> float:
        """
        Break-in magnitude q4 as a function of average SOC and DOD.

        Eq. 12 uses a product of two skewed-normal distributions:
            q4 = q4_max · SN_soc(SOC_avg) · SN_dod(DOD)

        Break-in is maximised at ~50 % SOC and ~20 % DOD.
        """
        p = self.params

        sn_soc = self._skewed_normal_pdf(
            soc_avg, p.sn_soc_loc, p.sn_soc_scale, p.sn_soc_alpha
        )
        sn_dod = self._skewed_normal_pdf(
            dod, p.sn_dod_loc, p.sn_dod_scale, p.sn_dod_alpha
        )

        # Normalise so that peak value = q4_max
        norm_soc = sn_soc / self._sn_soc_peak if self._sn_soc_peak > 0 else 0.0
        norm_dod = sn_dod / self._sn_dod_peak if self._sn_dod_peak > 0 else 0.0

        return p.q4_max * norm_soc * norm_dod

    # ─────────────────────────────────────────────────────────────────────
    # Eq. 11 — Break-in cycling algebraic expression
    # ─────────────────────────────────────────────────────────────────────

    def _breakin_loss_algebraic(
        self, efc: float, soc_avg: float, dod: float
    ) -> float:
        """
        Break-in cycling loss (Eq. 11):
            Q_loss_breakin = q4 · (1 - exp(-EFC / q5))

        Saturates within ~4 × q5 EFC.
        """
        if efc <= 0:
            return 0.0
        q4 = self._q4(soc_avg, dod)
        q5 = self.params.q5
        return q4 * (1.0 - math.exp(-efc / q5))

    # ─────────────────────────────────────────────────────────────────────
    # Eq. 15 — Long-term cycling parameter q7(DOD, C_rate)
    # ─────────────────────────────────────────────────────────────────────

    def _q7(self, dod: float, c_rate: float) -> float:
        """
        Long-term cycling rate q7 identified via SISSO (Eq. 15):
            q7 = q7_a · DOD^q7_b · (1 + q7_c · C_rate + q7_d · DOD · C_rate)

        Deeper cycles and higher C-rates increase long-term degradation.
        """
        p = self.params
        dod = max(dod, 0.01)
        c_rate = max(c_rate, 0.01)
        return p.q7_a * (dod ** p.q7_b) * (1.0 + p.q7_c * c_rate + p.q7_d * dod * c_rate)

    # ─────────────────────────────────────────────────────────────────────
    # Eq. 14 — Long-term cycling algebraic expression
    # ─────────────────────────────────────────────────────────────────────

    def _longterm_loss_algebraic(
        self, efc: float, dod: float, c_rate: float
    ) -> float:
        """
        Long-term cycling loss (Eq. 14):
            Q_loss_lt = q7 · EFC^q8

        Returns fractional capacity loss.
        """
        if efc <= 0:
            return 0.0
        q7 = self._q7(dod, c_rate)
        q8 = self.params.q8
        return q7 * (efc ** q8)

    # ─────────────────────────────────────────────────────────────────────
    # Eq. 20 — Path-independent incremental update
    # ─────────────────────────────────────────────────────────────────────

    def _incremental_calendar(
        self,
        time_days_new: float,
        temperature_c: float,
        soc: float,
    ) -> float:
        """
        Compute incremental calendar loss δQ_cal using Eq. 20.

        1.  Evaluate the algebraic expression at the new accumulated time.
        2.  Subtract the previous algebraic value.
        3.  The difference is the increment δQ_cal for this step.
        """
        new_val = self._calendar_loss_algebraic(time_days_new, temperature_c, soc)
        delta = new_val - self.state._prev_calendar_loss
        self.state._prev_calendar_loss = new_val
        return max(delta, 0.0)

    def _incremental_breakin(
        self,
        efc_new: float,
        soc_avg: float,
        dod: float,
    ) -> float:
        """Compute incremental break-in loss δQ_breakin using Eq. 20."""
        new_val = self._breakin_loss_algebraic(efc_new, soc_avg, dod)
        delta = new_val - self.state._prev_breakin_loss
        self.state._prev_breakin_loss = new_val
        return max(delta, 0.0)

    def _incremental_longterm(
        self,
        efc_new: float,
        dod: float,
        c_rate: float,
    ) -> float:
        """Compute incremental long-term cycling loss δQ_lt using Eq. 20."""
        new_val = self._longterm_loss_algebraic(efc_new, dod, c_rate)
        delta = new_val - self.state._prev_longterm_loss
        self.state._prev_longterm_loss = new_val
        return max(delta, 0.0)

    # ─────────────────────────────────────────────────────────────────────
    # Public API  (matches DegradationModel interface)
    # ─────────────────────────────────────────────────────────────────────

    def update_from_cycle(
        self,
        cycle_number: int,
        c_rate_charge: float = 1.0,
        c_rate_discharge: float = 1.0,
        temperature: float = 25.0,
        dod: float = 1.0,
        cycle_time_hours: float = 2.0,
        avg_soc: float = 0.5,
        capacity_throughput: float = 0.0,
    ) -> dict:
        """
        Update degradation state after one cycle.

        Combines all three components using path-independent incremental
        updates (Eq. 20).  The method signature is compatible with the
        existing ``DegradationModel.update_from_cycle()`` so it can act as
        a drop-in replacement.

        Args:
            cycle_number: 1-based cycle number.
            c_rate_charge: Charge C-rate for this cycle.
            c_rate_discharge: Discharge C-rate for this cycle.
            temperature: Average temperature during the cycle (°C).
            dod: Depth of discharge achieved (0–1).
            cycle_time_hours: Duration of this cycle (hours).
            avg_soc: Average SOC during the cycle (0–1).
            capacity_throughput: Ah throughput this cycle.

        Returns:
            Dictionary with degradation details for this cycle.
        """
        # --- Update accumulators ---
        cycle_time_days = cycle_time_hours / 24.0
        self.state.total_time_days += cycle_time_days

        if capacity_throughput > 0:
            self.state.total_ah_throughput += capacity_throughput
            self.state.equivalent_full_cycles += (
                capacity_throughput / (2.0 * self.nominal_capacity)
            )

        avg_c_rate = (c_rate_charge + c_rate_discharge) / 2.0

        # --- Incremental component losses (Eq. 20) ---
        delta_cal = self._incremental_calendar(
            self.state.total_time_days, temperature, avg_soc
        )
        delta_breakin = self._incremental_breakin(
            self.state.equivalent_full_cycles, avg_soc, dod
        )
        delta_longterm = self._incremental_longterm(
            self.state.equivalent_full_cycles, dod, avg_c_rate
        )

        # --- Update state ---
        self.state.calendar_capacity_loss += delta_cal
        self.state.breakin_capacity_loss += delta_breakin
        self.state.longterm_capacity_loss += delta_longterm
        self.state.total_capacity_loss = (
            self.state.calendar_capacity_loss
            + self.state.breakin_capacity_loss
            + self.state.longterm_capacity_loss
        )

        # Resistance growth (coupled to capacity loss)
        p = self.params
        delta_r = (
            p.r_cal_factor * delta_cal
            + p.r_cyc_factor * (delta_breakin + delta_longterm)
        )
        self.state.total_resistance_growth += delta_r

        # Stress tracking
        t_k = temperature + 273.15
        self.state.cumulative_temperature_stress += math.exp(
            -p.q1_Ea / (R_GAS * t_k)
        )
        self.state.cumulative_rate_stress += avg_c_rate

        # Cycle info for history
        cycle_info = {
            "cycle": cycle_number,
            "temperature": temperature,
            "c_rate_charge": c_rate_charge,
            "c_rate_discharge": c_rate_discharge,
            "dod": dod,
            "avg_soc": avg_soc,
            "cap_loss_cal": delta_cal,
            "cap_loss_breakin": delta_breakin,
            "cap_loss_longterm": delta_longterm,
            "cap_loss_cyc": delta_breakin + delta_longterm,  # Compat key
            "total_cap_loss": self.state.total_capacity_loss,
            "capacity_retention": self.get_capacity_retention(),
            # Component totals for visualisation
            "cum_calendar_loss": self.state.calendar_capacity_loss,
            "cum_breakin_loss": self.state.breakin_capacity_loss,
            "cum_longterm_loss": self.state.longterm_capacity_loss,
        }
        self.state.cycle_history.append(cycle_info)

        return cycle_info

    # ── Convenience wrappers matching DegradationModel interface ────────

    def calculate_cycle_degradation(
        self,
        cycle_number: int,
        c_rate_charge: float = 1.0,
        c_rate_discharge: float = 1.0,
        temperature: float = 25.0,
        dod: float = 1.0,
    ) -> tuple[float, float]:
        """Return (capacity_loss_increment, resistance_growth_increment)."""
        avg_c = (c_rate_charge + c_rate_discharge) / 2.0
        efc = self.state.equivalent_full_cycles + dod
        delta_b = self._breakin_loss_algebraic(efc, 0.5, dod) - self.state._prev_breakin_loss
        delta_l = self._longterm_loss_algebraic(efc, dod, avg_c) - self.state._prev_longterm_loss
        cap = max(delta_b, 0.0) + max(delta_l, 0.0)
        res = cap * self.params.r_cyc_factor
        return float(cap), float(res)

    def calculate_calendar_degradation(
        self,
        time_hours: float,
        temperature: float = 25.0,
        soc: float = 0.5,
    ) -> tuple[float, float]:
        """Return (capacity_loss_increment, resistance_growth_increment)."""
        t_new = self.state.total_time_days + time_hours / 24.0
        delta = self._calendar_loss_algebraic(t_new, temperature, soc) - self.state._prev_calendar_loss
        cap = max(delta, 0.0)
        res = cap * self.params.r_cal_factor
        return float(cap), float(res)

    def get_current_capacity(self) -> float:
        """Current capacity after degradation (Ah)."""
        return self.nominal_capacity * (1.0 - min(self.state.total_capacity_loss, 0.5))

    def get_current_resistance_factor(self) -> float:
        """Resistance multiplication factor (≥ 1)."""
        return 1.0 + self.state.total_resistance_growth

    def get_capacity_retention(self) -> float:
        """Capacity retention as fraction (0.5–1.0)."""
        return max(0.5, 1.0 - self.state.total_capacity_loss)

    def get_state_dict(self) -> dict:
        """Full degradation state as a dictionary."""
        return {
            "total_capacity_loss": self.state.total_capacity_loss,
            "calendar_capacity_loss": self.state.calendar_capacity_loss,
            "breakin_capacity_loss": self.state.breakin_capacity_loss,
            "longterm_capacity_loss": self.state.longterm_capacity_loss,
            "cyclic_capacity_loss": (
                self.state.breakin_capacity_loss + self.state.longterm_capacity_loss
            ),
            "total_resistance_growth": self.state.total_resistance_growth,
            "equivalent_full_cycles": self.state.equivalent_full_cycles,
            "total_ah_throughput": self.state.total_ah_throughput,
            "total_time_days": self.state.total_time_days,
            "total_time_hours": self.state.total_time_days * 24.0,
            "capacity_retention": self.get_capacity_retention(),
            "resistance_factor": self.get_current_resistance_factor(),
        }

    def get_component_breakdown(self) -> dict:
        """
        Return the three-component breakdown for visualization.

        Useful for creating stacked area charts showing how much each
        degradation mode contributes to total capacity loss.
        """
        return {
            "calendar": self.state.calendar_capacity_loss,
            "breakin": self.state.breakin_capacity_loss,
            "longterm": self.state.longterm_capacity_loss,
            "total": self.state.total_capacity_loss,
        }

    def predict_cycle_life(
        self,
        target_retention: float = 0.8,
        temperature: float = 25.0,
        c_rate: float = 1.0,
        dod: float = 1.0,
        soc_avg: float = 0.5,
        cycle_time_hours: float = 2.0,
    ) -> int:
        """
        Predict cycles to reach *target_retention* under constant conditions.

        Iterates the model forward under fixed conditions until the
        retention target is reached.

        Args:
            target_retention: Target capacity retention (e.g. 0.8 = 80 %).
            temperature: Constant operating temperature (°C).
            c_rate: Constant C-rate.
            dod: Constant depth of discharge.
            soc_avg: Constant average SOC.
            cycle_time_hours: Constant cycle duration.

        Returns:
            Predicted cycle count.
        """
        # Use a fresh copy of state to avoid side-effects
        import copy
        saved_state = copy.deepcopy(self.state)

        max_cycles = 50_000
        for n in range(1, max_cycles + 1):
            self.update_from_cycle(
                cycle_number=n,
                c_rate_charge=c_rate,
                c_rate_discharge=c_rate,
                temperature=temperature,
                dod=dod,
                cycle_time_hours=cycle_time_hours,
                avg_soc=soc_avg,
                capacity_throughput=2.0 * self.nominal_capacity * dod,
            )
            if self.get_capacity_retention() <= target_retention:
                self.state = saved_state
                return n

        self.state = saved_state
        return max_cycles

    def get_temperature_acceleration_factor(
        self, temperature: float, reference_temp: float = 25.0
    ) -> float:
        """Acceleration factor relative to the reference temperature."""
        t_k = temperature + 273.15
        t_ref = reference_temp + 273.15
        ea = self.params.q1_Ea
        return math.exp(-ea / R_GAS * (1.0 / t_k - 1.0 / t_ref))

    def reset(self) -> None:
        """Reset degradation state to initial conditions."""
        self.state = LFPPaperDegradationState()

    # ─────────────────────────────────────────────────────────────────────
    # Stress-factor helpers for heatmap visualisation
    # ─────────────────────────────────────────────────────────────────────

    def calendar_stress_rate(self, temperature_c: float, soc: float) -> float:
        """
        Instantaneous calendar degradation rate (per day) at given T, SOC.

        Useful for generating interactive heatmaps in a Streamlit dashboard.
        """
        q1 = self._q1(temperature_c, soc)
        q3 = self._q3(temperature_c, soc)
        # At t → 0 the derivative is q1/q2 + q3
        return q1 / self.params.q2 + q3

    def breakin_stress(self, soc_avg: float, dod: float) -> float:
        """
        Break-in magnitude q4 at given SOC_avg and DOD.

        Useful for generating heatmaps showing break-in sensitivity.
        """
        return self._q4(soc_avg, dod)

    def longterm_cycling_rate(self, dod: float, c_rate: float) -> float:
        """
        Long-term cycling rate q7 at given DOD and C-rate.

        Useful for generating heatmaps showing long-term stress factors.
        """
        return self._q7(dod, c_rate)
