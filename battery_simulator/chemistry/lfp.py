"""LFP (LiFePO4) / Graphite chemistry configuration."""

from dataclasses import dataclass

from battery_simulator.chemistry.base_chemistry import BaseChemistry, DegradationParameters


@dataclass
class LFPChemistry(BaseChemistry):
    """
    LFP (Lithium Iron Phosphate) / Graphite chemistry.
    
    Characteristics:
    - Excellent cycle life (2000+ cycles to 80% SOH)
    - High safety (thermal stability)
    - Lower energy density (~160 Wh/kg)
    - Flat voltage profile
    - Common in stationary storage and some EVs
    
    Degradation model:
    - Supports the standard Arrhenius-based model (backward compatible)
    - Also supports the three-component paper model from
      "ML-Assisted Identification of Accurate Battery Lifetime Models
      with Uncertainty" when ``use_paper_model=True``.
    
    Standard degradation characteristics:
    - E_a,cyc = 30 kJ/mol (lower temperature sensitivity)
    - z = 0.5 (square-root cycle dependence)
    - alpha = 0.25 (weak C-rate dependence - robust to high rates)
    - Moderate SOC dependence for calendar aging
    
    Paper model (three-component):
    - Calendar aging: sigmoidal + linear (Eq. 6, 7, 8)
    - Cycling break-in: saturation within ~4000 EFC (Eq. 11, 12)
    - Long-term cycling: power law (Eq. 14, 15)
    - Path-independent incremental updates (Eq. 20)
    """

    # Whether to use the three-component paper model for LFP degradation
    use_paper_model: bool = True

    def _init_parameters(self) -> None:
        """Initialize LFP-specific parameters."""
        # Chemistry identification
        self.name = "LFP-Graphite"
        self.cathode = "LFP"
        self.anode = "Graphite"

        # Voltage specifications (lower than NMC)
        self.voltage_nominal = 3.2
        self.voltage_max = 3.65
        self.voltage_min = 2.5

        # Capacity and resistance
        self.capacity = 3.0  # Ah
        self.resistance_initial = 0.030  # Ohm (typically lower than NMC)

        # Performance limits
        self.max_charge_rate = 1.0  # C
        self.max_discharge_rate = 3.0  # C (can handle higher rates)
        self.coulombic_efficiency = 0.9998
        self.energy_density = 160.0  # Wh/kg

        # Legacy degradation parameters (for backward compatibility)
        self.capacity_fade_rate = 0.02  # % per 100 cycles
        self.resistance_growth_rate = 0.01  # % per 100 cycles
        self.calendar_fade_rate = 0.01  # % per year at 25°C

        # Semi-empirical degradation parameters for LFP (standard Arrhenius)
        # Calibrated for ~10% capacity loss at 2000 cycles, 1C, 25°C
        self.degradation_params = DegradationParameters(
            # Cycle aging
            k_cyc=0.0022,  # Pre-factor calibrated for 10% loss @ 2000 cycles
            e_a_cyc=30000.0,  # 30 kJ/mol - lower temp sensitivity (stable structure)
            z=0.5,  # Square-root cycle dependence
            alpha=0.25,  # Weak C-rate dependence (robust to high rates)
            
            # Calendar aging
            k_cal=0.0008,  # Pre-factor for ~2% loss/year at 25°C, 50% SOC
            e_a_cal=27500.0,  # 27.5 kJ/mol (lower than NMC)
            b=0.5,  # Square-root time dependence
            beta_soc=0.5,  # Moderate SOC dependence (less sensitive than NMC)
            soc_ref=0.5,
            
            # Resistance growth (slower than NMC)
            k_resistance=0.0010,
            e_a_resistance=25000.0,
            
            # Reference conditions
            t_ref=298.0,
            c_ref=1.0,
        )

        # Paper model parameters (Table IV) for three-component degradation
        # These are used when use_paper_model is True
        self._paper_model_params = None  # Lazy-loaded via get_paper_params()

        # OCV vs SOC table (characteristic flat LFP curve)
        self.ocv_table = [
            [0.00, 2.50],
            [0.02, 2.80],
            [0.05, 3.00],
            [0.10, 3.15],
            [0.15, 3.22],
            [0.20, 3.26],
            [0.25, 3.28],
            [0.30, 3.29],
            [0.35, 3.30],
            [0.40, 3.31],
            [0.45, 3.32],
            [0.50, 3.32],  # Very flat plateau
            [0.55, 3.33],
            [0.60, 3.33],
            [0.65, 3.34],
            [0.70, 3.35],
            [0.75, 3.36],
            [0.80, 3.38],
            [0.85, 3.42],
            [0.90, 3.48],
            [0.95, 3.55],
            [1.00, 3.65],
        ]

    def get_paper_params(self):
        """
        Get the three-component paper model parameters (Table IV).

        Returns:
            LFPPaperParameters instance with calibrated values.
        """
        if self._paper_model_params is None:
            from battery_simulator.core.lfp_paper_degradation import LFPPaperParameters
            self._paper_model_params = LFPPaperParameters()
        return self._paper_model_params
