"""NMC811 (LiNi0.8Mn0.1Co0.1O2) / Graphite chemistry configuration."""

from dataclasses import dataclass

from battery_simulator.chemistry.base_chemistry import BaseChemistry, DegradationParameters


@dataclass
class NMC811Chemistry(BaseChemistry):
    """
    NMC811 / Graphite high-energy chemistry.
    
    Characteristics:
    - High energy density (~250 Wh/kg)
    - Moderate cycle life (~1000 cycles to 80% SOH)
    - Good rate capability
    - Common in EVs and consumer electronics
    
    Degradation characteristics:
    - E_a,cyc = 45 kJ/mol (moderate temperature sensitivity)
    - z = 0.6 (sub-linear cycle dependence)
    - alpha = 0.7 (significant C-rate dependence)
    - High SOC sensitivity for calendar aging
    """

    def _init_parameters(self) -> None:
        """Initialize NMC811-specific parameters."""
        # Chemistry identification
        self.name = "NMC811-Graphite"
        self.cathode = "NMC811"
        self.anode = "Graphite"

        # Voltage specifications
        self.voltage_nominal = 3.7
        self.voltage_max = 4.2
        self.voltage_min = 3.0

        # Capacity and resistance
        self.capacity = 3.0  # Ah (typical 18650)
        self.resistance_initial = 0.050  # Ohm

        # Performance limits
        self.max_charge_rate = 1.0  # C
        self.max_discharge_rate = 2.0  # C
        self.coulombic_efficiency = 0.9995
        self.energy_density = 250.0  # Wh/kg

        # Legacy degradation parameters (for backward compatibility)
        self.capacity_fade_rate = 0.05  # % per 100 cycles
        self.resistance_growth_rate = 0.02  # % per 100 cycles
        self.calendar_fade_rate = 0.03  # % per year at 25°C

        # Semi-empirical degradation parameters for NMC811
        # Calibrated for ~20% capacity loss at 1000 cycles, 1C, 25°C
        self.degradation_params = DegradationParameters(
            # Cycle aging: Q_loss = k * N^z * exp(-Ea/R*(1/T - 1/Tref)) * (C/Cref)^alpha
            k_cyc=0.0032,  # Pre-factor calibrated for 20% loss @ 1000 cycles
            e_a_cyc=45000.0,  # 45 kJ/mol - moderate temp sensitivity
            z=0.6,  # Sub-linear cycle dependence (SEI growth dominated)
            alpha=0.7,  # Significant C-rate dependence
            
            # Calendar aging: Q_loss = k * t^b * exp(-Ea/R*(1/T - 1/Tref)) * exp(beta*(SOC - SOCref))
            k_cal=0.0015,  # Pre-factor for ~5% loss/year at 25°C, 50% SOC
            e_a_cal=40000.0,  # 40 kJ/mol
            b=0.5,  # Square-root time dependence
            beta_soc=1.5,  # High SOC hurts (NMC sensitive to high SOC storage)
            soc_ref=0.5,
            
            # Resistance growth
            k_resistance=0.0020,
            e_a_resistance=35000.0,
            
            # Reference conditions
            t_ref=298.0,  # 25°C
            c_ref=1.0,
        )

        # OCV vs SOC table (characteristic NMC811 curve)
        self.ocv_table = [
            [0.00, 3.00],
            [0.02, 3.20],
            [0.05, 3.35],
            [0.10, 3.50],
            [0.15, 3.58],
            [0.20, 3.62],
            [0.25, 3.65],
            [0.30, 3.68],
            [0.35, 3.70],
            [0.40, 3.72],
            [0.45, 3.75],
            [0.50, 3.78],
            [0.55, 3.82],
            [0.60, 3.86],
            [0.65, 3.90],
            [0.70, 3.95],
            [0.75, 4.00],
            [0.80, 4.05],
            [0.85, 4.08],
            [0.90, 4.12],
            [0.95, 4.16],
            [1.00, 4.20],
        ]
