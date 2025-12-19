"""NCA (LiNiCoAlO2) / Silicon-Graphite chemistry configuration."""

from dataclasses import dataclass

from battery_simulator.chemistry.base_chemistry import BaseChemistry, DegradationParameters


@dataclass
class NCAChemistry(BaseChemistry):
    """
    NCA / Silicon-Graphite high-performance chemistry.
    
    Characteristics:
    - High energy density (~280 Wh/kg)
    - High power capability
    - Faster degradation (especially with Si anode)
    - Used in Tesla vehicles
    
    Degradation characteristics:
    - E_a,cyc = 50 kJ/mol (higher temperature sensitivity)
    - z = 0.6 (sub-linear cycle dependence)
    - alpha = 0.8 (strong C-rate dependence due to Si anode)
    - Very high SOC sensitivity for calendar aging
    """

    def _init_parameters(self) -> None:
        """Initialize NCA-specific parameters."""
        # Chemistry identification
        self.name = "NCA-SiGraphite"
        self.cathode = "NCA"
        self.anode = "Silicon-Graphite"

        # Voltage specifications
        self.voltage_nominal = 3.6
        self.voltage_max = 4.2
        self.voltage_min = 2.7

        # Capacity and resistance
        self.capacity = 3.5  # Ah (higher due to Si anode)
        self.resistance_initial = 0.040  # Ohm

        # Performance limits (high power)
        self.max_charge_rate = 2.0  # C (fast charging capable)
        self.max_discharge_rate = 5.0  # C (high power)
        self.coulombic_efficiency = 0.999
        self.energy_density = 280.0  # Wh/kg

        # Legacy degradation parameters (for backward compatibility)
        self.capacity_fade_rate = 0.08  # % per 100 cycles
        self.resistance_growth_rate = 0.04  # % per 100 cycles
        self.calendar_fade_rate = 0.05  # % per year at 25°C

        # Semi-empirical degradation parameters for NCA
        # Calibrated for ~25% capacity loss at 1000 cycles, 1C, 25°C
        self.degradation_params = DegradationParameters(
            # Cycle aging
            k_cyc=0.0040,  # Pre-factor calibrated for 25% loss @ 1000 cycles
            e_a_cyc=50000.0,  # 50 kJ/mol - higher temp sensitivity than NMC
            z=0.6,  # Sub-linear cycle dependence
            alpha=0.8,  # Strong C-rate dependence (Si anode expansion stress)
            
            # Calendar aging
            k_cal=0.0020,  # Pre-factor for ~7% loss/year at 25°C, 50% SOC
            e_a_cal=45000.0,  # 45 kJ/mol
            b=0.5,  # Square-root time dependence
            beta_soc=2.0,  # Very strong SOC dependence (NCA sensitive)
            soc_ref=0.5,
            
            # Resistance growth (faster due to Si anode)
            k_resistance=0.0030,
            e_a_resistance=40000.0,
            
            # Reference conditions
            t_ref=298.0,
            c_ref=1.0,
        )

        # OCV vs SOC table (NCA characteristic curve)
        self.ocv_table = [
            [0.00, 2.70],
            [0.02, 2.95],
            [0.05, 3.15],
            [0.10, 3.35],
            [0.15, 3.45],
            [0.20, 3.52],
            [0.25, 3.56],
            [0.30, 3.60],
            [0.35, 3.63],
            [0.40, 3.66],
            [0.45, 3.69],
            [0.50, 3.72],
            [0.55, 3.76],
            [0.60, 3.80],
            [0.65, 3.85],
            [0.70, 3.90],
            [0.75, 3.96],
            [0.80, 4.02],
            [0.85, 4.07],
            [0.90, 4.12],
            [0.95, 4.16],
            [1.00, 4.20],
        ]
