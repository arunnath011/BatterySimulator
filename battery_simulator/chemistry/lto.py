"""LTO (Li4Ti5O12) / LMO chemistry configuration."""

from dataclasses import dataclass

from battery_simulator.chemistry.base_chemistry import BaseChemistry, DegradationParameters


@dataclass
class LTOChemistry(BaseChemistry):
    """
    LTO (Lithium Titanate) / LMO ultra-long-life chemistry.
    
    Characteristics:
    - Exceptional cycle life (10,000+ cycles)
    - Very fast charging (5C+)
    - Excellent low-temperature performance
    - Lower energy density (~80 Wh/kg)
    - Zero-strain material (no SEI growth)
    - Used in grid storage and high-power applications
    
    Degradation characteristics:
    - E_a,cyc = 25 kJ/mol (low temperature sensitivity - stable structure)
    - z = 0.5 (square-root cycle dependence)
    - alpha = 0.1 (very weak C-rate dependence - exceptional rate capability)
    - Minimal SOC dependence for calendar aging
    """

    def _init_parameters(self) -> None:
        """Initialize LTO-specific parameters."""
        # Chemistry identification
        self.name = "LTO-LMO"
        self.cathode = "LMO"
        self.anode = "LTO"

        # Voltage specifications (lower voltage system)
        self.voltage_nominal = 2.3
        self.voltage_max = 2.8
        self.voltage_min = 1.5

        # Capacity and resistance
        self.capacity = 2.0  # Ah (lower due to LTO)
        self.resistance_initial = 0.020  # Ohm (very low)

        # Performance limits (exceptional)
        self.max_charge_rate = 5.0  # C (very fast charging)
        self.max_discharge_rate = 10.0  # C (extremely high power)
        self.coulombic_efficiency = 0.9999  # Near-perfect
        self.energy_density = 80.0  # Wh/kg (lower)

        # Legacy degradation parameters (for backward compatibility)
        self.capacity_fade_rate = 0.005  # % per 100 cycles
        self.resistance_growth_rate = 0.005  # % per 100 cycles
        self.calendar_fade_rate = 0.005  # % per year at 25°C

        # Semi-empirical degradation parameters for LTO
        # Calibrated for ~2% capacity loss at 10,000 cycles, 1C, 25°C
        self.degradation_params = DegradationParameters(
            # Cycle aging
            k_cyc=0.0002,  # Pre-factor calibrated for 2% loss @ 10000 cycles
            e_a_cyc=25000.0,  # 25 kJ/mol - very low temp sensitivity (zero-strain)
            z=0.5,  # Square-root cycle dependence
            alpha=0.1,  # Very weak C-rate dependence (exceptional rate capability)
            
            # Calendar aging
            k_cal=0.0003,  # Pre-factor for ~0.5% loss/year at 25°C, 50% SOC
            e_a_cal=22500.0,  # 22.5 kJ/mol (very low)
            b=0.4,  # Slightly sub-square-root time dependence
            beta_soc=0.2,  # Very weak SOC dependence (minimal calendar aging)
            soc_ref=0.5,
            
            # Resistance growth (minimal)
            k_resistance=0.0003,
            e_a_resistance=20000.0,
            
            # Reference conditions
            t_ref=298.0,
            c_ref=1.0,
        )

        # OCV vs SOC table (LTO characteristic curve)
        self.ocv_table = [
            [0.00, 1.50],
            [0.02, 1.70],
            [0.05, 1.90],
            [0.10, 2.10],
            [0.15, 2.20],
            [0.20, 2.28],
            [0.25, 2.32],
            [0.30, 2.35],
            [0.35, 2.38],
            [0.40, 2.40],
            [0.45, 2.42],
            [0.50, 2.44],
            [0.55, 2.46],
            [0.60, 2.48],
            [0.65, 2.50],
            [0.70, 2.52],
            [0.75, 2.55],
            [0.80, 2.58],
            [0.85, 2.62],
            [0.90, 2.68],
            [0.95, 2.74],
            [1.00, 2.80],
        ]
