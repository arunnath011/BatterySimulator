# Battery Test Data Simulator

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/version-1.1.0-orange.svg" alt="Version 1.1.0">
  <img src="https://img.shields.io/badge/tests-93%20passed-brightgreen.svg" alt="Tests: 93 passed">
</p>

<p align="center">
  <strong>Generate realistic lithium-ion battery cycling data for development, testing, and demonstration purposes.</strong>
</p>

<p align="center">
  The simulator produces CSV output files mimicking real battery cycler formats (Arbin, Neware, Biologic) with physics-based electrochemical modeling, semi-empirical degradation models, and optional PyBaMM integration for high-fidelity simulations.
</p>

---

## Features

| Feature | Description |
|---------|-------------|
| **Physics-Based Model** | Realistic voltage, SOC, temperature behavior using OCV lookup tables and RC circuit models |
| **4 Battery Chemistries** | NMC811, LFP, NCA, LTO with accurate OCV curves and chemistry-specific degradation parameters |
| **Semi-Empirical Degradation** | Arrhenius temperature dependence, C-rate effects, SOC-dependent calendar aging |
| **5 Test Protocols** | Formation, Cycle Life, Rate Capability, Calendar Aging, Reference Performance Test (RPT) |
| **4 Output Formats** | Generic CSV, Arbin, Neware, Biologic EC-Lab |
| **Flexible Configuration** | YAML config files or Python API |
| **CLI + Python API** | Full command-line interface and programmatic access |
| **Web Interface** | Interactive Streamlit app for easy simulation setup and visualization |
| **Realistic Noise** | Configurable measurement noise for voltage, current, temperature |
| **PyBaMM Integration** | Optional high-fidelity physics-based models (SPM, SPMe, DFN) |
| **Pack Simulation** | Multi-cell pack simulations with liionpack (series/parallel configurations) |
| **Automated Generator** | Batch data generation with randomized parameters and MQTT export |

---

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/BatterySimulator.git
cd BatterySimulator

# Install with Poetry (core dependencies)
poetry install

# Install with optional features
poetry install --extras webapp     # Streamlit web app
poetry install --extras pybamm     # High-fidelity PyBaMM models
poetry install --extras pack       # Pack simulation (includes PyBaMM + liionpack)
poetry install --extras all        # All optional features

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-org/BatterySimulator.git
cd BatterySimulator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Optional: Install high-fidelity simulation support
pip install pybamm

# Optional: Install pack simulation support (requires PyBaMM)
pip install pybamm liionpack

# Optional: Install MQTT export support
pip install paho-mqtt
```

---

## Quick Start

### Web Interface (Streamlit App)

The easiest way to get started is with our interactive web interface:

```bash
# Launch the Streamlit app
streamlit run battery_simulator/app.py
```

This opens an interactive dashboard where you can:
- **Select battery chemistry** (NMC811, LFP, NCA, LTO)
- **Configure test protocols** (Cycle Life, Formation, Rate Capability, RPT)
- **Adjust all simulation parameters** via intuitive controls
- **View real-time results** with interactive charts
- **Export data** in multiple formats (CSV, JSON)

### Command Line Interface

```bash
# Run a basic simulation (100 cycles, NMC811 chemistry)
battery-simulator run

# Customize the simulation
battery-simulator run \
    --chemistry LFP \
    --cycles 500 \
    --charge-rate 0.5 \
    --discharge-rate 1.0 \
    --output data/lfp_test.csv \
    --format arbin

# Generate a demo dataset
battery-simulator demo --chemistry NMC811 --cycles 500

# List available chemistries
battery-simulator list-chemistries

# List available protocols
battery-simulator list-protocols

# Generate an example configuration file
battery-simulator init-config --output config/my_config.yaml
```

### Python API

```python
from battery_simulator import BatterySimulator, Protocol
from battery_simulator.core.simulator import SimulationMode

# Create simulator with NMC811 chemistry (Fast empirical mode - default)
sim = BatterySimulator(
    chemistry="NMC811",
    capacity=3.0,      # Ah
    temperature=25.0   # C
)

# Define a cycle life test protocol
protocol = Protocol.cycle_life(
    charge_rate=1.0,           # 1C charge
    discharge_rate=1.0,        # 1C discharge
    cycles=1000,
    voltage_max=4.2,
    voltage_min=3.0,
    end_capacity_retention=0.80  # Stop at 80% capacity
)

# Run simulation and save results
results = sim.run(
    protocol=protocol,
    output_path="output/nmc811_cycle_life.csv",
    output_format="arbin",
    show_progress=True
)

# Access results
print(f"Test ID: {results.test_id}")
print(f"Cycles Completed: {results.cycles_completed}")
print(f"Simulation Mode: {results.simulation_mode}")
```

### High-Fidelity Mode (PyBaMM)

```python
from battery_simulator import BatterySimulator, Protocol
from battery_simulator.core.simulator import SimulationMode

# High-fidelity mode with PyBaMM physics-based models
sim_hifi = BatterySimulator(
    chemistry="NMC811",  # Or use PyBaMM parameter set: "Chen2020"
    mode=SimulationMode.HIGH_FIDELITY,
    pybamm_model="SPMe",  # SPM (fastest), SPMe (balanced), DFN (most accurate)
)

protocol = Protocol.cycle_life(cycles=100)
results = sim_hifi.run(protocol)

print(f"Backend: {results.backend_info}")  # Shows PyBaMM model used
```

### Pack Simulation Mode

```python
from battery_simulator import BatterySimulator, Protocol
from battery_simulator.core.simulator import SimulationMode

# Pack simulation with liionpack
sim_pack = BatterySimulator(
    chemistry="NMC811",
    mode=SimulationMode.PACK,
    pack_config={
        "series": 14,      # 14 cells in series (~50V)
        "parallel": 4,      # 4 parallel strings
        "cell_variation": 0.02,  # 2% cell-to-cell variation
    }
)

protocol = Protocol.cycle_life(cycles=50)
results = sim_pack.run(protocol)

print(f"Pack: {results.backend_info['pack_config']}")
```

### Using PyBaMM Parameter Sets

```python
from battery_simulator.chemistry import Chemistry

# List available PyBaMM parameter sets
pybamm_sets = Chemistry.list_pybamm_available()
print(f"Available: {pybamm_sets}")

# Create chemistry from PyBaMM parameters
chem = Chemistry.from_pybamm("Chen2020")  # LG M50 NMC811 cell
print(f"Chemistry: {chem.name}, Capacity: {chem.nominal_capacity} Ah")
print(f"Capacity Retention: {results.capacity_retention:.2%}")
print(f"Energy Throughput: {results.energy_throughput:.1f} Wh")
```

---

## Streamlit Web Application

The Battery Simulator includes a full-featured web interface built with Streamlit for interactive simulation setup and visualization.

> **Note:** The web app requires **Python 3.10+**. The core simulation library works with Python 3.9+.

### Installation with Web App

```bash
# Install with Streamlit support (requires Python 3.10+)
poetry install --extras webapp

# Or using pip
pip install -e ".[webapp]"
```

### Launching the App

```bash
# Using Poetry
poetry run streamlit run battery_simulator/app.py

# Or if already in the virtual environment
streamlit run battery_simulator/app.py

# Specify a custom port
streamlit run battery_simulator/app.py --server.port 8502
```

### App Features

#### 1. Simulation Mode Selection
- **Fast (Empirical)**: Quick simulations using lookup tables - always available
- **High-Fidelity (PyBaMM)**: Physics-based models with SPM/SPMe/DFN options - requires PyBaMM
- **Pack Simulation**: Multi-cell pack modeling - requires PyBaMM + liionpack
- Unavailable modes are shown with "not installed" suffix
- Detailed progress UI with estimated time for high-fidelity simulations

#### 2. Chemistry Selection
- Choose from 4 battery chemistries: **NMC811**, **LFP**, **NCA**, **LTO**
- View detailed chemistry specifications (voltage range, capacity, energy density)
- Chemistry parameters automatically adjust protocol defaults

#### 3. Cell Configuration
- **Capacity**: Set cell capacity (0.1 - 100 Ah)
- **Temperature**: Set operating temperature (-20C to 60C)

#### 4. Protocol Configuration

| Protocol | Available Parameters |
|----------|---------------------|
| **Cycle Life** | Cycles, Charge Rate, Discharge Rate, Voltage Limits, Rest Time |
| **Formation** | Cycles, Initial C-Rate |
| **Rate Capability** | Discharge Rates (list), Cycles per Rate, Charge Rate |
| **RPT** | Charge/Discharge Rate, Pulse Current |

#### 5. Output Settings
- Select output format: Generic, Arbin, Neware, Biologic
- Configure measurement noise levels
- Enable/disable degradation modeling

#### 6. Interactive Results Dashboard
- **Summary Metrics**: Cycles, Capacity Retention, Energy Throughput, Resistance
- **Degradation Charts**: Capacity retention and efficiency over cycles
- **Cycle Details**: Voltage, Current, SOC, Temperature profiles
- **Raw Data View**: Preview and explore the generated data
- **Export Options**: Download CSV data, JSON metadata, cycle summaries

#### 7. Automated Data Generator
- Generate batch data for multiple cells with randomized parameters
- Select multiple chemistries for varied datasets
- Configure cell-to-cell variation percentage
- Export to folder (CSV files) or MQTT broker
- Scheduled generation mode for continuous data production

---

## Supported Chemistries

| Chemistry | Cathode/Anode | Voltage Range | Capacity | Energy Density | Typical Cycle Life |
|-----------|---------------|---------------|----------|----------------|-------------------|
| **NMC811** | NMC811 / Graphite | 3.0 - 4.2 V | 3.0 Ah | 250 Wh/kg | 800-1000 cycles |
| **LFP** | LiFePO4 / Graphite | 2.5 - 3.65 V | 3.0 Ah | 160 Wh/kg | 2000+ cycles |
| **NCA** | NCA / Si-Graphite | 2.7 - 4.2 V | 3.5 Ah | 280 Wh/kg | 500-800 cycles |
| **LTO** | LMO / Li4Ti5O12 | 1.5 - 2.8 V | 2.0 Ah | 80 Wh/kg | 10,000+ cycles |

Each chemistry includes:
- Accurate OCV vs SOC curve (22 data points)
- Chemistry-specific semi-empirical degradation parameters
- Arrhenius activation energies for temperature effects
- C-rate sensitivity coefficients
- Coulombic efficiency values

---

## Test Protocols

### 1. Cycle Life Testing
Standard CC-CV charge / CC discharge cycling for degradation evaluation.

```python
protocol = Protocol.cycle_life(
    charge_rate=1.0,            # C-rate
    discharge_rate=1.0,         # C-rate
    cycles=1000,
    voltage_max=4.2,            # V
    voltage_min=3.0,            # V
    rest_time=300,              # seconds between steps
    end_capacity_retention=0.80 # Stop at 80% retention
)
```

### 2. Formation Cycling
Low-rate initial cycling for SEI layer formation on new cells.

```python
protocol = Protocol.formation(
    cycles=3,
    initial_rate=0.1,  # C/10
    rest_time=1800     # 30 min rest
)
```

### 3. Rate Capability
Multi-rate discharge testing to evaluate power performance.

```python
protocol = Protocol.rate_capability(
    rates=[0.2, 0.5, 1.0, 2.0, 3.0, 5.0],  # C-rates to test
    cycles_per_rate=3,
    charge_rate=0.5  # Fixed charge rate
)
```

### 4. Calendar Aging
Storage aging test with periodic capacity checkups.

```python
protocol = Protocol.calendar_aging(
    target_soc=0.5,            # Storage SOC
    storage_days=90,
    temperature=25.0,          # C
    checkup_interval_days=30
)
```

### 5. Reference Performance Test (RPT)
Comprehensive characterization including capacity, resistance, and rate tests.

```python
protocol = Protocol.rpt(
    charge_rate=0.33,          # C/3 for capacity test
    discharge_rate=0.33,
    pulse_soc_points=[0.2, 0.5, 0.8],  # SOC points for resistance
    pulse_current=1.0,         # C-rate for pulse
    pulse_duration=10.0        # seconds
)
```

---

## Output Formats

### Generic CSV
Standard format with all key measurements:
```csv
timestamp,test_time,cycle_index,step_index,step_type,current,voltage,capacity,energy,temperature,state_of_charge,power,internal_resistance
2024-01-15T10:00:00.000Z,0.0,1,1,charge_cc_cv,3.000000,3.500000,0.000000,0.000000,25.0000,0.0500,10.500000,0.0500
```

### Arbin Format
Compatible with Arbin cycler exports:
```csv
Data_Point,Test_Time(s),Date_Time,Cycle_Index,Step_Index,Current(A),Voltage(V),Charge_Capacity(Ah),Discharge_Capacity(Ah),Charge_Energy(Wh),Discharge_Energy(Wh),dV/dt(V/s),Internal_Resistance(Ohm),Temperature(C)
```

### Neware Format
Compatible with Neware cycler exports (uses mA/mAh units):
```csv
Record ID,Step ID,Status,Jump,Time,Voltage(V),Current(mA),CapaCity(mAh),Energy(mWh),CapaCity-Chg(mAh),CapaCity-DChg(mAh),Engy-Chg(mWh),Engy-DChg(mWh),Engy-Total(mWh)
```

### Biologic EC-Lab Format
Compatible with Biologic EC-Lab exports:
```csv
mode,ox/red,error,control changes,Ns changes,counter inc.,time/s,control/V/mA,Ewe/V,I/mA,dq/mA.h,Q charge/discharge/mA.h,half cycle,Ns
```

---

## Semi-Empirical Degradation Model

The simulator implements a physics-based semi-empirical degradation model that accurately captures the effects of temperature, C-rate, and SOC on battery aging.

### Cycle Aging Model

Capacity loss from cycling follows an Arrhenius-based model with C-rate dependence:

```
Q_loss,cyc = k_cyc * N^z * exp(-E_a,cyc/R * (1/T - 1/T_ref)) * (C_rate/C_ref)^alpha
```

Where:
- `N` = cycle number
- `z` = cycle exponent (0.5-0.6, sub-linear behavior)
- `E_a,cyc` = activation energy for cycle aging (J/mol)
- `T` = temperature (K), `T_ref` = 298 K (25C)
- `alpha` = C-rate exponent (chemistry-dependent)

### Calendar Aging Model

Capacity loss from storage follows a time-dependent Arrhenius model with SOC effects:

```
Q_loss,cal = k_cal * t^b * exp(-E_a,cal/R * (1/T - 1/T_ref)) * exp(beta_SOC * (SOC - SOC_ref))
```

Where:
- `t` = time (days)
- `b` = time exponent (typically 0.4-0.5)
- `E_a,cal` = activation energy for calendar aging (J/mol)
- `beta_SOC` = SOC sensitivity coefficient (high SOC accelerates aging)

### Chemistry-Specific Parameters

| Parameter | NMC811 | NCA | LFP | LTO |
|-----------|--------|-----|-----|-----|
| E_a,cyc (kJ/mol) | 45 | 50 | 30 | 25 |
| z (cycle exp) | 0.6 | 0.6 | 0.5 | 0.5 |
| alpha (C-rate exp) | 0.7 | 0.8 | 0.25 | 0.1 |
| E_a,cal (kJ/mol) | 40 | 45 | 27.5 | 22.5 |
| b (time exp) | 0.5 | 0.5 | 0.5 | 0.4 |
| beta_SOC | 1.5 | 2.0 | 0.5 | 0.2 |
| Target life @ 25C, 1C | 1000 cycles | 800 cycles | 2000 cycles | 10000 cycles |

### Temperature Acceleration Factors

The Arrhenius model captures how temperature affects aging rate:

| Temperature | NMC811 | LFP | LTO |
|-------------|--------|-----|-----|
| 10C | 0.4x | 0.5x | 0.6x |
| 25C (ref) | 1.0x | 1.0x | 1.0x |
| 35C | 1.8x | 1.5x | 1.4x |
| 45C | 3.2x | 2.2x | 1.9x |
| 55C | 5.5x | 3.2x | 2.5x |

### C-Rate Effects

Higher C-rates accelerate degradation, with chemistry-specific sensitivity:

| C-Rate | NMC811 | NCA | LFP | LTO |
|--------|--------|-----|-----|-----|
| 0.5C | 0.6x | 0.6x | 0.8x | 0.9x |
| 1C (ref) | 1.0x | 1.0x | 1.0x | 1.0x |
| 2C | 1.6x | 1.7x | 1.2x | 1.1x |
| 3C | 2.2x | 2.4x | 1.3x | 1.1x |

---

## Physics Model

### Voltage Model
```
V(t) = OCV(SOC) - I(t) * R_internal(SOC, T, age) - eta_polarization(I, SOC)
```

- **OCV(SOC)**: Open circuit voltage from chemistry-specific lookup table
- **R_internal**: Internal resistance (temperature and age dependent)
- **eta_polarization**: Polarization overpotential (first-order RC model)

### State of Charge
```
SOC(t) = SOC(t-1) + (I(t) * dt * eta_coulombic) / Q_capacity
```

- **eta_coulombic**: Coulombic efficiency (0.999-0.9999 depending on chemistry)
- **Q_capacity**: Current capacity (degrades over time)

### Temperature Model
```
dT = (I^2*R + I*eta) * dt / (m * Cp) - h*A*(T - T_amb) / (m * Cp)
```

---

## Configuration

Create a YAML configuration file for complex or repeatable simulations:

```yaml
# config/cycle_life_test.yaml

simulation:
  name: "NMC811 1C Cycle Life Test"
  output_dir: "./output"
  timing_mode: "instant"    # instant, accelerated, real_time
  speed_factor: 1000        # For accelerated mode
  data_rate: 1.0            # Hz

cell:
  chemistry: "NMC811"
  capacity: 3.0             # Ah
  form_factor: "cylindrical"

test_protocol:
  type: "cycle_life"
  cycles: 1000
  temperature: 25           # C

  cycling:
    charge_rate: 1.0
    discharge_rate: 1.0
    voltage_max: 4.2
    voltage_min: 3.0
    rest_time: 300          # seconds

  rpt_interval: 50          # RPT every 50 cycles

  end_conditions:
    capacity_retention: 0.80
    max_cycles: 1000

output:
  format: "arbin"
  cycle_data: true
  summary_data: true
  metadata: true

degradation:
  enable_capacity_fade: true
  enable_resistance_growth: true
  enable_sudden_failure: false
  failure_probability: 0.01

noise:
  voltage_noise: 0.001      # V (std dev)
  current_noise: 0.005      # A (std dev)
  temperature_noise: 0.5    # C (std dev)
```

Run with:
```bash
battery-simulator run --config config/cycle_life_test.yaml
```

---

## Project Structure

```
BatterySimulator/
├── battery_simulator/
│   ├── __init__.py              # Package exports
│   ├── cli.py                   # Command-line interface
│   ├── app.py                   # Streamlit web application
│   │
│   ├── core/
│   │   ├── simulator.py         # Main simulator orchestrator + SimulationMode
│   │   ├── battery_model.py     # Fast empirical battery model
│   │   ├── pybamm_model.py      # PyBaMM high-fidelity model wrapper
│   │   ├── pack_simulator.py    # liionpack pack simulation
│   │   ├── degradation.py       # Semi-empirical Arrhenius degradation
│   │   └── thermal_model.py     # Temperature modeling
│   │
│   ├── chemistry/
│   │   ├── base_chemistry.py    # Abstract base + DegradationParameters
│   │   ├── nmc811.py            # NMC811/Graphite parameters
│   │   ├── lfp.py               # LFP/Graphite parameters
│   │   ├── nca.py               # NCA/Si-Graphite parameters
│   │   ├── lto.py               # LTO/LMO parameters
│   │   ├── pybamm_params.py     # PyBaMM parameter library bridge
│   │   └── pack_config.py       # Pack topology configurations
│   │
│   ├── protocols/
│   │   ├── base_protocol.py     # Protocol base classes & steps
│   │   ├── formation.py         # Formation cycling
│   │   ├── cycle_life.py        # Cycle life testing
│   │   ├── rate_capability.py   # Rate capability study
│   │   ├── calendar_aging.py    # Calendar aging test
│   │   └── rpt.py               # Reference Performance Test
│   │
│   ├── outputs/
│   │   ├── base_writer.py       # Abstract writer class
│   │   ├── csv_writer.py        # Generic CSV format
│   │   ├── arbin_format.py      # Arbin cycler format
│   │   ├── neware_format.py     # Neware cycler format
│   │   └── biologic_format.py   # Biologic EC-Lab format
│   │
│   └── utils/
│       ├── config_loader.py     # YAML configuration parsing
│       ├── noise_generator.py   # Measurement noise simulation
│       └── validators.py        # Input/output validation
│
├── config/
│   └── examples/                # Example configuration files
│
├── tests/                       # Test suite (93 tests)
│   ├── test_battery_model.py
│   ├── test_chemistry.py
│   ├── test_degradation.py
│   ├── test_protocols.py
│   ├── test_simulator.py
│   ├── test_pybamm_integration.py  # PyBaMM/pack integration tests
│   └── test_app.py              # Streamlit app component tests
│
├── pyproject.toml               # Poetry configuration
├── requirements.txt             # pip requirements
└── README.md
```

---

## Examples

### Generate ML Training Data

```python
from battery_simulator import BatterySimulator, Protocol

# Generate datasets for multiple chemistries
chemistries = ["NMC811", "LFP", "NCA", "LTO"]

for chem in chemistries:
    sim = BatterySimulator(chemistry=chem, capacity=3.0)
    protocol = Protocol.cycle_life(cycles=500)
    
    results = sim.run(
        protocol=protocol,
        output_path=f"training_data/{chem}_500cycles.csv",
        output_format="generic"
    )
    
    print(f"{chem}: {results.capacity_retention:.2%} retention after {results.cycles_completed} cycles")
```

### Compare Chemistry Degradation at Different Temperatures

```python
import matplotlib.pyplot as plt
from battery_simulator import BatterySimulator, Protocol
from battery_simulator.core.simulator import SimulationConfig

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Compare chemistries at 25C
ax1 = axes[0]
for chem in ["NMC811", "LFP", "NCA", "LTO"]:
    sim = BatterySimulator(chemistry=chem, temperature=25.0)
    protocol = Protocol.cycle_life(cycles=500)
    results = sim.run(protocol=protocol, show_progress=False)
    
    cycles = results.cycle_summary["cycle_index"]
    retention = results.cycle_summary["capacity_retention"] * 100
    ax1.plot(cycles, retention, label=chem, linewidth=2)

ax1.set_xlabel("Cycle Number")
ax1.set_ylabel("Capacity Retention (%)")
ax1.set_title("Degradation at 25C")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Compare NMC811 at different temperatures
ax2 = axes[1]
for temp in [15, 25, 35, 45]:
    sim = BatterySimulator(chemistry="NMC811", temperature=float(temp))
    protocol = Protocol.cycle_life(cycles=500)
    results = sim.run(protocol=protocol, show_progress=False)
    
    cycles = results.cycle_summary["cycle_index"]
    retention = results.cycle_summary["capacity_retention"] * 100
    ax2.plot(cycles, retention, label=f"{temp}C", linewidth=2)

ax2.set_xlabel("Cycle Number")
ax2.set_ylabel("Capacity Retention (%)")
ax2.set_title("NMC811 Temperature Effects")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("degradation_analysis.png", dpi=150, bbox_inches="tight")
```

### Compare C-Rate Effects

```python
from battery_simulator import BatterySimulator, Protocol

# Compare degradation at different C-rates for NMC811
c_rates = [0.5, 1.0, 2.0, 3.0]

for rate in c_rates:
    sim = BatterySimulator(chemistry="NMC811", capacity=3.0)
    protocol = Protocol.cycle_life(
        charge_rate=rate,
        discharge_rate=rate,
        cycles=500
    )
    results = sim.run(protocol=protocol, show_progress=False)
    
    print(f"C-rate {rate}C: {results.capacity_retention:.2%} retention after 500 cycles")
```

### Real-Time Data Streaming

```python
from battery_simulator import BatterySimulator, Protocol
from battery_simulator.core.simulator import SimulationConfig, TimingMode

# Configure for accelerated streaming
config = SimulationConfig(
    timing_mode=TimingMode.ACCELERATED,
    speed_factor=100,  # 100x faster than real-time
    data_rate=1.0
)

sim = BatterySimulator(chemistry="NMC811", capacity=3.0, config=config)

# Register callback for live data
def on_data_point(data):
    print(f"Cycle {data['cycle_index']}: V={data['voltage']:.3f}V, SOC={data['state_of_charge']:.1%}")

sim.on_data(on_data_point)

# Run with streaming
protocol = Protocol.cycle_life(cycles=10)
sim.run(protocol=protocol, output_path="streaming_test.csv")
```

---

## Simulation Modes

The simulator supports three simulation modes, allowing you to trade off between speed and accuracy:

### Fast Mode (Default)

The fast empirical model uses lookup tables and simplified physics for rapid simulation. Ideal for:
- ML training data generation
- Quick prototyping
- Large-scale simulations (1000+ cycles)

```python
from battery_simulator import BatterySimulator
from battery_simulator.core.simulator import SimulationMode

sim = BatterySimulator(chemistry="NMC811", mode=SimulationMode.FAST)
```

### High-Fidelity Mode (PyBaMM)

Uses PyBaMM's physics-based electrochemical models for accurate simulations. The model extracts accurate OCV curves and parameters from PyBaMM during initialization, then uses optimized calculations for fast per-timestep updates.

| Model | Description | Speed | Accuracy |
|-------|-------------|-------|----------|
| SPM | Single Particle Model | Fastest | Good |
| SPMe | SPM with Electrolyte | Balanced | Better |
| DFN | Doyle-Fuller-Newman | Slowest | Best |

**Key Features:**
- Accurate OCV curves extracted from PyBaMM parameter sets
- Fast simulation speed (comparable to empirical model)
- Temperature-dependent resistance modeling
- Concentration polarization effects

```python
sim = BatterySimulator(
    chemistry="NMC811",
    mode=SimulationMode.HIGH_FIDELITY,
    pybamm_model="SPMe"  # or "SPM", "DFN"
)
```

**Requires:** `pip install pybamm`

### Pack Simulation Mode

Multi-cell pack simulations using liionpack with:
- Series/parallel configurations
- Cell-to-cell variation
- Thermal coupling between cells
- Individual cell monitoring

```python
sim = BatterySimulator(
    chemistry="NMC811",
    mode=SimulationMode.PACK,
    pack_config={"series": 14, "parallel": 4}
)
```

**Requires:** `pip install pybamm liionpack`

### Standard Pack Presets

Pre-configured pack topologies for common applications:

| Preset | Configuration | Voltage | Energy |
|--------|--------------|---------|--------|
| `ev_small` | 96s2p | ~350V | ~35 kWh |
| `ev_medium` | 108s4p | ~400V | ~80 kWh |
| `ev_large` | 120s6p | ~450V | ~135 kWh |
| `ess_module` | 14s4p | ~50V | ~28 kWh |
| `ebike` | 13s4p | ~48V | ~0.7 kWh |
| `power_tool` | 5s2p | ~20V | ~0.1 kWh |

```python
from battery_simulator.chemistry.pack_config import get_standard_pack

pack_cfg = get_standard_pack("ev_medium")
print(f"Pack: {pack_cfg.series}s{pack_cfg.parallel}p, {pack_cfg.pack_energy_kwh:.1f} kWh")
```

---

## PyBaMM Parameter Library

When using high-fidelity mode, you can access PyBaMM's extensive validated parameter library:

| Parameter Set | Chemistry | Description |
|--------------|-----------|-------------|
| `Chen2020` | NMC811-Graphite | LG M50 21700 cell |
| `Marquis2019` | NMC622-Graphite | Kokam pouch cell |
| `Prada2013` | LFP-Graphite | A123 26650 cell |
| `Ecker2015` | NMC532-Graphite | Kokam with aging |
| `NCA_Kim2011` | NCA-Graphite | Generic NCA |
| `Ramadass2004` | LCO-Graphite | Sony 18650 |

```python
from battery_simulator.chemistry import Chemistry

# List available parameter sets
sets = Chemistry.list_pybamm_available()
print(f"Available: {sets}")

# Create chemistry from PyBaMM parameters
chem = Chemistry.from_pybamm("Chen2020")
```

---

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run with coverage report
poetry run pytest --cov=battery_simulator --cov-report=html

# Run specific test file
poetry run pytest tests/test_battery_model.py -v
```

### Code Quality

```bash
# Format code with Black
poetry run black battery_simulator/

# Lint with Ruff
poetry run ruff check battery_simulator/

# Fix auto-fixable issues
poetry run ruff check battery_simulator/ --fix
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Acknowledgments

This simulator implements physics-based models drawing from battery research literature and real-world cycler data formats. The semi-empirical degradation models are based on Arrhenius kinetics and empirical correlations commonly used in battery aging research.

### References

- Schmalstieg et al., "A holistic aging model for Li(NiMnCo)O2 based 18650 lithium-ion batteries," Journal of Power Sources, 2014
- Wang et al., "Cycle-life model for graphite-LiFePO4 cells," Journal of Power Sources, 2011
- Bloom et al., "An accelerated calendar and cycle life study of Li-ion cells," Journal of Power Sources, 2001
