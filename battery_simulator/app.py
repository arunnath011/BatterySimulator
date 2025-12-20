"""
Streamlit Web Application for Battery Test Data Simulator.

Run with: streamlit run battery_simulator/app.py
"""

from __future__ import annotations

import io
import json
import os
import random
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from battery_simulator import BatterySimulator, Protocol
from battery_simulator.chemistry import Chemistry
from battery_simulator.core.simulator import SimulationConfig, TimingMode, SimulationMode


# Check for optional dependencies
def check_pybamm_available() -> bool:
    """Check if PyBaMM is installed."""
    try:
        from battery_simulator.core.pybamm_model import PYBAMM_AVAILABLE
        return PYBAMM_AVAILABLE
    except ImportError:
        return False


def check_liionpack_available() -> bool:
    """Check if liionpack is installed."""
    try:
        from battery_simulator.core.pack_simulator import LIIONPACK_AVAILABLE
        return LIIONPACK_AVAILABLE
    except ImportError:
        return False


def get_pybamm_parameter_sets() -> List[Dict[str, Any]]:
    """Get available PyBaMM parameter sets."""
    try:
        from battery_simulator.chemistry.pybamm_params import list_available_pybamm_chemistries
        return list_available_pybamm_chemistries()
    except ImportError:
        return []


def get_standard_packs() -> List[Dict[str, Any]]:
    """Get standard pack configurations."""
    try:
        from battery_simulator.chemistry.pack_config import list_standard_packs
        return list_standard_packs()
    except ImportError:
        return []


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Battery Test Data Simulator",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for light/dark mode compatibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .footer-link {
        opacity: 0.7;
    }
    
    .footer-link:hover {
        opacity: 1.0;
    }
    
    .generator-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-running {
        border-left: 4px solid #28a745;
    }
    
    .status-stopped {
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_data
def get_chemistry_info(chemistry_name: str) -> dict[str, Any]:
    """Get chemistry information for display."""
    chem = Chemistry.from_name(chemistry_name)
    return {
        "name": chem.name,
        "cathode": chem.cathode,
        "anode": chem.anode,
        "voltage_nominal": chem.voltage_nominal,
        "voltage_max": chem.voltage_max,
        "voltage_min": chem.voltage_min,
        "capacity": chem.capacity,
        "max_charge_rate": chem.max_charge_rate,
        "max_discharge_rate": chem.max_discharge_rate,
        "coulombic_efficiency": chem.coulombic_efficiency,
        "energy_density": chem.energy_density,
        "capacity_fade_rate": chem.capacity_fade_rate,
        "resistance_growth_rate": chem.resistance_growth_rate,
    }


def get_plotly_template() -> str:
    """Get Plotly template based on Streamlit theme."""
    return "plotly"


def create_voltage_profile_chart(df: pd.DataFrame) -> go.Figure:
    """Create voltage vs time chart."""
    fig = px.line(
        df,
        x="test_time",
        y="voltage",
        color="cycle_index",
        title="Voltage Profile",
        labels={"test_time": "Test Time (s)", "voltage": "Voltage (V)", "cycle_index": "Cycle"},
        template=get_plotly_template(),
    )
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    return fig


def create_soc_chart(df: pd.DataFrame) -> go.Figure:
    """Create SOC vs time chart."""
    fig = px.line(
        df,
        x="test_time",
        y="state_of_charge",
        color="cycle_index",
        title="State of Charge Profile",
        labels={"test_time": "Test Time (s)", "state_of_charge": "SOC", "cycle_index": "Cycle"},
        template=get_plotly_template(),
    )
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    return fig


def create_capacity_retention_chart(cycle_summary: pd.DataFrame) -> go.Figure:
    """Create capacity retention over cycles chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cycle_summary["cycle_index"],
        y=cycle_summary["capacity_retention"] * 100,
        mode="lines+markers",
        name="Capacity Retention",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6),
    ))
    
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="red",
        annotation_text="80% EOL Threshold",
        annotation_position="right",
    )
    
    fig.update_layout(
        title="Capacity Retention Over Cycles",
        xaxis_title="Cycle Number",
        yaxis_title="Capacity Retention (%)",
        height=400,
        yaxis=dict(range=[70, 105]),
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    return fig


def create_efficiency_chart(cycle_summary: pd.DataFrame) -> go.Figure:
    """Create coulombic and energy efficiency chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cycle_summary["cycle_index"],
        y=cycle_summary["coulombic_efficiency"] * 100,
        mode="lines",
        name="Coulombic Efficiency",
        line=dict(color="#2ca02c", width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=cycle_summary["cycle_index"],
        y=cycle_summary["energy_efficiency"] * 100,
        mode="lines",
        name="Energy Efficiency",
        line=dict(color="#ff7f0e", width=2),
    ))
    
    fig.update_layout(
        title="Cycle Efficiency",
        xaxis_title="Cycle Number",
        yaxis_title="Efficiency (%)",
        height=400,
        yaxis=dict(range=[85, 102]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    return fig


def create_temperature_chart(df: pd.DataFrame) -> go.Figure:
    """Create temperature profile chart."""
    fig = px.line(
        df,
        x="test_time",
        y="temperature",
        color="cycle_index",
        title="Temperature Profile",
        labels={"test_time": "Test Time (s)", "temperature": "Temperature (C)", "cycle_index": "Cycle"},
        template=get_plotly_template(),
    )
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    return fig


def create_combined_cycle_chart(df: pd.DataFrame, cycle_num: int) -> go.Figure:
    """Create combined chart for a single cycle."""
    cycle_data = df[df["cycle_index"] == cycle_num]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Voltage", "Current", "SOC", "Temperature"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    fig.add_trace(
        go.Scatter(x=cycle_data["test_time"], y=cycle_data["voltage"], 
                   mode="lines", name="Voltage", line=dict(color="#1f77b4")),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=cycle_data["test_time"], y=cycle_data["current"],
                   mode="lines", name="Current", line=dict(color="#ff7f0e")),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=cycle_data["test_time"], y=cycle_data["state_of_charge"] * 100,
                   mode="lines", name="SOC", line=dict(color="#2ca02c")),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=cycle_data["test_time"], y=cycle_data["temperature"],
                   mode="lines", name="Temperature", line=dict(color="#d62728")),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Current (A)", row=1, col=2)
    fig.update_yaxes(title_text="SOC (%)", row=2, col=1)
    fig.update_yaxes(title_text="Temp (C)", row=2, col=2)
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    
    fig.update_layout(
        height=500,
        title_text=f"Cycle {cycle_num} Details",
        showlegend=False,
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def run_simulation(
    chemistry: str,
    capacity: float,
    protocol_type: str,
    protocol_params: dict,
    output_format: str,
    enable_degradation: bool,
    noise_voltage: float,
    noise_current: float,
    noise_temperature: float,
    output_path: str | None = None,
    simulation_mode: str = "fast",
    pybamm_model: str = "SPMe",
    pybamm_parameter_set: str | None = None,
    pack_config: dict | None = None,
) -> tuple[Any, pd.DataFrame, str]:
    """Run the battery simulation with given parameters."""
    
    # Map mode string to enum
    mode_map = {
        "fast": SimulationMode.FAST,
        "high_fidelity": SimulationMode.HIGH_FIDELITY,
        "pack": SimulationMode.PACK,
    }
    mode = mode_map.get(simulation_mode, SimulationMode.FAST)
    
    config = SimulationConfig(
        timing_mode=TimingMode.INSTANT,
        enable_degradation=enable_degradation,
        noise_voltage=noise_voltage,
        noise_current=noise_current,
        noise_temperature=noise_temperature,
        simulation_mode=mode,
        pybamm_model=pybamm_model,
        pybamm_parameter_set=pybamm_parameter_set,
        pack_config=pack_config,
    )
    
    sim = BatterySimulator(
        chemistry=chemistry,
        capacity=capacity,
        temperature=protocol_params.get("temperature", 25.0),
        config=config,
    )
    
    if protocol_type == "Cycle Life":
        protocol = Protocol.cycle_life(
            charge_rate=protocol_params["charge_rate"],
            discharge_rate=protocol_params["discharge_rate"],
            cycles=protocol_params["cycles"],
            voltage_max=protocol_params["voltage_max"],
            voltage_min=protocol_params["voltage_min"],
            rest_time=protocol_params["rest_time"],
        )
    elif protocol_type == "Formation":
        protocol = Protocol.formation(
            cycles=protocol_params["cycles"],
            initial_rate=protocol_params["initial_rate"],
            voltage_max=protocol_params["voltage_max"],
            voltage_min=protocol_params["voltage_min"],
        )
    elif protocol_type == "Rate Capability":
        protocol = Protocol.rate_capability(
            rates=protocol_params["rates"],
            cycles_per_rate=protocol_params["cycles_per_rate"],
            charge_rate=protocol_params["charge_rate"],
            voltage_max=protocol_params["voltage_max"],
            voltage_min=protocol_params["voltage_min"],
        )
    elif protocol_type == "RPT":
        protocol = Protocol.rpt(
            charge_rate=protocol_params["charge_rate"],
            discharge_rate=protocol_params["discharge_rate"],
            pulse_current=protocol_params["pulse_current"],
        )
    else:
        protocol = Protocol.cycle_life(cycles=protocol_params["cycles"])
    
    if output_path is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name
    
    results = sim.run(
        protocol=protocol,
        output_path=output_path,
        output_format=output_format.lower(),
        show_progress=False,
    )
    
    df = pd.read_csv(output_path)
    
    return results, df, output_path


def generate_random_cell_params(
    base_capacity: float,
    base_temperature: float,
    variation_capacity: float = 0.05,
    variation_temperature: float = 2.0,
) -> dict:
    """Generate random cell parameters with variations."""
    return {
        "capacity": base_capacity * (1 + random.uniform(-variation_capacity, variation_capacity)),
        "temperature": base_temperature + random.uniform(-variation_temperature, variation_temperature),
        "charge_rate_variation": random.uniform(0.95, 1.05),
        "discharge_rate_variation": random.uniform(0.95, 1.05),
    }


def run_batch_simulation(
    chemistries: list[str],
    cycler_format: str,
    num_cells: int,
    cycles_per_cell: int,
    base_capacity: float,
    base_temperature: float,
    charge_rate: float,
    discharge_rate: float,
    output_folder: str,
    enable_degradation: bool = True,
    cell_variation: float = 0.05,
) -> list[dict]:
    """Run batch simulation for multiple cells."""
    results_list = []
    
    os.makedirs(output_folder, exist_ok=True)
    
    for cell_idx in range(num_cells):
        chemistry = random.choice(chemistries)
        cell_params = generate_random_cell_params(
            base_capacity, base_temperature, cell_variation
        )
        
        chem_info = get_chemistry_info(chemistry)
        
        protocol_params = {
            "temperature": cell_params["temperature"],
            "cycles": cycles_per_cell,
            "charge_rate": charge_rate * cell_params["charge_rate_variation"],
            "discharge_rate": discharge_rate * cell_params["discharge_rate_variation"],
            "voltage_max": chem_info["voltage_max"],
            "voltage_min": chem_info["voltage_min"],
            "rest_time": 300,
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cell_{cell_idx+1:03d}_{chemistry}_{timestamp}.csv"
        output_path = os.path.join(output_folder, filename)
        
        try:
            results, df, _ = run_simulation(
                chemistry=chemistry,
                capacity=cell_params["capacity"],
                protocol_type="Cycle Life",
                protocol_params=protocol_params,
                output_format=cycler_format,
                enable_degradation=enable_degradation,
                noise_voltage=0.001,
                noise_current=0.005,
                noise_temperature=0.5,
                output_path=output_path,
            )
            
            results_list.append({
                "cell_id": cell_idx + 1,
                "chemistry": chemistry,
                "capacity": cell_params["capacity"],
                "temperature": cell_params["temperature"],
                "cycles_completed": results.cycles_completed,
                "capacity_retention": results.capacity_retention,
                "output_file": filename,
                "status": "success",
            })
        except Exception as e:
            results_list.append({
                "cell_id": cell_idx + 1,
                "chemistry": chemistry,
                "status": "failed",
                "error": str(e),
            })
    
    return results_list


def publish_to_mqtt(
    data: dict,
    broker: str,
    port: int,
    topic: str,
    username: str | None = None,
    password: str | None = None,
) -> bool:
    """Publish data to MQTT broker."""
    try:
        import paho.mqtt.client as mqtt
        
        client = mqtt.Client()
        if username and password:
            client.username_pw_set(username, password)
        
        client.connect(broker, port, 60)
        payload = json.dumps(data, default=str)
        result = client.publish(topic, payload)
        client.disconnect()
        
        return result.rc == mqtt.MQTT_ERR_SUCCESS
    except ImportError:
        st.error("MQTT support requires paho-mqtt. Install with: pip install paho-mqtt")
        return False
    except Exception as e:
        st.error(f"MQTT publish failed: {e}")
        return False


# =============================================================================
# Page: Single Simulation
# =============================================================================

def page_single_simulation():
    """Single simulation page."""
    st.markdown('<p class="main-header">Battery Test Data Simulator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Generate realistic lithium-ion battery cycling data</p>',
        unsafe_allow_html=True
    )
    
    # Check available backends
    pybamm_available = check_pybamm_available()
    liionpack_available = check_liionpack_available()
    
    with st.sidebar:
        st.header("Simulation Configuration")
        
        # Simulation Mode Selector
        st.subheader("1. Simulation Mode")
        
        # Always show all modes, but indicate availability
        mode_options = ["Fast (Empirical)"]
        mode_values = ["fast"]
        mode_disabled = [False]
        
        # High-Fidelity mode
        if pybamm_available:
            mode_options.append("High-Fidelity (PyBaMM)")
        else:
            mode_options.append("High-Fidelity (PyBaMM) - not installed")
        mode_values.append("high_fidelity")
        mode_disabled.append(not pybamm_available)
        
        # Pack mode
        if liionpack_available:
            mode_options.append("Pack Simulation")
        else:
            mode_options.append("Pack Simulation - not installed")
        mode_values.append("pack")
        mode_disabled.append(not liionpack_available)
        
        selected_mode_idx = st.selectbox(
            "Select Simulation Mode",
            options=range(len(mode_options)),
            format_func=lambda x: mode_options[x],
            index=0,
            help="Fast: Quick empirical model. High-Fidelity: Physics-based PyBaMM (pip install pybamm). Pack: Multi-cell pack simulation (pip install liionpack).",
            key="sim_mode",
        )
        
        # Check if selected mode is available
        if mode_disabled[selected_mode_idx]:
            st.warning(f"Selected mode requires additional packages. Using Fast mode instead.")
            simulation_mode = "fast"
        else:
            simulation_mode = mode_values[selected_mode_idx]
        
        # Mode-specific settings
        pybamm_model = "SPMe"
        pybamm_parameter_set = None
        pack_config = None
        
        if simulation_mode == "high_fidelity":
            with st.expander("PyBaMM Settings", expanded=True):
                pybamm_model = st.selectbox(
                    "Electrochemical Model",
                    options=["SPM", "SPMe", "DFN"],
                    index=1,
                    help="SPM: Fastest. SPMe: Balanced. DFN: Most accurate.",
                    key="pybamm_model",
                )
                
                param_sets = get_pybamm_parameter_sets()
                if param_sets:
                    param_names = ["Auto-detect"] + [p["name"] for p in param_sets]
                    selected_param = st.selectbox(
                        "Parameter Set",
                        options=param_names,
                        index=0,
                        help="Choose a validated PyBaMM parameter set or auto-detect from chemistry.",
                        key="pybamm_params",
                    )
                    if selected_param != "Auto-detect":
                        pybamm_parameter_set = selected_param
                
                st.info("High-fidelity mode uses physics-based models for more accurate simulations.")
        
        elif simulation_mode == "pack":
            with st.expander("Pack Configuration", expanded=True):
                # Standard pack presets
                std_packs = get_standard_packs()
                pack_names = ["Custom"] + [p["name"] for p in std_packs]
                
                selected_pack_preset = st.selectbox(
                    "Pack Preset",
                    options=pack_names,
                    index=0,
                    key="pack_preset",
                )
                
                if selected_pack_preset == "Custom":
                    pack_series = st.number_input(
                        "Cells in Series (Ns)",
                        min_value=1,
                        max_value=200,
                        value=14,
                        key="pack_series",
                    )
                    pack_parallel = st.number_input(
                        "Cells in Parallel (Np)",
                        min_value=1,
                        max_value=50,
                        value=4,
                        key="pack_parallel",
                    )
                    pack_variation = st.slider(
                        "Cell-to-Cell Variation (%)",
                        min_value=0,
                        max_value=10,
                        value=2,
                        key="pack_variation",
                    ) / 100.0
                else:
                    # Get preset values
                    preset = next((p for p in std_packs if p["name"] == selected_pack_preset), None)
                    if preset:
                        pack_series = preset["topology"]["series"]
                        pack_parallel = preset["topology"]["parallel"]
                        st.write(f"Series: {pack_series}, Parallel: {pack_parallel}")
                        st.write(f"Total cells: {preset['topology']['total_cells']}")
                        st.write(f"Energy: {preset['electrical']['energy_kwh']:.2f} kWh")
                    pack_variation = 0.02
                
                pack_config = {
                    "series": pack_series,
                    "parallel": pack_parallel,
                    "cell_variation": pack_variation,
                }
                
                st.info(f"Pack: {pack_series}s{pack_parallel}p ({pack_series * pack_parallel} cells)")
        
        st.markdown("---")
        
        st.subheader("2. Battery Chemistry")
        chemistry = st.selectbox(
            "Select Chemistry",
            options=["NMC811", "LFP", "NCA", "LTO"],
            index=0,
            help="Choose the battery chemistry type",
            key="single_chemistry",
        )
        
        chem_info = get_chemistry_info(chemistry)
        with st.expander("Chemistry Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nominal Voltage", f"{chem_info['voltage_nominal']} V")
                st.metric("Max Voltage", f"{chem_info['voltage_max']} V")
                st.metric("Min Voltage", f"{chem_info['voltage_min']} V")
            with col2:
                st.metric("Energy Density", f"{chem_info['energy_density']} Wh/kg")
                st.metric("Max Charge Rate", f"{chem_info['max_charge_rate']}C")
                st.metric("Max Discharge Rate", f"{chem_info['max_discharge_rate']}C")
        
        st.subheader("3. Cell Parameters")
        capacity = st.number_input(
            "Capacity (Ah)",
            min_value=0.1,
            max_value=100.0,
            value=3.0,
            step=0.1,
            help="Nominal cell capacity in Amp-hours",
            key="single_capacity",
        )
        
        temperature = st.number_input(
            "Temperature (C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=1.0,
            help="Operating temperature",
            key="single_temperature",
        )
        
        st.subheader("4. Test Protocol")
        protocol_type = st.selectbox(
            "Select Protocol",
            options=["Cycle Life", "Formation", "Rate Capability", "RPT"],
            index=0,
            help="Choose the test protocol type",
            key="single_protocol",
        )
        
        protocol_params = {"temperature": temperature}
        
        if protocol_type == "Cycle Life":
            with st.expander("Cycle Life Parameters", expanded=True):
                protocol_params["cycles"] = st.number_input(
                    "Number of Cycles", min_value=1, max_value=10000, value=10, step=1,
                    key="single_cycles",
                )
                protocol_params["charge_rate"] = st.slider(
                    "Charge Rate (C)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    key="single_charge_rate",
                )
                protocol_params["discharge_rate"] = st.slider(
                    "Discharge Rate (C)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    key="single_discharge_rate",
                )
                protocol_params["voltage_max"] = st.number_input(
                    "Max Voltage (V)", min_value=2.0, max_value=5.0, value=chem_info["voltage_max"], step=0.05,
                    key="single_voltage_max",
                )
                protocol_params["voltage_min"] = st.number_input(
                    "Min Voltage (V)", min_value=1.0, max_value=4.0, value=chem_info["voltage_min"], step=0.05,
                    key="single_voltage_min",
                )
                protocol_params["rest_time"] = st.number_input(
                    "Rest Time (s)", min_value=0, max_value=3600, value=300, step=60,
                    key="single_rest_time",
                )
        
        elif protocol_type == "Formation":
            with st.expander("Formation Parameters", expanded=True):
                protocol_params["cycles"] = st.number_input(
                    "Number of Cycles", min_value=1, max_value=10, value=3, step=1,
                    key="single_form_cycles",
                )
                protocol_params["initial_rate"] = st.slider(
                    "Initial C-Rate", min_value=0.05, max_value=0.5, value=0.1, step=0.05,
                    key="single_initial_rate",
                )
                protocol_params["voltage_max"] = chem_info["voltage_max"]
                protocol_params["voltage_min"] = chem_info["voltage_min"]
        
        elif protocol_type == "Rate Capability":
            with st.expander("Rate Capability Parameters", expanded=True):
                rates_input = st.text_input(
                    "Discharge Rates (C)", value="0.2, 0.5, 1.0, 2.0",
                    key="single_rates",
                )
                protocol_params["rates"] = [float(r.strip()) for r in rates_input.split(",")]
                protocol_params["cycles_per_rate"] = st.number_input(
                    "Cycles per Rate", min_value=1, max_value=10, value=2, step=1,
                    key="single_cycles_per_rate",
                )
                protocol_params["charge_rate"] = st.slider(
                    "Charge Rate (C)", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                    key="single_rate_charge",
                )
                protocol_params["cycles"] = len(protocol_params["rates"]) * protocol_params["cycles_per_rate"]
                protocol_params["voltage_max"] = chem_info["voltage_max"]
                protocol_params["voltage_min"] = chem_info["voltage_min"]
        
        elif protocol_type == "RPT":
            with st.expander("RPT Parameters", expanded=True):
                protocol_params["charge_rate"] = st.slider(
                    "Charge Rate (C)", min_value=0.1, max_value=1.0, value=0.33, step=0.01,
                    key="single_rpt_charge",
                )
                protocol_params["discharge_rate"] = st.slider(
                    "Discharge Rate (C)", min_value=0.1, max_value=1.0, value=0.33, step=0.01,
                    key="single_rpt_discharge",
                )
                protocol_params["pulse_current"] = st.slider(
                    "Pulse Current (C)", min_value=0.5, max_value=3.0, value=1.0, step=0.1,
                    key="single_pulse_current",
                )
                protocol_params["cycles"] = 1
        
        st.subheader("5. Output Settings")
        output_format = st.selectbox(
            "Output Format",
            options=["Generic", "Arbin", "Neware", "Biologic"],
            index=0,
            key="single_format",
        )
        
        with st.expander("Advanced Settings"):
            enable_degradation = st.checkbox(
                "Enable Degradation", value=True, key="single_degradation",
            )
            st.markdown("**Measurement Noise (Std Dev)**")
            noise_voltage = st.number_input(
                "Voltage Noise (V)", min_value=0.0, max_value=0.01, value=0.001,
                step=0.0001, format="%.4f", key="single_noise_v",
            )
            noise_current = st.number_input(
                "Current Noise (A)", min_value=0.0, max_value=0.05, value=0.005,
                step=0.001, format="%.3f", key="single_noise_i",
            )
            noise_temperature = st.number_input(
                "Temperature Noise (C)", min_value=0.0, max_value=2.0, value=0.5,
                step=0.1, key="single_noise_t",
            )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("Run Simulation", type="primary", use_container_width=True)
    
    if run_button:
        mode_label = {
            "fast": "Fast empirical",
            "high_fidelity": f"PyBaMM ({pybamm_model})",
            "pack": f"Pack ({pack_config['series'] if pack_config else 1}s{pack_config['parallel'] if pack_config else 1}p)",
        }.get(simulation_mode, "Fast")
        
        # Get cycle count for progress tracking
        total_cycles = protocol_params.get("cycles", 10)
        
        # Show detailed progress for high-fidelity and pack modes
        if simulation_mode in ["high_fidelity", "pack"]:
            # Create progress UI elements
            st.markdown("---")
            st.subheader("Simulation Progress")
            
            # Info box about expected duration
            if simulation_mode == "high_fidelity":
                model_speed = {"SPM": "fast", "SPMe": "medium", "DFN": "slow"}.get(pybamm_model, "medium")
                st.info(f"""
                **High-Fidelity Simulation Started**
                - Model: {pybamm_model} ({model_speed} speed)
                - Chemistry: {chemistry}
                - Cycles: {total_cycles}
                - PyBaMM simulations are more accurate but slower than fast mode.
                - Estimated time: {total_cycles * (1 if model_speed == 'fast' else 3 if model_speed == 'medium' else 10)} - {total_cycles * (3 if model_speed == 'fast' else 10 if model_speed == 'medium' else 30)} seconds
                """)
            else:
                pack_cells = (pack_config.get('series', 1) * pack_config.get('parallel', 1)) if pack_config else 1
                st.info(f"""
                **Pack Simulation Started**
                - Pack Configuration: {pack_config.get('series', 1)}s{pack_config.get('parallel', 1)}p ({pack_cells} cells)
                - Chemistry: {chemistry}
                - Cycles: {total_cycles}
                - Pack simulations model each cell individually.
                """)
            
            progress_bar = st.progress(0, text="Initializing simulation...")
            status_text = st.empty()
            metrics_container = st.empty()
            
            # Update progress during initialization
            progress_bar.progress(5, text="Loading PyBaMM model and parameters...")
            
            try:
                # Update progress
                progress_bar.progress(10, text=f"Starting {mode_label} simulation...")
                status_text.text(f"Running {total_cycles} cycles with {pybamm_model if simulation_mode == 'high_fidelity' else 'pack'} model...")
                
                results, df, output_path = run_simulation(
                    chemistry=chemistry,
                    capacity=capacity,
                    protocol_type=protocol_type,
                    protocol_params=protocol_params,
                    output_format=output_format,
                    enable_degradation=enable_degradation,
                    noise_voltage=noise_voltage,
                    noise_current=noise_current,
                    noise_temperature=noise_temperature,
                    simulation_mode=simulation_mode,
                    pybamm_model=pybamm_model,
                    pybamm_parameter_set=pybamm_parameter_set,
                    pack_config=pack_config,
                )
                
                # Complete progress
                progress_bar.progress(100, text="Simulation complete!")
                status_text.text(f"Completed {results.cycles_completed} cycles successfully.")
                
                # Show quick metrics
                with metrics_container.container():
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        st.metric("Cycles Done", results.cycles_completed)
                    with mc2:
                        st.metric("Capacity Retention", f"{results.capacity_retention:.1%}")
                    with mc3:
                        st.metric("Duration", f"{(results.end_time - results.start_time).total_seconds():.1f}s")
                
                st.session_state["results"] = results
                st.session_state["df"] = df
                st.session_state["run_complete"] = True
                st.session_state["simulation_mode"] = simulation_mode
                
                st.success(f"Simulation completed successfully! (Mode: {mode_label})")
                
                Path(output_path).unlink(missing_ok=True)
                
            except Exception as e:
                progress_bar.progress(100, text="Simulation failed")
                st.error(f"Simulation failed: {str(e)}")
                status_text.text(f"Error: {str(e)}")
                st.session_state["run_complete"] = False
        
        else:
            # Fast mode - use simple spinner
            with st.spinner(f"Running {mode_label} simulation... This may take a moment."):
                try:
                    results, df, output_path = run_simulation(
                        chemistry=chemistry,
                        capacity=capacity,
                        protocol_type=protocol_type,
                        protocol_params=protocol_params,
                        output_format=output_format,
                        enable_degradation=enable_degradation,
                        noise_voltage=noise_voltage,
                        noise_current=noise_current,
                        noise_temperature=noise_temperature,
                        simulation_mode=simulation_mode,
                        pybamm_model=pybamm_model,
                        pybamm_parameter_set=pybamm_parameter_set,
                        pack_config=pack_config,
                    )
                    
                    st.session_state["results"] = results
                    st.session_state["df"] = df
                    st.session_state["run_complete"] = True
                    st.session_state["simulation_mode"] = simulation_mode
                    
                    st.success(f"Simulation completed successfully! (Mode: {mode_label})")
                    
                    Path(output_path).unlink(missing_ok=True)
                    
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")
                    st.session_state["run_complete"] = False
    
    if st.session_state.get("run_complete", False):
        results = st.session_state["results"]
        df = st.session_state["df"]
        sim_mode = st.session_state.get("simulation_mode", "fast")
        
        st.markdown("---")
        st.header("Simulation Results")
        
        # Show backend info
        if hasattr(results, 'backend_info') and results.backend_info:
            backend_type = results.backend_info.get("type", "fast")
            if backend_type == "pybamm":
                st.info(f"Simulated with PyBaMM ({results.backend_info.get('model', 'SPMe')})")
            elif backend_type == "pack":
                pack_cfg = results.backend_info.get("pack_config", {})
                st.info(f"Pack simulation: {pack_cfg.get('series', 1)}s{pack_cfg.get('parallel', 1)}p")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Cycles Completed", f"{results.cycles_completed}")
        with col2:
            st.metric("Capacity Retention", f"{results.capacity_retention:.1%}",
                      delta=f"{(results.capacity_retention - 1) * 100:.2f}%")
        with col3:
            st.metric("Energy Throughput", f"{results.energy_throughput:.1f} Wh")
        with col4:
            st.metric("Final Resistance", f"{results.resistance_final * 1000:.1f} mOhm",
                      delta=f"+{(results.resistance_final / results.resistance_initial - 1) * 100:.1f}%")
        with col5:
            st.metric("Test Duration", f"{(results.end_time - results.start_time).total_seconds():.1f}s")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Degradation Analysis", "Cycle Details", "Raw Data", "Export"
        ])
        
        with tab1:
            if not results.cycle_summary.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_capacity_retention_chart(results.cycle_summary), use_container_width=True)
                with col2:
                    st.plotly_chart(create_efficiency_chart(results.cycle_summary), use_container_width=True)
        
        with tab2:
            max_cycle = int(df["cycle_index"].max()) if "cycle_index" in df.columns else 1
            selected_cycle = st.slider("Select Cycle to View", min_value=1, max_value=max_cycle, value=1)
            
            if "cycle_index" in df.columns:
                st.plotly_chart(create_combined_cycle_chart(df, selected_cycle), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_voltage_profile_chart(df[df["cycle_index"] <= min(5, max_cycle)]), use_container_width=True)
            with col2:
                if "temperature" in df.columns:
                    st.plotly_chart(create_temperature_chart(df[df["cycle_index"] <= min(5, max_cycle)]), use_container_width=True)
        
        with tab3:
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            st.markdown(f"**Total rows:** {len(df):,}")
            st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        with tab4:
            st.subheader("Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download CSV Data",
                    data=csv_buffer.getvalue(),
                    file_name=f"battery_simulation_{results.test_id}.csv",
                    mime="text/csv",
                )
            
            with col2:
                metadata = results.to_dict()
                st.download_button(
                    label="Download Metadata (JSON)",
                    data=json.dumps(metadata, indent=2, default=str),
                    file_name=f"battery_simulation_{results.test_id}_metadata.json",
                    mime="application/json",
                )
            
            if not results.cycle_summary.empty:
                csv_summary = io.StringIO()
                results.cycle_summary.to_csv(csv_summary, index=False)
                st.download_button(
                    label="Download Cycle Summary",
                    data=csv_summary.getvalue(),
                    file_name=f"battery_simulation_{results.test_id}_summary.csv",
                    mime="text/csv",
                )


# =============================================================================
# Page: Automated Data Generator
# =============================================================================

def page_automated_generator():
    """Automated data generator page."""
    st.markdown('<p class="main-header">Automated Data Generator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Generate batch battery data automatically with randomized parameters</p>',
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Battery Configuration")
        
        available_chemistries = ["NMC811", "LFP", "NCA", "LTO"]
        selected_chemistries = st.multiselect(
            "Select Chemistries (random selection per cell)",
            options=available_chemistries,
            default=["NMC811", "LFP"],
            help="Multiple chemistries will be randomly assigned to each cell",
        )
        
        if not selected_chemistries:
            st.warning("Please select at least one chemistry")
            selected_chemistries = ["NMC811"]
        
        num_cells = st.number_input(
            "Number of Cells to Simulate",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            help="Each cell will have slightly randomized parameters",
        )
        
        cycles_per_cell = st.number_input(
            "Cycles per Cell",
            min_value=1,
            max_value=1000,
            value=10,
            step=1,
        )
        
        st.subheader("2. Cell Parameters")
        
        base_capacity = st.number_input(
            "Base Capacity (Ah)",
            min_value=0.1,
            max_value=100.0,
            value=3.0,
            step=0.1,
        )
        
        base_temperature = st.number_input(
            "Base Temperature (C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=1.0,
        )
        
        charge_rate = st.slider(
            "Base Charge Rate (C)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )
        
        discharge_rate = st.slider(
            "Base Discharge Rate (C)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )
        
        cell_variation = st.slider(
            "Cell-to-Cell Variation (%)",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            help="Random variation in capacity, temperature, and rates",
        ) / 100.0
    
    with col2:
        st.subheader("3. Output Configuration")
        
        cycler_format = st.selectbox(
            "Cycler Data Format",
            options=["Generic", "Arbin", "Neware", "Biologic"],
            index=0,
        )
        
        export_method = st.radio(
            "Export Method",
            options=["Folder Export", "MQTT Publish"],
            index=0,
            help="Choose how to export the generated data",
        )
        
        if export_method == "Folder Export":
            output_folder = st.text_input(
                "Output Folder Path",
                value="./generated_data",
                help="Folder where CSV files will be saved",
            )
            
            st.info(f"Files will be saved to: {os.path.abspath(output_folder)}")
        
        else:  # MQTT
            st.subheader("MQTT Configuration")
            mqtt_broker = st.text_input("MQTT Broker", value="localhost")
            mqtt_port = st.number_input("MQTT Port", min_value=1, max_value=65535, value=1883)
            mqtt_topic = st.text_input("MQTT Topic", value="battery/simulation/data")
            
            with st.expander("MQTT Authentication (Optional)"):
                mqtt_username = st.text_input("Username", value="")
                mqtt_password = st.text_input("Password", value="", type="password")
        
        st.subheader("4. Generation Settings")
        
        enable_degradation = st.checkbox("Enable Degradation Model", value=True)
        
        generation_mode = st.radio(
            "Generation Mode",
            options=["Single Batch", "Scheduled Generation"],
            index=0,
        )
        
        if generation_mode == "Scheduled Generation":
            export_frequency = st.selectbox(
                "Export Frequency",
                options=["Every 1 minute", "Every 5 minutes", "Every 15 minutes", "Every 30 minutes", "Every hour"],
                index=1,
            )
            
            freq_map = {
                "Every 1 minute": 60,
                "Every 5 minutes": 300,
                "Every 15 minutes": 900,
                "Every 30 minutes": 1800,
                "Every hour": 3600,
            }
            frequency_seconds = freq_map[export_frequency]
            
            st.warning("Scheduled generation will run in the background. Use 'Stop Generator' to halt.")
    
    st.markdown("---")
    
    # Summary
    st.subheader("Generation Summary")
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Total Cells", num_cells)
    with summary_cols[1]:
        st.metric("Cycles per Cell", cycles_per_cell)
    with summary_cols[2]:
        st.metric("Total Cycles", num_cells * cycles_per_cell)
    with summary_cols[3]:
        st.metric("Chemistries", len(selected_chemistries))
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if generation_mode == "Single Batch":
            generate_button = st.button("Generate Data", type="primary", use_container_width=True)
        else:
            generate_button = st.button("Start Scheduled Generator", type="primary", use_container_width=True)
    
    with col2:
        if generation_mode == "Scheduled Generation":
            stop_button = st.button("Stop Generator", type="secondary", use_container_width=True)
            if stop_button:
                st.session_state["generator_running"] = False
                st.info("Generator stopped.")
    
    # Handle generation
    if generate_button:
        if generation_mode == "Single Batch":
            with st.spinner(f"Generating data for {num_cells} cells..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if export_method == "Folder Export":
                    results_list = []
                    
                    for i in range(num_cells):
                        status_text.text(f"Simulating cell {i+1}/{num_cells}...")
                        progress_bar.progress((i + 1) / num_cells)
                        
                        chemistry = random.choice(selected_chemistries)
                        cell_params = generate_random_cell_params(
                            base_capacity, base_temperature, cell_variation
                        )
                        chem_info = get_chemistry_info(chemistry)
                        
                        protocol_params = {
                            "temperature": cell_params["temperature"],
                            "cycles": cycles_per_cell,
                            "charge_rate": charge_rate * cell_params["charge_rate_variation"],
                            "discharge_rate": discharge_rate * cell_params["discharge_rate_variation"],
                            "voltage_max": chem_info["voltage_max"],
                            "voltage_min": chem_info["voltage_min"],
                            "rest_time": 300,
                        }
                        
                        os.makedirs(output_folder, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"cell_{i+1:03d}_{chemistry}_{timestamp}.csv"
                        output_path = os.path.join(output_folder, filename)
                        
                        try:
                            results, df, _ = run_simulation(
                                chemistry=chemistry,
                                capacity=cell_params["capacity"],
                                protocol_type="Cycle Life",
                                protocol_params=protocol_params,
                                output_format=cycler_format,
                                enable_degradation=enable_degradation,
                                noise_voltage=0.001,
                                noise_current=0.005,
                                noise_temperature=0.5,
                                output_path=output_path,
                            )
                            
                            results_list.append({
                                "Cell ID": i + 1,
                                "Chemistry": chemistry,
                                "Capacity (Ah)": f"{cell_params['capacity']:.2f}",
                                "Temperature (C)": f"{cell_params['temperature']:.1f}",
                                "Cycles": results.cycles_completed,
                                "Retention": f"{results.capacity_retention:.1%}",
                                "File": filename,
                                "Status": "Success",
                            })
                        except Exception as e:
                            results_list.append({
                                "Cell ID": i + 1,
                                "Chemistry": chemistry,
                                "Status": f"Failed: {str(e)[:30]}",
                            })
                    
                    progress_bar.progress(1.0)
                    status_text.text("Generation complete!")
                    
                    st.success(f"Generated {len(results_list)} cell datasets in {output_folder}")
                    
                    # Show results table
                    st.subheader("Generation Results")
                    results_df = pd.DataFrame(results_list)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Save summary
                    summary_path = os.path.join(output_folder, f"generation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    results_df.to_csv(summary_path, index=False)
                    st.info(f"Summary saved to: {summary_path}")
                
                else:  # MQTT
                    st.info("Generating and publishing to MQTT...")
                    
                    for i in range(num_cells):
                        status_text.text(f"Simulating and publishing cell {i+1}/{num_cells}...")
                        progress_bar.progress((i + 1) / num_cells)
                        
                        chemistry = random.choice(selected_chemistries)
                        cell_params = generate_random_cell_params(
                            base_capacity, base_temperature, cell_variation
                        )
                        chem_info = get_chemistry_info(chemistry)
                        
                        protocol_params = {
                            "temperature": cell_params["temperature"],
                            "cycles": cycles_per_cell,
                            "charge_rate": charge_rate * cell_params["charge_rate_variation"],
                            "discharge_rate": discharge_rate * cell_params["discharge_rate_variation"],
                            "voltage_max": chem_info["voltage_max"],
                            "voltage_min": chem_info["voltage_min"],
                            "rest_time": 300,
                        }
                        
                        try:
                            results, df, output_path = run_simulation(
                                chemistry=chemistry,
                                capacity=cell_params["capacity"],
                                protocol_type="Cycle Life",
                                protocol_params=protocol_params,
                                output_format=cycler_format,
                                enable_degradation=enable_degradation,
                                noise_voltage=0.001,
                                noise_current=0.005,
                                noise_temperature=0.5,
                            )
                            
                            # Publish summary to MQTT
                            mqtt_data = {
                                "cell_id": i + 1,
                                "chemistry": chemistry,
                                "capacity": cell_params["capacity"],
                                "temperature": cell_params["temperature"],
                                "cycles_completed": results.cycles_completed,
                                "capacity_retention": results.capacity_retention,
                                "timestamp": datetime.now().isoformat(),
                            }
                            
                            publish_to_mqtt(
                                mqtt_data,
                                mqtt_broker,
                                mqtt_port,
                                mqtt_topic,
                                mqtt_username if mqtt_username else None,
                                mqtt_password if mqtt_password else None,
                            )
                            
                            Path(output_path).unlink(missing_ok=True)
                            
                        except Exception as e:
                            st.warning(f"Cell {i+1} failed: {e}")
                    
                    progress_bar.progress(1.0)
                    status_text.text("MQTT publishing complete!")
                    st.success(f"Published {num_cells} cell datasets to MQTT topic: {mqtt_topic}")
        
        else:  # Scheduled Generation
            st.session_state["generator_running"] = True
            st.info(f"Scheduled generator started. Will generate every {export_frequency.lower()}.")
            st.warning("Note: In Streamlit, scheduled tasks won't persist after page reload. For production use, consider a separate background service.")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application with navigation."""
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Mode",
        options=["Single Simulation", "Automated Generator"],
        index=0,
    )
    
    st.sidebar.markdown("---")
    
    if page == "Single Simulation":
        page_single_simulation()
    else:
        page_automated_generator()
    
    # Footer with backend status
    st.markdown("---")
    
    # Show backend availability
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Fast Mode: Available")
    with col2:
        pybamm_status = "Available" if check_pybamm_available() else "Not installed"
        st.caption(f"PyBaMM: {pybamm_status}")
    with col3:
        liionpack_status = "Available" if check_liionpack_available() else "Not installed"
        st.caption(f"liionpack: {liionpack_status}")
    
    st.markdown(
        """
        <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
            Battery Test Data Simulator v1.1.0 | 
            <a href="https://github.com/arunnath011/BatterySimulator" target="_blank" class="footer-link">GitHub</a> |
            Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
