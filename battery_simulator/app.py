"""
Streamlit Web Application for Battery Test Data Simulator.

Run with: streamlit run battery_simulator/app.py
"""

from __future__ import annotations

import io
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from battery_simulator import BatterySimulator, Protocol
from battery_simulator.chemistry import Chemistry
from battery_simulator.core.simulator import SimulationConfig, TimingMode


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
    /* Header styling - uses CSS variables for theme compatibility */
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
    
    /* Tab styling with theme-aware colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    /* Ensure links are visible in both modes */
    .footer-link {
        opacity: 0.7;
    }
    
    .footer-link:hover {
        opacity: 1.0;
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
    # Plotly templates that work well with light/dark modes
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
    
    # Add 80% threshold line
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
    
    # Voltage
    fig.add_trace(
        go.Scatter(x=cycle_data["test_time"], y=cycle_data["voltage"], 
                   mode="lines", name="Voltage", line=dict(color="#1f77b4")),
        row=1, col=1
    )
    
    # Current
    fig.add_trace(
        go.Scatter(x=cycle_data["test_time"], y=cycle_data["current"],
                   mode="lines", name="Current", line=dict(color="#ff7f0e")),
        row=1, col=2
    )
    
    # SOC
    fig.add_trace(
        go.Scatter(x=cycle_data["test_time"], y=cycle_data["state_of_charge"] * 100,
                   mode="lines", name="SOC", line=dict(color="#2ca02c")),
        row=2, col=1
    )
    
    # Temperature
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
    
    # Add grid to all subplots
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
) -> tuple[Any, pd.DataFrame, str]:
    """Run the battery simulation with given parameters."""
    
    # Create simulation config
    config = SimulationConfig(
        timing_mode=TimingMode.INSTANT,
        enable_degradation=enable_degradation,
        noise_voltage=noise_voltage,
        noise_current=noise_current,
        noise_temperature=noise_temperature,
    )
    
    # Create simulator
    sim = BatterySimulator(
        chemistry=chemistry,
        capacity=capacity,
        temperature=protocol_params.get("temperature", 25.0),
        config=config,
    )
    
    # Create protocol based on type
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
    
    # Create temp file for output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        output_path = f.name
    
    # Run simulation
    results = sim.run(
        protocol=protocol,
        output_path=output_path,
        output_format=output_format.lower(),
        show_progress=False,
    )
    
    # Read the generated data
    df = pd.read_csv(output_path)
    
    # Clean up
    Path(output_path).unlink(missing_ok=True)
    
    return results, df, output_path


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">Battery Test Data Simulator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Generate realistic lithium-ion battery cycling data for development, testing, and demonstration</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("Simulation Configuration")
        
        # Chemistry Selection
        st.subheader("1. Battery Chemistry")
        chemistry = st.selectbox(
            "Select Chemistry",
            options=["NMC811", "LFP", "NCA", "LTO"],
            index=0,
            help="Choose the battery chemistry type",
        )
        
        # Display chemistry info
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
        
        # Cell Parameters
        st.subheader("2. Cell Parameters")
        capacity = st.number_input(
            "Capacity (Ah)",
            min_value=0.1,
            max_value=100.0,
            value=3.0,
            step=0.1,
            help="Nominal cell capacity in Amp-hours",
        )
        
        temperature = st.number_input(
            "Temperature (C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=1.0,
            help="Operating temperature",
        )
        
        # Protocol Selection
        st.subheader("3. Test Protocol")
        protocol_type = st.selectbox(
            "Select Protocol",
            options=["Cycle Life", "Formation", "Rate Capability", "RPT"],
            index=0,
            help="Choose the test protocol type",
        )
        
        # Protocol-specific parameters
        protocol_params = {"temperature": temperature}
        
        if protocol_type == "Cycle Life":
            with st.expander("Cycle Life Parameters", expanded=True):
                protocol_params["cycles"] = st.number_input(
                    "Number of Cycles",
                    min_value=1,
                    max_value=10000,
                    value=10,
                    step=1,
                    help="Total cycles to simulate",
                )
                protocol_params["charge_rate"] = st.slider(
                    "Charge Rate (C)",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                )
                protocol_params["discharge_rate"] = st.slider(
                    "Discharge Rate (C)",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                )
                protocol_params["voltage_max"] = st.number_input(
                    "Max Voltage (V)",
                    min_value=2.0,
                    max_value=5.0,
                    value=chem_info["voltage_max"],
                    step=0.05,
                )
                protocol_params["voltage_min"] = st.number_input(
                    "Min Voltage (V)",
                    min_value=1.0,
                    max_value=4.0,
                    value=chem_info["voltage_min"],
                    step=0.05,
                )
                protocol_params["rest_time"] = st.number_input(
                    "Rest Time (s)",
                    min_value=0,
                    max_value=3600,
                    value=300,
                    step=60,
                )
        
        elif protocol_type == "Formation":
            with st.expander("Formation Parameters", expanded=True):
                protocol_params["cycles"] = st.number_input(
                    "Number of Cycles",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                )
                protocol_params["initial_rate"] = st.slider(
                    "Initial C-Rate",
                    min_value=0.05,
                    max_value=0.5,
                    value=0.1,
                    step=0.05,
                )
                protocol_params["voltage_max"] = chem_info["voltage_max"]
                protocol_params["voltage_min"] = chem_info["voltage_min"]
        
        elif protocol_type == "Rate Capability":
            with st.expander("Rate Capability Parameters", expanded=True):
                rates_input = st.text_input(
                    "Discharge Rates (C)",
                    value="0.2, 0.5, 1.0, 2.0",
                    help="Comma-separated C-rates to test",
                )
                protocol_params["rates"] = [float(r.strip()) for r in rates_input.split(",")]
                protocol_params["cycles_per_rate"] = st.number_input(
                    "Cycles per Rate",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                )
                protocol_params["charge_rate"] = st.slider(
                    "Charge Rate (C)",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                )
                protocol_params["cycles"] = len(protocol_params["rates"]) * protocol_params["cycles_per_rate"]
                protocol_params["voltage_max"] = chem_info["voltage_max"]
                protocol_params["voltage_min"] = chem_info["voltage_min"]
        
        elif protocol_type == "RPT":
            with st.expander("RPT Parameters", expanded=True):
                protocol_params["charge_rate"] = st.slider(
                    "Charge Rate (C)",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.33,
                    step=0.01,
                )
                protocol_params["discharge_rate"] = st.slider(
                    "Discharge Rate (C)",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.33,
                    step=0.01,
                )
                protocol_params["pulse_current"] = st.slider(
                    "Pulse Current (C)",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                )
                protocol_params["cycles"] = 1
        
        # Output Configuration
        st.subheader("4. Output Settings")
        output_format = st.selectbox(
            "Output Format",
            options=["Generic", "Arbin", "Neware", "Biologic"],
            index=0,
            help="Choose the output file format",
        )
        
        # Advanced Settings
        with st.expander("Advanced Settings"):
            enable_degradation = st.checkbox(
                "Enable Degradation",
                value=True,
                help="Simulate capacity fade and resistance growth",
            )
            
            st.markdown("**Measurement Noise (Std Dev)**")
            noise_voltage = st.number_input(
                "Voltage Noise (V)",
                min_value=0.0,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f",
            )
            noise_current = st.number_input(
                "Current Noise (A)",
                min_value=0.0,
                max_value=0.05,
                value=0.005,
                step=0.001,
                format="%.3f",
            )
            noise_temperature = st.number_input(
                "Temperature Noise (C)",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1,
            )
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button(
            "Run Simulation",
            type="primary",
            use_container_width=True,
        )
    
    # Run simulation when button is clicked
    if run_button:
        with st.spinner("Running simulation... This may take a moment."):
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
                )
                
                # Store results in session state
                st.session_state["results"] = results
                st.session_state["df"] = df
                st.session_state["run_complete"] = True
                
                st.success("Simulation completed successfully!")
                
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                st.session_state["run_complete"] = False
    
    # Display results if available
    if st.session_state.get("run_complete", False):
        results = st.session_state["results"]
        df = st.session_state["df"]
        
        st.markdown("---")
        st.header("Simulation Results")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                "Cycles Completed",
                f"{results.cycles_completed}",
            )
        with col2:
            st.metric(
                "Capacity Retention",
                f"{results.capacity_retention:.1%}",
                delta=f"{(results.capacity_retention - 1) * 100:.2f}%",
            )
        with col3:
            st.metric(
                "Energy Throughput",
                f"{results.energy_throughput:.1f} Wh",
            )
        with col4:
            st.metric(
                "Final Resistance",
                f"{results.resistance_final * 1000:.1f} mOhm",
                delta=f"+{(results.resistance_final / results.resistance_initial - 1) * 100:.1f}%",
            )
        with col5:
            st.metric(
                "Test Duration",
                f"{(results.end_time - results.start_time).total_seconds():.1f}s",
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "Degradation Analysis",
            "Cycle Details",
            "Raw Data",
            "Export",
        ])
        
        with tab1:
            # Capacity retention chart
            if not results.cycle_summary.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        create_capacity_retention_chart(results.cycle_summary),
                        use_container_width=True,
                    )
                with col2:
                    st.plotly_chart(
                        create_efficiency_chart(results.cycle_summary),
                        use_container_width=True,
                    )
        
        with tab2:
            # Cycle selector
            max_cycle = int(df["cycle_index"].max()) if "cycle_index" in df.columns else 1
            selected_cycle = st.slider(
                "Select Cycle to View",
                min_value=1,
                max_value=max_cycle,
                value=1,
            )
            
            # Combined cycle chart
            if "cycle_index" in df.columns:
                st.plotly_chart(
                    create_combined_cycle_chart(df, selected_cycle),
                    use_container_width=True,
                )
            
            # Voltage and SOC profiles
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_voltage_profile_chart(df[df["cycle_index"] <= min(5, max_cycle)]),
                    use_container_width=True,
                )
            with col2:
                if "temperature" in df.columns:
                    st.plotly_chart(
                        create_temperature_chart(df[df["cycle_index"] <= min(5, max_cycle)]),
                        use_container_width=True,
                    )
        
        with tab3:
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            st.markdown(f"**Total rows:** {len(df):,}")
            st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        with tab4:
            st.subheader("Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download CSV Data",
                    data=csv_buffer.getvalue(),
                    file_name=f"battery_simulation_{results.test_id}.csv",
                    mime="text/csv",
                )
            
            with col2:
                # JSON metadata download
                metadata = results.to_dict()
                st.download_button(
                    label="Download Metadata (JSON)",
                    data=json.dumps(metadata, indent=2, default=str),
                    file_name=f"battery_simulation_{results.test_id}_metadata.json",
                    mime="application/json",
                )
            
            # Cycle summary download
            if not results.cycle_summary.empty:
                csv_summary = io.StringIO()
                results.cycle_summary.to_csv(csv_summary, index=False)
                st.download_button(
                    label="Download Cycle Summary",
                    data=csv_summary.getvalue(),
                    file_name=f"battery_simulation_{results.test_id}_summary.csv",
                    mime="text/csv",
                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
            Battery Test Data Simulator v1.0.0 | 
            <a href="https://github.com/your-org/BatterySimulator" target="_blank" class="footer-link">GitHub</a> |
            Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
