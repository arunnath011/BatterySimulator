"""Command-line interface for battery simulator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from battery_simulator import BatterySimulator, CycleResult
from battery_simulator.chemistry import Chemistry
from battery_simulator.core.simulator import (
    SimulationConfig,
    TimingMode,
    StreamingMode,
    StreamingConfig,
)
from battery_simulator.protocols import Protocol


@click.group()
@click.version_option(version="1.0.0", prog_name="battery-simulator")
def main():
    """
    Battery Test Data Simulator
    
    Generate realistic lithium-ion battery cycling data for development,
    testing, and demonstration purposes.
    """
    pass


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML configuration file",
)
@click.option(
    "--chemistry",
    type=click.Choice(["NMC811", "LFP", "NCA", "LTO"], case_sensitive=False),
    default="NMC811",
    help="Battery chemistry type",
)
@click.option(
    "--capacity",
    type=float,
    default=3.0,
    help="Cell capacity in Ah",
)
@click.option(
    "--cycles",
    type=int,
    default=100,
    help="Number of cycles to simulate",
)
@click.option(
    "--protocol",
    type=click.Choice(["cycle_life", "formation", "rate_capability", "rpt"]),
    default="cycle_life",
    help="Test protocol type",
)
@click.option(
    "--charge-rate",
    type=float,
    default=1.0,
    help="Charge C-rate",
)
@click.option(
    "--discharge-rate",
    type=float,
    default=1.0,
    help="Discharge C-rate",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="./output/simulation.csv",
    help="Output file path",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["generic", "arbin", "neware", "biologic"]),
    default="generic",
    help="Output file format",
)
@click.option(
    "--timing",
    type=click.Choice(["instant", "accelerated", "real_time"]),
    default="instant",
    help="Timing mode for simulation",
)
@click.option(
    "--speed",
    type=float,
    default=100.0,
    help="Speed factor for accelerated mode",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bar",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--streaming",
    is_flag=True,
    help="Enable streaming mode - output data after each cycle completes",
)
@click.option(
    "--streaming-callback",
    type=click.Choice(["print", "json", "none"]),
    default="print",
    help="Streaming callback type (print: show progress, json: output JSON per cycle)",
)
def run(
    config: Path | None,
    chemistry: str,
    capacity: float,
    cycles: int,
    protocol: str,
    charge_rate: float,
    discharge_rate: float,
    output: Path,
    output_format: str,
    timing: str,
    speed: float,
    no_progress: bool,
    seed: int | None,
    streaming: bool,
    streaming_callback: str,
):
    """
    Run a battery simulation.
    
    Examples:
    
    \b
    # Basic simulation with defaults
    battery-simulator run
    
    \b
    # Custom chemistry and protocol
    battery-simulator run --chemistry LFP --cycles 500 --protocol cycle_life
    
    \b
    # Output in Arbin format
    battery-simulator run --format arbin --output data/test.csv
    
    \b
    # Using configuration file
    battery-simulator run --config config/my_test.yaml
    """
    click.echo("=" * 60)
    click.echo("Battery Test Data Simulator")
    click.echo("=" * 60)

    # Load configuration if provided
    if config:
        click.echo(f"Loading configuration from: {config}")
        from battery_simulator.utils.config_loader import load_config

        cfg = load_config(config)
        chemistry = cfg.cell.chemistry
        capacity = cfg.cell.capacity
        cycles = cfg.protocol.cycles
        protocol = cfg.protocol.type
        charge_rate = cfg.protocol.charge_rate
        discharge_rate = cfg.protocol.discharge_rate
        output_format = cfg.output.format
        timing = cfg.timing_mode
        speed = cfg.speed_factor
        seed = cfg.random_seed

    # Display simulation parameters
    click.echo(f"\nSimulation Parameters:")
    click.echo(f"  Chemistry: {chemistry}")
    click.echo(f"  Capacity: {capacity} Ah")
    click.echo(f"  Cycles: {cycles}")
    click.echo(f"  Protocol: {protocol}")
    click.echo(f"  Charge Rate: {charge_rate}C")
    click.echo(f"  Discharge Rate: {discharge_rate}C")
    click.echo(f"  Output Format: {output_format}")
    click.echo(f"  Output Path: {output}")
    click.echo(f"  Timing Mode: {timing}")
    click.echo(f"  Streaming Mode: {'Enabled' if streaming else 'Disabled'}")
    if seed:
        click.echo(f"  Random Seed: {seed}")
    click.echo("")

    # Create timing mode enum
    timing_mode = TimingMode(timing)
    
    # Create streaming config if enabled
    streaming_config = StreamingConfig(
        mode=StreamingMode.PER_CYCLE if streaming else StreamingMode.BATCH,
        flush_after_cycle=True,
    )

    # Create simulation config
    sim_config = SimulationConfig(
        timing_mode=timing_mode,
        speed_factor=speed,
        random_seed=seed,
        enable_degradation=True,
        streaming=streaming_config,
    )

    # Create simulator
    click.echo("Initializing simulator...")
    sim = BatterySimulator(
        chemistry=chemistry,
        capacity=capacity,
        config=sim_config,
    )

    # Create protocol
    click.echo("Creating protocol...")
    if protocol == "cycle_life":
        test_protocol = Protocol.cycle_life(
            charge_rate=charge_rate,
            discharge_rate=discharge_rate,
            cycles=cycles,
        )
    elif protocol == "formation":
        test_protocol = Protocol.formation(cycles=min(cycles, 5))
    elif protocol == "rate_capability":
        test_protocol = Protocol.rate_capability()
    elif protocol == "rpt":
        test_protocol = Protocol.rpt()
    else:
        test_protocol = Protocol.cycle_life(cycles=cycles)

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Define streaming callback if needed
    def print_cycle_callback(result: CycleResult) -> None:
        """Print cycle completion info."""
        metrics = result.cumulative_metrics
        click.echo(
            f"  Cycle {result.cycle_index}: "
            f"Retention={metrics['capacity_retention']:.2%}, "
            f"Resistance={metrics['resistance_current']*1000:.2f}mΩ, "
            f"Energy={metrics['total_energy']:.2f}Wh"
        )
    
    def json_cycle_callback(result: CycleResult) -> None:
        """Output JSON per cycle."""
        click.echo(json.dumps(result.to_dict()))

    # Run simulation
    click.echo("Running simulation...")
    
    if streaming:
        # Use streaming mode
        if streaming_callback == "print":
            sim.on_cycle_complete(print_cycle_callback)
        elif streaming_callback == "json":
            sim.on_cycle_complete(json_cycle_callback)
        
        # Run with streaming generator - exhaust the generator to complete
        generator = sim.run_streaming(
            protocol=test_protocol,
            output_path=output,
            output_format=output_format,
            show_progress=not no_progress and streaming_callback != "json",
        )
        
        # Iterate through all cycles
        try:
            while True:
                next(generator)
        except StopIteration as e:
            # Generator returns the final results
            results = e.value
        
        # Fallback: get results from simulator if generator didn't return them
        if results is None:
            results = sim.get_streaming_results()
    else:
        # Standard batch mode
        results = sim.run(
            protocol=test_protocol,
            output_path=output,
            output_format=output_format,
            show_progress=not no_progress,
        )

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("Simulation Complete!")
    click.echo("=" * 60)
    click.echo(f"\nResults Summary:")
    click.echo(f"  Test ID: {results.test_id}")
    click.echo(f"  Cycles Completed: {results.cycles_completed}")
    click.echo(f"  Capacity Retention: {results.capacity_retention:.2%}")
    click.echo(f"  Energy Throughput: {results.energy_throughput:.2f} Wh")
    click.echo(f"  Equivalent Full Cycles: {results.equivalent_full_cycles:.1f}")
    if results.failure_mode:
        click.echo(f"  Failure Mode: {results.failure_mode} (cycle {results.failure_cycle})")
    click.echo(f"\nOutput written to: {output}")

    # Save metadata
    metadata_path = output.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    click.echo(f"Metadata written to: {metadata_path}")


@main.command()
@click.option(
    "--chemistry",
    type=click.Choice(["NMC811", "LFP", "NCA", "LTO"], case_sensitive=False),
    default="NMC811",
    help="Battery chemistry type",
)
@click.option(
    "--cycles",
    type=int,
    default=500,
    help="Number of cycles",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="./demo_data",
    help="Output directory",
)
def demo(chemistry: str, cycles: int, output: Path):
    """
    Generate demo dataset for demonstrations.
    
    Creates a ready-to-use dataset showcasing battery cycling with
    realistic degradation.
    """
    click.echo("Generating demo dataset...")

    output.mkdir(parents=True, exist_ok=True)

    sim = BatterySimulator(chemistry=chemistry, capacity=3.0)
    protocol = Protocol.cycle_life(cycles=cycles)

    output_file = output / f"demo_{chemistry.lower()}_{cycles}cycles.csv"

    results = sim.run(
        protocol=protocol,
        output_path=output_file,
        output_format="arbin",
        show_progress=True,
    )

    click.echo(f"\nDemo dataset generated!")
    click.echo(f"  File: {output_file}")
    click.echo(f"  Cycles: {results.cycles_completed}")
    click.echo(f"  Final Capacity: {results.capacity_retention:.2%}")


@main.command()
def list_chemistries():
    """List available battery chemistries."""
    click.echo("\nAvailable Battery Chemistries:")
    click.echo("-" * 40)

    chemistries = [
        ("NMC811", "NMC811/Graphite - High energy density (~250 Wh/kg)"),
        ("LFP", "LFP/Graphite - Long cycle life, high safety"),
        ("NCA", "NCA/Si-Graphite - High power, fast charging"),
        ("LTO", "LTO/LMO - Ultra-long life (10,000+ cycles)"),
    ]

    for name, description in chemistries:
        chem = Chemistry.from_name(name)
        click.echo(f"\n  {name}")
        click.echo(f"    {description}")
        click.echo(f"    Voltage: {chem.voltage_min}-{chem.voltage_max}V")
        click.echo(f"    Capacity: {chem.capacity} Ah")


@main.command()
def list_protocols():
    """List available test protocols."""
    click.echo("\nAvailable Test Protocols:")
    click.echo("-" * 40)

    protocols = [
        ("cycle_life", "Standard cycling for degradation testing"),
        ("formation", "Low-rate initial cycling for SEI formation"),
        ("rate_capability", "Multi-rate discharge testing"),
        ("calendar_aging", "Storage aging with periodic checkups"),
        ("rpt", "Reference Performance Test (comprehensive)"),
    ]

    for name, description in protocols:
        click.echo(f"\n  {name}")
        click.echo(f"    {description}")


@main.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="./config/example_config.yaml",
    help="Output path for example config",
)
def init_config(output: Path):
    """Generate example configuration file."""
    example_config = """# Battery Simulator Configuration
# ================================

simulation:
  name: "NMC811 Cycle Life Test"
  output_dir: "./output"
  timing_mode: "instant"  # instant, accelerated, or real_time
  speed_factor: 1000      # For accelerated mode
  data_rate: 1.0          # Hz

cell:
  chemistry: "NMC811"     # NMC811, LFP, NCA, or LTO
  capacity: 3.0           # Ah
  form_factor: "cylindrical"

test_protocol:
  type: "cycle_life"
  cycles: 1000
  temperature: 25

  cycling:
    charge_rate: 1.0
    discharge_rate: 1.0
    voltage_max: 4.2
    voltage_min: 3.0
    rest_time: 300

  rpt_interval: 50        # RPT every 50 cycles

  end_conditions:
    capacity_retention: 0.80
    max_cycles: 1000

output:
  format: "arbin"         # generic, arbin, neware, or biologic
  cycle_data: true
  summary_data: true
  metadata: true

degradation:
  enable_capacity_fade: true
  enable_resistance_growth: true
  enable_sudden_failure: false
  failure_probability: 0.01

noise:
  voltage_noise: 0.001    # V (std dev)
  current_noise: 0.005    # A
  temperature_noise: 0.5  # °C
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(example_config)

    click.echo(f"Example configuration written to: {output}")
    click.echo("\nEdit this file and run:")
    click.echo(f"  battery-simulator run --config {output}")


if __name__ == "__main__":
    main()

