#!/usr/bin/env python3
import os
import json
import typer
import matplotlib.pyplot as plt
from parse_logs import parse_experiment_logs, download_remote_log, plot_experiments_variables

app = typer.Typer()

@app.command()
def plot(
    file_paths: list[str] = typer.Argument(
        ..., help="Paths to JSONL metrics log files (remote paths if --remote is given)"
    ),
    remote: str = typer.Option(
        None,
        "-r",
        "--remote",
        help="Remote host spec ([user@]host) to fetch file via scp",
    ),
    names: list[str] = typer.Option(
        None,
        "-n",
        "--name",
        help="Experiment name(s). Defaults to basename of file.",
    ),
    variables: list[str] = typer.Option(
        ["avg_reward", "avg_max_reward_in_group", "avg_output_tokens", "avg_kl_div"],
        "-v",
        "--variable",
        help="List of variables to plot.",
    ),
    x_key: str = typer.Option(
        "total_samples_accumulated",
        "-x",
        "--x_key",
        help="Key for x-axis values.",
    ),
    smooth: bool = typer.Option(
        False,
        "-s",
        "--smooth",
        help="Enable smoothing of the plotted lines.",
    ),
    smooth_window: int = typer.Option(
        10,
        "--smooth-window",
        help="Window size for smoothing (centered moving average).",
    ),
    output: str = typer.Option(
        None,
        "-o",
        "--output",
        help="Save path for output SVG file. If omitted, shows plot interactively.",
    ),
):
    """
    Plot specified variables from a JSONL metrics log.
    """
    # Handle multiple file paths and optional remote download
    local_paths = []
    for fp in file_paths:
        if remote:
            typer.echo(f"Fetching remote file {fp} from {remote}...")
            local_fp = download_remote_log(fp, remote)
        else:
            local_fp = fp
        local_paths.append(local_fp)

    # Print available keys for each experiment
    typer.secho("\nAvailable keys per experiment:", fg=typer.colors.CYAN, bold=True)
    for lp in local_paths:
        typer.secho(f"\n{os.path.basename(lp)}:", fg=typer.colors.YELLOW, bold=True)
        with open(lp, "r") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
        if not lines:
            typer.secho("  No records found.", fg=typer.colors.RED)
        else:
            first = json.loads(lines[0])
            for key in sorted(first.keys()):
                typer.secho(f"  • {key}", fg=typer.colors.GREEN)

    # Load and parse logs for each file
    steps_list = []
    for lp in local_paths:
        typer.echo(f"Loading metrics from {lp}")
        with open(lp, "r") as f:
            log_text = f.read()
        steps_list.append(parse_experiment_logs(log_text))

    # Default names to basenames if not provided
    if not names:
        names = [os.path.basename(lp) for lp in local_paths]
    elif len(names) != len(local_paths):
        typer.secho("Number of --name flags must match number of input files", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    fig = plot_experiments_variables(
        steps_list,
        names,
        variables,
        x_key=x_key,
        smooth=smooth,
        smooth_window=smooth_window,
        save_path=output,
    )
    # If no output file specified, display interactively
    if output is None:
        plt.show()


@app.command()
def info(
    file_path: str = typer.Argument(..., help="Path to JSONL metrics log file")
):
    """
    Print summary info about a JSONL metrics log (first record and its keys).
    """
    with open(file_path, "r") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    if not lines:
        typer.secho("No records found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    first = json.loads(lines[0])
    typer.echo("First record:")
    typer.echo(first)
    typer.echo("\nKeys:")
    for key in first.keys():
        typer.echo(f"- {key}")


if __name__ == "__main__":
    app() 

'''
python plot_cli.py plot \
  /new_data/experiments_rh/deepscaler_r1_qwen1.5b_1.2e-6_monitoring_v2/training_metrics.jsonl.bk \
  --remote-host rh-h100-02 \
  --remote-user lab \
  -n deepscaler_r1_distilled \
  -v avg_max_reward_in_group \
  -v avg_output_tokens \
  -v entropy \
  -v perc_truncated_samples \
  -v perc_with_0_advantage \
  -x total_samples_accumulated \
  -s \
  --smooth-window 40 \
  -o experiment_plots.svg
'''