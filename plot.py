#!/usr/bin/env python3
import os
import json
import typer
import matplotlib.pyplot as plt
from parse_logs import parse_experiment_logs, download_remote_log, plot_experiments_variables

app = typer.Typer()

@app.command()
def plot(
    file_path: str = typer.Argument(
        ..., help="Path to JSONL metrics log file (remote path if --remote is given)"
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
    # Fetch remote or use local file
    if remote:
        typer.echo(f"Fetching remote file {file_path} from {remote}...")
        local_path = download_remote_log(file_path, remote)
    else:
        local_path = file_path

    typer.echo(f"Loading metrics from {local_path}")
    with open(local_path, "r") as f:
        log_text = f.read()

    steps = parse_experiment_logs(log_text)
    if not names:
        names = [os.path.basename(local_path)]

    fig = plot_experiments_variables(
        [steps],
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