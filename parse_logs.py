from copy import deepcopy
import re
import numpy as np
import json
import os

def parse_experiment_logs(log_text):
    """
    Parses JSON Lines experiment logs into a list of dicts.
    Each line should be a valid JSON object.

    Parameters:
        log_text (str): The raw log text in JSON Lines format.

    Returns:
        List[dict]: A list where each dict is the parsed JSON object for each non-empty line.
    """
    return [json.loads(line) for line in log_text.splitlines() if line.strip()]

import matplotlib.pyplot as plt

def convert_to_float(value_str):
    """
    Converts a string value to float.
    It removes common extra characters such as brackets and units.
    Returns None if conversion fails.
    """
    if value_str is None:
        return None
    if isinstance(value_str, (int, float)):
        return float(value_str)
    # Remove brackets and common non-numeric characters.
    clean_str = value_str.replace("[", "").replace("]", "")
    # Split by whitespace in case there are extra words (like "seconds").
    tokens = clean_str.split()
    for token in tokens:
        try:
            return float(token)
        except ValueError:
            continue
    return None

def plot_experiments_variables(experiments, experiment_names, variables, x_key="samples trained on", smooth=False, smooth_window=10, save_path=None):
    """
    Plots specified variables for multiple experiments in vertically stacked subplots.
    
    Parameters:
      experiments (list of list of dict): Each element is an experiment represented as a list of step dictionaries.
      experiment_names (list of str): Names corresponding to each experiment.
      variables (list of str): List of variable keys to plot. Each variable will be plotted in its own subplot.
      x_key (str): Key to use for x-axis values (default: "samples trained on").
      smooth (bool): If True, plots a smoothed line (moving average) with the raw line made transparent.
      smooth_window (int): Number of steps over which to compute the moving average (default: 10).
      save_path (str, optional): If provided, saves the figure to this path. Should end with '.svg' for SVG format.
    
    Returns:
      matplotlib.figure.Figure: The created figure object.
    """
    if len(experiments) != len(experiment_names):
        raise ValueError("The number of experiments and experiment_names must be equal.")

    n_vars = len(variables)
    # Create subplots arranged vertically.
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4 * n_vars), sharex=True)
    
    # If only one variable is provided, ensure axes is iterable.
    if n_vars == 1:
        axes = [axes]
    
    for ax, variable in zip(axes, variables):
        for exp, name in zip(experiments, experiment_names):
            x_values = []
            y_values = []
            
            # Extract and convert x and y values from each step.
            for step in exp:
                x_val = convert_to_float(step.get(x_key, None))
                y_val = convert_to_float(step.get(variable, None))
                if x_val is not None and y_val is not None:
                    x_values.append(x_val)
                    y_values.append(y_val)
            
            # Sort the values by x-axis to ensure proper plotting.
            sorted_pairs = sorted(zip(x_values, y_values), key=lambda pair: pair[0])
            if sorted_pairs:
                x_values, y_values = zip(*sorted_pairs)
            else:
                continue
            
            # Plot raw data and optionally a smoothed version.
            if smooth and len(y_values) >= smooth_window:
                # Plot raw data with full opacity
                ax.plot(x_values, y_values, marker='o', label=f'{name} (raw)', alpha=0.3)
                
                # Only smooth the interior points where we have enough data
                half_window = smooth_window // 2
                
                # Create arrays for the interior points only
                interior_x = x_values[half_window:-half_window]
                interior_y = []
                
                # Calculate smoothed values for interior points only
                for i in range(half_window, len(y_values) - half_window):
                    # Use a centered window for each point
                    window_start = i - half_window
                    window_end = i + half_window + 1  # +1 because slice end is exclusive
                    interior_y.append(np.mean(y_values[window_start:window_end]))
                
                # Plot only the smoothed interior points
                ax.plot(interior_x, interior_y, label=f'{name} (smoothed)', linewidth=2)
            else:
                ax.plot(x_values, y_values, marker='o', label=name)
        
        # x-axis labels handled on the bottom subplot only
        ax.set_ylabel(variable)
        ax.set_title(f'{variable} vs {x_key}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    # hide x tick labels for all but the bottom subplot
    for ax in axes[:-1]:
        ax.label_outer()
    # adjust vertical spacing between subplots
    fig.subplots_adjust(hspace=0.3)
    # set shared x-axis label on bottom subplot
    axes[-1].set_xlabel(x_key)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

def download_remote_log(remote_file, remote):
    import subprocess
    import tempfile
    import os

    # Create a unique temporary file for this download
    name, ext = os.path.splitext(os.path.basename(remote_file))
    fd, local_file = tempfile.mkstemp(prefix=f"{name}_", suffix=ext)
    os.close(fd)
    scp_target = f"{remote}:{remote_file}"
    scp_cmd = ["scp", "-r", scp_target, local_file]
    print(f"Copying JSONL metrics to local: {' '.join(scp_cmd)}")
    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error copying metrics:", result.stderr)
    else:
        print("Successfully copied metrics to", local_file)
    return local_file

def get_steps_from_remote_log(remote_file, remote=None):
    """
    Retrieve steps from a JSONL metrics log. If 'remote' is provided,
    the file is fetched via scp from that alias; otherwise local file is used.
    """
    if remote:
        local_file = download_remote_log(remote_file, remote)
    else:
        local_file = remote_file
    with open(local_file, "r") as f:
        log_text = f.read()

    steps = parse_experiment_logs(log_text)
    return steps

if __name__ == "__main__":
    # Demo / ad-hoc usage (customize as needed)
    remote_file = "path/to/your/log.jsonl"
    exp_names = ["my_experiment"]

    steps = get_steps_from_remote_log(remote_file)
    # Demo plotting; customize variables and settings as needed
    svg_output_path = "experiment_plots.svg"
    fig = plot_experiments_variables(
        [steps],
        exp_names,
        x_key="total_samples_accumulated",
        variables=[
            "avg_max_reward_in_group",
            "avg_output_tokens",
            "entropy",
            "perc_truncated_samples",
            "perc_with_0_advantage",
        ],
        smooth=True,
        smooth_window=40,
        save_path=svg_output_path,
    )
    plt.figure(fig.number)
    plt.show()

    # Example summary calculation
    sum_of_time_taken = sum(convert_to_float(s.get("time_per_batch")) for s in steps)
    print(sum_of_time_taken / len(steps))

