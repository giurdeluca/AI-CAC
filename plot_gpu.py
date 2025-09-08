"""
This script is used to plot GPU usage from nvidia-smi.
Tutorial:
Keep 2 terminals open, one to run the script and the other to monitor the GPU.

On the monitor terminal, run:
```
nvidia-smi --id=0 --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,nounits -lms 1000 > gpu_usage_report.txt
```
Running this command will continuously monitor and update the GPU metrics while the command is active. So stop it manually when you're done.
It will save a gpu_usage_report.txt in your current working directory.

Then, to plot results,
```
python plot_gpu.py /path/to/gpu_usage_report.txt /path/to/output_file.png
```
"""

import csv
import datetime
import matplotlib.pyplot as plt
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Parse GPU usage report and plot data.")
parser.add_argument('input_filename', type=str, help="Input CSV file containing GPU usage data")
parser.add_argument('output_filename', type=str, help="Path of the output PNG file")
args = parser.parse_args()

timestamps = []
utilizations = []
memory_used = []

with open(args.input_filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header line
    for row in reader:
        timestamp_str = row[0]
        utilization = float(row[1])
        memory = float(row[2])  # Column index 2 corresponds to memory.used

        # Convert timestamp string to datetime object
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S.%f")
        timestamps.append(timestamp)
        utilizations.append(utilization)
        memory_used.append(memory)

# Calculate elapsed time from the starting timestamp
elapsed_time = [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps]

# Plot GPU utilization and memory usage on the same plot
fig, ax1 = plt.subplots()

# Plot GPU utilization percentage
color = 'tab:blue'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('GPU Utilization (%)', color=color)
ax1.plot(elapsed_time, utilizations, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for GPU memory usage
ax2 = ax1.twinx()

# Plot GPU memory usage 
color = 'tab:red'
ax2.set_ylabel('GPU Memory Used (MiB)', color=color)
ax2.plot(elapsed_time, memory_used, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

# Adjust plot margins
fig.tight_layout()

# Save the plot as a PNG file without cutting off
output_filename = f"{args.output_filename}"
plt.savefig(output_filename, bbox_inches='tight')

# Show the plot
plt.show()
