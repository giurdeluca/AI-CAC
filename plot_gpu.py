import csv
import datetime
import matplotlib.pyplot as plt
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Parse GPU usage report and plot data.")
parser.add_argument('filename', type=str, help="Input CSV file containing GPU usage data")
args = parser.parse_args()
base_filename = os.path.splitext(args.filename)[0]

timestamps = []
utilizations = []
memory_used = []

with open(args.filename, 'r') as file:
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
output_filename = f"{base_filename}_plot.png"
plt.savefig(output_filename, bbox_inches='tight')

# Show the plot
plt.show()
