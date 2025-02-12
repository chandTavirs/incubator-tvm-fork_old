import os
import glob

# Set the directory containing log files
log_directory = "logs/tuning_logs/vta_1x16x16"  # Change this to your log directory
output_file = "logs/tuning_logs/merged.log"  # Name of the output file

# Get a list of all .log files in the directory
log_files = sorted(glob.glob(os.path.join(log_directory, "*.log")))

# Merge all logs into one file
with open(output_file, "w") as outfile:
    for log_file in log_files:
        with open(log_file, "r") as infile:
            outfile.write(infile.read())  # Append content

print(f"All logs merged into {output_file}")