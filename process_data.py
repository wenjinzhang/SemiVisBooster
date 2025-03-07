import re
import numpy as np

file_name = 

# Read the content of the text file
with open(file_name, 'r') as file:
    lines = file.readlines()

# Regular expression pattern to capture iteration index and util_ratio
pattern = re.compile(r'\[(\d+) iteration .* train/util_ratio: ([0-9.]+)')

# Lists to store extracted information
iteration_indices = []
util_ratios = []

# Iterate through each line
for line in lines:
    match = pattern.search(line)
    if match:
        iteration_index = int(match.group(1))
        util_ratio = float(match.group(2))
        iteration_indices.append(iteration_index)
        util_ratios.append(util_ratio)

# Convert lists to NumPy arrays
iteration_indices_array = np.array(iteration_indices)
util_ratios_array = np.array(util_ratios)

# Save the arrays to a NumPy binary file (.npy)
np.save('iteration_util_data.npy', np.column_stack((iteration_indices_array, util_ratios_array)))