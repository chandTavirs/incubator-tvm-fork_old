import json
import os
import csv
import re
log_files_dir = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/profiling_data/rcg"
json_files_dir = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/profiling_results/ro_uart/rcg/archive"
final_samples = {}



for filename in os.listdir(log_files_dir):
    if filename.endswith('csvs'):
        continue
    network_id = '_'.join(filename.split('_')[:-1])
    if network_id in final_samples.keys():
        continue

    samples = []
    for sample_file in os.listdir(log_files_dir):
        sample_file_network = '_'.join(sample_file.split('_')[:-1])
        if network_id == sample_file_network:
            samples.append(sample_file)
    # samples = [sample_file for sample_file in os.listdir(log_files_dir) if network_id == '_'.join(sample_file.split('_')[:-1])]

    sorted_samples = sorted(samples, key=lambda x: os.path.getsize(os.path.join(log_files_dir, x)), reverse=True)

    if sorted_samples:
        final_samples[network_id] = sorted_samples[0]

broken_files = []

for filename in final_samples.values():
    with open(os.path.join(log_files_dir, filename), 'rb') as myfile:
        byte_array = bytearray()
        while read_line := myfile.read(1):
            byte_array.append(int.from_bytes(read_line, "big"))

    if len(byte_array) == 0:
        broken_files.append(filename)
        continue

    i = 0
    num_layers = 0
    ro_values = []
    while i < len(byte_array) - 1:
        current_bytes = byte_array[i:i + 2]
        value = int.from_bytes(current_bytes, byteorder='big', signed=False)

        if 18000 <= value <= 22000 or value == 55555:
            ro_values.append(value)
            i += 2
            if value == 55555:
                num_layers += 1
        else:
            # Find the next pair of bytes that form a valid integer
            j = i + 1
            while j < len(byte_array) - 1:
                next_bytes = byte_array[j:j + 2]
                next_value = int.from_bytes(next_bytes, byteorder='big', signed=True)
                if 18000 <= next_value <= 22000 or value == 55555:
                    i = j
                    break
                j += 2
            else:
                # If no valid integer is found, exit the loop
                break



    # if len(ro_values) < 10000:
    #     broken_files.append(filename)
    #     continue

    actual_layer_count = 0
    for json_filename in os.listdir(json_files_dir):
        if '_'.join(filename.split('_')[:-1]) == json_filename.split('.')[0]:
            with open(os.path.join(json_files_dir, json_filename), 'r') as myfile:
                json_file_data = json.load(myfile)
                layer_sequence = json_file_data['layer_sequence']
                actual_layer_count = layer_sequence.count('conv') + layer_sequence.count('maxpool')
                break

    if actual_layer_count != num_layers:
        broken_files.append(filename)
        continue

    csv_filename = filename.split('.')[0]+'.csv'
    with open(os.path.join(log_files_dir,'csvs','try',csv_filename),'w') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(['RO_VALUES'])

        for row in ro_values:
            writer.writerow([row])

print("Broken files : ",broken_files)
