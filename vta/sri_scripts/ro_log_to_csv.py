import os
import csv

log_files_dir = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/profiling_data/rcg"

final_samples = {}

for filename in os.listdir(log_files_dir):
    wkl_name = '_'.join(filename.split('_')[:-1])
    if wkl_name in final_samples.keys():
        continue

    samples = [sample_file for sample_file in os.listdir(log_files_dir) if wkl_name in sample_file]

    sorted_samples = sorted(samples, key=lambda x: os.path.getsize(os.path.join(log_files_dir, x)), reverse=True)

    if sorted_samples:
        final_samples[wkl_name] = sorted_samples[0]

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
    ro_values = []
    while i < len(byte_array) - 1:
        current_bytes = byte_array[i:i + 2]
        value = int.from_bytes(current_bytes, byteorder='big', signed=False)

        if 18000 <= value <= 22000 or value == 55555:
            ro_values.append(value)
            i += 2
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



    if len(ro_values) == 0:
        broken_files.append(filename)
        continue

    csv_filename = filename.split('.')[0]+'.csv'
    with open(os.path.join(log_files_dir,'csvs',csv_filename),'w') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(['RO_VALUES'])

        for row in ro_values:
            writer.writerow([row])

print("Broken files : ",broken_files)
