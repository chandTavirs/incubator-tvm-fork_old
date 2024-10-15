import json
import os

json_files_dir = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/profiling_results/ro_uart/rcg/archive"
network_log = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/profiling_results/ro_uart/rcg/archive/networks_profiled_actual_2.log"


network_log_lines = []
for json_filename in os.listdir(json_files_dir):
    if json_filename.endswith('.json'):
        with open(os.path.join(json_files_dir, json_filename), 'r') as myfile:
            json_file_data = json.load(myfile)
            layer_sequence = json_file_data['network_nodes']
            network_log_lines.append(json_filename.split('.')[0]+"::"+layer_sequence+'\n')

with open(network_log,'w') as myfile:
    myfile.writelines(network_log_lines)