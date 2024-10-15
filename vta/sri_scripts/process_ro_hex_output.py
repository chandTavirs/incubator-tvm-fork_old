import csv
cur_pos = 0
ro_values = []
with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/resnet18_ro_with_6m_final.log', 'rb') as myfile:
    # while True:
    #
    #     read_byte = myfile.read(1)
    #     if read_byte == b'\n':
    #         cur_pos = myfile.tell()
    #
    #         if myfile.read(4).endswith(b'\n'):
    #             cur_pos = myfile.tell()
    #             break
    #         else:
    #             myfile.seek(cur_pos)
    # myfile.seek(cur_pos)
    byte_array = bytearray()
    while read_line := myfile.read(1):
        byte_array.append(int.from_bytes(read_line, "big"))

# ba = ba.split(b'\n')
#
# segments = [byte_array[i:i+2] for i in range(0, len(byte_array), 2)]
#
# # Print the segments
# for segment in segments:
#     print(int.from_bytes(segment, byteorder='big', signed=False))
#
# bla
print(len(byte_array))

i = 0
count_bound = 0
while i < len(byte_array) - 1:
    current_bytes = byte_array[i:i + 2]
    value = int.from_bytes(current_bytes, byteorder='big', signed=False)

    if 24000 <= value <= 28000 or value == 55555:
        ro_values.append(value)
        i += 2
        if value == 55555:
            count_bound += 1
    else:
        # Find the next pair of bytes that form a valid integer
        j = i + 1
        while j < len(byte_array) - 1:
            next_bytes = byte_array[j:j + 2]
            next_value = int.from_bytes(next_bytes, byteorder='big', signed=True)
            if 24000 <= next_value <= 28000 or value == 55555:
                i = j
                break
            j += 2
        else:
            # If no valid integer is found, exit the loop
            break

print(len(ro_values))
print("Num 55555: ", count_bound)

# for val in ba:
#     if len(val) == 2:
#         integer_ro = int.from_bytes(val, "big")
#         if integer_ro <= 2**16:
#             ro_values.append(integer_ro)
    # elif len(val) == 1:
    #     integer_ro = int.from_bytes(val+b'0a', "big")
    #     if integer_ro <= 2**16:
    #         ro_values.append(integer_ro)

with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/resnet18_ro_with_6m_final.csv', 'w') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(['RO_VALUE'])

    for row in ro_values:
        writer.writerow([row])