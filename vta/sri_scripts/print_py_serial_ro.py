import serial
from multiprocessing import Process, Value
import time
import csv


def reset_serial_port(port="/dev/ttyUSB3", baud=6000000):
    ser = serial.Serial(port, baud)
    ser.write(b'password\n')
    time.sleep(2)
    ser.write(b'password\n')
    ser.close()

def write_password(ser):
    ser.write(b'password\n')

def close_ser(ser):
    ser.close()


shared_flag = Value('i', 0)

def read_serial_port(ser):
    flag = False
    bytes_read = bytearray()
    # with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/raw_no_el_12m_334_non_block.log', 'wb') as myfile:
    while True:
        # byte_line = ser.read(4096)
        # if len(byte_line) > 0:
        #     flag = True
        #     #print(byte_line)
        #     myfile.write(byte_line)
        # elif flag:
        #     break
        if ser.in_waiting > 0 and shared_flag.value == 0:
            # flag = True
            read = ser.read(ser.in_waiting)
            bytes_read.extend(read)
            # elif flag:
            #     break
        elif shared_flag.value == 1:
            with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/raw_no_el_12m_334_non_block.log', 'wb') as myfile:
                myfile.write(bytes_read)
            break

port = '/dev/ttyUSB3'
baud = 12000000

ser = serial.Serial(port, baud)
ser.timeout = 0


read_process = Process(target=read_serial_port, args=(ser, ))

read_process.start()

write_password(ser)

time.sleep(0.5)

write_password(ser)

time.sleep(0.5)

shared_flag.value = 1

read_process.join(2)
if read_process.is_alive():
    print("terminating")
    read_process.terminate()

close_ser(ser)

cur_pos = 0
ro_values = []
with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/raw_no_el_12m_334_non_block.log', 'rb') as myfile:
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

# segments = [byte_array[i:i+2] for i in range(0, len(byte_array), 2)]
#
# # Print the segments
# for segment in segments:
#     print(int.from_bytes(segment, byteorder='big', signed=False))
#
# bla
print(len(byte_array))
i = 0
while i < len(byte_array) - 1:
    current_bytes = byte_array[i:i + 2]
    value = int.from_bytes(current_bytes, byteorder='big', signed=False)

    if 5000 <= value <= 7000:
        ro_values.append(value)
        i += 2
    else:
        # Find the next pair of bytes that form a valid integer
        j = i + 1
        while j < len(byte_array) - 1:
            next_bytes = byte_array[j:j + 2]
            next_value = int.from_bytes(next_bytes, byteorder='big', signed=True)
            if 5000 <= next_value <= 7000:
                i = j
                break
            j += 2
        else:
            # If no valid integer is found, exit the loop
            break

print(len(ro_values))

with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/raw_no_el_12m_334_non_block.csv', 'w') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(['RO_VALUES'])

    for row in ro_values:
        writer.writerow([row])


