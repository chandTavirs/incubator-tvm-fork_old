import multiprocessing

import serial
import time
def reset_serial_port(port="/dev/ttyUSB3", baud=921600):
    ser = serial.Serial(port, baud)
    ser.write(b'Ppassword\n')
    time.sleep(2)
    # ser.write(b'password\n')
    ser.close()

def send_sampling_rate(port="/dev/ttyUSB3", baud=921600, sampling_rate=50000):
    ser = serial.Serial(port, baud)
    data = bytearray()
    data.append(0x53)  # 8'h50 (just the hexadecimal value 0x50)
    data.extend(sampling_rate.to_bytes(4, 'big'))
    data.append(0x0A)
    ser.write(data)
    ser.close()


def poll_serial_port(port="/dev/ttyUSB3", baud=921600, log_file="uart_sniffer_data/tmp.log"):
    ser = serial.Serial(port, baud, timeout=5)
    flag = False
    layer_count = 0
    # with open(log_file, 'w') as myfile:
    while True:
        byte_line = ser.read(20)
        chunks = [byte_line[i:i + 4] for i in range(0, len(byte_line), 4)]
        int_line = [int.from_bytes(bytearray(val), "big") for val in chunks]
        line = ':'.join(str(item) for item in int_line)
        if(line is not None and line != ""):
            flag = True
            layer_count += 1
            print(line)
            # if args.print_to_console:
            #     print(line)
        elif flag:
            # print("Layer count:: {}".format(layer_count))
            print("Layer count:: {}".format(layer_count))
            ser.close()
            break


port = '/dev/ttyUSB3'
baud = 921600

reset_serial_port(port=port, baud=baud)
send_sampling_rate(port=port, baud=baud, sampling_rate=250000)
# send_sampling_rate(port=port, baud=baud, sampling_rate=125000)

serial_read_process = multiprocessing.Process(target=poll_serial_port, args=(port, baud))

serial_read_process.start()
time.sleep(1)

print("starting polling subprocess for single exec")


time.sleep(1)
print("Exiting polling process...")



serial_read_process.join(10)


# reset_serial_port(port,baud)
if serial_read_process.is_alive():
    serial_read_process.terminate()

reset_serial_port(port=port, baud=baud)