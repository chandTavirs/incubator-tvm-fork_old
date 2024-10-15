import time

import serial

def reset_serial_port(port="/dev/ttyUSB3", baud=921600):
    ser = serial.Serial(port, baud)
    ser.write(b'password\n')
    time.sleep(2)
    ser.write(b'password\n')
    ser.close()

def write_password(port="/dev/ttyUSB3", baud=921600):
    ser = serial.Serial(port, baud)
    ser.write(b'password\n')

def print_serial_port(port="/dev/ttyUSB3", baud=921600):
    ser = serial.Serial(port, baud, timeout=5)
    flag = False
    layer_count = 0
    while True:
        byte_line = ser.read(20)
        chunks = [byte_line[i:i + 4] for i in range(0, len(byte_line), 4)]
        int_line = [int.from_bytes(bytearray(val), "big") for val in chunks]
        line = ':'.join(str(item) for item in int_line)
        if(line is not None and line != ""):
            flag = True
            layer_count += 1
            print(line+'\n')
            # if args.print_to_console:
            #     print(line)
        elif flag:
            # print("Layer count:: {}".format(layer_count))
            ser.close()
            break


# write_password()
print_serial_port()
