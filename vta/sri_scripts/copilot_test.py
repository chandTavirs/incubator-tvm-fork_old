# read from serial port tty/USB3. baud rate 115200. write code
import serial


def read_serial():
    ser = serial.Serial('/dev/ttyUSB3', 115200)
    while True:
        print(ser.readline())
    ser.close()
