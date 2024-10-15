import serial


def write_password(ser):
    ser.write(b'password\n')

def close_ser(ser):
    ser.close()

port = '/dev/ttyUSB3'
baud = 6000000

ser = serial.Serial(port, baud)
ser.timeout = 0


write_password(ser)

close_ser(ser)