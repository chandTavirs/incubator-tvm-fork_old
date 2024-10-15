from pynq import MMIO
import argparse
import time

SLOT_0_OFFSET = 0x00
SLOT_1_OFFSET = 0x04
SLOT_2_OFFSET = 0x08
SLOT_3_OFFSET = 0x0C
SLOT_4_OFFSET = 0x10



parser = argparse.ArgumentParser(description='Poll trojan for side channel information')
parser.add_argument('--base-address', type=str, default="0xA0010000", help='base address of trojan peripheral')
parser.add_argument('--size', type=str, default="0x10000", help='size of trojan peripheral in memory')
parser.add_argument('--offset', type=str, default="0x0000",
                        help='offset of register to be read')
parser.add_argument('--enable', action='store_true', help='write 0x00000001 to ro control register to enable trojan')
parser.add_argument('--disable', action='store_true', help='write 0x00000000 to ro control register to disable trojan')

parser.add_argument('--poll', action='store_true', help='keep polling register')
parser.add_argument('--auto-stop', action='store_true', help='stop polling once model stops execution')
args = parser.parse_args()

base_addr = int(args.base_address, 16)
size = int(args.size, 16)
offset = int(args.offset, 16)

trojan = MMIO(base_addr, size)

flag = False
counter = 0
stop_counter = 20
loop_counter = 0 
slot_0_readings = []
slot_1_readings = []
slot_2_readings = []
slot_3_readings = []
slot_4_readings = []



if args.poll:
    print(['FETCH_READ_VOL','STORE_WRITE_VOL','LOAD_READ_VOL','COMPUTE_UOP_READ_VOL','COMPUTE_DATA_READ_VOL'].join('\t'))
    while True:
        #time.sleep(1/1000000.0)
        for i in range(10):
            loop_counter = loop_counter+1
        slot_0_data = trojan.read(SLOT_0_OFFSET)
        slot_1_data = trojan.read(SLOT_1_OFFSET)
        slot_2_data = trojan.read(SLOT_2_OFFSET)
        slot_3_data = trojan.read(SLOT_3_OFFSET)
        slot_4_data = trojan.read(SLOT_4_OFFSET)

        if slot_0_data != 0:
            flag = True
            slot_0_readings.append(slot_0_data)
        if slot_1_data != 0:
            slot_1_readings.append(slot_1_data)
        if slot_2_data != 0:
            slot_2_readings.append(slot_2_data)
        if slot_3_data != 0:
            slot_3_readings.append(slot_3_data)
        if slot_4_data != 0:
            slot_4_readings.append(slot_4_data)
        
        if flag and slot_0_data == 0 and slot_1_data == 0 and slot_2_data == 0 and slot_3_data == 0 and slot_4_data == 0:    
            counter += 1
            if counter >= stop_counter:
                break

    assert all(len(lst) == len(slot_0_readings) for lst in [slot_0_readings, slot_1_readings, slot_2_readings, slot_3_readings, slot_4_readings])

    for i, _ in enumerate(slot_0_readings):
        print([str(slot_0_readings[i]), str(slot_1_readings[i]), str(slot_2_readings[i]), str(slot_3_readings[i]), str(slot_4_readings[i])].join('\t'))




    

elif args.enable:
    print("Enabling ROs...")
    trojan.write(0x0004,0x00000001)
    print("RO control register value:: {}".format(trojan.read(0x0004)))
elif args.disable:
    print("Disabling ROs...")
    trojan.write(0x0004,0x00000000)
    print("RO control register value:: {}".format(trojan.read(0x0004)))    
else:
    print("Data read from {} :: {}".format(args.offset, trojan.read(offset)))
