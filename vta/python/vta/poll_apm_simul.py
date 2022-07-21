from axi_perf_mon_simul import APM
import time
import argparse


parser = argparse.ArgumentParser(description='AXI Performance Monitor Poller')
parser.add_argument('--slots', type=str, default="0",
                     help='the AXI perfmon slot to be read back. 0 - overall; 1 - fetch; 2 - load; 3 - compute uop; 4 - compute data; 5 - store')
parser.add_argument('--stop_count', type=int, default=20,
                     help='stop polling if <stop_count> consecutive readings are zero')
args = parser.parse_args()
slots = [int(slot) for slot in args.slots.split(',')]
apm = APM()


print('waiting for logging to start...')
print('*'*70)
prev_write = -1
prev_read = -1

count = 0
stop_count = 0
stop_count_max = args.stop_count
while True:
   #time.sleep(0.005)
   #apm.apm_metric_cnt_reset()
   #apm.apm_metric_cnt_enable()
   apm.read_metrics(slots)
   if apm.recording:
      curr_write = apm.total_write[0]
      curr_read = apm.total_read[0]
      count+=1
      if curr_write == prev_write and curr_read == prev_read:
         stop_count+=1
         if stop_count == stop_count_max:
            break
      else:
         stop_count=0
      prev_write = curr_write
      prev_read = curr_read 
      

for slot in slots:
    for i, write_bytes in enumerate(apm.write_bytes_records[slot]):
        print("slot {} write bytes = {}  read bytes = {}  write b/w = {}  read b/w = {}".format(slot, write_bytes, apm.read_bytes_records[slot][i],
                                                                                         apm.write_bw_records[slot][i], apm.read_bw_records[slot][i])) 


    print("*"*70)
    print("slot {} total write bytes = {} and total read bytes = {}".format(slot, apm.total_write[slot], apm.total_read[slot]))
    print("*"*70+"\n")

print("total samples = {}".format(count-stop_count_max))
