from pynq import MMIO
import time

class APM():
    SAMPLE_REGISTER = 0x002C
    
    SLOT_ADDRESSES = {#0:{"wb":0x0100,"wt":0x0110,"wl":0x0120,"rb":0x0130,"rt":0x0140,"rl":0x0150},
                      0:{"wb":0x0200,"wt":0x0210,"wl":0x0220,"rb":0x0230,"rt":0x0240,"rl":0x0250},
                      1:{"wb":0x0260,"wt":0x0270,"wl":0x0280,"rb":0x0290,"rt":0x02a0,"rl":0x02b0},
                      2:{"wb":0x0600,"wt":0x0610,"wl":0x0620,"rb":0x0630,"rt":0x0640,"rl":0x0650},
                      3:{"wb":0x0660,"wt":0x0670,"wl":0x0680,"rb":0x0690,"rt":0x06a0,"rl":0x06b0},
		      4:{"wb":0x0800,"wt":0x0810,"wl":0x0820,"rb":0x0830,"rt":0x0840,"rl":0x0850},
                      5:{"wb":0x0860,"wt":0x0870,"wl":0x0880,"rb":0x0890,"rt":0x08a0,"rl":0x08b0},
                      6:{"wb":0x0a00,"wt":0x0a10,"wl":0x0a20,"rb":0x0a30,"rt":0x0a40,"rl":0x0a50}, 
                      7:{"wb":0x0a60,"wt":0x0a70,"wl":0x0a80,"rb":0x0a90,"rt":0x0aa0,"rl":0x0ab0}}

    CONTROL_REGISTER = 0x0300

    ENABLE_MASK = 0x00000001
    RESET_MASK = 0x00000002

    START_ADDRESS = 0x43C10000

    prev_samples = 0
    fpga_clk1 = 100 * 1000000
    perfmon = None
    write_bytes_records = {}
    read_bytes_records = {}
    write_bw_records = {}
    read_bw_records = {}
    total_write = {}
    total_read = {}
    recording = False
    def __init__(self):
        print("initializing APM...\n")
        for slot in range(6):
            self.write_bytes_records[slot] = []
            self.read_bytes_records[slot] = []
            self.write_bw_records[slot] = []
            self.read_bw_records[slot] = [] 
            self.total_write[slot] = 0
            self.total_read[slot] = 0
            
        self.init()
        self.apm_metric_cnt_reset()
        self.apm_metric_cnt_enable() 
        self.prev_samples = float(self.apm_read_reg(self.SAMPLE_REGISTER)) 
        
    def init(self):
        self.perfmon = MMIO(self.START_ADDRESS, 0x10000)
        
    def apm_metric_cnt_enable(self):
        self.perfmon.write_reg(self.CONTROL_REGISTER, self.ENABLE_MASK)
        #print("CONTROL REGISTER....",self.apm_read_reg(self.CONTROL_REGISTER))

    def apm_metric_cnt_reset(self):
        self.perfmon.write_reg(self.CONTROL_REGISTER, self.RESET_MASK)

    def apm_read_reg(self,offset):
        return self.perfmon.read_reg(offset)

    def apm_write_reg(self, offset, value):
        self.perfmon.write_reg(offset,value)

    def reset(self):
        for slot in range(6):
            self.write_bytes_records[slot] = []
            self.read_bytes_records[slot] = []
            self.write_bw_records[slot] = []
            self.read_bw_records[slot] = [] 
            self.total_write[slot] = 0
            self.total_read[slot] = 0
        self.apm_metric_cnt_reset()

    def read_metrics(self,slots):
            #while(True):
            self.init()
            #print("CONTROL REGISTER....",self.apm_read_reg(self.CONTROL_REGISTER))
            #if self.prev_samples == 0:
            #   time.sleep(1)
            #   self.prev_samples = self.apm_read_reg(self.SAMPLE_REGISTER)
            #   self.prev_samples = self.apm_read_reg(self.SAMPLE_REGISTER)
            cur_samples = float(self.apm_read_reg(self.SAMPLE_REGISTER))
            #print(self.prev_samples)
            #print(cur_samples)
            samples = cur_samples - self.prev_samples
            self.prev_samples = cur_samples
            seconds = samples / self.fpga_clk1
            
            #for i in range(1):
            
            for slot in slots:
                write_bytes = self.apm_read_reg(self.SLOT_ADDRESSES[slot]["wb"])
                write_bw = write_bytes / (seconds*1000*1000)
                write_lat_tot = self.apm_read_reg(self.SLOT_ADDRESSES[slot]["wl"])
                write_t = self.apm_read_reg(self.SLOT_ADDRESSES[slot]["wt"])
                read_bytes = self.apm_read_reg(self.SLOT_ADDRESSES[slot]["rb"])
                read_bw = read_bytes / (seconds*1000*1000)
                read_lat_tot = self.apm_read_reg(self.SLOT_ADDRESSES[slot]["rl"])
                read_t = self.apm_read_reg(self.SLOT_ADDRESSES[slot]["rt"])
        
                if write_lat_tot == 0 or read_lat_tot == 0:
                   write_lat = 0
                   read_lat = 0
                else:
                   write_lat = write_bytes / write_lat_tot
                   read_lat = read_bytes / read_lat_tot 

                if write_bytes != 0 or read_bytes != 0 or self.recording:
                   self.recording = True
                   self.total_read[slot] += read_bytes
                   self.total_write[slot] += write_bytes
                   #with open("/home/xilinx/apm_output.txt","a") as myfile:
                   #    myfile.write("SLOT_{}:: Write bytes {}, write transactions {}, write bw {bw:.3f} MBytes/sec, write latency average {lt}\n".format(slot,write_bytes, write_t,  bw=write_bw,
                   #                                                                       lt=write_lat))
                   #    myfile.write("SLOT_{}:: Read bytes {}, read transactions {}, read bw {bw:.3f} MBytes/sec, read latency average {lt}\n".format(slot,read_bytes, read_t, bw=read_bw,
                   #                                                                    lt=read_lat))
                   self.read_bytes_records[slot].append(read_bytes)
                   self.write_bytes_records[slot].append(write_bytes)
                   self.read_bw_records[slot].append(read_bw)
                   self.write_bw_records[slot].append(write_bw)
                   #print("slot {} write bytes = {}  read bytes = {}  write b/w = {}  read b/w = {}".format(slot, write_bytes, read_bytes, write_bw, read_bw)) 
