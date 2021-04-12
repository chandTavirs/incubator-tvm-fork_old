#!/usr/bin/env python3

# Modified by contributors from Intel Labs

'''
VTA instruction cycle estimation.
'''

import sys

class cyclist():

  alu_insn = ('ADD', 'ADDI', 'MAX', 'MAXI', 'MIN', 'MINI', 'SHR', 'SHRI')
  mem_insn = ('INP', 'WGT', 'ACC', 'UOP', 'OUT')
  cycles_fix     = {'GEM': 1, 'ALU': 1, 'OUT': 0}
  cycles_stage   = {'GEM': 1, 'ALU': 5, 'OUT': 1.5}
  cycles_latency = {'GEM': 6, 'ALU': 5, 'OUT': 4}
  
  def __init__(self, trace):
    self.fn = trace
    self.prev = None
    self.cycle = 0
    self.pulse = {}
    self.stall = {}
    self.insn = {}
    self.iseq = []
    self.last = {}
    self.parse()
    self.analyze()

  def parse(self):
    fp = open(self.fn)
    for line in fp:
      tok = line.split()
      evt = tok[0]
      if evt == 'CYCLE':
        self.cycle = int(tok[1],16)
      elif evt.endswith('PULSE'):
        '''For some reasons load pulses ends one clock cycle after
        retirement, which should not happen. For now add the pulse
        to the last executed instruction.'''
        k = evt[1:4]
        self.pulse[k] = int(tok[1],16) + 1
        self.insn[self.last[k]]['epulse'] = self.pulse[k]
      elif evt == 'STALL':
        self.stall[tok[1]] = self.stall.get(tok[1], 0) + 1
      elif evt == 'EXE':
        if self.prev:
          self.iseq.append((self.prev, self.insn[self.prev]))
        self.prev = tok[2]
        self.insn[tok[2]] = {'E': self.cycle, 'I': tok[1]}
        if tok[1] in cycler.mem_insn:
          self.insn[tok[2]]['bpulse'] = self.pulse.get(tok[1], 0)
        elif tok[1] in cycler.alu_insn:
          self.last_alu = tok[2]
        self.last[tok[1]] = tok[2]
      elif evt == 'RET':
        self.insn[tok[2]]['R'] = self.cycle
        self.insn[tok[2]]['epulse'] = self.pulse.get(tok[1], 0)
      elif evt.endswith('LOOP'):
        N = int(tok[1],16)*int(tok[2],16)*(int(tok[4],16)-int(tok[3],16))
        last = self.last['GEM'] if evt.startswith('GEM') else self.last_alu
        self.insn[last]['iter'] = N

  def analyze(self):
    for k, v in self.iseq:
      I = v['I']
      cycles = v['R'] - v['E']
      latency = cycler.cycles_latency.get(I, 4)
      stage = cycler.cycles_stage.get(I, 1)
      fix = cycler.cycles_fix.get(I, 0)
      if I == 'GEM' or I == 'ALU':
        iterations = v['iter']
        estimate = fix + latency + (iterations-1) * stage
        print(I, iterations, cycles, estimate)
      elif I in cycler.mem_insn:
        pulses = v['epulse'] - v['bpulse']
        #estimate = fix + max(latency, iterations * stage)
        estimate = latency*((pulses+15)//16) + pulses * stage
        print(I, pulses, cycles, estimate)
    print('PULSE:', self.pulse)
    print('STALL:', self.stall)


if __name__ == '__main__':
  for trace in sys.argv[1:]:
    cyclist(trace)
