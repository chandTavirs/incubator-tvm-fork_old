#!/usr/bin/env python3

# Modified by contributors from Intel Labs

'''
VTA instruction decoder.
'''

import ctypes
from vta.beh.datatype import * 

class decoder():

  opcode_to_insn = {
    0 : ('memory', 'LOAD'),
    1 : ('memory', 'STORE'),
    2 : ('gemm', 'GEMM'),
    3 : ('generic', 'FNSH'),
    4 : ('alu', 'ALU')
  }

  alu_opcode_to_insn = {
    0 : 'MIN',
    1 : 'MAX',
    2 : 'ADD',
    3 : 'SHX',
    4 : 'CLP',
    5 : 'MOV',
    6 : 'MUL'
  }

  memory_type_to_insn = {
    0 : 'UOP',
    1 : 'WGT',
    2 : 'INP',
    3 : 'ACC',
    4 : 'OUT'
  }

  def __init__(self, bits=128):
    self.bits = bits
    self.fmt = {}
    self._parse()

  def _parse(self, fmt='generic', spec=INSTRUCTION_FMT, fields=[]):
    i = fields
    for elem in spec:
      if isinstance(elem, tuple):
        i.append((elem[0], ctypes.c_uint64, int(elem[1])))
      elif isinstance(elem, dict):
        for k, v in elem.items():
          self._parse(k, v, list(i))
    self.fmt[fmt] = i

  def check(self):
    for k, v in self.fmt.items():
      size = 0
      print(k)
      for n, _, s in v:
        print(f'  {n:>16}: {s}')
        size += int(s)
      n = 'total'
      print(f'  {n:>16}: {size}')

  def _repr(self, fields):
    op, s = None, ''
    for f, _, _ in fields:
      v = getattr(self,f)
      if f == 'opcode':
        op = v
        v = f'{v} [{decoder.opcode_to_insn[v][1]}]'
      if f == 'alu_opcode':
        v = f'{v} [{decoder.alu_opcode_to_insn[v]}]'
      if f == 'memory_type':
        if op == 1: v = 4
        v = f'{v} [{decoder.memory_type_to_insn[v]}]'
      s += f'  {f:>16}: {v}\n'
    return s

dec = decoder()

class generic(ctypes.LittleEndianStructure, decoder):
  _fields_ = dec.fmt['generic']
  def __repr__(self):
    return self._repr(self._fields_)

class memory(ctypes.LittleEndianStructure, decoder):
  _fields_ = dec.fmt['memory']
  def __repr__(self):
    return self._repr(self._fields_)

class alu(ctypes.LittleEndianStructure, decoder):
  _fields_ = dec.fmt['alu']
  def __repr__(self):
    return self._repr(self._fields_)

class gemm(ctypes.LittleEndianStructure, decoder):
  _fields_ = dec.fmt['gemm']
  def __repr__(self):
    return self._repr(self._fields_)

class word(ctypes.LittleEndianStructure):
  _fields_ = [('lo', ctypes.c_uint64, 64),
              ('hi', ctypes.c_uint64, 64)]
  def __repr__(self):
    return f'{self.hi:016x}{self.lo:016x}'

class insn(ctypes.Union):
  _fields_ = [('generic', generic),
              ('memory', memory),
              ('alu', alu),
              ('gemm', gemm),
              ('word', word)]

  def __init__(self, blob):
    self.word.hi = blob >> 64
    self.word.lo = blob

  def __repr__(self):
    opcode = decoder.opcode_to_insn[self.generic.opcode][0]
    return str(getattr(self, opcode))


if __name__ == '__main__':
  import sys
  for b in sys.argv[1:]:
    blob = int(b, 16)
    i = insn(blob)
    print(i)
