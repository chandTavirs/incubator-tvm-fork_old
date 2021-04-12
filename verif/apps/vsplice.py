#!/usr/bin/env python3

# Modified by contributors from Intel Labs

'''
Separate design modules contained in a design file
into individual module files.

usage: vsplice.py <design.v>
'''

import sys, os

design = sys.argv[1]
idle, module = 0, 1
state = idle
ctx = ''

with open(design) as design_fp:
  for line in design_fp:
    tok = line.split()
    if state == idle:
      if tok[0] == 'module':
        state = module
        name = tok[1].rstrip('(')
        fp = open(name+'.v', 'w')
        fp.write(line)
      else:
        ctx += line
    elif state == module:
      fp.write(line)
      if tok[0] == 'endmodule':
        fp.close()
        print(f'Spliced module {name}')
        state = idle

design = os.path.basename(design)
#if os.path.exists(design):
#  os.rename(design, design+'.org')
fp = open(design, 'w')
fp.write(ctx)
fp.close() 
if ctx:
  print(f'TOP context present in {design}')
