
# Modified by contributors from Intel Labs

import re
import json

insts = []

with open( "simLOG", "rt") as fp:
    p_inst = re.compile( '^Inst: (.*)$')
    p_matrices = re.compile( '^(inp|acc_i|acc_o|wgt|uop): (.*)$')
    for line in fp:
        line = line.rstrip( '\n')
        m = p_matrices.match(line)
        if m:
            key = m.groups()[0]
            arr = m.groups()[1].split(' ')
            idx = arr[0]
            vec = arr[1:]
            insts[-1][key].append( {"idx": idx, "vec": vec})
            continue
        m = p_inst.match(line)
        if m:
            tbl = {}
            inst = m.groups()[0].split(' ')
            assert inst.pop(0) == 'reset:'
            tbl['reset'] = inst.pop(0)
            assert inst.pop(0) == 'uop_{begin,end}:'
            tbl['uop_begin'] = inst.pop(0)
            tbl['uop_end'] = inst.pop(0)
            assert inst.pop(0) == 'lp_{0,1}:'            
            tbl['lp_0'] = inst.pop(0)
            tbl['lp_1'] = inst.pop(0)
            assert inst.pop(0) == '{acc,inp,wgt}_{0,1}:'            
            tbl['acc_0'] = inst.pop(0)
            tbl['acc_1'] = inst.pop(0)
            tbl['inp_0'] = inst.pop(0)
            tbl['inp_1'] = inst.pop(0)
            tbl['wgt_0'] = inst.pop(0)
            tbl['wgt_1'] = inst.pop(0)
            assert not inst

            insts.append( {'inst': tbl, 'inp': [], 'acc_i': [], 'acc_o': [], 'wgt': [], 'uop': []})
            continue
        print( 'Extra:', line)

for (idx,inst) in enumerate(insts):
    with open( f"jsons/simLOG{idx:04d}.json", "wt") as fp:
        json.dump( inst, fp=fp, indent=2)
